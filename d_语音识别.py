import os
import glob
from pathlib import Path
import pysrt
import faster_whisper.vad
import faster_whisper.transcribe
from unittest.mock import patch
from faster_whisper import WhisperModel
from settings import config

extensions = ("wav", "mp3", "aac")
pattern = os.path.join(config["path"]["audio"], "*.*")
files = [
    f for f in glob.glob(pattern)
    if f.lower().endswith(extensions)
]

if not files:
    print(f"在 {config['path']['audio']} 中未找到音频文件")
    exit(0)

audio_tasks = []
for file_str in files:
    file_path = Path(file_str)
    basename = file_path.stem
    # 构造 VAD 输入和 ASR 输出路径
    vad_log_path = os.path.join(config["path"]["vad"], f"{basename}.srt")
    asr_log_path = os.path.join(config["path"]["asr"], f"{basename}.srt")

    if not os.path.exists(vad_log_path):
        print(f"跳过：找不到 VAD 字幕文件 {vad_log_path}")
        continue
    audio_tasks.append((str(file_path), vad_log_path, asr_log_path))

print('设备:', config["device"], '类型:', config["compute_type"])


# 2. 定义补丁函数（用于注入你之前 VAD 得到的时间轴）
def get_custom_vad_patch(audio, vad_parameters=None):
    if hasattr(faster_whisper.vad, "_custom_segments"):
        segment_count = len(faster_whisper.vad._custom_segments)
        print(f"  [Patch Success] 注入来自 SRT 的 {segment_count} 个片段")
        return faster_whisper.vad._custom_segments
    print("  [Patch Warning] 未发现预设片段数据！")
    return []


# 3. 初始化 Whisper 模型
model = WhisperModel(
    model_size_or_path=config["model"]["asr"],
    device=config["device"],
    compute_type=config["compute_type"],
    download_root=config["path"]["model"]
)

# 4. 开始推理循环
for audio_path, vad_log_path, asr_log_path in audio_tasks:
    print(f"\n开始推理: {Path(audio_path).name}")

    # 加载 VAD 得到的片段
    vad_subs = pysrt.open(vad_log_path)
    custom_chunks = []
    sr = config["vad"]["sample_rate"]  # 16000

    for sub in vad_subs:
        custom_chunks.append({
            'start': int(sub.start.ordinal / 1000 * sr),
            'end': int(sub.end.ordinal / 1000 * sr)
        })

    # 临时挂载到 faster_whisper 模块上
    faster_whisper.vad._custom_segments = custom_chunks

    # 使用 patch 强制 Whisper 使用我们的时间轴，而不是让它重新检测 VAD
    with patch(
            "faster_whisper.transcribe.get_speech_timestamps",
            side_effect=get_custom_vad_patch
    ):
        segments, _ = model.transcribe(
            audio=audio_path,
            language="ja",
            task='translate',
            vad_filter=True,  # 必须开启，Patch 才会生效
            condition_on_previous_text=True,
            log_progress=True
        )

        asr_subs = pysrt.SubRipFile()
        for segment in segments:
            text = segment.text.strip()
            if not text: continue

            new_sub = pysrt.SubRipItem(
                index=len(asr_subs) + 1,
                start=pysrt.SubRipTime.from_ordinal(int(segment.start * 1000)),
                end=pysrt.SubRipTime.from_ordinal(int(segment.end * 1000)),
                text=text
            )
            asr_subs.append(new_sub)
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s]: {text}")
            asr_subs.save(asr_log_path, encoding='utf-8')

    print(f"处理完成，保存至: {asr_log_path}")

print("\n所有 ASR 任务处理完毕！")