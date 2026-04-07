import os
import pysrt
import faster_whisper.vad
import faster_whisper.transcribe  # 必须显式导入，以便 patch 定位
from unittest.mock import patch
import torch
from faster_whisper import WhisperModel
##############################################################################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type ="int8"
path_model="model"
asr_language="ja"
asr_model="4evergr8/whisper-large-v2-translate-zh-v0.2-st-ct2"
path_audio="1audio"
path_vad="2vad"
path_asr="3asr"



print('设备:', device, '类型:', compute_type)

def get_custom_vad_patch(audio, vad_parameters=None):
    """
    这个函数会被注入到 faster_whisper.transcribe 内部。
    """
    # 打印测试：如果补丁生效，你会看到这行输出
    if hasattr(faster_whisper.vad, "_custom_segments"):
        segment_count = len(faster_whisper.vad._custom_segments)
        print(f"  [Patch Success] 补丁函数触发：已注入来自 SRT 的 {segment_count} 个片段")
        return faster_whisper.vad._custom_segments

    print("  [Patch Warning] 补丁函数已触发，但未发现预设片段数据！")
    return []
model = WhisperModel(
    model_size_or_path=asr_model,
    device=device,
    compute_type=compute_type,
    download_root=path_model
)

for filename in os.listdir(path_audio):
    if not filename.endswith((".wav", ".mp3", ".flac")):
        continue

    audio_path = os.path.join(path_audio, filename)
    basename = os.path.splitext(filename)[0]
    vad_log_path = os.path.join(path_vad, f"{basename}.srt")
    asr_log_path = os.path.join(path_asr, f"{basename}.srt")

    if not os.path.exists(vad_log_path):
        print(f"跳过：找不到 VAD 字幕文件 {vad_log_path}")
        continue

    # 1. 解析 SRT 得到采样点区间
    vad_subs = pysrt.open(vad_log_path)
    custom_chunks = []
    for sub in vad_subs:
        custom_chunks.append({
            'start': int(sub.start.ordinal / 1000 * 16000),
            'end': int(sub.end.ordinal / 1000 * 16000)
        })

    # 将数据存入模块变量，供补丁函数读取
    faster_whisper.vad._custom_segments = custom_chunks
    print(f"\n开始推理: {audio_path}")
    with patch("faster_whisper.transcribe.get_speech_timestamps", side_effect=get_custom_vad_patch):

        segments, _ = model.transcribe(
            audio=audio_path,
            language=asr_language,
            task="translate",
            vad_filter=True,  # 必须为 True 才会触发调用
            condition_on_previous_text=True,
            log_progress=True
        )

        asr_subs = pysrt.SubRipFile()

        # 3. 迭代处理流式输出的结果
        for segment in segments:
            text = segment.text.strip()
            # 创建新的 SRT 条目
            new_sub = pysrt.SubRipItem(
                index=len(asr_subs) + 1,
                start=pysrt.SubRipTime.from_ordinal(int(segment.start * 1000)),
                end=pysrt.SubRipTime.from_ordinal(int(segment.end * 1000)),
                text=text
            )
            asr_subs.append(new_sub)
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s]: {text}")
            # 保存 ASR 结果
            asr_subs.save(asr_log_path, encoding='utf-8')


    print(f"处理完成，保存至: {asr_log_path}")