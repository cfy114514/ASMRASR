import os
import pysrt
import numpy as np
import faster_whisper.vad
from faster_whisper import WhisperModel
from config import asr_illusion, path_audio, path_sub, device, compute_type, asr_model, path_model, asr_language



print('设备:', device, '类型:', compute_type)

# 初始化模型
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
    vad_log_path = os.path.join(path_sub, f"1vad-{basename}.srt")
    asr_log_path = os.path.join(path_sub, f"{basename}.srt")

    if not os.path.exists(vad_log_path):
        print(f"跳过：找不到 VAD 字幕文件 {vad_log_path}")
        continue

    # 1. 读取 SRT 并转换为采样点格式 (Whisper 默认 16000Hz)
    vad_subs = pysrt.open(vad_log_path)
    custom_chunks = []
    for sub in vad_subs:
        custom_chunks.append({
            'start': int(sub.start.ordinal / 1000 * 16000),
            'end': int(sub.end.ordinal / 1000 * 16000)
        })

    # 将解析好的时间戳压入补丁变量中
    faster_whisper.vad._custom_segments = custom_chunks

    print(f"\n开始推理: {audio_path}")

    # 2. 调用 transcribe
    # 注意：必须设置 vad_filter=True 才能触发我们的补丁函数
    segments, _ = model.transcribe(
        audio=audio_path,
        language=asr_language,
        task="translate",
        vad_filter=True,  # 触发补丁
        condition_on_previous_text=True,  # 保证上下文连续
        log_progress=True

    )

    asr_subs = pysrt.SubRipFile()

    # 3. 迭代处理结果
    for segment in segments:
        text = segment.text.strip()
        if not text or text in asr_illusion:
            continue

        # 因为使用了 VAD 注入模式，这里的 segment.start 就是绝对时间（秒）
        new_sub = pysrt.SubRipItem(
            index=len(asr_subs) + 1,
            start=pysrt.SubRipTime.from_ordinal(int(segment.start * 1000)),
            end=pysrt.SubRipTime.from_ordinal(int(segment.end * 1000)),
            text=text
        )
        asr_subs.append(new_sub)
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s]: {text}")

    # 保存结果
    asr_subs.save(asr_log_path, encoding='utf-8')
    print(f"处理完成，保存至: {asr_log_path}")