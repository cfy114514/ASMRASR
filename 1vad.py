import os
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import WhisperFeatureExtractor
import librosa
import pysrt
from config import path_model, vad_model, vad_positive, vad_silence, vad_speech, vad_negivative, path_sub, path_audio

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# ===== 加载模型 =====
repo_dir = snapshot_download(
    repo_id=vad_model,
    cache_dir=path_model  # 可选，自定义缓存目录
)
model_path = os.path.join(repo_dir, "model.onnx")
session = ort.InferenceSession(model_path)
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    vad_model,
    cache_dir=path_model
)

# ===== 遍历音频 =====
for filename in os.listdir(path_audio):
    if not filename.lower().endswith((".wav", ".mp3", ".flac")):
        continue

    audio_path = os.path.join(path_audio, filename)
    print(f"\n处理音频: {audio_path}")

    basename = os.path.splitext(filename)[0]
    srt_path = os.path.join(path_sub, f"1vad-{basename}.srt")
    """
        if os.path.exists(srt_path):
        print("字幕已存在，跳过")
        continue
    """

    # ===== 读取音频 =====
    audio, sr = librosa.load(audio_path, sr=16000)

    all_probs = []

    chunk_samples = int(30 * sr)

    # ===== 分块推理 =====
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]

        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        inputs = feature_extractor(chunk, sampling_rate=sr, return_tensors="np")

        outputs = session.run(
            None,
            {session.get_inputs()[0].name: inputs.input_features}
        )

        # sigmoid
        logits = outputs[0][0]
        probs = 1 / (1 + np.exp(-logits))

        all_probs.append(probs)

    # 拼接
    all_probs = np.concatenate(all_probs)

    frame_duration_s = 30 / len(probs)

    min_silence_frames = int(vad_silence / frame_duration_s)
    min_speech_frames = int(vad_speech / frame_duration_s)

    # ===== 状态机 =====
    speech_segments = []
    triggered = False
    start_frame = 0
    temp_end = 0

    for i, prob in enumerate(all_probs):
        # 开始
        if prob >= vad_positive and not triggered:
            triggered = True
            start_frame = i
            temp_end = 0
            continue

        # 处理中
        if triggered:
            if prob < vad_negivative:
                if temp_end == 0:
                    temp_end = i

                if i - temp_end >= min_silence_frames:
                    end_frame = temp_end

                    if end_frame - start_frame >= min_speech_frames:
                        speech_segments.append((
                            start_frame * frame_duration_s,
                            end_frame * frame_duration_s
                        ))

                    triggered = False
                    temp_end = 0
            else:
                temp_end = 0

    # 结尾
    if triggered:
        end_frame = len(all_probs)
        if end_frame - start_frame >= min_speech_frames:
            speech_segments.append((
                start_frame * frame_duration_s,
                end_frame * frame_duration_s
            ))

    # ===== 写 SRT =====
    # ===== 写 SRT =====
    srt_file = pysrt.SubRipFile()

    for idx, (start, end) in enumerate(speech_segments, 1):
        start_frame = int(start / frame_duration_s)
        end_frame = int(end / frame_duration_s)

        # 取该段所有帧的置信度
        segment_probs = all_probs[start_frame:end_frame]

        # 转成字符串（可控制精度）
        prob_text = " ".join(f"{p:.3f}" for p in segment_probs)

        srt_item = pysrt.SubRipItem(
            index=idx,
            start=pysrt.SubRipTime(seconds=start),
            end=pysrt.SubRipTime(seconds=end),
            text=prob_text
        )
        srt_file.append(srt_item)

    srt_file.save(srt_path)
