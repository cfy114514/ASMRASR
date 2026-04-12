import torch
# ====================== 配置 ======================
config = {"path": {
    "pre": "0pre",
    "audio": "1audio",
    "vad": "2vad",
    "asr": "3asr",
    "model": "model",
    "ffmpeg": "ffmpeg",
}, "model": {
    "sep": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "vad": "4evergr8/pyannote-segmentation-3.0",
    "asr": "4evergr8/whisper-large-v2-translate-zh-v0.2-st-ct2",
}, "vad": {
    "min_duration_on": 0.3,
    "min_duration_off": 0.1,
    "sample_rate": 16000,
}, "sep": {
    "sample_rate": 44100,
    "chunk_duration":1800
# ====================== 设备与精度动态计算 ======================
}, "device": "cuda" if torch.cuda.is_available() else "cpu"}
config["compute_type"] = "float16" if config["device"] == "cuda" else "int8"