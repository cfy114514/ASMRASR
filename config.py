import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type ="int8"
path_audio="audio"
path_sub="sub"
path_model="model"
vad_model="4evergr8/pyannote-segmentation-3.0" # VAD模型，来自Pyannote，“修复了原版模型强制登陆的bug”
vad_min_duration_on= 0.3
vad_min_duration_off= 1.25
asr_language="ja"
asr_model="chickenrice0721/whisper-large-v2-translate-zh-v0.2-lt-ct2"
asr_prompt="はぁっ…んっ…あんっ…お耳の中、れろれろって舐めちゃうね…じゅるっ、じゅぽっ…もっと気持ちいい声、出して？ はあはあ…んふぅ…"
asr_illusion = {"ご視聴ありがとうございました"}
