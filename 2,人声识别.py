import os
import pysrt
import torch
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
#############################################################################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
path_audio="1audio"
path_vad="2vad"
path_model="model"
vad_model="4evergr8/pyannote-segmentation-3.0"
vad_min_duration_on=0.3
vad_min_duration_off=0.1
##############################################################################################################################
print('设备:', device)
vad = Model.from_pretrained(checkpoint=vad_model, cache_dir=path_model)
vad.to(torch.device(device))
vad_pipeline = VoiceActivityDetection(segmentation=vad)
vad_pipeline.instantiate({
    "min_duration_on": vad_min_duration_on,
    "min_duration_off": vad_min_duration_off,
})

for filename in os.listdir(path_audio):
    if not filename.endswith((".wav", ".mp3", ".flac")):
        continue
    audio_path = os.path.join(path_audio, filename)
    print(f"\n处理音频: {audio_path}")

    basename = os.path.splitext(filename)[0]
    vad_log_path = os.path.join(path_vad, f"{basename}.srt")
    """
        if os.path.exists(vad_log_path):
        print('VAD记录存在，跳过')
        continue
    """


    vad_result = vad_pipeline(audio_path)
    srt = pysrt.SubRipFile()

    for idx, (segment, _, score) in enumerate(vad_result.itertracks(yield_label=True), start=1):
        sub_item = pysrt.SubRipItem(
            index=idx,
            start=pysrt.SubRipTime.from_ordinal(int(segment.start * 1000)),
            end=pysrt.SubRipTime.from_ordinal(int(segment.end * 1000)),
            text=f"默认文本{idx}"
        )
        srt.append(sub_item)

    srt.save(vad_log_path)
    print(f"VAD记录写入: {vad_log_path}")