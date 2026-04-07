import os
import shutil
import subprocess
from audio_separator.separator import Separator
#############################################################################################################################
path_pre="0pre"
path_model="model"
path_audio="1audio"
sep_model="MDX23C-8KFFT-InstVoc_HQ_2.ckpt"
#############################################################################################################################
for filename in os.listdir(path_pre):
    if filename.lower().endswith((".wav", ".mp3", ".flac")):
        basename = os.path.splitext(filename)[0]

        slice_path = os.path.join(path_pre, f"{basename}-slice")
        split_path = os.path.join(path_pre, f"{basename}-split")
        audio_path = os.path.join(path_pre, filename)

        if not os.path.isfile(audio_path):
            continue

        # ===== 切片 =====
        if os.path.exists(slice_path):
            shutil.rmtree(slice_path)
        os.makedirs(slice_path)

        segment_length = 1200

        command = [
            "ffmpeg",
            "-i", audio_path,
            "-f", "segment",
            "-segment_time", str(segment_length),
            "-c", "copy",
            os.path.join(slice_path, "%03d.wav")
        ]
        subprocess.run(command, check=True)

        # ===== 分离 =====
        if not os.path.exists(split_path):
            os.makedirs(split_path)

        separator = Separator(
            model_file_dir=path_model,
            output_dir=split_path,
            output_single_stem="vocals",
            sample_rate=16000,
        )
        separator.load_model(model_filename=sep_model)

        for file in os.listdir(slice_path):
            if file.endswith(".wav"):
                slice_basename = os.path.splitext(file)[0]

                exists = any(name.startswith(slice_basename) for name in os.listdir(split_path))
                if exists:
                    print(f"跳过: {file}")
                    continue

                separator.separate(os.path.join(slice_path, file))

        # ===== 拼接 =====
        file_list = sorted(
            [f for f in os.listdir(split_path) if f.endswith(".wav")],
            key=lambda x: int(x[:3])
        )

        list_path = os.path.join(path_pre, f"{basename}_list.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for f_name in file_list:
                full_path = os.path.join(split_path, f_name)
                f.write(f"file '{full_path}'\n")

        output_path = os.path.join(path_audio, f"{basename}.wav")

        command = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            output_path
        ]
        subprocess.run(command, check=True)

        print(f"完成人声分离: {output_path}")


