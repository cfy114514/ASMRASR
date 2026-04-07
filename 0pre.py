import os
import subprocess


path_pre="0pre"
os.system("chcp 65001")

for filename in os.listdir(path_pre):
    if filename.lower().endswith((".mp4", ".mkv", ".avi", ".mov", ".flv")):
        video_path = os.path.join(path_pre, filename)

        if os.path.isfile(video_path):
            basename = os.path.splitext(filename)[0]
            audio_output_path = os.path.join(path_pre, f"{basename}.wav")

            if os.path.exists(audio_output_path):
                print(f"已存在音频，跳过: {audio_output_path}")
                continue

            command = [
                "ffmpeg",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                audio_output_path
            ]

            subprocess.run(command, check=True)
            print(f"已提取音频: {audio_output_path}")


