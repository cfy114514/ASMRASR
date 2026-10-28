# Linux / 云端运行

新增的 `cloud_asr.py` 是独立的轻量入口，保留原 Windows 四阶段流程不变。
目标是 **日语音频 → 带时间轴的中文字幕 SRT**，不使用收费翻译 API。
请使用你有权处理的音频。输入音频在运行机器本地推理，不上传到翻译服务；首次运行会联网下载公开模型和依赖。

## 安装与运行

建议 Python 3.10–3.12、独立虚拟环境。先运行：

```bash
bash scripts/setup_cloud.sh
.venv-cloud/bin/python cloud_asr.py /path/to/japanese.wav --output-dir 3asr
```

支持单文件或目录（不递归），包括 WAV、MP3、FLAC、AAC、M4A、OGG、OPUS 和常见视频容器。
能否解码还取决于文件实际编码。PyAV 自带解码库，此入口不会安装或修改系统 FFmpeg。

默认 CPU / int8，4 个线程。GPU 需另行具备匹配 CTranslate2 的 CUDA/cuBLAS/cuDNN，不能靠安装脚本保证：

```bash
.venv-cloud/bin/python cloud_asr.py 1audio --device cuda --compute-type float16
```

首次运行下载约 3.09 GB 的日→中微调 Whisper 权重，另留依赖、缓存和音频空间。
CPU 长音频可能很慢；先用 30–60 秒样本测耗时与内存，再运行整段。没有承诺固定速度或最低内存。

## 与旧流程的区别

- 此入口使用 faster-whisper 自带的 **Silero VAD**，而非 Pyannote 分段或私有函数 monkey patch。
- 不加载 Torch、人声分离模型，不生成 `2vad` 中间 SRT；直接输出到 `3asr`。
- 适合清晰人声。低声耳语、背景音乐和长静音需试听核对。`--no-vad` 可比较漏字情况，但可能增加静音幻觉。
- 使用仓库原先的日→中模型，固定日语输入。`--model` 可给本地模型目录；替换成标准 Whisper 不能保证输出中文。
- `requirements-cloud.txt` 固定入口依赖，`requirements-cloud-lock.txt` 是本次 Linux/Python 3.12 环境的完整依赖快照；其他系统请先用入口依赖安装验证。
- 保留原安装器、脚本和 requirements.txt，旧流程仍可使用。

## 安全重跑与离线

```bash
python cloud_asr.py 1audio --dry-run              # 只检查路径，不下载、不推理
python cloud_asr.py input.wav --offline          # 模型已缓存或指定本地模型时
python cloud_asr.py input.wav --overwrite        # 明确覆盖既有同名字幕
python -m unittest discover -s tests -v
```

同目录同名不同扩展名输入会报错，防止覆盖；默认不覆盖既有字幕。
每个文件推理完成后原子写入 UTF-8 SRT，异常不会留下伪装成完成结果的半份字幕。
批处理前面已完成的文件会保留；重跑时请单独选择未完成文件或明确使用覆盖参数。
无检测到人声会生成空 SRT 并打印提示，不代表音频确实无语音。

## 验证边界

自动测试覆盖路径、格式、重复文件名、时间戳、错误恢复、UTF-8 输出和模型调用参数。
它们不等于日语识别/中文翻译质量验收；真实样本字幕仍需和音频核对。
