
<div align="center">
  <img src="https://api.iconify.design/material-symbols:movie.svg" height="36" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:music-note.svg" height="36" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:noise-control-on.svg" height="36" style="margin: 0 0px;">

  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 10px;">
  <img src="https://api.iconify.design/material-symbols:language-chinese-pinyin.svg" height="36" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:subtitles.svg" height="36" style="margin: 0 0px;">
</div>


<div align="center">
  <img src="https://api.iconify.design/material-symbols:music-note.svg" height="36" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:noise-control-on.svg" height="36" style="margin: 0 0px;">

  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:language-chinese-pinyin.svg" height="36" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:double-arrow.svg" height="20" style="margin: 0 0px;">
  <img src="https://api.iconify.design/material-symbols:subtitles.svg" height="36" style="margin: 0 0px;">

</div>


<h3 align="center">ASMRASR</h3>

  <p align="center">
    用机器学习为DL上的日语音频生成中文字幕
    <br>
    同样适用于日语视频的汉化
    <br>
    <a href="https://github.com/4evergr8/asmrasr/issues/new">🐞故障报告</a>
    ·
    <a href="https://github.com/4evergr8/asmrasr/issues/new">🏹功能请求</a>
  </p>




## 使用方法
### 本地运行（无需独显，但是会很慢）
* 安装Python3.10
* 打包下载此仓库，使用Pycharm打开项目
* 0pre文件夹放置待处理视频和混合音频,1audio文件夹存放纯净人声音频
* 安装依赖后依次运行所有代码
* 在3asr文件夹中获取识别结果


### 云端运行（不稳定，调试中）
* 点击<a href="https://colab.research.google.com/github/4evergr8/ASMRASR/blob/main/Colab.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" width="80">
</a>在Colab中打开项目
* 选择GPU运行时，点击全部运行，并允许访问云盘文件
* 将要处理的混合音频和视频上传至0pre文件夹
* 将要处理的纯净音频上传至1audio文件夹
* 要等狠久

## 主要功能
* 为日语音频生成中文字幕
* 支持视频提取音频,人声背景分离
* 无需独显，核显运行
* 支持MP3和WAV



## 项目原理
* 使用demucs提取人声
* 用Pyannote裁剪出人声,写入srt文件
* 使用faster-whisper进行带时间戳转写,对其中的VAD实现进行替换,换成之前的识别结果
* 输出srt文件
## 感谢
* chickenrice0721的Whisper微调模型
* faster-whisper和Pyannote
* Google的Colab帮了我很大的忙，Kaggle就是一坨

