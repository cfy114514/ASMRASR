@echo off
chcp 65001
setlocal
.venv\Scripts\python.exe 0,提取音频.py
.venv\Scripts\python.exe 1,人声分离.py
.venv\Scripts\python.exe 2,人声识别.py
.venv\Scripts\python.exe 3,生成字幕.py
popd
pause
endlocal