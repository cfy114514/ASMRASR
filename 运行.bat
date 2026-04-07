@echo off
chcp 65001
setlocal
.venv\Scripts\python.exe 2,人声识别.py
.venv\Scripts\python.exe 3,生成字幕.py
popd
pause
endlocal