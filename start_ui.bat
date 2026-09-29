@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

REM ============================================================
REM  FinResearch-Agent 可视化界面 · 双击启动
REM
REM  为什么要有这个文件：
REM  本机装了不止一个 Python，PyCharm 默认挑的那个里面
REM  一个第三方包都没有，直接运行会报
REM      ModuleNotFoundError: No module named 'langgraph'
REM  看着像代码错，其实只是解释器选错了。
REM
REM  这个 bat 写死了装了依赖的那个解释器路径，
REM  所以不依赖 PyCharm 设置，双击就能用。
REM ============================================================

set "PY=C:\Users\17667\AppData\Local\Programs\Python\Python312\python.exe"

if not exist "%PY%" (
    echo.
    echo [错误] 找不到 Python 3.12：
    echo        %PY%
    echo.
    echo 请修改本文件里的 PY 变量，指向你机器上装了依赖的 python.exe。
    echo.
    pause
    exit /b 1
)

echo 使用解释器：%PY%
echo 正在启动，浏览器会自动打开...
echo.

"%PY%" web_ui.py

echo.
echo 服务已停止。
pause
