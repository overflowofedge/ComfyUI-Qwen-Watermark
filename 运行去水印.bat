@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if "%~1"=="" (
    python scripts\run_workflow.py samples\demo_watermarked.png --output outputs\demo
) else (
    python scripts\run_workflow.py "%~1"
)
pause
