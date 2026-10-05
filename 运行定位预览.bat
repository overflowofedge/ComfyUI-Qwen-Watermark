@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if "%~1"=="" (
    python scripts\run_workflow.py samples\demo_watermarked.png --detect-only --output outputs\detection
) else (
    python scripts\run_workflow.py "%~1" --detect-only --output outputs\detection
)
pause
