@echo off
REM Install FFmpeg on Windows

echo ============================================================
echo FFmpeg Installation Helper
echo ============================================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if errorlevel 1 (
    echo This script requires administrator privileges.
    echo Please right-click and select "Run as administrator"
    pause
    exit /b 1
)

echo Checking if Chocolatey is installed...
where choco >nul 2>&1
if errorlevel 1 (
    echo.
    echo Chocolatey is not installed.
    echo.
    echo Option 1: Install Chocolatey ^(recommended^)
    echo   Visit: https://chocolatey.org/install
    echo   Or run in PowerShell ^(as admin^):
    echo   Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ^^^(^^^(New-Object System.Net.WebClient^^^).DownloadString^^^('https://community.chocolatey.org/install.ps1'^^^)^^^)
    echo.
    echo Option 2: Install FFmpeg manually
    echo   1. Download from: https://ffmpeg.org/download.html
    echo   2. Extract to C:\ffmpeg
    echo   3. Add C:\ffmpeg\bin to your PATH
    echo.
    pause
    exit /b 1
)

echo.
echo Installing FFmpeg with Chocolatey...
choco install ffmpeg -y

echo.
echo Checking FFmpeg installation...
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo.
    echo ⚠ FFmpeg installation may have failed.
    echo Please close and reopen your terminal, then try: ffmpeg -version
    echo.
) else (
    echo.
    echo ✓ FFmpeg installed successfully!
    ffmpeg -version
    echo.
)

pause
