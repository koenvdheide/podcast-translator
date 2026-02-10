@echo off
REM Example: Translate the original "Tell me a History" podcast

echo ============================================================
echo Podcast Translation Pipeline
echo ============================================================
echo.

REM Translate the original podcast
python podcast_translator.py "https://tellmeahistory.net/podlove/file/181/s/webplayer/c/episode/tmah_063_Sich_ein_Prachtbild_machen.mp3" --title "tmah063 - Sich ein Prachtbild machen" --podcast "Tell me a History" --duration "01:48:34"

pause
