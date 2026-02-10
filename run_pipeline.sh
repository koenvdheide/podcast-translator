#!/bin/bash
# Example: Translate the original "Tell me a History" podcast

echo "============================================================"
echo "Podcast Translation Pipeline"
echo "============================================================"
echo

# Check dependencies
if ! python3 -c "import requests" &> /dev/null; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Check for API keys
if [ -z "$DEEPL_API_KEY" ]; then
    echo "WARNING: DEEPL_API_KEY is not set!"
    echo "Get a FREE DeepL API key at: https://www.deepl.com/pro-api"
    echo
fi

if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "WARNING: HUGGINGFACE_TOKEN is not set!"
    echo "Speaker diarization requires a token from: https://huggingface.co"
    echo
fi

# Translate the original podcast
python3 podcast_translator.py "https://tellmeahistory.net/podlove/file/181/s/webplayer/c/episode/tmah_063_Sich_ein_Prachtbild_machen.mp3" --title "tmah063 - Sich ein Prachtbild machen" --podcast "Tell me a History" --duration "01:48:34"
