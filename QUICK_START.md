# Quick Start Guide - Podcast Translation Pipeline

## 🎉 Now Using FREE DeepL API!

This pipeline has been updated to use **DeepL's FREE API** instead of paid services.

## ✅ What You Need

1. **DeepL API Key** (FREE - 500k characters/month)
2. **Hugging Face Token** (FREE - for speaker diarization)
3. **FFmpeg** (for audio processing)
4. **Python 3.8+**

## 🚀 Setup (5 minutes)

### Step 1: Get Free DeepL API Key

1. Visit: https://www.deepl.com/pro-api
2. Click "Sign up for free"
3. Choose **"DeepL API Free"**
4. Verify your email
5. Copy your API key (ends with `:fx`)

### Step 2: Install FFmpeg

**Windows (as administrator):**
```bash
choco install ffmpeg
```

Or run:
```bash
install_ffmpeg.bat
```

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

### Step 3: Get Free Hugging Face Token (for Speaker Diarization)

1. Visit: https://huggingface.co and sign up (free)
2. Accept model agreements:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Create token at: https://huggingface.co/settings/tokens
4. Copy your token

### Step 4: Set Your API Keys

**Windows:**
```bash
set DEEPL_API_KEY=your-key-here:fx
set HUGGINGFACE_TOKEN=your-hf-token-here
```

**Mac/Linux:**
```bash
export DEEPL_API_KEY=your-key-here:fx
export HUGGINGFACE_TOKEN=your-hf-token-here
```

### Step 5: Run the Pipeline

```bash
run_pipeline.bat
```

## ⏱️ Timeline

1. **Download**: 2-5 minutes (88MB file)
2. **Transcribe**: 30-60 minutes (local Whisper)
3. **Speaker Diarization**: 1.5-3 minutes (Pyannote)
4. **Translate**: 1-2 minutes (DeepL API)
5. **Format**: 1 second (automatic)

**Total**: ~1 hour

## 💰 Cost

**$0** - Completely FREE! 🎉

- Transcription: FREE (local Whisper)
- Speaker Diarization: FREE (Pyannote + Hugging Face)
- Translation: FREE (DeepL free tier)

## 📁 Output Files

Everything saved to `podcast_output/`:

- `tmah063_german.mp3` - Downloaded audio
- `transcript_german.txt` - German transcription
- `diarization.json` - Speaker identification data
- `transcript_english.txt` - English translation (raw)
- `transcript_english_formatted.txt` - **English translation (formatted)** ⭐

**💡 Best for reading**: `transcript_english_formatted.txt` - includes speaker labels, timestamps, and proper paragraphs!

## 🔍 Verify Setup

Before running the pipeline:

```bash
python check_requirements.py
```

This checks:
- ✅ Python version
- ✅ FFmpeg installation
- ✅ Required packages
- ✅ API keys configuration (DeepL + Hugging Face)

## ❓ Troubleshooting

### FFmpeg not found
```bash
# Install FFmpeg first
install_ffmpeg.bat  # Windows (as admin)
```

### DeepL API key issues
- Make sure your key ends with `:fx` (free tier)
- Verify your email address at DeepL
- Check you selected "DeepL API Free" not "DeepL Pro"

### Hugging Face token issues
- Make sure you've accepted the model agreements (see links above)
- Token should be created at https://huggingface.co/settings/tokens
- Token needs read access (default)
- **Note**: Pipeline will continue without diarization if token is missing

### Quota exceeded
- Free tier: 500k characters/month
- This podcast: ~200k characters
- You can translate 2-3 podcasts per month for free

## 📊 Character Usage

| Item | Characters |
|------|-----------|
| This podcast | ~200,000 |
| Free tier limit | 500,000/month |
| **Remaining** | **~300,000** |

You have room for 1-2 more podcasts this month!

## 🎯 Next Steps

1. **First time?**
   - Run `check_requirements.py` to verify setup
   - Install any missing components

2. **Ready to go?**
   - Make sure FFmpeg is installed
   - Set `DEEPL_API_KEY` and `HUGGINGFACE_TOKEN`
   - Run `run_pipeline.bat`

3. **Need help?**
   - Check [README_podcast_translator.md](README_podcast_translator.md)
   - Verify API key at: https://www.deepl.com/account/summary

## 🌟 Why DeepL?

✅ **FREE** - 500k characters/month at no cost
✅ **High Quality** - Excellent for German→English
✅ **Fast** - Processes large chunks efficiently
✅ **Easy Setup** - No credit card required
✅ **Perfect for this use case** - Made for translation

## 📝 Files in This Directory

- **run_pipeline.bat** - Start here! (Windows)
- **podcast_translator.py** - Main translation script
- **check_requirements.py** - Verify your setup
- **install_ffmpeg.bat** - FFmpeg installer
- **README_podcast_translator.md** - Full documentation

## 🔄 What's Different?

### Changed:
- ❌ ~~Anthropic Claude API~~ (paid)
- ✅ **DeepL API** (FREE!)

### Stays the same:
- ✅ Local Whisper transcription
- ✅ High-quality translation
- ✅ Same output format
- ✅ Resume capability

## 🎊 Ready to Start!

### Quick Examples

```bash
# 1. Translate the original podcast
run_pipeline.bat  # Windows
./run_pipeline.sh # Mac/Linux

# 2. Translate your own audio file
python podcast_translator.py "your_audio.mp3"

# 3. Translate from URL
python podcast_translator.py "https://example.com/podcast.mp3"

# 4. With metadata
python podcast_translator.py "audio.mp3" --title "Episode 5" --podcast "My Show"
```

Your translated audio will be ready in the `podcast_output/` directory! 🚀
