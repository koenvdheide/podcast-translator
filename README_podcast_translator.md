# Podcast Translation Pipeline: German to English

This pipeline downloads, transcribes, and translates the German podcast "Tell me a History" (tmah063) to English using FREE DeepL API.

## Episode Details
- **Title**: tmah063 Sich ein Prachtbild machen
- **Duration**: 01:48:34
- **Size**: ~88 MB
- **Guest**: Dr. Christiane Czygan (Istanbul Orient Institute)
- **Topic**: 1554 illuminated manuscript with poetry by Ottoman Sultan Süleyman I

## Features

1. **Download**: Fetches the MP3 file from the podcast website
2. **Transcribe**: Converts German audio to text using OpenAI Whisper
3. **Translate**: Translates German transcript to English using **DeepL API FREE** ✅
4. **Resume**: Automatically skips completed steps if you re-run

## Prerequisites

### Required:
- **DeepL API key** for translation (FREE tier available!)
  - Get your FREE API key from: https://www.deepl.com/pro-api
  - Choose "DeepL API Free" - 500,000 characters/month
  - Set as environment variable: `DEEPL_API_KEY`

### For Transcription (choose one):

#### Option 1: Local Whisper (Recommended for this 88MB file)
- Python 3.8+
- FFmpeg installed on your system
- Sufficient disk space (~5GB for Whisper large model)
- No API costs for transcription

#### Option 2: OpenAI Whisper API
- OpenAI API key
- **Note**: The file is 88MB, which exceeds Whisper API's 25MB limit
- You'll need to split the audio file first

## Installation

### 1. Install Python dependencies

**Windows:**
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Mac/Linux:**
```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg (required for audio processing)

**Windows:**
```bash
# Using chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

### 3. For local Whisper (recommended for this 88MB file)

**Windows:**
```bash
python -m pip install openai-whisper
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Mac/Linux:**
```bash
pip install openai-whisper torch
```

**Note**: For GPU support on Windows, visit https://pytorch.org/get-started/locally/ to get the correct PyTorch installation command for your CUDA version.

## Usage

### Basic Usage

```bash
# Translate a German podcast from URL
python podcast_translator.py "https://example.com/podcast.mp3"

# Translate a local audio file
python podcast_translator.py "my_audio.mp3"

# Translate with custom languages
python podcast_translator.py "audio.mp3" --source-lang de --target-lang EN-GB
```

### Advanced Usage

```bash
# Add episode metadata
python podcast_translator.py "audio.mp3" \
    --title "Episode 5: Berlin Wall" \
    --podcast "History Matters"

# Custom output directory
python podcast_translator.py "audio.mp3" --output-dir "translations"

# Skip speaker diarization (faster)
python podcast_translator.py "audio.mp3" --no-diarization

# Translate French to Spanish
python podcast_translator.py "audio.mp3" --source-lang fr --target-lang ES
```

### Command-Line Options

- `audio_source` (required): URL or path to audio file
- `--source-lang`, `-s`: Source language code (default: de)
- `--target-lang`, `-t`: Target language code (default: EN-US)
- `--output-dir`, `-o`: Output directory (default: podcast_output)
- `--output-name`, `-n`: Base name for output files (auto-generated)
- `--title`: Episode title (optional)
- `--podcast`: Podcast name (optional)
- `--duration`: Episode duration (auto-detected)
- `--no-diarization`: Skip speaker diarization

### Supported Languages

DeepL supports these language codes:
- **Source**: BG, CS, DA, DE, EL, EN, ES, ET, FI, FR, HU, ID, IT, JA, KO, LT, LV, NB, NL, PL, PT, RO, RU, SK, SL, SV, TR, UK, ZH
- **Target**: EN-US, EN-GB, DE, FR, ES, IT, PT-PT, PT-BR, etc.

See [DeepL API documentation](https://www.deepl.com/docs-api/translate-text/) for full list.

### Setup API Keys

**1. Get Your FREE DeepL API Key**

1. Visit: https://www.deepl.com/pro-api
2. Sign up for "DeepL API Free" (500,000 chars/month)
3. Verify your email
4. Copy your API key (it will end with `:fx`)

**2. Set Your API Keys**

```bash
# Windows
set DEEPL_API_KEY=your-api-key-here:fx
set HUGGINGFACE_TOKEN=your-hf-token-here

# Mac/Linux
export DEEPL_API_KEY=your-api-key-here:fx
export HUGGINGFACE_TOKEN=your-hf-token-here
```

**Note**: Hugging Face token is needed for speaker diarization. Get it at https://huggingface.co/settings/tokens

## Output

All files are saved to `podcast_output/` directory:

- `tmah063_german.mp3` - Downloaded German audio
- `transcript_german.txt` - German transcription
- `transcript_german.json` - Detailed transcription with timestamps
- `transcript_english.txt` - English translation (raw)
- `transcript_english_formatted.txt` - **English translation (formatted for easy reading)** ⭐

**💡 Recommended**: Open `transcript_english_formatted.txt` for the best reading experience with proper paragraphs and formatting!

## Pipeline Steps

### Step 1: Download MP3 (88MB)
- Downloads from: `https://tellmeahistory.net/podlove/file/181/s/webplayer/c/episode/tmah_063_Sich_ein_Prachtbild_machen.mp3`
- Shows progress bar
- Skips if already downloaded

### Step 2: Transcribe German Audio
- Uses Whisper "large" model for best accuracy
- Takes ~30-60 minutes depending on your hardware
- GPU recommended but not required
- Saves transcript with timestamps

### Step 3: Translate to English
- Uses DeepL for high-quality translation
- Processes in chunks to handle long text
- Preserves conversational tone and historical accuracy
- Takes ~1-2 minutes

### Step 4: Format for Easy Reading
- Automatically formats the translation
- Adds proper paragraph breaks
- Includes header with episode info
- Creates easy-to-read version
- Takes ~1 second

## Cost Estimates

### Using DeepL Free API + Local Whisper (Recommended):
- Transcription: **FREE** (uses your computer)
- DeepL translation: **FREE** (500k chars/month free tier)
- **Total**: **$0** 🎉

### Character Usage:
- This podcast transcript: ~150k-200k characters
- DeepL Free tier: 500k characters/month
- **Result**: Fits easily! Can translate 2-3 podcasts per month for free

## Troubleshooting

### "File exceeds Whisper API limit (25MB)"
✓ The script automatically falls back to local Whisper
✓ Or manually split the audio into smaller chunks

### "Whisper not installed"
```bash
pip install openai-whisper torch
```

### "FFmpeg not found"
Install FFmpeg (see Installation section above)

### Out of memory during transcription
Use a smaller Whisper model:
```python
model = whisper.load_model("medium")  # or "base"
```

### Translation is slow
- Translation with DeepL is typically very fast (1-2 minutes)
- The script shows progress for each chunk
- DeepL processes large chunks efficiently (up to 50k characters)

## Customization

### Use different Whisper model
Edit line 89 in `podcast_translator.py`:
```python
model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
```

### Use different target language
Edit the `translate_text` call in `podcast_translator.py`:
```python
# Change target language
target_lang="EN-GB"  # British English
# or
target_lang="FR"     # French
target_lang="ES"     # Spanish
target_lang="IT"     # Italian
# See DeepL docs for all supported languages
```

### Change TTS voice
Edit line 225 in `podcast_translator.py`:
```python
voice="nova",  # Options: alloy, echo, fable, onyx, nova, shimmer
```

## Alternative Approaches

### For splitting large audio files:
```bash
ffmpeg -i input.mp3 -f segment -segment_time 600 -c copy output_%03d.mp3
```
This splits into 10-minute chunks.

### For using alternative translation services:
Consider DeepL API or Google Translate for potentially better German→English quality:
```bash
pip install deep-translator
```

## License

This is a utility script. The podcast content belongs to "Tell me a History" (tellmeahistory.net).

## Support

For issues with:
- The podcast content: Contact tellmeahistory.net
- This script: Check the troubleshooting section above
- OpenAI API: See OpenAI documentation
- Whisper issues: See OpenAI Whisper GitHub repo
