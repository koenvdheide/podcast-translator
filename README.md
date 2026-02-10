# Podcast Translation Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A powerful, **FREE** Python pipeline for transcribing and translating audio files. Built for podcasters, journalists, researchers, and content creators who work with multilingual audio content.

## Features

- **Audio Transcription** - Uses OpenAI Whisper (local or API) for high-quality speech-to-text
- **Multi-Language Translation** - Powered by DeepL's free API (500k chars/month)
- **Speaker Diarization** - Identifies and labels different speakers using Pyannote
- **Timestamps** - Adds timestamps to transcripts for easy navigation
- **Smart Formatting** - Automatically formats transcripts for readability
- **100% Free** - Uses only free tiers and open-source tools
- **Resume Capability** - Skips completed steps if pipeline is interrupted

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/koenvdheide/podcast-translator.git
cd podcast-translator

# Install as a package
pip install .

# Or install for development
pip install -e .
```

### Setup API Keys

1. **DeepL API Key** (FREE - 500k chars/month)
   - Sign up at: https://www.deepl.com/pro-api
   - Choose "DeepL API Free"

2. **Hugging Face Token** (FREE - for speaker diarization)
   - Sign up at: https://huggingface.co
   - Accept model agreements for pyannote

```bash
# Set environment variables
export DEEPL_API_KEY=your-key-here:fx
export HUGGINGFACE_TOKEN=your-token-here
```

### Basic Usage

```bash
# Translate a German podcast from URL
python podcast_translator.py "https://example.com/podcast.mp3"

# Translate a local audio file
python podcast_translator.py "my_audio.mp3"

# Translate with custom languages
python podcast_translator.py "audio.mp3" --source-lang fr --target-lang ES

# With metadata
python podcast_translator.py "audio.mp3" \
    --title "Episode 5" \
    --podcast "My Show" \
    --source-lang de \
    --target-lang EN-GB
```

## Documentation

For detailed command-line options, run:

```bash
python podcast_translator.py --help
```

See the [Quick Start](#-quick-start) section above for usage examples.

## Use Cases

- Translate podcasts and interviews
- Transcribe multilingual meetings
- Create subtitles from audio
- Archive spoken content with searchable text
- Research audio data with speaker identification

## Requirements

- Python 3.8+
- FFmpeg (for audio processing)
- 8GB+ RAM (for local Whisper transcription)

## Supported Languages

**Translation** (via DeepL):
- Source: BG, CS, DA, DE, EL, EN, ES, ET, FI, FR, HU, ID, IT, JA, KO, LT, LV, NB, NL, PL, PT, RO, RU, SK, SL, SV, TR, UK, ZH
- Target: EN-US, EN-GB, DE, FR, ES, IT, PT-PT, PT-BR, and more

See [DeepL API documentation](https://www.deepl.com/docs-api/translate-text/) for the complete list.

## Example Output

```
[00:00:05] [SPEAKER_00]: Hello and welcome to the 63rd episode of Tell Me Your History.
My name is Marvin and today I'm talking to Dr. Christiane Czygan from the Istanbul
Orient Institute.

[00:03:42] [SPEAKER_01]: That's a fascinating manuscript. In the 16th century, poetry
played an important role at the Ottoman court...
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [DeepL](https://www.deepl.com/) - Translation API
- [Pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization

## Support

- Report bugs via [GitHub Issues](https://github.com/koenvdheide/podcast-translator/issues)
- Questions? Open a [Discussion](https://github.com/koenvdheide/podcast-translator/discussions)
