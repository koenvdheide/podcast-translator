# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-09

### Added
- 🎙️ Audio transcription using OpenAI Whisper (local and API)
- 🌍 Multi-language translation powered by DeepL API
- 👥 Speaker diarization using Pyannote.audio
- ⏱️ Automatic timestamp generation in transcripts
- 📝 Smart transcript formatting with paragraphs
- 🔄 Resume capability (skips completed steps)
- 🎯 Command-line interface with argparse
- 📂 Per-episode subdirectory organization
- 🔍 Auto-detection of audio duration via ffprobe
- 🆓 Support for free API tiers (DeepL Free, Hugging Face)

### Features
- Support for both URL and local file input
- Configurable source and target languages
- Optional metadata (title, podcast name, duration)
- Dynamic output filenames based on language codes
- Speaker labels in formatted transcripts
- Graceful degradation when API keys are missing
- Cross-platform support (Windows, Mac, Linux)

### Documentation
- Comprehensive README with quick start guide
- Full documentation in README_podcast_translator.md
- Quick start guide in QUICK_START.md
- Command-line help and usage examples
- Troubleshooting guide

### Initial Release
This is the first public release of the Podcast Translation Pipeline.

## [Unreleased]

### Planned
- [ ] Batch processing for multiple files
- [ ] GUI interface
- [ ] Additional translation services (Google Translate, Azure)
- [ ] Export to subtitle formats (SRT, VTT)
- [ ] Web interface
- [ ] Docker container support

---

[1.0.0]: https://github.com/yourusername/podcast-translator/releases/tag/v1.0.0
