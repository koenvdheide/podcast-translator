#!/usr/bin/env python3
"""
Podcast Translation Pipeline: German to English
Downloads, transcribes, and translates a German podcast to English.
"""

import os
import sys
import requests
from pathlib import Path
from typing import Optional
import json
import subprocess
import shutil
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs
import re
import argparse
import time


@dataclass
class TranslationConfig:
    """Configuration for podcast translation pipeline."""
    audio_source: str          # URL or file path
    output_dir: Path
    base_name: str             # Base name for output files
    source_lang: str = "de"
    target_lang: str = "EN-US"
    episode_title: Optional[str] = None
    podcast_name: Optional[str] = None
    duration: Optional[str] = None
    enable_diarization: bool = True


class PodcastTranslator:
    """Pipeline for downloading, transcribing, and translating podcasts."""

    def __init__(self, config: TranslationConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create episode-specific subdirectory
        self.episode_dir = self.output_dir / config.base_name
        self.episode_dir.mkdir(parents=True, exist_ok=True)

    def check_ffmpeg(self) -> bool:
        """Check if FFmpeg is installed and accessible."""
        return shutil.which("ffmpeg") is not None

    def parse_audio_source(self, source: str) -> tuple[str, Optional[Path]]:
        """Determine if source is URL or local file."""
        if source.startswith(('http://', 'https://')):
            return ('url', None)
        else:
            path = Path(source)
            if path.exists():
                return ('file', path.resolve())
            else:
                raise FileNotFoundError(f"Local file not found: {source}")

    @staticmethod
    def generate_base_name(source: str) -> str:
        """Generate base filename from URL or file path."""
        if source.startswith(('http://', 'https://')):
            # Extract from URL
            parsed = urlparse(source)
            filename = Path(parsed.path).stem
            if not filename or filename == 'file':
                # Fallback: use query params or timestamp
                params = parse_qs(parsed.query)
                filename = params.get('id', [f'podcast_{int(time.time())}'])[0]
        else:
            # Extract from local path
            filename = Path(source).stem

        # Clean filename (remove special chars)
        filename = re.sub(r'[^\w\-_]', '_', filename)
        return filename

    def get_audio_duration(self, audio_path: Path) -> str:
        """Auto-detect audio duration using ffprobe."""
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(audio_path)
            ], capture_output=True, text=True, check=True)

            seconds = float(result.stdout.strip())
            return self._format_timestamp(seconds)
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
            print(f"⚠ Could not auto-detect duration: {e}")
            return None

    def prepare_audio(self, source: str) -> Path:
        """
        Prepare audio file (download URL or validate local file).
        Replaces download_mp3().
        """
        source_type, local_path = self.parse_audio_source(source)

        if source_type == 'url':
            filename = f"{self.config.base_name}_{self.config.source_lang}.mp3"
            return self._download_from_url(source, filename)
        else:
            # Validate local file format
            valid_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac']
            if local_path.suffix.lower() not in valid_extensions:
                print(f"⚠ Warning: {local_path.suffix} may not be supported by Whisper")

            print(f"✓ Using local file: {local_path}")
            return local_path

    def _download_from_url(self, url: str, filename: str) -> Path:
        """Download MP3 from URL."""
        output_path = self.episode_dir / filename

        if output_path.exists():
            print(f"✓ File already exists: {output_path}")
            return output_path

        print(f"Downloading audio from {url}...")
        print(f"This may take a while...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end="", flush=True)

        print(f"\n✓ Downloaded to: {output_path}")
        return output_path

    def download_mp3(self, url: str, filename: str = "podcast.mp3") -> Path:
        """Download MP3 file from URL."""
        output_path = self.output_dir / filename

        if output_path.exists():
            print(f"✓ File already exists: {output_path}")
            return output_path

        print(f"Downloading MP3 from {url}...")
        print(f"This may take a while (file size: ~88 MB)...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end="", flush=True)

        print(f"\n✓ Downloaded to: {output_path}")
        return output_path

    def transcribe_audio(self, audio_path: Path) -> dict:
        """Transcribe audio using OpenAI Whisper API."""
        lang = self.config.source_lang
        transcript_path = self.episode_dir / f"transcript_{lang}.txt"
        transcript_json = self.episode_dir / f"transcript_{lang}.json"

        if transcript_path.exists():
            print(f"✓ Transcript already exists: {transcript_path}")
            with open(transcript_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return {"text": text, "language": lang}

        print("Transcribing audio with Whisper...")
        print("Note: This requires OpenAI API key (set OPENAI_API_KEY env variable)")

        try:
            from openai import OpenAI
            client = OpenAI()

            # Whisper API has a 25MB limit, so we might need to split the file
            file_size_mb = audio_path.stat().st_size / (1024 * 1024)

            if file_size_mb > 25:
                print(f"⚠ File size ({file_size_mb:.1f}MB) exceeds Whisper API limit (25MB)")
                print("Consider using local Whisper model or splitting the audio file")
                return self._transcribe_with_local_whisper(audio_path, language)

            with open(audio_path, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=lang,
                    response_format="verbose_json"
                )

            # Save transcript
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript.text)

            with open(transcript_json, 'w', encoding='utf-8') as f:
                json.dump(transcript.model_dump(), f, ensure_ascii=False, indent=2)

            print(f"✓ Transcription saved to: {transcript_path}")
            return transcript.model_dump()

        except ImportError:
            print("⚠ OpenAI library not installed. Install with: pip install openai")
            return self._transcribe_with_local_whisper(audio_path)
        except Exception as e:
            print(f"⚠ Error with OpenAI API: {e}")
            return self._transcribe_with_local_whisper(audio_path)

    def _transcribe_with_local_whisper(self, audio_path: Path) -> dict:
        """Transcribe using local Whisper model (fallback)."""
        lang = self.config.source_lang
        transcript_path = self.episode_dir / f"transcript_{lang}.txt"
        transcript_json = self.episode_dir / f"transcript_{lang}.json"

        # Check for FFmpeg first
        if not self.check_ffmpeg():
            print("\n" + "=" * 60)
            print("⚠ ERROR: FFmpeg is not installed or not in PATH")
            print("=" * 60)
            print("\nWhisper requires FFmpeg to process audio files.")
            print("\nTo install FFmpeg on Windows:")
            print("  1. Using Chocolatey (recommended):")
            print("     choco install ffmpeg")
            print("\n  2. Using winget:")
            print("     winget install ffmpeg")
            print("\n  3. Manual installation:")
            print("     - Download from: https://ffmpeg.org/download.html")
            print("     - Extract the archive")
            print("     - Add the 'bin' folder to your system PATH")
            print("\nAfter installation, restart your terminal and try again.")
            print("=" * 60)
            sys.exit(1)

        try:
            import whisper
            print("Using local Whisper model (this may take a while)...")
            print("Loading model...")

            # Use 'large' model for best accuracy, or 'base'/'medium' for speed
            model = whisper.load_model("large")

            print("Transcribing...")
            result = model.transcribe(
                str(audio_path),
                language=lang,
                verbose=True
            )

            # Save transcript
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(result['text'])

            # Save detailed JSON with segments/timestamps
            with open(transcript_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"✓ Transcription saved to: {transcript_path}")
            return result

        except ImportError:
            print("⚠ Whisper not installed. Install with: pip install openai-whisper")
            print("Or use the OpenAI API by setting OPENAI_API_KEY")
            sys.exit(1)

    def diarize_audio(self, audio_path: Path) -> dict:
        """Perform speaker diarization using Pyannote."""
        diarization_path = self.episode_dir / "diarization.json"

        if diarization_path.exists():
            print(f"✓ Diarization already exists: {diarization_path}")
            with open(diarization_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        print("Performing speaker diarization...")

        # Check for Hugging Face token
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            print("\n⚠ WARNING: HUGGINGFACE_TOKEN not found")
            print("Speaker diarization requires a free Hugging Face token.")
            print("\nTo get a token:")
            print("  1. Sign up at: https://huggingface.co")
            print("  2. Accept model agreements:")
            print("     - https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("     - https://huggingface.co/pyannote/segmentation-3.0")
            print("  3. Create token at: https://huggingface.co/settings/tokens")
            print("  4. Set it: set HUGGINGFACE_TOKEN=your-token-here")
            print("\nSkipping diarization...")
            return None

        try:
            from pyannote.audio import Pipeline

            # Load pre-trained pipeline
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )

            # Perform diarization
            diarization = pipeline(str(audio_path))

            # Convert to JSON-serializable format
            speakers = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })

            # Save diarization results
            with open(diarization_path, 'w', encoding='utf-8') as f:
                json.dump(speakers, f, ensure_ascii=False, indent=2)

            print(f"✓ Found {len(set(s['speaker'] for s in speakers))} speakers")
            print(f"✓ Diarization saved to: {diarization_path}")

            return speakers

        except ImportError:
            print("⚠ Pyannote.audio not installed. Install with: pip install pyannote.audio")
            return None
        except Exception as e:
            print(f"⚠ Diarization error: {e}")
            return None

    def align_speakers_to_segments(self, segments: list, speakers: list) -> list:
        """Assign speaker labels to transcript segments."""
        if not speakers or not segments:
            return segments

        print("Aligning speakers with transcript segments...")

        aligned_segments = []
        for segment in segments:
            seg_start = segment['start']
            seg_end = segment['end']
            seg_mid = (seg_start + seg_end) / 2  # Use midpoint for assignment

            # Find speaker active at segment midpoint
            assigned_speaker = "UNKNOWN"
            for speaker in speakers:
                if speaker['start'] <= seg_mid < speaker['end']:
                    assigned_speaker = speaker['speaker']
                    break

            # Add speaker to segment
            aligned_segment = segment.copy()
            aligned_segment['speaker'] = assigned_speaker
            aligned_segments.append(aligned_segment)

        # Print speaker distribution
        speaker_counts = {}
        for seg in aligned_segments:
            spk = seg.get('speaker', 'UNKNOWN')
            speaker_counts[spk] = speaker_counts.get(spk, 0) + 1

        print(f"✓ Aligned {len(aligned_segments)} segments:")
        for speaker, count in sorted(speaker_counts.items()):
            print(f"  - {speaker}: {count} segments")

        return aligned_segments

    def translate_text(self, text: str) -> str:
        """Translate text using DeepL API (Free tier: 500k chars/month)."""
        target_lang_code = self._get_target_lang_code()
        translation_path = self.episode_dir / f"transcript_{target_lang_code}.txt"

        if translation_path.exists():
            print(f"✓ Translation already exists: {translation_path}")
            with open(translation_path, 'r', encoding='utf-8') as f:
                return f.read()

        source_lang = self.config.source_lang.upper()
        target_lang = self.config.target_lang
        print(f"Translating text from {source_lang} to {target_lang} using DeepL...")

        api_key = os.getenv("DEEPL_API_KEY")
        if not api_key:
            print("\n⚠ ERROR: DEEPL_API_KEY not found")
            print("\nGet a FREE DeepL API key:")
            print("  1. Sign up at: https://www.deepl.com/pro-api")
            print("  2. Choose the 'DeepL API Free' plan (500k chars/month)")
            print("  3. Set the key: set DEEPL_API_KEY=your-key-here")
            sys.exit(1)

        # Determine API endpoint based on key type
        # Free API keys end with ":fx"
        if api_key.endswith(":fx"):
            api_url = "https://api-free.deepl.com/v2/translate"
        else:
            api_url = "https://api.deepl.com/v2/translate"

        # DeepL can handle large texts, but let's chunk to be safe
        max_chunk_size = 50000  # DeepL can handle up to 128KB
        chunks = self._split_text(text, max_chunk_size)

        print(f"Text length: {len(text):,} characters")
        print(f"Number of chunks: {len(chunks)}")

        translated_chunks = []
        for i, chunk in enumerate(chunks, 1):
            print(f"Translating chunk {i}/{len(chunks)}... ({len(chunk):,} chars)")

            try:
                response = requests.post(
                    api_url,
                    headers={
                        "Authorization": f"DeepL-Auth-Key {api_key}",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    data={
                        "text": chunk,
                        "source_lang": source_lang,
                        "target_lang": self.config.target_lang,
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    translated_chunks.append(result['translations'][0]['text'])
                elif response.status_code == 456:
                    print("\n⚠ ERROR: Quota exceeded!")
                    print("DeepL Free tier limit: 500,000 characters/month")
                    print("Consider upgrading or waiting until next month.")
                    sys.exit(1)
                else:
                    print(f"\n⚠ ERROR: DeepL API error {response.status_code}")
                    print(response.text)
                    sys.exit(1)

            except Exception as e:
                print(f"\n⚠ ERROR: {e}")
                sys.exit(1)

        translated_text = "\n\n".join(translated_chunks)

        # Save translation
        with open(translation_path, 'w', encoding='utf-8') as f:
            f.write(translated_text)

        print(f"✓ Translation saved to: {translation_path}")
        return translated_text

    def _split_text(self, text: str, max_size: int) -> list[str]:
        """Split text into chunks at sentence boundaries."""
        if len(text) <= max_size:
            return [text]

        chunks = []
        current_chunk = ""

        # Split by paragraphs first, then sentences
        paragraphs = text.split('\n\n')

        for para in paragraphs:
            if len(current_chunk) + len(para) <= max_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _get_target_lang_code(self) -> str:
        """Convert target language code to filename-safe format."""
        return self.config.target_lang.replace('-', '_').lower()

    def _generate_header(self) -> str:
        """Generate header with available metadata."""
        if not (self.config.podcast_name or self.config.episode_title or self.config.duration):
            # No metadata provided - skip header
            return ""

        lines = []

        # Title line
        if self.config.podcast_name:
            lines.append(f"# Podcast Transcript: {self.config.podcast_name}")
        else:
            lines.append("# Podcast Transcript")

        # Episode info
        if self.config.episode_title:
            lines.append(f"Episode: {self.config.episode_title}")

        if self.config.duration:
            lines.append(f"Duration: {self.config.duration}")

        # Language info
        lines.append(f"Source Language: {self.config.source_lang.upper()}")
        lines.append(f"Target Language: {self.config.target_lang.upper()}")
        lines.append("Translation: DeepL API")
        lines.append("")
        lines.append("---")
        lines.append("")

        return "\n".join(lines)

    def format_transcript(self, text: str, segments: list = None) -> str:
        """Format transcript for better readability."""
        target_lang_code = self._get_target_lang_code()
        formatted_path = self.episode_dir / f"transcript_{target_lang_code}_formatted.txt"

        if formatted_path.exists():
            print(f"✓ Formatted transcript already exists: {formatted_path}")
            with open(formatted_path, 'r', encoding='utf-8') as f:
                return f.read()

        print("Formatting transcript for better readability...")

        if segments:
            # Format with timestamps from segments
            print(f"Using {len(segments)} segments with timestamps...")
            formatted_paragraphs = []
            current_para = []
            current_speaker = None
            last_timestamp = None

            for segment in segments:
                timestamp = self._format_timestamp(segment['start'])
                speaker = segment.get('speaker', None)
                segment_text = segment['text'].strip()

                # Add speaker label + timestamp when speaker changes or new paragraph
                if (last_timestamp is None or
                    speaker != current_speaker or
                    len(current_para) >= 5 or
                    segment['start'] - last_timestamp > 300):  # 5 min gap

                    if current_para:
                        formatted_paragraphs.append(' '.join(current_para))

                    # Format: [HH:MM:SS] [SPEAKER]: text
                    if speaker:
                        current_para = [f"[{timestamp}] [{speaker}]: {segment_text}"]
                    else:
                        current_para = [f"[{timestamp}] {segment_text}"]

                    current_speaker = speaker
                    last_timestamp = segment['start']
                else:
                    current_para.append(segment_text)

            if current_para:
                formatted_paragraphs.append(' '.join(current_para))

            formatted_text = '\n\n'.join(formatted_paragraphs)
        else:
            # Fallback to sentence-based formatting (original method)
            import re

            # Add line breaks after sentence endings followed by space
            text = re.sub(r'([.!?])\s+', r'\1\n', text)

            # Group sentences into paragraphs (every 4-6 sentences)
            lines = text.split('\n')
            paragraphs = []
            current_para = []

            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    current_para.append(line)
                    # Create new paragraph every 5 sentences or at topic changes
                    if len(current_para) >= 5:
                        paragraphs.append(' '.join(current_para))
                        current_para = []

            # Add remaining sentences
            if current_para:
                paragraphs.append(' '.join(current_para))

            # Join paragraphs with double newlines
            formatted_text = '\n\n'.join(paragraphs)

        # Add header
        header = self._generate_header()
        formatted_text = header + formatted_text if header else formatted_text

        # Save formatted version
        with open(formatted_path, 'w', encoding='utf-8') as f:
            f.write(formatted_text)

        print(f"✓ Formatted transcript saved to: {formatted_path}")
        return formatted_text

    def generate_audio(self, text: str, output_filename: str = "podcast_english.mp3") -> Optional[Path]:
        """Generate English audio from translated text (optional)."""
        output_path = self.output_dir / output_filename

        if output_path.exists():
            print(f"✓ Audio already exists: {output_path}")
            return output_path

        print("Generating English audio (this may take a while)...")

        try:
            from openai import OpenAI
            client = OpenAI()

            # OpenAI TTS has a 4096 character limit
            # For long podcasts, we'd need to split and concatenate
            if len(text) > 4000:
                print("⚠ Text is too long for single TTS generation")
                print("Consider using a different TTS service or splitting the audio")
                return None

            response = client.audio.speech.create(
                model="tts-1-hd",
                voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
                input=text[:4096]  # Limit to first portion
            )

            response.stream_to_file(output_path)
            print(f"✓ Audio saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"⚠ Could not generate audio: {e}")
            return None


def create_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Podcast Translation Pipeline: Transcribe and translate audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate a German podcast from URL
  python podcast_translator.py https://example.com/podcast.mp3

  # Translate a local audio file
  python podcast_translator.py my_audio.mp3

  # Translate with custom languages
  python podcast_translator.py audio.mp3 --source-lang de --target-lang EN-GB

  # Translate with metadata
  python podcast_translator.py audio.mp3 --title "Episode 5" --podcast "My Show"

  # Translate French to Spanish
  python podcast_translator.py audio.mp3 --source-lang fr --target-lang ES
        """
    )

    # Required positional argument
    parser.add_argument(
        'audio_source',
        help='URL or path to audio file (MP3, WAV, M4A, etc.)'
    )

    # Language options
    parser.add_argument(
        '--source-lang', '-s',
        default='de',
        help='Source language code (default: de for German)'
    )
    parser.add_argument(
        '--target-lang', '-t',
        default='EN-US',
        help='Target language code for DeepL (default: EN-US)'
    )

    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        default='podcast_output',
        help='Output directory (default: podcast_output)'
    )
    parser.add_argument(
        '--output-name', '-n',
        help='Base name for output files (auto-generated if not provided)'
    )

    # Metadata options
    parser.add_argument(
        '--title',
        help='Episode title (optional)'
    )
    parser.add_argument(
        '--podcast',
        help='Podcast name (optional)'
    )
    parser.add_argument(
        '--duration',
        help='Episode duration (auto-detected if not provided)'
    )

    # Feature toggles
    parser.add_argument(
        '--no-diarization',
        action='store_true',
        help='Skip speaker diarization'
    )

    return parser


def main():
    """Main pipeline execution."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Generate base name if not provided
    output_name = args.output_name or PodcastTranslator.generate_base_name(args.audio_source)

    # Create configuration
    config = TranslationConfig(
        audio_source=args.audio_source,
        output_dir=Path(args.output_dir),
        base_name=output_name,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        episode_title=args.title,
        podcast_name=args.podcast,
        duration=args.duration,  # May be None, will auto-detect later
        enable_diarization=not args.no_diarization
    )

    # Display configuration
    print("=" * 60)
    print("Podcast Translation Pipeline")
    print("=" * 60)
    print()
    print(f"Audio Source: {config.audio_source}")
    print(f"Output Directory: {config.output_dir / config.base_name}")
    print(f"Source Language: {config.source_lang.upper()}")
    print(f"Target Language: {config.target_lang.upper()}")
    if config.episode_title:
        print(f"Episode: {config.episode_title}")
    if config.podcast_name:
        print(f"Podcast: {config.podcast_name}")
    print()

    # Check for API keys
    has_deepl = bool(os.getenv("DEEPL_API_KEY"))
    if not has_deepl:
        print("⚠ WARNING: DEEPL_API_KEY not found in environment variables")
        print("Translation requires DeepL API key (FREE tier available).")
        print("\nGet a FREE DeepL API key:")
        print("  1. Visit: https://www.deepl.com/pro-api")
        print("  2. Sign up for 'DeepL API Free' (500k chars/month)")
        print("  3. Set it with: set DEEPL_API_KEY=your-key-here")
        print()
        response = input("Continue without API key? (transcription only) [y/N]: ")
        if response.lower() != 'y':
            sys.exit(0)

    # Initialize translator with config
    translator = PodcastTranslator(config)

    try:
        # Step 1: Prepare audio (download or validate)
        print("\n" + "=" * 60)
        print("STEP 1: Preparing Audio")
        print("=" * 60)
        audio_path = translator.prepare_audio(config.audio_source)

        # Auto-detect duration if not provided
        if not config.duration:
            print("Auto-detecting audio duration...")
            config.duration = translator.get_audio_duration(audio_path)
            if config.duration:
                print(f"✓ Duration: {config.duration}")

        # Step 2: Transcribe audio
        print("\n" + "=" * 60)
        print(f"STEP 2: Transcribing {config.source_lang.upper()} audio")
        print("=" * 60)
        transcript = translator.transcribe_audio(audio_path)
        source_text = transcript.get("text", "")

        if not source_text:
            print("⚠ No transcript text found")
            sys.exit(1)

        print(f"\nTranscript preview (first 500 chars):")
        print("-" * 60)
        print(source_text[:500] + "...")
        print("-" * 60)

        # Step 2.5: Speaker diarization (if enabled)
        speakers = None
        if config.enable_diarization:
            print("\n" + "=" * 60)
            print("STEP 2.5: Identifying speakers (diarization)")
            print("=" * 60)
            speakers = translator.diarize_audio(audio_path)

            # Align speakers with transcript segments if both available
            if speakers and 'segments' in transcript:
                transcript_segments = transcript['segments']
                aligned_segments = translator.align_speakers_to_segments(
                    transcript_segments,
                    speakers
                )
                # Update transcript with speaker-aligned segments
                transcript['segments'] = aligned_segments

        # Step 3: Translate to target language
        print("\n" + "=" * 60)
        print(f"STEP 3: Translating to {config.target_lang.upper()}")
        print("=" * 60)
        translated_text = translator.translate_text(source_text)

        print(f"\nTranslation preview (first 500 chars):")
        print("-" * 60)
        print(translated_text[:500] + "...")
        print("-" * 60)

        # Step 4: Format transcript for readability
        print("\n" + "=" * 60)
        print("STEP 4: Formatting transcript for readability")
        print("=" * 60)

        # Load segment data if available
        segments = transcript.get('segments', None)
        if segments:
            print(f"✓ Using {len(segments)} segments with timestamps")

        formatted_text = translator.format_transcript(translated_text, segments=segments)

        print(f"\nFormatted preview (first 500 chars):")
        print("-" * 60)
        print(formatted_text[:500] + "...")
        print("-" * 60)

        # Summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"\nOutput files in: {translator.episode_dir}")
        print(f"  - Audio: {audio_path.name}")
        print(f"  - {config.source_lang.upper()} transcript: transcript_{config.source_lang}.txt")
        if speakers:
            print(f"  - Diarization: diarization.json")
        target_code = translator._get_target_lang_code()
        print(f"  - {config.target_lang.upper()} translation: transcript_{target_code}.txt")
        print(f"  - {config.target_lang.upper()} formatted: transcript_{target_code}_formatted.txt ⭐")
        if segments:
            speaker_count = len(set(s.get('speaker', 'UNKNOWN') for s in segments if 'speaker' in s))
            print(f"    (Includes {len(segments)} timestamps, {speaker_count} speakers)")
        print()
        print("💡 Tip: Open the formatted file for the best reading experience!")
        if segments:
            print("   Timestamps allow you to jump to specific parts of the audio!")
            if speakers:
                print("   Speaker labels show who is speaking at each point!")
        print()

    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n⚠ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
