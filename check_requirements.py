#!/usr/bin/env python3
"""
Check if all requirements are installed for the podcast translation pipeline.
"""

import sys
import shutil
import os


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠ WARNING: Python 3.8+ is recommended")
        return False
    return True


def check_ffmpeg():
    """Check if FFmpeg is installed."""
    if shutil.which("ffmpeg"):
        print("✓ FFmpeg is installed")
        return True
    else:
        print("✗ FFmpeg is NOT installed")
        print("\n  To install FFmpeg on Windows:")
        print("    choco install ffmpeg")
        print("  Or download from: https://ffmpeg.org/download.html")
        return False


def check_python_package(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False


def check_api_keys():
    """Check if API keys are set."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if anthropic_key:
        print(f"✓ ANTHROPIC_API_KEY is set ({anthropic_key[:10]}...)")
    else:
        print("✗ ANTHROPIC_API_KEY is NOT set")
        print("  Set it with: set ANTHROPIC_API_KEY=your-api-key-here")

    if openai_key:
        print(f"✓ OPENAI_API_KEY is set ({openai_key[:10]}...)")
    else:
        print("ℹ OPENAI_API_KEY is NOT set (optional - only needed for OpenAI Whisper API)")

    return bool(anthropic_key)


def main():
    """Main check function."""
    print("=" * 60)
    print("Podcast Translation Pipeline - Requirements Check")
    print("=" * 60)
    print()

    all_good = True

    print("Python Version:")
    all_good &= check_python_version()
    print()

    print("System Dependencies:")
    all_good &= check_ffmpeg()
    print()

    print("Python Packages (Core):")
    all_good &= check_python_package("requests")
    all_good &= check_python_package("anthropic")
    print()

    print("Python Packages (Transcription):")
    has_whisper = check_python_package("openai-whisper", "whisper")
    has_torch = check_python_package("torch")
    all_good &= (has_whisper and has_torch)
    print()

    print("API Keys:")
    all_good &= check_api_keys()
    print()

    print("=" * 60)
    if all_good:
        print("✓ All requirements are met! You're ready to run the pipeline.")
    else:
        print("⚠ Some requirements are missing. Please install them first.")
        print("\nQuick install command:")
        print("  python -m pip install -r requirements.txt")
        print("  python -m pip install openai-whisper")
        print("  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print("=" * 60)

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
