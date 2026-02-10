#!/usr/bin/env python3
"""
Setup script for the Podcast Translation Pipeline.
Makes the package installable via pip.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements from requirements.txt
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="podcast-translator",
    version="1.0.0",
    author="Koen van der Heide",
    author_email="koenjvanderheide@gmail.com", 
    description="A FREE pipeline for transcribing and translating audio files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/koenvdheide/podcast-translator",
    packages=find_packages(),
    py_modules=['podcast_translator', 'check_requirements'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'whisper': [
            'openai-whisper>=20231117',
            'torch>=2.0.0',
        ],
        'dev': [
            'pytest>=7.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'podcast-translator=podcast_translator:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        'podcast', 'translation', 'transcription', 'whisper',
        'deepl', 'audio', 'speech-to-text', 'nlp', 'diarization'
    ],
    project_urls={
        "Bug Reports": "https://github.com/koenvdheide/podcast-translator/issues",
        "Documentation": "https://github.com/koenvdheide/podcast-translator#readme",
        "Source": "https://github.com/koenvdheide/podcast-translator",
    },
)
