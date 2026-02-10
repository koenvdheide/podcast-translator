# Contributing to Podcast Translation Pipeline

First off, thank you for considering contributing to this project! 🎉

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **System information** (OS, Python version)
- **Relevant logs or error messages**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear use case** - Why is this enhancement useful?
- **Describe the solution** you'd like
- **Describe alternatives** you've considered
- **Additional context** or screenshots

### Pull Requests

1. **Fork the repo** and create your branch from `main`
2. **Make your changes**
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Follow the code style** (we use `black` for formatting)
6. **Write clear commit messages**
7. **Submit a pull request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/podcast-translator.git
cd podcast-translator

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Code Style

- We use **Black** for Python formatting
- Follow **PEP 8** guidelines
- Add **docstrings** to new functions
- Keep functions **focused and small**
- Write **clear variable names**

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=podcast_translator

# Format code
black podcast_translator.py check_requirements.py

# Lint code
flake8 podcast_translator.py
```

## Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests

Examples:
```
Add support for Spanish translation
Fix duration auto-detection for WAV files
Update documentation for new CLI options
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue with the "question" label or start a discussion!
