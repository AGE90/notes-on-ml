# Contributing Guide

Thank you for your interest in contributing to **Notes on ML**! This guide will help you get started.

## Code of Conduct

Please read and follow our [Code of Conduct](code_of_conduct.md).

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported
2. Use the bug report template
3. Include steps to reproduce
4. Add relevant logs and screenshots
5. Specify your environment

### Suggesting Features

1. Check if the feature has been requested
2. Use the feature request template
3. Explain the use case
4. Describe the expected behavior
5. Consider implementation complexity

### Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and formatters
5. Submit a pull request

## Development Setup

### Prerequisites

- Python >= 3.9
- Poetry
- Git
- A code editor

### Setting Up

1. Fork and clone:
   ```bash
   git clone https://github.com/YOUR_USERNAME/notes-on-ml.git
   cd notes-on-ml
   ```

2. Add upstream:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/notes-on-ml.git
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```

4. Install pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```

### Development Workflow

1. Update your fork:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```

3. Make your changes
4. Run tests and formatters:
   ```bash
   poetry run pytest
   poetry run black .
   poetry run ruff check .
   poetry run mypy src
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: your feature description"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/your-feature
   ```

7. Create a pull request

## Code Style

### Python Code

- Follow PEP 8
- Use type hints
- Write docstrings
- Keep lines under 88 characters

### Documentation

- Use Google-style docstrings
- Update relevant documentation
- Include examples
- Add type hints

### Testing

- Write unit tests
- Aim for high coverage
- Use pytest fixtures
- Test edge cases

## Pull Request Process

1. Update documentation
2. Add tests if needed
3. Ensure all tests pass
4. Update the changelog
5. Request review
6. Address feedback
7. Wait for approval

## Review Process

1. Code review
2. Documentation review
3. Test coverage check
4. Style check
5. Final approval

## Release Process

1. Update version
2. Update changelog
3. Create release branch
4. Run final tests
5. Create release
6. Deploy

## Getting Help

- Check the documentation
- Search existing issues
- Ask in discussions
- Contact maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License. 