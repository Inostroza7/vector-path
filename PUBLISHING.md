# Publishing Guide for vector-path

This guide explains how to publish the `vector-path` package to PyPI.

## Prerequisites

1. **PyPI Account**: Create accounts on both:
   - Test PyPI: https://test.pypi.org/account/register/
   - PyPI: https://pypi.org/account/register/

2. **API Tokens**: Generate API tokens for uploading:
   - Test PyPI: https://test.pypi.org/manage/account/token/
   - PyPI: https://pypi.org/manage/account/token/

3. **Configure credentials** in `~/.pypirc`:
   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = pypi-your-api-token-here

   [testpypi]
   username = __token__
   password = pypi-your-test-api-token-here
   ```

## Pre-Publication Checklist

Before publishing, ensure:

- [ ] All tests pass: `pytest`
- [ ] Code is formatted: `black vector_path tests examples`
- [ ] Linting passes: `ruff check vector_path tests examples`
- [ ] Type checking passes: `mypy vector_path`
- [ ] Version number is updated in `vector_path/__init__.py`
- [ ] CHANGELOG.md is updated with release notes
- [ ] README.md is accurate and up-to-date
- [ ] All Azure OpenAI credentials are in .env (not hardcoded)
- [ ] pyproject.toml metadata is correct (author, URLs, etc.)

## Update Version

1. Update version in [vector_path/__init__.py](vector_path/__init__.py):
   ```python
   __version__ = "0.1.0"  # Update this
   ```

2. Update version in [pyproject.toml](pyproject.toml):
   ```toml
   [project]
   version = "0.1.0"  # Update this
   ```

3. Add release notes to [CHANGELOG.md](CHANGELOG.md)

## Build the Package

```bash
# Activate virtual environment
source venv/bin/activate

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build the package
python -m build
```

This creates two files in `dist/`:
- `vector_path-X.Y.Z.tar.gz` (source distribution)
- `vector_path-X.Y.Z-py3-none-any.whl` (wheel)

## Test the Build

Check that the distribution is valid:

```bash
twine check dist/*
```

## Publish to Test PyPI (Recommended First)

Test your package on Test PyPI before publishing to the real PyPI:

```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ vector-path
```

Note: `--extra-index-url` is needed because dependencies (openai, numpy, etc.) are on the real PyPI.

## Publish to PyPI

Once you've tested on Test PyPI:

```bash
# Upload to PyPI
twine upload dist/*
```

## Post-Publication

1. **Tag the release** in git:
   ```bash
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push origin v0.1.0
   ```

2. **Create GitHub Release** (if using GitHub):
   - Go to repository → Releases → Create new release
   - Select the tag you just created
   - Copy release notes from CHANGELOG.md
   - Attach the built distributions from `dist/`

3. **Test installation**:
   ```bash
   pip install vector-path
   ```

## Version Numbering (Semantic Versioning)

Follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: Add functionality (backwards-compatible)
- **PATCH**: Bug fixes (backwards-compatible)

Examples:
- `0.1.0` → `0.1.1` (bug fix)
- `0.1.1` → `0.2.0` (new feature)
- `0.2.0` → `1.0.0` (breaking change)

## Quick Release Script

Save this as `scripts/release.sh`:

```bash
#!/bin/bash
set -e

echo "🚀 Starting release process..."

# Run tests
echo "Running tests..."
pytest

# Check code quality
echo "Checking code quality..."
ruff check vector_path tests examples
black --check vector_path tests examples
mypy vector_path

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

# Build
echo "Building package..."
python -m build

# Check distribution
echo "Checking distribution..."
twine check dist/*

echo "✅ Build complete! Ready to publish."
echo ""
echo "To publish to Test PyPI:"
echo "  twine upload --repository testpypi dist/*"
echo ""
echo "To publish to PyPI:"
echo "  twine upload dist/*"
```

Make it executable:
```bash
chmod +x scripts/release.sh
```

## Troubleshooting

### "File already exists" error
If you get this error, you've already uploaded this version. Increment the version number.

### Import errors after installation
Make sure the package structure is correct and `__init__.py` properly exports all public APIs.

### Missing dependencies
Ensure all runtime dependencies are listed in `pyproject.toml` under `dependencies`.

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Semantic Versioning](https://semver.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
