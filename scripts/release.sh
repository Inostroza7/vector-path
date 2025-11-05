#!/bin/bash
# Release script for vector-path package

set -e  # Exit on error

echo "🚀 Starting release process for vector-path..."
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Not in a virtual environment!"
    echo "   Activate venv first: source venv/bin/activate"
    exit 1
fi

# Run tests
echo "📋 Step 1: Running tests..."
pytest --cov=vector_path --cov-report=term-missing
if [ $? -ne 0 ]; then
    echo "❌ Tests failed! Fix errors before releasing."
    exit 1
fi
echo "✅ Tests passed!"
echo ""

# Check code quality with ruff
echo "🔍 Step 2: Checking code quality (ruff)..."
ruff check vector_path tests examples
if [ $? -ne 0 ]; then
    echo "⚠️  Linting issues found. Fix them or continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo "✅ Linting passed!"
echo ""

# Check formatting with black
echo "🎨 Step 3: Checking code formatting (black)..."
black --check vector_path tests examples
if [ $? -ne 0 ]; then
    echo "⚠️  Code formatting issues found."
    echo "   Run: black vector_path tests examples"
    echo "   Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo "✅ Formatting is correct!"
echo ""

# Type checking with mypy
echo "🔬 Step 4: Running type checker (mypy)..."
mypy vector_path
if [ $? -ne 0 ]; then
    echo "⚠️  Type checking issues found. Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo "✅ Type checking passed!"
echo ""

# Clean previous builds
echo "🧹 Step 5: Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info vector_path.egg-info
echo "✅ Clean complete!"
echo ""

# Build the package
echo "📦 Step 6: Building package..."
python -m build
if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi
echo "✅ Build successful!"
echo ""

# Check distribution
echo "🔍 Step 7: Checking distribution..."
twine check dist/*
if [ $? -ne 0 ]; then
    echo "❌ Distribution check failed!"
    exit 1
fi
echo "✅ Distribution is valid!"
echo ""

# Show what was built
echo "📦 Built packages:"
ls -lh dist/
echo ""

echo "✅ Release build complete!"
echo ""
echo "Next steps:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Test on Test PyPI (recommended):"
echo "   twine upload --repository testpypi dist/*"
echo ""
echo "2. Install from Test PyPI to verify:"
echo "   pip install --index-url https://test.pypi.org/simple/ \\"
echo "     --extra-index-url https://pypi.org/simple/ vector-path"
echo ""
echo "3. If everything works, publish to PyPI:"
echo "   twine upload dist/*"
echo ""
echo "4. Tag the release:"
echo "   git tag -a v\$(python -c 'import vector_path; print(vector_path.__version__)') -m 'Release version X.Y.Z'"
echo "   git push origin --tags"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
