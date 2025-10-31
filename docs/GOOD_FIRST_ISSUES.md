# Good First Issues Guide

Welcome to RLAF! This guide will help you find and tackle your first contribution.

## What are Good First Issues?

Good First Issues are beginner-friendly tasks that:
- Are well-defined and scoped
- Don't require deep knowledge of the entire codebase
- Have clear acceptance criteria
- Usually take 1-5 hours to complete

## Finding Good First Issues

Browse issues with the `good first issue` label:
- [Good First Issues](https://github.com/cogniolab/cognio-rlaf/labels/good%20first%20issue)
- [Help Wanted](https://github.com/cogniolab/cognio-rlaf/labels/help%20wanted)

## Types of Contributions

### 1. Documentation Improvements
**Difficulty:** Easy
**Time:** 1-2 hours
**Skills needed:** Writing, Markdown

Examples:
- Fix typos or grammatical errors
- Improve code examples in docstrings
- Add missing documentation sections
- Create tutorials or guides

**Where to start:**
- `docs/` directory
- `README.md`
- Docstrings in `rlaf/` modules

### 2. Adding Tests
**Difficulty:** Easy-Medium
**Time:** 2-3 hours
**Skills needed:** Python, pytest

Examples:
- Add unit tests for existing functions
- Improve test coverage
- Add integration tests
- Add edge case tests

**Where to start:**
- `tests/` directory
- Run `pytest --cov=rlaf` to see coverage gaps

### 3. Creating Examples
**Difficulty:** Medium
**Time:** 3-5 hours
**Skills needed:** Python, understanding of use cases

Examples:
- Create domain-specific examples (healthcare, finance, etc.)
- Add more detailed tutorials
- Create Jupyter notebooks
- Add example datasets

**Where to start:**
- `examples/` directory
- Look at existing examples as templates

### 4. Adding New Critics
**Difficulty:** Medium
**Time:** 3-5 hours
**Skills needed:** Python, understanding of evaluation metrics

Examples:
- Implement domain-specific critics
- Add new evaluation perspectives
- Create critic templates

**Where to start:**
- `rlaf/agents/critic.py`
- Study existing critic implementations

### 5. Code Quality Improvements
**Difficulty:** Easy-Medium
**Time:** 1-3 hours
**Skills needed:** Python, code review

Examples:
- Fix linting warnings
- Improve type hints
- Refactor code for clarity
- Add better error messages

**Where to start:**
- Run `ruff check rlaf/`
- Run `mypy rlaf/`

## Step-by-Step: Your First Contribution

### 1. Set Up Your Environment

```bash
# Fork the repository on GitHub first, then:

# Clone your fork
git clone https://github.com/YOUR_USERNAME/cognio-rlaf.git
cd cognio-rlaf

# Add upstream remote
git remote add upstream https://github.com/cogniolab/cognio-rlaf.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### 2. Find an Issue

1. Browse [Good First Issues](https://github.com/cogniolab/cognio-rlaf/labels/good%20first%20issue)
2. Read the issue description carefully
3. Comment on the issue: "I'd like to work on this!"
4. Wait for maintainer confirmation (usually < 24 hours)

### 3. Create a Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b fix/issue-123-description
```

Branch naming conventions:
- `fix/issue-123-bug-description` - Bug fixes
- `feat/issue-123-feature-name` - New features
- `docs/issue-123-doc-update` - Documentation
- `test/issue-123-test-name` - Tests

### 4. Make Your Changes

Follow the coding standards:

```bash
# Write your code
# ...

# Format with Black
black rlaf/ tests/

# Lint with Ruff
ruff check rlaf/ tests/

# Type check with MyPy
mypy rlaf/

# Run tests
pytest tests/
```

### 5. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a clear message
git commit -m "Fix: Improve error handling in critic aggregation

- Add try-catch for edge cases
- Improve error messages
- Add tests for error scenarios

Fixes #123"
```

Commit message format:
```
<type>: <short summary>

<detailed description>

<footer>
```

Types: `Fix`, `Feat`, `Docs`, `Test`, `Refactor`, `Style`, `Chore`

### 6. Push and Create PR

```bash
# Push to your fork
git push origin fix/issue-123-description
```

Then on GitHub:
1. Click "Compare & pull request"
2. Fill out the PR template
3. Link the related issue
4. Submit the PR

### 7. Respond to Reviews

- Be open to feedback
- Make requested changes promptly
- Ask questions if something is unclear
- Update your PR based on comments

## Common Tasks

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_critics.py

# With coverage
pytest --cov=rlaf tests/

# Verbose output
pytest -v tests/
```

### Code Quality Checks

```bash
# Format code
black rlaf/ tests/

# Check formatting (without changing files)
black --check rlaf/ tests/

# Lint
ruff check rlaf/ tests/

# Fix auto-fixable linting issues
ruff check --fix rlaf/ tests/

# Type checking
mypy rlaf/
```

### Building Documentation

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build docs (if using Sphinx)
cd docs
make html
```

## Tips for Success

### Do:
✅ Ask questions if you're stuck
✅ Start with small contributions
✅ Read existing code to understand patterns
✅ Test your changes thoroughly
✅ Follow the code style guide
✅ Update documentation when needed
✅ Be patient with the review process

### Don't:
❌ Work on issues without confirmation
❌ Submit massive PRs with many changes
❌ Ignore CI failures
❌ Take feedback personally
❌ Force push to PR branches after review
❌ Copy code without understanding it

## Getting Help

If you're stuck:

1. **Read the docs**: Check `docs/` and `README.md`
2. **Check examples**: Look at `examples/` for patterns
3. **Ask in the issue**: Comment on the issue you're working on
4. **Search existing issues**: Someone may have had the same question
5. **Email maintainers**: moses@cogniolab.com

## Recognition

Contributors are recognized:
- In PR comments and reviews
- In release notes
- In `CONTRIBUTORS.md` (coming soon)
- On our website (coming soon)

## Next Steps

After your first contribution:

1. **Look for more issues**: Try slightly harder ones
2. **Help others**: Answer questions from new contributors
3. **Propose features**: Suggest new ideas
4. **Review PRs**: Provide feedback on others' contributions
5. **Become a regular contributor**: Join the core team

## Contribution Ideas

Not sure where to start? Here are some ideas:

### Quick Wins (< 1 hour)
- Fix typos in documentation
- Add missing docstrings
- Improve code comments
- Update dependencies in `requirements.txt`

### Medium Tasks (2-4 hours)
- Add unit tests for uncovered code
- Create a new example
- Write a tutorial
- Implement a new critic perspective

### Bigger Projects (1-2 days)
- Integrate with a new framework (Hugging Face, W&B)
- Implement a new algorithm variant
- Create comprehensive benchmarks
- Build a new feature

## Resources

- [Contributing Guidelines](../CONTRIBUTING.md)
- [Code of Conduct](../CODE_OF_CONDUCT.md)
- [Security Policy](../SECURITY.md)
- [Documentation](../docs/)
- [GitHub Issues](https://github.com/cogniolab/cognio-rlaf/issues)

---

**Ready to contribute? Pick an issue and let's build something amazing together!** 🚀
