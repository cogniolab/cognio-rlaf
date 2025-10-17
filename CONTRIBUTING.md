# Contributing to RLAF

Thank you for your interest in contributing to RLAF! 🎉

## How to Contribute

### 1. Report Bugs

Open an issue with:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)

### 2. Suggest Enhancements

Open an issue with:
- Clear use case
- Proposed solution
- Examples or mockups

### 3. Submit Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass (`pytest`)
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings (Google style)
- Format with `black`
- Lint with `ruff`

```bash
# Format code
black rlaf/

# Lint
ruff check rlaf/

# Type check
mypy rlaf/
```

### Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=rlaf tests/
```

## Areas for Contribution

### High Priority

1. **New Critic Perspectives**
   - Domain-specific critics (healthcare, finance, legal)
   - Custom evaluation metrics
   - Multi-modal critics (image, audio)

2. **Algorithm Implementations**
   - Complete KAT-style multi-stage training
   - Integration with TRL library
   - Custom reward shaping strategies

3. **Examples & Tutorials**
   - More domain examples
   - Step-by-step tutorials
   - Video walkthroughs

4. **Benchmarks**
   - Comparison with Open-AgentRL
   - Comparison with ARPO
   - Multi-domain benchmark suite

### Medium Priority

5. **Performance Optimization**
   - Parallel critic evaluation
   - Batch processing improvements
   - Memory optimization

6. **Integrations**
   - Hugging Face integration
   - Weights & Biases logging
   - MLflow tracking

7. **Documentation**
   - API reference
   - Architecture diagrams
   - Best practices guide

### Future Ideas

8. **Advanced Features**
   - Self-improving critics
   - Hierarchical critic ensembles
   - Federated training
   - Critic marketplace

## Development Setup

```bash
# Clone repo
git clone https://github.com/cogniolab/cognio-rlaf.git
cd cognio-rlaf

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest
```

## Questions?

- Open a GitHub issue
- Email: moses@cogniolab.com
- Join our Discord (coming soon)

## Code of Conduct

Be respectful and constructive. We're building this together!

---

**Thank you for contributing to RLAF!** 🚀
