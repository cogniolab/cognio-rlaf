# RLAF Documentation

Welcome to the RLAF (Reinforcement Learning from Agentic Feedback) documentation!

## 📚 Table of Contents

### Articles
- [**Introduction to RLAF**](articles/introduction-to-rlaf.md) - Comprehensive overview of RLAF, its innovations, and how it builds on ARPO, Open-AgentRL, and KAT-Dev
- [**Latest RLAIF/RLAF Developments (2024-2025)**](articles/latest-rlaif-developments-2024-2025.md) - 🆕 Cutting-edge research breakthroughs including Direct-RLAIF, RLTHF, Chain-of-Thought critics, Constitutional AI, self-improvement, and more

### Guides
- [Quick Start Guide](../README.md#-quick-start) - Get started in 30 seconds
- [Examples Overview](../examples/) - Working examples (ITSM, code generation, simple demo)

### API Reference
- [Core Classes](api/core.md) - BaseAgent, Feedback, AgentResponse (Coming Soon)
- [Algorithms](api/algorithms.md) - ARPO, GRPO-TCR, PPO, DPO (Coming Soon)

## 🎯 Quick Links

- [Main README](../README.md)
- [GitHub Repository](https://github.com/cogniolab/cognio-rlaf)
- [Contributing Guidelines](../CONTRIBUTING.md)
- [License](../LICENSE)

## 📖 What is RLAF?

RLAF (Reinforcement Learning from Agentic Feedback) is a unified framework for training AI agents using multi-perspective critic ensembles.

Instead of single scalar rewards, RLAF uses multiple specialized critics that evaluate agent responses from different perspectives:
- ✅ Accuracy
- ✅ Reasoning
- ✅ Tool Use
- ✅ Code Quality
- ✅ Policy Compliance
- ✅ Speed
- ✅ Safety

## 🚀 Getting Started

```bash
# Install
pip install git+https://github.com/cogniolab/cognio-rlaf.git

# Run examples
python examples/simple_demo.py
python examples/itsm_agent.py
python examples/code_generation.py
```

## 🤝 Community

- **Issues**: [GitHub Issues](https://github.com/cogniolab/cognio-rlaf/issues)
- **Discussions**: [GitHub Discussions](https://github.com/cogniolab/cognio-rlaf/discussions)
- **Email**: moses@cogniolab.com

---

**Built with ❤️ by [Cognio Lab](https://cogniolab.com)**
