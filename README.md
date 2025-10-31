# RLAF: Reinforcement Learning from Agentic Feedback

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/cogniolab/cognio-rlaf/workflows/CI/badge.svg)](https://github.com/cogniolab/cognio-rlaf/actions)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

**A unified framework for training AI agents using multi-perspective critic ensembles.**

RLAF (Reinforcement Learning from Agentic Feedback) combines innovations from the latest research in agentic reinforcement learning:
- **ARPO** (July 2025): Adaptive rollout based on entropy
- **Open-AgentRL** (Oct 2025): GRPO-TCR with tool-call reasoning
- **KAT-Dev** (Sept 2025): Multi-stage training pipeline

## 🎯 Why RLAF?

Traditional RL uses **single scalar rewards**. RLAF uses **multi-perspective critic ensembles**:

```python
# Traditional RL: Single reward
reward = 0.75  # Good? Bad? Why?

# RLAF: Multi-critic feedback
feedbacks = [
    Feedback(critic="accuracy", score=0.9, reasoning="Factually correct"),
    Feedback(critic="policy", score=0.6, reasoning="SLA violation risk"),
    Feedback(critic="efficiency", score=0.8, reasoning="Could be faster"),
]
# Aggregated reward: 0.77 (with rich context!)
```

**Key Benefits:**
- 🎭 **Multi-perspective evaluation** - Accuracy, reasoning, tool use, code quality, policy compliance
- 🔄 **Algorithm-agnostic** - Supports ARPO, GRPO-TCR, PPO, DPO
- 🏭 **Production-ready** - Not just research, built for real applications
- 🌐 **Cross-domain** - ITSM, code generation, reasoning tasks, chatbots

## 📚 Documentation

- **[Introduction to RLAF](docs/articles/introduction-to-rlaf.md)** - Comprehensive article on RLAF's innovations and how it builds on ARPO, Open-AgentRL, and KAT-Dev
- **[Full Documentation](docs/)** - Guides, API reference, and more

## 🚀 Quick Start

### Installation

```bash
pip install rlaf
```

Or install from source:

```bash
git clone https://github.com/cogniolab/cognio-rlaf.git
cd cognio-rlaf
pip install -e .
```

### Minimal Example (30 seconds)

```python
import asyncio
from rlaf import RLAFTrainer
from rlaf.agents import ActorAgent, CriticAgent, CriticEnsemble
from rlaf.core.trainer import TrainingConfig

async def main():
    # 1. Create actor (agent to train)
    actor = ActorAgent(
        name="my-agent",
        model="claude-3-5-sonnet-20241022",
        api_key="your-api-key"
    )

    # 2. Create multi-critic ensemble
    critics = CriticEnsemble([
        CriticAgent("accuracy-critic", "accuracy", api_key="your-api-key"),
        CriticAgent("reasoning-critic", "reasoning", api_key="your-api-key"),
    ])

    # 3. Configure training
    config = TrainingConfig(algorithm="arpo", max_iterations=10)

    # 4. Train!
    trainer = RLAFTrainer(actor=actor, critics=critics, config=config)
    results = await trainer.train(your_dataset)

asyncio.run(main())
```

## 📚 Examples

### ITSM Agent Training

Train an IT service management agent to triage incidents:

```bash
python examples/itsm_agent.py
```

**Features:**
- Actor: ITSM triage agent
- Critics: Accuracy, policy compliance, speed
- Algorithm: ARPO (adaptive exploration)

### Code Generation Agent

Train a Python code generation agent:

```bash
python examples/code_generation.py
```

**Features:**
- Actor: Code generator
- Critics: Correctness, code quality, efficiency
- Algorithm: GRPO-TCR (tool-call reasoning)

### Simple Demo

See `examples/simple_demo.py` for a minimal working example.

## 🏗️ Architecture

### Core Components

```
rlaf/
├── agents/          # Actor and Critic agents
│   ├── actor.py     # Agent being trained
│   └── critic.py    # Evaluation agents
├── algorithms/      # RL algorithms
│   ├── arpo.py      # Adaptive RPO (entropy-based)
│   ├── grpo_tcr.py  # Tool-call reasoning (Open-AgentRL)
│   ├── ppo.py       # Proximal Policy Optimization
│   └── dpo.py       # Direct Preference Optimization
├── feedback/        # Feedback collection
│   └── collector.py # Multi-critic aggregation
├── rewards/         # Reward computation
│   └── aggregator.py # Feedback → RL rewards
└── core/
    ├── base.py      # Base classes
    └── trainer.py   # Main trainer
```

### Multi-Critic Feedback Flow

```
Input Task
    ↓
[Actor] generates response
    ↓
[Critics] evaluate from multiple perspectives
    ├─ Accuracy Critic → score: 0.9
    ├─ Reasoning Critic → score: 0.8
    ├─ Tool Use Critic → score: 0.7
    └─ Policy Critic → score: 0.85
    ↓
[Feedback Collector] aggregates (weighted avg, voting, debate)
    ↓
[Reward Aggregator] converts to RL reward (with bonuses/penalties)
    ↓
[Algorithm] updates policy (ARPO/GRPO-TCR/PPO/DPO)
```

## 🔬 Algorithms

### ARPO: Adaptive Reinforcement Policy Optimization

**From July 2025 paper (arXiv:2507.19849)**

Key innovation: Entropy-based adaptive rollout
- High uncertainty → more exploration
- Low confidence → increase batch size
- Adaptive learning rate scaling

```python
config = TrainingConfig(
    algorithm="arpo",
    entropy_threshold=0.8,
    adaptive_rollout=True
)
```

### GRPO-TCR: Tool-Call Reasoning

**From Open-AgentRL (Oct 13, 2025)**

Key innovation: Deliberative reasoning before tool calls
- 4B model outperforms 32B models
- Selective tool use (avoid over-calling)
- SOTA on AIME, GPQA, LiveCodeBench

```python
config = TrainingConfig(
    algorithm="grpo-tcr",
    tool_call_reasoning=True,
    deliberative_mode=True
)
```

### KAT-Style: Multi-Stage Training

**From KAT-Dev (Sept 2025)**

3-stage pipeline:
1. **Mid-training**: Enhance LLM-as-agent capabilities
2. **RFT**: Reinforcement fine-tuning with teacher trajectories
3. **Agentic RL**: Full RL with critic ensemble

```python
config = TrainingConfig(
    algorithm="kat",
    multi_stage=True,
    stages=["mid_train", "rft", "agentic_rl"]
)
```

## 🎨 Critic Perspectives

RLAF supports multiple critic perspectives:

| Perspective | Evaluates | Example Use Case |
|-------------|-----------|------------------|
| `accuracy` | Factual correctness | Q&A, reasoning |
| `reasoning` | Logical soundness | Math, planning |
| `tool_use` | Tool efficiency | Agent workflows |
| `code_quality` | Code quality | Code generation |
| `policy` | SLA/rule compliance | ITSM, enterprise |
| `speed` | Response efficiency | Real-time systems |
| `safety` | Security/ethics | Production deployment |

Create custom perspectives:

```python
custom_critic = CriticAgent(
    name="domain-expert",
    perspective="medical_accuracy",  # Custom perspective
    model="claude-3-5-sonnet-20241022",
    api_key="your-key"
)
```

## 📊 Reward Aggregation Strategies

RLAF offers multiple ways to aggregate multi-critic feedback:

### 1. Weighted Average (default)
```python
# Confidence-weighted average
config.reward_aggregation = "weighted_average"
```

### 2. Voting
```python
# Majority vote on quality threshold
config.reward_aggregation = "voting"
```

### 3. Debate
```python
# Highest-confidence critic wins
config.reward_aggregation = "debate"
```

### 4. Consensus
```python
# Accept only high-agreement feedback
config.reward_aggregation = "consensus"
```

## 🛠️ Configuration

### TrainingConfig

```python
from rlaf.core.trainer import TrainingConfig

config = TrainingConfig(
    # Algorithm
    algorithm="arpo",  # arpo, grpo-tcr, kat, ppo, dpo

    # Training
    max_iterations=1000,
    batch_size=32,
    learning_rate=3e-4,

    # ARPO-specific
    entropy_threshold=0.8,
    adaptive_rollout=True,

    # GRPO-TCR-specific
    tool_call_reasoning=True,
    deliberative_mode=True,

    # Rewards
    reward_aggregation="weighted_average",

    # Logging
    checkpoint_every=100,
    eval_every=50,
)
```

### BaseConfig

```python
from rlaf.core.base import BaseConfig

config = BaseConfig(
    model_name="claude-3-5-sonnet-20241022",
    temperature=0.7,
    max_tokens=2048,
    num_critics=3,
)
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Run examples:

```bash
# Simple demo
python examples/simple_demo.py

# ITSM agent
export ANTHROPIC_API_KEY="your-key"
python examples/itsm_agent.py

# Code generation
python examples/code_generation.py
```

## 📈 Benchmarks

**Comprehensive benchmarks comparing RLAF with baseline methods are now available!**

### Quick Results

| Method | ITSM Triage | Code Generation | Reasoning | Avg. Score | Training Time |
|--------|-------------|-----------------|-----------|------------|---------------|
| **RLAF (ARPO)** | **87.3%** | **82.5%** | **79.8%** | **83.2%** | 3.2h |
| **RLAF (GRPO-TCR)** | 85.1% | **84.2%** | 81.3% | **83.5%** | 4.1h |
| Open-AgentRL | 82.4% | 80.1% | **82.1%** | 81.5% | 5.3h |
| PPO | 76.2% | 74.3% | 73.1% | 74.5% | 6.1h |
| DPO | 74.8% | 76.5% | 71.9% | 74.4% | 4.8h |

**Key Findings:**
- ✅ **12.4% improvement** over supervised fine-tuning
- ✅ **35% faster training** than Open-AgentRL
- ✅ **43% cost savings** with intelligent model routing
- ✅ **40% fewer samples** needed to reach 80% performance vs PPO

**See full benchmarks:** [benchmarks/README.md](benchmarks/README.md)

### Run Benchmarks Yourself

```bash
# Run all benchmarks
python benchmarks/run_all.py

# Generate charts
python benchmarks/visualize.py
```

## 🤝 Contributing

We welcome contributions from everyone! Whether you're fixing a typo, adding tests, or implementing new features, your contributions are valued.

### Quick Start for Contributors

- **New to the project?** Check out our [Good First Issues Guide](docs/GOOD_FIRST_ISSUES.md)
- **Ready to contribute?** Read our [Contributing Guidelines](CONTRIBUTING.md)
- **Found a security issue?** See our [Security Policy](SECURITY.md)
- **Want to report a bug?** Use our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.yml)
- **Have a feature idea?** Use our [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.yml)

### Areas We Need Help

- **New critic perspectives** - Domain-specific critics (healthcare, finance, legal)
- **Additional RL algorithms** - REINFORCE, A2C, SAC variants
- **Domain-specific examples** - Real-world use cases and tutorials
- **Performance optimizations** - Parallel processing, memory efficiency
- **Documentation** - Guides, tutorials, API docs
- **Tests** - Unit tests, integration tests, benchmarks

### Recognition

All contributors are recognized in our release notes and will be featured in our Contributors Hall of Fame.

**Join us in making AI agents smarter!** 🚀

## 📝 Citation

If you use RLAF in your research, please cite:

```bibtex
@software{rlaf2025,
  title = {RLAF: Reinforcement Learning from Agentic Feedback},
  author = {Cognio Lab},
  year = {2025},
  url = {https://github.com/cogniolab/cognio-rlaf}
}
```

## 🔗 Related Work

RLAF builds on these excellent projects:

- **ARPO** (July 2025): [arXiv:2507.19849](https://arxiv.org/abs/2507.19849)
- **Open-AgentRL** (Oct 2025): [GitHub](https://github.com/Gen-Verse/Open-AgentRL)
- **KAT-Dev** (Sept 2025): [Skywork AI Blog](https://skywork.ai/blog/agentic-reinforcement-learning-code-generation/)
- **IBM Multi-Agent Learning**: [Research Blog](https://research.ibm.com/blog/what-is-multi-agent-system)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Anthropic for Claude API
- OpenAI for RL research foundations
- Open-AgentRL team at Gen-Verse
- ARPO authors
- KAT-Dev team at Skywork/Kuaishou

---

**Built with ❤️ by [Cognio Lab](https://cogniolab.com)**

*Making AI agents smarter through multi-perspective feedback.*
