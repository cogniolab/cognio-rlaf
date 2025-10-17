# RLAF: Reinforcement Learning from Agentic Feedback

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

Coming soon: Comparison with Open-AgentRL, ARPO, and baseline RL methods.

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas:
- New critic perspectives
- Additional RL algorithms
- Domain-specific examples
- Performance optimizations

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
