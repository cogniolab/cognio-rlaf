# RLAF API Documentation

## Overview

RLAF (Reinforcement Learning from Agentic Feedback) provides a unified framework for training AI agents using multi-perspective critic ensembles. This API enables seamless integration of diverse feedback sources for robust agent training.

## Installation

```bash
pip install cognio-rlaf
```

## Core Classes

### Agent

The primary interface for training agents with agentic feedback.

```python
from rlaf import Agent, CriticEnsemble

# Initialize agent
agent = Agent(
    model_name="gpt-4",
    max_tokens=2048,
    temperature=0.7
)

# Configure critic ensemble
critics = CriticEnsemble(
    critics=[
        "safety_critic",
        "accuracy_critic",
        "efficiency_critic"
    ],
    aggregation="weighted_mean"
)

# Train with feedback
feedback = agent.train(
    task="summarize_text",
    data=training_data,
    critics=critics,
    epochs=10
)
```

### CriticEnsemble

Manages multiple critic perspectives for comprehensive feedback.

```python
from rlaf import CriticEnsemble, SafetyCritic, AccuracyCritic

ensemble = CriticEnsemble(
    critics=[
        SafetyCritic(threshold=0.8),
        AccuracyCritic(metric="f1"),
        EfficientyCritic(max_latency_ms=500)
    ],
    weights=[0.3, 0.5, 0.2]
)

# Get aggregated feedback
feedback_scores = ensemble.evaluate(
    agent_output="model response",
    reference="ground truth"
)
```

### Trainer

High-level training orchestrator.

```python
from rlaf import Trainer, TrainerConfig

config = TrainerConfig(
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=20,
    critic_update_frequency=5,
    gradient_accumulation_steps=4
)

trainer = Trainer(config=config)

results = trainer.train(
    agent=agent,
    critics=ensemble,
    dataset=train_dataset,
    val_dataset=val_dataset,
    callbacks=[
        LoggingCallback(),
        CheckpointCallback(save_interval=100)
    ]
)
```

## Configuration

### TrainerConfig

```python
from rlaf import TrainerConfig

config = TrainerConfig(
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=20,
    critic_update_frequency=5,
    use_distributed=True,
    device="cuda",
    seed=42,
    log_interval=10
)
```

## Advanced Usage

### Custom Critics

```python
from rlaf import BaseCritic

class CustomCritic(BaseCritic):
    def evaluate(self, output: str, reference: str) -> float:
        # Custom evaluation logic
        score = self._compute_metric(output, reference)
        return score
    
    def get_feedback(self) -> dict:
        return {"score": self.score, "explanation": self.explanation}

custom_critic = CustomCritic(weight=0.25)
ensemble.add_critic(custom_critic)
```

### Callbacks

```python
from rlaf import Callback

class MetricsCallback(Callback):
    def on_epoch_end(self, epoch: int, metrics: dict):
        wandb.log({"epoch": epoch, **metrics})

trainer.train(callbacks=[MetricsCallback()])
```

## Error Handling

```python
from rlaf import RLAFException, ValidationError

try:
    results = trainer.train(agent, critics, dataset)
except ValidationError as e:
    print(f"Validation failed: {e}")
except RLAFException as e:
    print(f"RLAF error: {e}")
```

## Performance Optimization

- Use `use_distributed=True` for multi-GPU training
- Enable gradient accumulation for larger batch sizes
- Batch critic evaluations using `batch_evaluate()`
- Cache critic outputs when possible

## Monitoring

```python
from rlaf import MetricsCollector

collector = MetricsCollector()
results = trainer.train(
    agent=agent,
    critics=ensemble,
    dataset=dataset,
    metrics_collector=collector
)

collector.plot_training_curves()
collector.export_metrics("metrics.json")
```

## API Reference

- `Agent.train()` - Train agent with feedback
- `CriticEnsemble.evaluate()` - Get aggregated critic scores
- `Trainer.train()` - Full training pipeline
- `BaseCritic.evaluate()` - Individual critic evaluation

## Support

For issues and feature requests, visit: https://github.com/cognio/cognio-rlaf