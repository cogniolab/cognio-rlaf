# RLAF API Documentation

## Overview

RLAF (Reinforcement Learning from Agentic Feedback) provides a unified framework for training AI agents using multi-perspective critic ensembles. This API enables seamless integration of diverse feedback sources for improved agent learning.

## Core Classes

### RLAFTrainer

Main training orchestrator for RLAF framework.

```python
from cognio_rlaf import RLAFTrainer, CriticEnsemble

trainer = RLAFTrainer(
    model_name="gpt-3.5-turbo",
    ensemble_size=3,
    learning_rate=1e-4,
    batch_size=32
)
```

**Parameters:**
- `model_name` (str): Base model identifier
- `ensemble_size` (int): Number of critic perspectives (default: 3)
- `learning_rate` (float): Optimization learning rate
- `batch_size` (int): Training batch size

### CriticEnsemble

Multi-perspective feedback aggregator.

```python
ensemble = CriticEnsemble(
    critics=["technical", "domain_expert", "user_preference"],
    aggregation_method="weighted_mean",
    weights=[0.4, 0.35, 0.25]
)
```

**Methods:**
- `add_critic(name: str, weight: float) -> None`: Register new critic
- `get_feedback(trajectory: List[Experience]) -> Dict[str, Any]`: Aggregate feedback
- `update_weights(weights: Dict[str, float]) -> None`: Adjust critic weights

### Agent

Base agent interface for RLAF training.

```python
from cognio_rlaf import Agent

agent = Agent(
    policy_model="checkpoint.pt",
    action_space=env.action_space,
    state_encoder="default"
)

action, log_prob = agent.act(observation, training=True)
```

## Training Loop

```python
from cognio_rlaf import RLAFTrainer, CriticEnsemble, Agent
import gymnasium as gym

env = gym.make("CartPole-v1")
agent = Agent(policy_model="policy.pt", action_space=env.action_space)
ensemble = CriticEnsemble(critics=["reward", "safety", "efficiency"])
trainer = RLAFTrainer(model_name="base-agent", ensemble_size=3)

for epoch in range(100):
    trajectories = []
    for episode in range(10):
        obs, _ = env.reset()
        done = False
        trajectory = []
        
        while not done:
            action, log_prob = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append((obs, action, reward, log_prob))
            done = terminated or truncated
        
        trajectories.append(trajectory)
    
    # Aggregate feedback from ensemble
    feedback = ensemble.get_feedback(trajectories)
    
    # Update agent
    loss = trainer.train_step(agent, trajectories, feedback)
    print(f"Epoch {epoch}: Loss={loss:.4f}")
```

## Advanced Configuration

```python
config = {
    "trainer": {
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
        "gradient_clip": 1.0
    },
    "ensemble": {
        "aggregation": "weighted_mean",
        "diversity_penalty": 0.1,
        "critic_update_freq": 5
    },
    "agent": {
        "entropy_coeff": 0.01,
        "gae_lambda": 0.95,
        "num_steps": 2048
    }
}

trainer = RLAFTrainer(**config)
```

## Callback System

```python
from cognio_rlaf import Callback

class MetricsCallback(Callback):
    def on_epoch_end(self, epoch, metrics):
        print(f"Epoch {epoch}: Reward={metrics['avg_reward']:.2f}")

trainer.add_callback(MetricsCallback())
```

## Error Handling

```python
from cognio_rlaf.exceptions import (
    CriticConfigError,
    TrajectoryError,
    ModelLoadError
)

try:
    trainer.train_step(agent, trajectories, feedback)
except TrajectoryError as e:
    print(f"Invalid trajectory: {e}")
except CriticConfigError as e:
    print(f"Ensemble configuration error: {e}")
```

## Performance Tips

1. Use checkpoint saving for fault tolerance
2. Parallelize trajectory collection across environments
3. Monitor critic agreement to detect training instability
4. Adjust ensemble weights based on historical performance

## See Also

- Documentation: https://github.com/cognio-rlaf/docs
- Examples: https://github.com/cognio-rlaf/examples
- Issue Tracker: https://github.com/cognio-rlaf/issues