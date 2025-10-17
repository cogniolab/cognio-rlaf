"""
PPO: Proximal Policy Optimization

Standard RL algorithm for agentic systems.
Used as baseline and fallback in RLAF.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..core.base import AgentResponse
from ..rewards.aggregator import RewardSignal

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm."""

    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    batch_size: int = 32
    num_epochs: int = 3
    gae_lambda: float = 0.95


class PPOAlgorithm:
    """
    Proximal Policy Optimization for agentic RL.

    Standard PPO with multi-critic reward integration.
    """

    def __init__(self, config: Optional[PPOConfig] = None):
        self.config = config or PPOConfig()
        self.iteration = 0
        logger.info("Initialized PPO algorithm")

    def compute_loss(
        self,
        responses: List[AgentResponse],
        reward_signals: List[RewardSignal],
        old_log_probs: List[float],
        values: List[float],
    ) -> Dict[str, float]:
        """Compute PPO loss."""
        rewards = [rs.reward for rs in reward_signals]

        # Compute advantages using GAE
        advantages = self._compute_advantages(rewards, values)

        # Policy loss with clipping
        policy_loss = self._compute_policy_loss(
            old_log_probs, advantages
        )

        # Value loss
        value_loss = self._compute_value_loss(values, rewards)

        # Entropy bonus
        entropy = 0.01  # Placeholder

        total_loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * entropy
        )

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }

    def _compute_advantages(
        self, rewards: List[float], values: List[float]
    ) -> List[float]:
        """Compute GAE advantages."""
        # Simplified GAE
        advantages = [r - v for r, v in zip(rewards, values)]
        return advantages

    def _compute_policy_loss(
        self, old_log_probs: List[float], advantages: List[float]
    ) -> float:
        """PPO clipped policy loss."""
        # Placeholder
        return sum((1 - a) ** 2 for a in advantages) / len(advantages)

    def _compute_value_loss(
        self, values: List[float], rewards: List[float]
    ) -> float:
        """MSE value loss."""
        return sum((v - r) ** 2 for v, r in zip(values, rewards)) / len(values)

    def update_policy(self, loss_dict: Dict[str, float]) -> Dict[str, Any]:
        """Update policy."""
        self.iteration += 1
        logger.info(f"PPO update #{self.iteration}: loss={loss_dict['total_loss']:.4f}")
        return {"iteration": self.iteration}
