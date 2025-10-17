"""
ARPO: Adaptive Reinforcement Policy Optimization

Based on July 2025 paper: arXiv:2507.19849
Key innovation: Entropy-based adaptive rollout for efficient exploration.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import math

from ..core.base import AgentResponse, Feedback
from ..rewards.aggregator import RewardSignal

logger = logging.getLogger(__name__)


@dataclass
class ARPOConfig:
    """Configuration for ARPO algorithm."""

    # Entropy-based adaptation
    entropy_threshold: float = 0.8  # High uncertainty trigger
    adaptive_rollout: bool = True
    rollout_multiplier: float = 1.5  # Increase rollouts when uncertain

    # Policy optimization
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # Training dynamics
    batch_size: int = 32
    num_epochs: int = 3
    gae_lambda: float = 0.95  # Generalized Advantage Estimation
    max_grad_norm: float = 0.5


class ARPOAlgorithm:
    """
    ARPO: Adaptive Reinforcement Policy Optimization.

    Key features from paper:
    1. Entropy-based adaptive rollout: Low confidence → more exploration
    2. Dynamic batch sizing based on uncertainty
    3. Adaptive learning rate scaling
    4. Multi-critic feedback integration

    Example:
        >>> arpo = ARPOAlgorithm(config=ARPOConfig())
        >>> loss = arpo.compute_loss(responses, rewards, feedback_list)
        >>> arpo.update_policy(loss)
    """

    def __init__(self, config: Optional[ARPOConfig] = None):
        """
        Initialize ARPO algorithm.

        Args:
            config: ARPO configuration
        """
        self.config = config or ARPOConfig()
        self.iteration = 0
        self.entropy_history: List[float] = []

        logger.info(
            f"Initialized ARPO (entropy_threshold={self.config.entropy_threshold})"
        )

    def compute_loss(
        self,
        responses: List[AgentResponse],
        reward_signals: List[RewardSignal],
        feedback_list: List[List[Feedback]],
        old_log_probs: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Compute ARPO loss with adaptive components.

        Args:
            responses: Actor responses
            reward_signals: Reward signals from aggregator
            feedback_list: Multi-critic feedback
            old_log_probs: Log probabilities from old policy (for PPO-style clipping)

        Returns:
            Dict with loss components
        """
        # Calculate entropy from feedback uncertainty
        entropy = self._calculate_entropy(feedback_list)
        self.entropy_history.append(entropy)

        # Check if we need adaptive rollout
        needs_exploration = entropy < self.config.entropy_threshold

        if needs_exploration and self.config.adaptive_rollout:
            logger.info(
                f"High entropy detected ({entropy:.3f}), triggering adaptive rollout"
            )
            # In practice: Sample more trajectories, increase batch size
            # For now, we'll adjust loss weighting

        # Policy loss (placeholder - in practice, use actual policy gradients)
        rewards = [rs.reward for rs in reward_signals]
        policy_loss = self._compute_policy_loss(responses, rewards, old_log_probs)

        # Value loss
        value_loss = self._compute_value_loss(responses, rewards)

        # Entropy bonus (encourage exploration)
        entropy_bonus = self.config.entropy_coef * entropy

        # Total loss
        total_loss = (
            policy_loss + self.config.value_coef * value_loss - entropy_bonus
        )

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "entropy_bonus": entropy_bonus,
            "needs_exploration": needs_exploration,
        }

    def _calculate_entropy(self, feedback_list: List[List[Feedback]]) -> float:
        """
        Calculate entropy from multi-critic feedback uncertainty.

        High variance in critic scores → high entropy → more exploration needed.
        """
        if not feedback_list:
            return 0.0

        # Collect all critic scores
        all_scores = []
        for feedbacks in feedback_list:
            scores = [f.score for f in feedbacks]
            all_scores.extend(scores)

        if not all_scores:
            return 0.0

        # Calculate mean and variance
        mean = sum(all_scores) / len(all_scores)
        variance = sum((s - mean) ** 2 for s in all_scores) / len(all_scores)

        # Normalize to 0-1 scale (higher variance → higher entropy)
        # Max variance is 0.25 when scores are binary (0 or 1)
        normalized_entropy = min(1.0, variance / 0.25)

        logger.debug(
            f"Entropy calculation: mean={mean:.3f}, var={variance:.3f}, "
            f"entropy={normalized_entropy:.3f}"
        )

        return normalized_entropy

    def _compute_policy_loss(
        self,
        responses: List[AgentResponse],
        rewards: List[float],
        old_log_probs: Optional[List[float]] = None,
    ) -> float:
        """
        Compute policy gradient loss with PPO-style clipping.

        In production, this would:
        1. Compute log probs of actions under current policy
        2. Compute advantage estimates
        3. Apply PPO clipping
        """
        # Placeholder: Simple policy gradient
        # Real implementation would use actual policy network

        if old_log_probs is None:
            # No clipping, standard policy gradient
            policy_loss = sum((1 - r) ** 2 for r in rewards) / len(rewards)
        else:
            # PPO-style clipped loss
            policy_loss = 0.0
            for i, (reward, old_lp) in enumerate(zip(rewards, old_log_probs)):
                # Simplified: In practice, compute ratio = new_prob / old_prob
                ratio = 1.0  # Placeholder
                advantage = reward - 0.5  # Placeholder advantage

                # Clipped objective
                clipped_ratio = max(
                    min(ratio, 1 + self.config.clip_ratio),
                    1 - self.config.clip_ratio,
                )
                loss = -min(ratio * advantage, clipped_ratio * advantage)
                policy_loss += loss

            policy_loss = policy_loss / len(rewards)

        return policy_loss

    def _compute_value_loss(
        self, responses: List[AgentResponse], rewards: List[float]
    ) -> float:
        """
        Compute value function loss.

        In production, this trains a value network to predict returns.
        """
        # Placeholder: MSE between predicted and actual returns
        # Real implementation would use actual value network
        predicted_values = [0.5] * len(rewards)  # Placeholder predictions

        value_loss = sum(
            (pred - actual) ** 2 for pred, actual in zip(predicted_values, rewards)
        ) / len(rewards)

        return value_loss

    def should_increase_rollout(self) -> bool:
        """
        Determine if rollout should be increased based on entropy.

        Returns True if recent entropy is below threshold.
        """
        if not self.entropy_history or not self.config.adaptive_rollout:
            return False

        # Check recent entropy (last 5 iterations)
        recent_entropy = self.entropy_history[-5:]
        avg_entropy = sum(recent_entropy) / len(recent_entropy)

        return avg_entropy < self.config.entropy_threshold

    def get_adaptive_batch_size(self) -> int:
        """
        Get adaptive batch size based on entropy.

        High entropy → larger batches for more exploration.
        """
        if not self.entropy_history:
            return self.config.batch_size

        recent_entropy = self.entropy_history[-1]

        if recent_entropy < self.config.entropy_threshold:
            # High uncertainty: increase batch size
            return int(self.config.batch_size * self.config.rollout_multiplier)
        else:
            # Normal batch size
            return self.config.batch_size

    def get_adaptive_learning_rate(self) -> float:
        """
        Get adaptive learning rate based on entropy.

        High entropy → lower learning rate (more cautious updates).
        """
        if not self.entropy_history:
            return self.config.learning_rate

        recent_entropy = self.entropy_history[-1]

        if recent_entropy < self.config.entropy_threshold:
            # High uncertainty: reduce learning rate
            return self.config.learning_rate * 0.5
        else:
            # Normal learning rate
            return self.config.learning_rate

    def update_policy(self, loss_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Update policy using ARPO algorithm.

        Args:
            loss_dict: Loss components from compute_loss()

        Returns:
            Update statistics
        """
        self.iteration += 1

        # In production, this would:
        # 1. Compute gradients
        # 2. Clip gradients
        # 3. Apply optimizer step
        # 4. Update value network

        # Placeholder update
        adaptive_lr = self.get_adaptive_learning_rate()
        adaptive_batch = self.get_adaptive_batch_size()

        logger.info(
            f"ARPO update #{self.iteration}: "
            f"loss={loss_dict['total_loss']:.4f}, "
            f"entropy={loss_dict['entropy']:.3f}, "
            f"lr={adaptive_lr:.2e}, "
            f"batch_size={adaptive_batch}"
        )

        return {
            "iteration": self.iteration,
            "learning_rate": adaptive_lr,
            "batch_size": adaptive_batch,
            "entropy": loss_dict["entropy"],
            "needs_exploration": loss_dict["needs_exploration"],
        }

    def reset(self):
        """Reset algorithm state."""
        self.iteration = 0
        self.entropy_history.clear()
        logger.info("ARPO algorithm reset")
