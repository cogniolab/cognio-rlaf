"""
DPO: Direct Preference Optimization

Alternative to PPO that learns from preference pairs.
Useful for multi-critic scenarios where critics provide rankings.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..core.base import AgentResponse, Feedback

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Configuration for DPO algorithm."""

    learning_rate: float = 5e-7
    beta: float = 0.1  # KL penalty coefficient
    batch_size: int = 32
    num_epochs: int = 3


class DPOAlgorithm:
    """
    Direct Preference Optimization for agentic RL.

    Learns from preference pairs instead of scalar rewards.
    Multi-critic feedback naturally provides preferences.
    """

    def __init__(self, config: Optional[DPOConfig] = None):
        self.config = config or DPOConfig()
        self.iteration = 0
        logger.info("Initialized DPO algorithm")

    def compute_loss(
        self,
        response_pairs: List[Tuple[AgentResponse, AgentResponse]],
        feedback_pairs: List[Tuple[List[Feedback], List[Feedback]]],
        log_prob_pairs: List[Tuple[float, float]],
        ref_log_prob_pairs: List[Tuple[float, float]],
    ) -> Dict[str, float]:
        """
        Compute DPO loss from preference pairs.

        Args:
            response_pairs: (preferred, non-preferred) response pairs
            feedback_pairs: Feedback for each response
            log_prob_pairs: Log probs for each response
            ref_log_prob_pairs: Reference model log probs

        Returns:
            Loss dict
        """
        total_loss = 0.0

        for i, ((lp_w, lp_l), (ref_w, ref_l)) in enumerate(
            zip(log_prob_pairs, ref_log_prob_pairs)
        ):
            # DPO loss: -log(σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))
            logit = self.config.beta * ((lp_w - lp_l) - (ref_w - ref_l))
            loss = -self._log_sigmoid(logit)
            total_loss += loss

        avg_loss = total_loss / len(response_pairs) if response_pairs else 0.0

        return {"total_loss": avg_loss, "dpo_loss": avg_loss}

    def _log_sigmoid(self, x: float) -> float:
        """Numerically stable log sigmoid."""
        if x >= 0:
            return -self._log1p_exp(-x)
        else:
            return x - self._log1p_exp(x)

    def _log1p_exp(self, x: float) -> float:
        """log(1 + exp(x))"""
        import math

        if x > 20:
            return x
        else:
            return math.log1p(math.exp(x))

    def create_preference_pairs(
        self,
        responses: List[AgentResponse],
        feedback_list: List[List[Feedback]],
    ) -> List[Tuple[int, int]]:
        """
        Create preference pairs from multi-critic feedback.

        Returns indices of (preferred, non-preferred) pairs.
        """
        pairs = []

        # Compute aggregate scores
        scores = []
        for feedbacks in feedback_list:
            avg_score = sum(f.score for f in feedbacks) / len(feedbacks)
            scores.append(avg_score)

        # Create pairs
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                if scores[i] > scores[j]:
                    pairs.append((i, j))  # i is preferred
                elif scores[j] > scores[i]:
                    pairs.append((j, i))  # j is preferred

        return pairs

    def update_policy(self, loss_dict: Dict[str, float]) -> Dict[str, Any]:
        """Update policy."""
        self.iteration += 1
        logger.info(f"DPO update #{self.iteration}: loss={loss_dict['total_loss']:.4f}")
        return {"iteration": self.iteration}
