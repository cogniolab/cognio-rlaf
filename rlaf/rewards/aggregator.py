"""Reward Aggregator - Converts feedback to RL reward signals."""

import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from ..core.base import Feedback
from ..feedback.collector import AggregatedFeedback, FeedbackCollector

logger = logging.getLogger(__name__)


@dataclass
class RewardSignal:
    """Reward signal for reinforcement learning."""

    reward: float  # Final reward value
    feedback_score: float  # Raw aggregated feedback score
    bonus: float = 0.0  # Bonus reward (e.g., for tool use, speed)
    penalty: float = 0.0  # Penalty (e.g., for policy violations)
    components: Dict[str, float] = None  # Breakdown of reward components
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.components is None:
            self.components = {}
        if self.metadata is None:
            self.metadata = {}


class RewardAggregator:
    """
    Aggregates multi-critic feedback into RL reward signals.

    This is a key component of RLAF:
    - Converts structured feedback to scalar rewards
    - Supports reward shaping and bonuses
    - Handles multi-objective optimization
    - Integrates with different RL algorithms

    Example:
        >>> aggregator = RewardAggregator(
        ...     feedback_collector=FeedbackCollector(strategy="weighted_average"),
        ...     reward_scale=1.0,
        ...     use_bonus=True
        ... )
        >>> reward = aggregator.compute_reward(feedback_list, context)
    """

    def __init__(
        self,
        feedback_collector: Optional[FeedbackCollector] = None,
        reward_scale: float = 1.0,
        use_bonus: bool = True,
        bonus_weights: Optional[Dict[str, float]] = None,
        clip_range: Optional[tuple] = None,
    ):
        """
        Initialize reward aggregator.

        Args:
            feedback_collector: FeedbackCollector instance
            reward_scale: Scale factor for rewards
            use_bonus: Whether to add bonus rewards
            bonus_weights: Weights for different bonus types
            clip_range: (min, max) range to clip rewards
        """
        self.feedback_collector = feedback_collector or FeedbackCollector()
        self.reward_scale = reward_scale
        self.use_bonus = use_bonus
        self.clip_range = clip_range

        # Default bonus weights
        self.bonus_weights = bonus_weights or {
            "tool_efficiency": 0.1,  # Bonus for efficient tool use
            "speed": 0.05,  # Bonus for fast responses
            "consensus": 0.1,  # Bonus for high critic consensus
            "safety": 0.15,  # Bonus for safe outputs
        }

        logger.info(f"Initialized RewardAggregator (scale={reward_scale})")

    def compute_reward(
        self,
        feedbacks: List[Feedback],
        context: Optional[Dict[str, Any]] = None,
    ) -> RewardSignal:
        """
        Compute reward signal from critic feedbacks.

        Args:
            feedbacks: List of Feedback from critics
            context: Optional context (task info, metrics, etc.)

        Returns:
            RewardSignal with computed reward and breakdown
        """
        context = context or {}

        # Aggregate feedback
        aggregated = self.feedback_collector.aggregate(feedbacks)

        # Base reward from feedback score
        base_reward = aggregated.aggregated_score * self.reward_scale

        # Compute bonuses
        bonus = 0.0
        bonus_components = {}

        if self.use_bonus:
            bonus, bonus_components = self._compute_bonuses(
                aggregated, feedbacks, context
            )

        # Compute penalties
        penalty, penalty_components = self._compute_penalties(feedbacks, context)

        # Final reward
        final_reward = base_reward + bonus - penalty

        # Clip if range specified
        if self.clip_range:
            final_reward = max(
                self.clip_range[0], min(self.clip_range[1], final_reward)
            )

        # Build components breakdown
        components = {
            "base_reward": base_reward,
            "bonus": bonus,
            "penalty": penalty,
            **{f"bonus_{k}": v for k, v in bonus_components.items()},
            **{f"penalty_{k}": v for k, v in penalty_components.items()},
        }

        return RewardSignal(
            reward=final_reward,
            feedback_score=aggregated.aggregated_score,
            bonus=bonus,
            penalty=penalty,
            components=components,
            metadata={
                "consensus_level": aggregated.consensus_level,
                "num_critics": len(feedbacks),
                "aggregation_method": aggregated.aggregation_method,
            },
        )

    def _compute_bonuses(
        self,
        aggregated: AggregatedFeedback,
        feedbacks: List[Feedback],
        context: Dict[str, Any],
    ) -> tuple[float, Dict[str, float]]:
        """Compute bonus rewards."""
        bonuses = {}

        # Consensus bonus: High critic agreement
        if aggregated.consensus_level > 0.8:
            bonuses["consensus"] = (
                self.bonus_weights["consensus"] * aggregated.consensus_level
            )

        # Tool efficiency bonus (from context)
        if "tool_calls" in context:
            tool_efficiency = self._calculate_tool_efficiency(context["tool_calls"])
            if tool_efficiency > 0.7:
                bonuses["tool_efficiency"] = (
                    self.bonus_weights["tool_efficiency"] * tool_efficiency
                )

        # Speed bonus (from context)
        if "response_time" in context:
            speed_score = self._calculate_speed_score(context["response_time"])
            if speed_score > 0.5:
                bonuses["speed"] = self.bonus_weights["speed"] * speed_score

        # Safety bonus: Check for safety-focused feedback
        safety_feedbacks = [f for f in feedbacks if "safety" in f.critic_name.lower()]
        if safety_feedbacks:
            avg_safety_score = sum(f.score for f in safety_feedbacks) / len(
                safety_feedbacks
            )
            if avg_safety_score > 0.8:
                bonuses["safety"] = self.bonus_weights["safety"] * avg_safety_score

        total_bonus = sum(bonuses.values())
        return total_bonus, bonuses

    def _compute_penalties(
        self, feedbacks: List[Feedback], context: Dict[str, Any]
    ) -> tuple[float, Dict[str, float]]:
        """Compute penalty reductions."""
        penalties = {}

        # Policy violation penalty
        policy_feedbacks = [f for f in feedbacks if "policy" in f.critic_name.lower()]
        if policy_feedbacks:
            avg_policy_score = sum(f.score for f in policy_feedbacks) / len(
                policy_feedbacks
            )
            if avg_policy_score < 0.5:
                # Low policy score = penalty
                penalties["policy_violation"] = 0.2 * (1.0 - avg_policy_score)

        # Error penalty (from context)
        if context.get("has_errors", False):
            penalties["errors"] = 0.3

        # Timeout penalty
        if context.get("timeout", False):
            penalties["timeout"] = 0.15

        # Harmful content penalty
        safety_feedbacks = [f for f in feedbacks if "safety" in f.critic_name.lower()]
        if safety_feedbacks:
            avg_safety_score = sum(f.score for f in safety_feedbacks) / len(
                safety_feedbacks
            )
            if avg_safety_score < 0.3:
                penalties["harmful_content"] = 0.5 * (1.0 - avg_safety_score)

        total_penalty = sum(penalties.values())
        return total_penalty, penalties

    def _calculate_tool_efficiency(self, tool_calls: List[Dict]) -> float:
        """
        Calculate tool use efficiency score.

        Considers:
        - Number of redundant calls
        - Success rate
        - Appropriateness
        """
        if not tool_calls:
            return 1.0  # No tools = perfect efficiency

        # Count successful calls
        successful = sum(1 for call in tool_calls if call.get("success", False))
        success_rate = successful / len(tool_calls)

        # Penalize excessive tool calls (>5 is probably redundant)
        redundancy_penalty = max(0, (len(tool_calls) - 5) * 0.1)

        efficiency = success_rate - redundancy_penalty
        return max(0.0, min(1.0, efficiency))

    def _calculate_speed_score(self, response_time: float) -> float:
        """
        Calculate speed bonus score.

        Fast responses get bonus, slow responses get penalty.
        Uses sigmoid function centered at acceptable time.
        """
        # Acceptable time: 2 seconds
        # Fast: <1s gets bonus
        # Slow: >5s gets penalty
        acceptable_time = 2.0

        if response_time < acceptable_time:
            # Fast = high score
            speed_score = 1.0 - (response_time / acceptable_time) * 0.5
        else:
            # Slow = low score
            speed_score = max(0.0, 1.0 - (response_time - acceptable_time) / 5.0)

        return max(0.0, min(1.0, speed_score))

    def batch_compute_rewards(
        self,
        feedback_batches: List[List[Feedback]],
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RewardSignal]:
        """
        Compute rewards for batch of feedback lists.

        Args:
            feedback_batches: List of feedback lists (one per response)
            contexts: Optional list of context dicts

        Returns:
            List of RewardSignal objects
        """
        if contexts is None:
            contexts = [{}] * len(feedback_batches)

        rewards = []
        for feedbacks, context in zip(feedback_batches, contexts):
            reward = self.compute_reward(feedbacks, context)
            rewards.append(reward)

        logger.info(
            f"Computed {len(rewards)} rewards (avg: {sum(r.reward for r in rewards) / len(rewards):.3f})"
        )
        return rewards

    def get_reward_statistics(self, rewards: List[RewardSignal]) -> Dict[str, float]:
        """Get statistics from list of rewards."""
        if not rewards:
            return {}

        reward_values = [r.reward for r in rewards]

        return {
            "mean": sum(reward_values) / len(reward_values),
            "min": min(reward_values),
            "max": max(reward_values),
            "std": (
                sum((r - sum(reward_values) / len(reward_values)) ** 2 for r in reward_values)
                / len(reward_values)
            )
            ** 0.5,
            "avg_bonus": sum(r.bonus for r in rewards) / len(rewards),
            "avg_penalty": sum(r.penalty for r in rewards) / len(rewards),
            "avg_consensus": sum(r.metadata["consensus_level"] for r in rewards)
            / len(rewards),
        }
