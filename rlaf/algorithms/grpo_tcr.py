"""
GRPO-TCR: Generalized RPO with Tool-Call Reasoning

Based on Open-AgentRL (Oct 13, 2025)
Key innovation: Deliberative reasoning with selective tool calls.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ..core.base import AgentResponse, Feedback
from ..rewards.aggregator import RewardSignal

logger = logging.getLogger(__name__)


class ReasoningMode(Enum):
    """Reasoning modes for GRPO-TCR."""

    DELIBERATIVE = "deliberative"  # Explicit reasoning steps
    DIRECT = "direct"  # Direct tool calls
    HYBRID = "hybrid"  # Mix of both


@dataclass
class GRPOTCRConfig:
    """Configuration for GRPO-TCR algorithm."""

    # Tool-call reasoning
    tool_call_reasoning: bool = True
    deliberative_mode: bool = True
    reasoning_mode: ReasoningMode = ReasoningMode.HYBRID

    # Data requirements (from Open-AgentRL paper)
    sft_samples: int = 3000
    rl_samples: int = 30000

    # Optimization
    learning_rate: float = 5e-6
    kl_coef: float = 0.05  # KL divergence coefficient
    clip_ratio: float = 0.2

    # Tool-specific
    tool_success_bonus: float = 0.1
    tool_efficiency_weight: float = 0.15
    max_tool_calls: int = 10


class GRPOTCRAlgorithm:
    """
    GRPO-TCR: Generalized Reward Policy Optimization with Tool-Call Reasoning.

    From Open-AgentRL paper (Gen-Verse, Oct 2025):
    - Deliberative reasoning before tool calls
    - Selective tool use (avoid over-calling)
    - 4B model outperforms 32B models
    - SOTA on AIME, GPQA, LiveCodeBench

    Example:
        >>> grpo = GRPOTCRAlgorithm(config=GRPOTCRConfig())
        >>> loss = grpo.compute_loss(responses, rewards, feedback_list)
        >>> grpo.update_policy(loss)
    """

    def __init__(self, config: Optional[GRPOTCRConfig] = None):
        """
        Initialize GRPO-TCR algorithm.

        Args:
            config: GRPO-TCR configuration
        """
        self.config = config or GRPOTCRConfig()
        self.iteration = 0
        self.tool_call_stats: List[Dict[str, Any]] = []

        logger.info(
            f"Initialized GRPO-TCR (mode={self.config.reasoning_mode.value}, "
            f"TCR={'enabled' if self.config.tool_call_reasoning else 'disabled'})"
        )

    def compute_loss(
        self,
        responses: List[AgentResponse],
        reward_signals: List[RewardSignal],
        feedback_list: List[List[Feedback]],
        old_log_probs: Optional[List[float]] = None,
        ref_log_probs: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Compute GRPO-TCR loss.

        Args:
            responses: Actor responses
            reward_signals: Reward signals from aggregator
            feedback_list: Multi-critic feedback
            old_log_probs: Log probs from old policy
            ref_log_probs: Log probs from reference model (for KL penalty)

        Returns:
            Dict with loss components
        """
        # Detect tool call patterns
        tool_stats = self._analyze_tool_calls(responses)
        self.tool_call_stats.append(tool_stats)

        # Base policy loss
        rewards = [rs.reward for rs in reward_signals]
        policy_loss = self._compute_policy_loss(responses, rewards, old_log_probs)

        # KL divergence penalty (keep close to reference model)
        kl_penalty = self._compute_kl_penalty(old_log_probs, ref_log_probs)

        # Tool-call reasoning reward shaping
        tcr_bonus = 0.0
        if self.config.tool_call_reasoning and tool_stats["has_tool_calls"]:
            tcr_bonus = self._compute_tcr_bonus(tool_stats, feedback_list)

        # Total loss
        total_loss = policy_loss + self.config.kl_coef * kl_penalty - tcr_bonus

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "kl_penalty": kl_penalty,
            "tcr_bonus": tcr_bonus,
            "tool_calls": tool_stats["num_tool_calls"],
            "tool_efficiency": tool_stats["efficiency"],
        }

    def _analyze_tool_calls(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """
        Analyze tool call patterns in responses.

        Checks for:
        - Number of tool calls
        - Tool call efficiency
        - Deliberative reasoning presence
        """
        total_tool_calls = 0
        successful_calls = 0
        has_reasoning = 0
        has_tool_calls = False

        for response in responses:
            metadata = response.metadata or {}

            # Check for tool calls
            if "tool_calls" in metadata:
                has_tool_calls = True
                tool_calls = metadata["tool_calls"]
                total_tool_calls += len(tool_calls)

                # Count successful calls
                successful_calls += sum(
                    1 for call in tool_calls if call.get("success", False)
                )

            # Check for reasoning (keywords in content)
            if self._has_deliberative_reasoning(response.content):
                has_reasoning += 1

        # Calculate efficiency
        efficiency = (
            successful_calls / total_tool_calls if total_tool_calls > 0 else 1.0
        )

        # Reasoning ratio
        reasoning_ratio = has_reasoning / len(responses) if responses else 0.0

        return {
            "num_tool_calls": total_tool_calls,
            "successful_calls": successful_calls,
            "efficiency": efficiency,
            "has_tool_calls": has_tool_calls,
            "reasoning_ratio": reasoning_ratio,
        }

    def _has_deliberative_reasoning(self, content: str) -> bool:
        """
        Check if content contains deliberative reasoning.

        Looks for reasoning keywords like: "Let me think", "First,", "Therefore", etc.
        """
        reasoning_keywords = [
            "let me think",
            "first,",
            "second,",
            "therefore",
            "because",
            "reasoning:",
            "analysis:",
            "step 1:",
            "to solve this",
            "my approach",
        ]

        content_lower = content.lower()
        return any(keyword in content_lower for keyword in reasoning_keywords)

    def _compute_policy_loss(
        self,
        responses: List[AgentResponse],
        rewards: List[float],
        old_log_probs: Optional[List[float]],
    ) -> float:
        """
        Compute policy loss with PPO-style clipping.

        GRPO uses similar approach to PPO but with tool-aware rewards.
        """
        if old_log_probs is None:
            # Standard policy gradient
            policy_loss = sum((1 - r) ** 2 for r in rewards) / len(rewards)
        else:
            # PPO-style clipped loss
            policy_loss = 0.0
            for i, (reward, old_lp) in enumerate(zip(rewards, old_log_probs)):
                # Placeholder: ratio = exp(new_lp - old_lp)
                ratio = 1.0  # Simplified
                advantage = reward - 0.5

                # Clipped objective
                clipped_ratio = max(
                    min(ratio, 1 + self.config.clip_ratio),
                    1 - self.config.clip_ratio,
                )
                loss = -min(ratio * advantage, clipped_ratio * advantage)
                policy_loss += loss

            policy_loss = policy_loss / len(rewards)

        return policy_loss

    def _compute_kl_penalty(
        self, log_probs: Optional[List[float]], ref_log_probs: Optional[List[float]]
    ) -> float:
        """
        Compute KL divergence penalty to keep policy close to reference.

        KL(π || π_ref) keeps model from deviating too much.
        """
        if log_probs is None or ref_log_probs is None:
            return 0.0

        # KL divergence: E[log(π) - log(π_ref)]
        kl = sum(lp - rlp for lp, rlp in zip(log_probs, ref_log_probs)) / len(
            log_probs
        )

        return max(0.0, kl)  # KL is non-negative

    def _compute_tcr_bonus(
        self, tool_stats: Dict[str, Any], feedback_list: List[List[Feedback]]
    ) -> float:
        """
        Compute Tool-Call Reasoning bonus.

        Rewards:
        - High tool efficiency
        - Deliberative reasoning before tool calls
        - Appropriate tool usage
        """
        bonus = 0.0

        # Efficiency bonus
        efficiency = tool_stats["efficiency"]
        if efficiency > 0.7:
            bonus += self.config.tool_efficiency_weight * efficiency

        # Reasoning bonus (deliberative mode)
        if self.config.deliberative_mode:
            reasoning_ratio = tool_stats["reasoning_ratio"]
            if reasoning_ratio > 0.5:
                bonus += 0.1 * reasoning_ratio

        # Success bonus
        if tool_stats["successful_calls"] > 0:
            bonus += self.config.tool_success_bonus

        # Penalty for excessive tool calls
        if tool_stats["num_tool_calls"] > self.config.max_tool_calls:
            bonus -= 0.2 * (
                tool_stats["num_tool_calls"] - self.config.max_tool_calls
            ) / self.config.max_tool_calls

        logger.debug(
            f"TCR bonus: {bonus:.3f} "
            f"(efficiency={efficiency:.2f}, "
            f"reasoning={tool_stats['reasoning_ratio']:.2f})"
        )

        return max(0.0, bonus)  # Non-negative bonus

    def update_policy(self, loss_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Update policy using GRPO-TCR.

        Args:
            loss_dict: Loss components

        Returns:
            Update statistics
        """
        self.iteration += 1

        # In production:
        # 1. Compute gradients
        # 2. Clip gradients
        # 3. Apply optimizer
        # 4. Update KL target if needed

        logger.info(
            f"GRPO-TCR update #{self.iteration}: "
            f"loss={loss_dict['total_loss']:.4f}, "
            f"tool_calls={loss_dict['tool_calls']}, "
            f"efficiency={loss_dict['tool_efficiency']:.2f}"
        )

        return {
            "iteration": self.iteration,
            "learning_rate": self.config.learning_rate,
            "tool_stats": self.tool_call_stats[-1] if self.tool_call_stats else {},
        }

    def get_training_phase(self) -> str:
        """
        Get current training phase based on iteration.

        Open-AgentRL uses:
        - Phase 1: SFT (3K samples)
        - Phase 2: RL (30K samples)
        """
        if self.iteration < self.config.sft_samples:
            return "sft"
        elif self.iteration < self.config.sft_samples + self.config.rl_samples:
            return "rl"
        else:
            return "completed"

    def reset(self):
        """Reset algorithm state."""
        self.iteration = 0
        self.tool_call_stats.clear()
        logger.info("GRPO-TCR algorithm reset")
