"""
RLAF Trainer - Unified training for Agentic RL

Supports multiple algorithms:
- ARPO (Adaptive Reinforcement Policy Optimization)
- GRPO-TCR (Generalized RPO with Tool-Call Reasoning) - from Open-AgentRL
- KAT-style (Multi-stage training) - from KAT-Dev
- PPO (Proximal Policy Optimization)
- DPO (Direct Preference Optimization)
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import asyncio

from .base import BaseAgent, BaseConfig, AgentResponse, Feedback

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for RLAF training."""

    # Algorithm selection
    algorithm: str = "arpo"  # arpo, grpo-tcr, kat, ppo, dpo

    # Direct-RLAIF mode (2024-2025 research)
    direct_feedback: bool = False  # Skip reward model, query critics directly
    feedback_frequency: str = "every_step"  # How often to query critics in direct mode

    # RLTHF - Targeted Human Feedback (2025)
    use_human_feedback: bool = False  # Enable human-in-the-loop
    human_feedback_threshold: float = 0.6  # Request human input if consensus < threshold
    human_feedback_budget: float = 0.07  # Max fraction of human annotations (7%)
    human_feedback_mode: str = "uncertainty"  # uncertainty, random, critical

    # Online Iterative RLHF (2025)
    online_mode: bool = False  # Enable continuous learning from production
    update_frequency: str = "hourly"  # Update interval for online mode
    feedback_buffer_size: int = 1000  # Min feedback before update

    # ARPO-specific (from July 2025 paper)
    entropy_threshold: float = 0.8  # High uncertainty trigger
    adaptive_rollout: bool = True

    # GRPO-TCR-specific (from Open-AgentRL Oct 2025)
    tool_call_reasoning: bool = True
    deliberative_mode: bool = True
    sft_samples: int = 3000
    rl_samples: int = 30000

    # KAT-style (from KAT-Dev Sept 2025)
    multi_stage: bool = False
    stages: List[str] = None  # ["mid_train", "rft", "agentic_rl"]

    # Training hyperparameters
    max_iterations: int = 1000
    checkpoint_every: int = 100
    eval_every: int = 50

    def __post_init__(self):
        """Validate algorithm-specific configs."""
        valid_algorithms = ["arpo", "grpo-tcr", "kat", "ppo", "dpo"]
        if self.algorithm not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of {valid_algorithms}")

        if self.multi_stage and not self.stages:
            self.stages = ["mid_train", "rft", "agentic_rl"]


class RLAFTrainer:
    """
    Unified trainer for Agentic Reinforcement Learning.

    Combines innovations from:
    - ARPO (2025): Adaptive rollout based on entropy
    - Open-AgentRL (Oct 2025): GRPO-TCR with tool-call reasoning
    - KAT-Dev (Sept 2025): Multi-stage training pipeline
    - Constitutional AI: Self-critique mechanisms

    Example:
        >>> from rlaf import RLAFTrainer, ActorAgent, CriticEnsemble
        >>>
        >>> trainer = RLAFTrainer(
        ...     actor=ActorAgent(),
        ...     critics=CriticEnsemble([...]),
        ...     config=TrainingConfig(algorithm="grpo-tcr")
        ... )
        >>> trainer.train(dataset)
    """

    def __init__(
        self,
        actor: BaseAgent,
        critics: Union[BaseAgent, List[BaseAgent]],
        config: Optional[TrainingConfig] = None,
        human_provider: Optional[Any] = None,
    ):
        """
        Initialize RLAF trainer.

        Args:
            actor: The agent being trained
            critics: Single critic or ensemble of critics
            config: Training configuration
            human_provider: Optional human feedback provider for RLTHF
        """
        self.actor = actor
        self.critics = critics if isinstance(critics, list) else [critics]
        self.config = config or TrainingConfig()
        self.human_provider = human_provider

        self.iteration = 0
        self.training_history: List[Dict[str, Any]] = []
        self.human_feedback_count = 0  # Track human annotations used

        logger.info(
            f"Initialized RLAFTrainer with {self.config.algorithm} algorithm"
        )
        logger.info(f"Actor: {self.actor.name}")
        logger.info(f"Critics: {[c.name for c in self.critics]}")

        # Log new features
        if self.config.direct_feedback:
            logger.info("Direct-RLAIF mode enabled (no reward model)")
        if self.config.use_human_feedback:
            logger.info(
                f"RLTHF enabled (budget: {self.config.human_feedback_budget*100:.1f}%)"
            )
        if self.config.online_mode:
            logger.info(f"Online mode enabled (updates: {self.config.update_frequency})")

    async def train(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the actor using agentic RL.

        Args:
            dataset: Training examples

        Returns:
            Training results with metrics
        """
        logger.info(f"Starting training with {len(dataset)} examples")

        if self.config.multi_stage:
            return await self._train_multi_stage(dataset)
        else:
            return await self._train_single_stage(dataset)

    async def _train_single_stage(
        self, dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Single-stage training (ARPO, GRPO-TCR, PPO, DPO)."""
        results = {"losses": [], "rewards": [], "metrics": {}}

        for iteration in range(self.config.max_iterations):
            self.iteration = iteration

            # Sample batch
            batch = self._sample_batch(dataset)

            # Forward pass: Actor generates responses
            responses = await self._actor_forward(batch)

            # Critic evaluation: Get multi-perspective feedback
            feedback_list = await self._critic_evaluation(responses, batch)

            # Compute rewards
            rewards = self._compute_rewards(feedback_list)

            # Algorithm-specific update
            loss = await self._update_actor(responses, rewards, feedback_list)

            # Logging
            results["losses"].append(loss)
            results["rewards"].extend(rewards)

            if iteration % self.config.eval_every == 0:
                metrics = await self._evaluate(dataset)
                results["metrics"][f"iter_{iteration}"] = metrics
                logger.info(
                    f"Iteration {iteration}: Loss={loss:.4f}, "
                    f"Avg Reward={sum(rewards)/len(rewards):.4f}"
                )

        return results

    async def _train_multi_stage(
        self, dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Multi-stage training (KAT-style)."""
        results = {}

        for stage in self.config.stages:
            logger.info(f"Starting stage: {stage}")

            if stage == "mid_train":
                # Mid-training: Enhance LLM-as-agent capabilities
                stage_results = await self._mid_training(dataset)

            elif stage == "rft":
                # Reinforcement Fine-Tuning with teacher trajectories
                stage_results = await self._reinforcement_fine_tuning(dataset)

            elif stage == "agentic_rl":
                # Full agentic RL with critic ensemble
                stage_results = await self._train_single_stage(dataset)

            results[stage] = stage_results

        return results

    async def _actor_forward(
        self, batch: List[Dict[str, Any]]
    ) -> List[AgentResponse]:
        """Actor generates responses for batch."""
        responses = []
        for item in batch:
            response = await self.actor.process(item)
            responses.append(response)
        return responses

    async def _critic_evaluation(
        self, responses: List[AgentResponse], batch: List[Dict[str, Any]]
    ) -> List[List[Feedback]]:
        """
        Multi-critic evaluation (RLAF's core innovation).

        Each response is evaluated by ALL critics, providing
        multi-perspective feedback.

        NEW (2025): Supports RLTHF - requests human feedback when needed.
        """
        all_feedback = []

        for idx, (response, item) in enumerate(zip(responses, batch)):
            feedback_for_response = []

            # Evaluate with each critic
            for critic in self.critics:
                # Critic processes (response, context) pair
                critic_input = {"response": response, "context": item}
                critic_response = await critic.process(critic_input)

                # Parse feedback from critic response
                feedback = self._parse_critic_feedback(
                    critic_response, critic.name
                )
                feedback_for_response.append(feedback)

            # RLTHF: Check if human feedback is needed
            if self.config.use_human_feedback and self.human_provider:
                should_request_human = self._should_request_human_feedback(
                    feedback_for_response, idx
                )

                if should_request_human:
                    human_feedback = await self._get_human_feedback(
                        response, item, feedback_for_response
                    )
                    if human_feedback:
                        feedback_for_response.append(human_feedback)
                        self.human_feedback_count += 1
                        logger.info(
                            f"Human feedback requested ({self.human_feedback_count} total)"
                        )

            all_feedback.append(feedback_for_response)

        return all_feedback

    def _parse_critic_feedback(
        self, critic_response: AgentResponse, critic_name: str
    ) -> Feedback:
        """Parse critic response into structured feedback."""
        # In production, this would parse JSON/structured output
        # For now, create basic feedback
        return Feedback(
            critic_name=critic_name,
            score=0.75,  # Placeholder
            reasoning=critic_response.content,
            suggestions=[],
            confidence=1.0,
        )

    def _compute_rewards(
        self, feedback_list: List[List[Feedback]]
    ) -> List[float]:
        """
        Aggregate multi-critic feedback into reward signals.

        Supports multiple strategies:
        - weighted_average: Confidence-weighted average of scores
        - voting: Majority vote on quality
        - debate: Critics argue, judge decides
        """
        rewards = []

        for feedback_for_item in feedback_list:
            if self.config.reward_aggregation == "weighted_average":
                # Weighted average by confidence
                total_score = sum(
                    f.score * f.confidence for f in feedback_for_item
                )
                total_confidence = sum(f.confidence for f in feedback_for_item)
                reward = (
                    total_score / total_confidence if total_confidence > 0 else 0.0
                )

            elif self.config.reward_aggregation == "voting":
                # Simple majority vote (score > 0.5 = positive)
                votes = [1 if f.score > 0.5 else 0 for f in feedback_for_item]
                reward = sum(votes) / len(votes)

            else:  # debate
                # For debate, highest-confidence critic wins
                reward = max(feedback_for_item, key=lambda f: f.confidence).score

            rewards.append(reward)

        return rewards

    async def _update_actor(
        self,
        responses: List[AgentResponse],
        rewards: List[float],
        feedback_list: List[List[Feedback]],
    ) -> float:
        """
        Update actor based on algorithm.

        This is where ARPO, GRPO-TCR, PPO, DPO differ.
        """
        if self.config.algorithm == "arpo":
            return await self._arpo_update(responses, rewards, feedback_list)

        elif self.config.algorithm == "grpo-tcr":
            return await self._grpo_tcr_update(responses, rewards, feedback_list)

        elif self.config.algorithm in ["ppo", "dpo"]:
            return await self._standard_rl_update(responses, rewards)

        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

    async def _arpo_update(
        self,
        responses: List[AgentResponse],
        rewards: List[float],
        feedback_list: List[List[Feedback]],
    ) -> float:
        """
        ARPO: Adaptive Reinforcement Policy Optimization.

        Key: Entropy-based adaptive rollout (from July 2025 paper).
        """
        # Calculate entropy from feedback uncertainty
        avg_confidence = sum(
            sum(f.confidence for f in fb) / len(fb) for fb in feedback_list
        ) / len(feedback_list)

        # Low confidence = high entropy = explore more
        if avg_confidence < self.config.entropy_threshold:
            logger.info("High entropy detected, increasing exploration")
            # In practice: Sample more trajectories here

        # Standard policy gradient update (placeholder)
        loss = sum((1 - r) ** 2 for r in rewards) / len(rewards)
        return loss

    async def _grpo_tcr_update(
        self,
        responses: List[AgentResponse],
        rewards: List[float],
        feedback_list: List[List[Feedback]],
    ) -> float:
        """
        GRPO-TCR: Generalized RPO with Tool-Call Reasoning.

        Key: Deliberative reasoning with selective tool calls
        (from Open-AgentRL Oct 2025).
        """
        # Check if this batch involved tool calls
        tool_call_detected = any("tool" in r.metadata for r in responses)

        if tool_call_detected and self.config.tool_call_reasoning:
            logger.info("Tool-call reasoning mode activated")
            # In practice: Apply TCR-specific reward shaping

        # Placeholder update
        loss = sum((1 - r) ** 2 for r in rewards) / len(rewards)
        return loss

    async def _standard_rl_update(
        self, responses: List[AgentResponse], rewards: List[float]
    ) -> float:
        """Standard PPO/DPO update."""
        # Placeholder: In practice, use TRL library
        loss = sum((1 - r) ** 2 for r in rewards) / len(rewards)
        return loss

    async def _mid_training(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """KAT-style mid-training stage."""
        logger.info("Mid-training: Enhancing LLM-as-agent capabilities")
        # Placeholder
        return {"stage": "mid_train", "status": "completed"}

    async def _reinforcement_fine_tuning(
        self, dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """KAT-style RFT with teacher trajectories."""
        logger.info("RFT: Training with teacher trajectories")
        # Placeholder
        return {"stage": "rft", "status": "completed"}

    def _sample_batch(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sample batch from dataset."""
        import random

        return random.sample(dataset, min(self.config.batch_size, len(dataset)))

    async def _evaluate(self, dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate current model."""
        # Placeholder: Run on eval set
        return {"accuracy": 0.85, "avg_reward": 0.72}

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        logger.info(f"Saving checkpoint to {path}")
        # Placeholder: Save actor weights, optimizer state

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        logger.info(f"Loading checkpoint from {path}")
        # Placeholder: Load actor weights, optimizer state

    def _should_request_human_feedback(
        self, feedback_list: List[Feedback], iteration_idx: int
    ) -> bool:
        """
        Determine if human feedback should be requested (RLTHF).

        Uses uncertainty-based strategy: request human input when
        critic consensus is low.
        """
        if not self.config.use_human_feedback:
            return False

        # Check if we've exceeded budget
        total_samples = self.iteration * self.config.batch_size + iteration_idx + 1
        human_ratio = self.human_feedback_count / total_samples if total_samples > 0 else 0

        if human_ratio >= self.config.human_feedback_budget:
            return False  # Budget exhausted

        # Calculate consensus level
        if len(feedback_list) < 2:
            return False

        scores = [f.score for f in feedback_list]
        avg_score = sum(scores) / len(scores)
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        consensus = 1.0 - min(1.0, variance * 2)  # Low variance = high consensus

        # Request human feedback if consensus is below threshold
        return consensus < self.config.human_feedback_threshold

    async def _get_human_feedback(
        self,
        response: AgentResponse,
        context: Dict[str, Any],
        ai_feedback: List[Feedback],
    ) -> Optional[Feedback]:
        """
        Request human feedback for a response (RLTHF).

        Args:
            response: Actor's response
            context: Task context
            ai_feedback: Existing AI feedback

        Returns:
            Human feedback or None
        """
        if not self.human_provider:
            return None

        try:
            # Request human evaluation
            human_eval = await self.human_provider.request_feedback(
                response=response,
                context=context,
                ai_feedback=ai_feedback,
            )

            return Feedback(
                critic_name="human",
                score=human_eval.get("score", 0.5),
                reasoning=human_eval.get("reasoning", "Human evaluation"),
                suggestions=human_eval.get("suggestions", []),
                confidence=1.0,  # Human feedback has highest confidence
            )
        except Exception as e:
            logger.error(f"Failed to get human feedback: {e}")
            return None
