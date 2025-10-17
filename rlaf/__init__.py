"""
RLAF: Reinforcement Learning from Agentic Feedback
===================================================

A novel framework for training AI models using multi-agent critic ensembles.

Key Components:
- Actor: The agent being trained
- Critics: Ensemble of specialized evaluator agents
- Feedback: Structured feedback from multiple perspectives
- Rewards: Aggregated reward signals for RL training

Example:
    >>> from rlaf import RLAFTrainer, ActorAgent, CriticEnsemble
    >>>
    >>> # Define your actor
    >>> actor = ActorAgent(model="gpt-4")
    >>>
    >>> # Create critic ensemble
    >>> critics = CriticEnsemble([
    ...     AccuracyCritic(),
    ...     ClarityCritic(),
    ...     PolicyCritic()
    ... ])
    >>>
    >>> # Train with RLAF
    >>> trainer = RLAFTrainer(actor=actor, critics=critics)
    >>> trainer.train(dataset)

Author: Cognio Lab
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Cognio Lab"
__email__ = "dev@cogniolab.com"

from .core.trainer import RLAFTrainer
from .core.base import BaseAgent, BaseConfig
from .agents.actor import ActorAgent
from .agents.critic import CriticAgent, CriticEnsemble
from .feedback.collector import FeedbackCollector
from .rewards.aggregator import RewardAggregator

__all__ = [
    "RLAFTrainer",
    "BaseAgent",
    "BaseConfig",
    "ActorAgent",
    "CriticAgent",
    "CriticEnsemble",
    "FeedbackCollector",
    "RewardAggregator",
]
