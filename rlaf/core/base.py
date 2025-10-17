"""Base classes for RLAF framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class AgentRole(Enum):
    """Agent role in RLAF system."""
    ACTOR = "actor"
    CRITIC = "critic"
    JUDGE = "judge"


@dataclass
class BaseConfig:
    """Base configuration for RLAF components."""

    # Model configuration
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048

    # Training configuration
    learning_rate: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 3

    # RLAF specific
    num_critics: int = 3
    reward_aggregation: str = "weighted_average"  # weighted_average, voting, debate

    # Logging
    log_level: str = "INFO"
    wandb_project: Optional[str] = None

    # Advanced
    use_ppo: bool = True
    use_dpo: bool = False
    clip_ratio: float = 0.2

    def __post_init__(self):
        """Validate configuration."""
        if self.num_critics < 1:
            raise ValueError("num_critics must be at least 1")

        if self.reward_aggregation not in ["weighted_average", "voting", "debate"]:
            raise ValueError(f"Invalid reward_aggregation: {self.reward_aggregation}")


@dataclass
class Feedback:
    """Structured feedback from a critic agent."""

    critic_name: str
    score: float  # 0.0 to 1.0
    reasoning: str
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate feedback."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0 and 1, got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


@dataclass
class AgentResponse:
    """Response from an agent."""

    content: str
    role: AgentRole
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "role": self.role.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class BaseAgent(ABC):
    """Abstract base class for all RLAF agents."""

    def __init__(self, name: str, role: AgentRole, config: Optional[BaseConfig] = None):
        """
        Initialize base agent.

        Args:
            name: Agent identifier
            role: Agent role (actor, critic, judge)
            config: Configuration object
        """
        self.name = name
        self.role = role
        self.config = config or BaseConfig()
        self._history: List[AgentResponse] = []

    @abstractmethod
    async def process(self, input_data: Any) -> AgentResponse:
        """
        Process input and generate response.

        Args:
            input_data: Input to process

        Returns:
            AgentResponse with generated content
        """
        pass

    def add_to_history(self, response: AgentResponse):
        """Add response to history."""
        self._history.append(response)

    def get_history(self) -> List[AgentResponse]:
        """Get agent's response history."""
        return self._history.copy()

    def clear_history(self):
        """Clear response history."""
        self._history.clear()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', role={self.role.value})"
