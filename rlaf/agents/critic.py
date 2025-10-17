"""Critic Agents - Multi-perspective evaluation ensemble."""

import logging
from typing import Any, Dict, List, Optional
import json
import anthropic

from ..core.base import BaseAgent, BaseConfig, AgentRole, AgentResponse, Feedback

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """
    Critic agent that evaluates actor performance.

    Critics provide structured feedback from specific perspectives:
    - Accuracy: Is the output correct?
    - Reasoning: Is the logic sound? (Open-AgentRL inspired)
    - Tool Use: Are tools used effectively? (ARPO inspired)
    - Code Quality: Is code clean and efficient?
    - Policy Compliance: Does it follow rules/SLA?

    Example:
        >>> critic = CriticAgent(
        ...     name="accuracy-critic",
        ...     perspective="accuracy",
        ...     model="claude-3-5-sonnet-20241022"
        ... )
        >>> feedback = await critic.evaluate(actor_response, context)
    """

    def __init__(
        self,
        name: str,
        perspective: str,
        model: str = "claude-3-5-sonnet-20241022",
        config: Optional[BaseConfig] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize critic agent.

        Args:
            name: Critic name
            perspective: What to evaluate (accuracy, reasoning, tool_use, etc.)
            model: Model to use
            config: Configuration
            api_key: API key
        """
        super().__init__(name=name, role=AgentRole.CRITIC, config=config)

        self.perspective = perspective
        self.model = model
        self.api_key = api_key

        # Initialize model client
        if "claude" in model.lower():
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = None
            logger.warning(f"Using mock critic for {name}")

    async def process(self, input_data: Any) -> AgentResponse:
        """
        Evaluate actor response and provide feedback.

        Args:
            input_data: Dict with {"response": AgentResponse, "context": dict}

        Returns:
            AgentResponse with structured feedback
        """
        response = input_data.get("response")
        context = input_data.get("context", {})

        # Generate critique
        critique = await self._generate_critique(response, context)

        # Create response
        critic_response = AgentResponse(
            content=critique,
            role=AgentRole.CRITIC,
            metadata={
                "perspective": self.perspective,
                "evaluated_agent": response.role.value if response else "unknown",
            },
        )

        self.add_to_history(critic_response)
        return critic_response

    async def _generate_critique(
        self, actor_response: AgentResponse, context: Dict[str, Any]
    ) -> str:
        """Generate critique based on perspective."""
        prompt = self._build_critique_prompt(actor_response, context)

        if self.client and "claude" in self.model.lower():
            return await self._claude_critique(prompt)
        else:
            return self._mock_critique(actor_response)

    def _build_critique_prompt(
        self, actor_response: AgentResponse, context: Dict[str, Any]
    ) -> str:
        """Build prompt for critique generation."""
        perspective_instructions = {
            "accuracy": "Evaluate if the response is factually correct and achieves the task goal.",
            "reasoning": "Assess the logical reasoning and thought process. Check for coherence and soundness.",
            "tool_use": "Evaluate tool usage: Are tools called appropriately? Is there unnecessary tool usage?",
            "code_quality": "Review code quality: readability, efficiency, best practices, and potential bugs.",
            "policy": "Check policy compliance: SLA requirements, security guidelines, business rules.",
            "speed": "Evaluate response efficiency: Is it unnecessarily verbose or slow?",
            "safety": "Assess safety: Are there harmful outputs, security risks, or ethical concerns?",
        }

        instruction = perspective_instructions.get(
            self.perspective, "Evaluate overall quality."
        )

        return f"""You are a specialized critic agent focusing on: {self.perspective}

{instruction}

**Context/Task:**
{json.dumps(context, indent=2)}

**Actor's Response:**
{actor_response.content if actor_response else "No response"}

**Your Task:**
Provide structured feedback in JSON format:
{{
    "score": <float 0.0-1.0>,
    "reasoning": "<your detailed analysis>",
    "suggestions": ["<improvement 1>", "<improvement 2>", ...],
    "confidence": <float 0.0-1.0>
}}

Be specific, actionable, and constructive."""

    async def _claude_critique(self, prompt: str) -> str:
        """Generate critique using Claude."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.3,  # Lower temp for consistent evaluation
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Claude critique failed: {e}")
            return self._mock_critique_json()

    def _mock_critique(self, actor_response: AgentResponse) -> str:
        """Mock critique for testing."""
        return json.dumps(
            {
                "score": 0.75,
                "reasoning": f"[Mock {self.perspective} critique] Response appears reasonable.",
                "suggestions": ["Add more detail", "Consider edge cases"],
                "confidence": 0.8,
            }
        )

    def _mock_critique_json(self) -> str:
        """Mock critique in JSON format."""
        return json.dumps(
            {
                "score": 0.7,
                "reasoning": "Mock evaluation due to API error",
                "suggestions": [],
                "confidence": 0.5,
            }
        )


class CriticEnsemble:
    """
    Ensemble of critic agents for multi-perspective evaluation.

    This is RLAF's core innovation: Multiple specialized critics
    provide diverse feedback, creating richer reward signals.

    Example:
        >>> ensemble = CriticEnsemble([
        ...     CriticAgent("accuracy-critic", "accuracy"),
        ...     CriticAgent("reasoning-critic", "reasoning"),
        ...     CriticAgent("tool-critic", "tool_use")
        ... ])
        >>> feedback_list = await ensemble.evaluate_all(response, context)
    """

    def __init__(self, critics: List[CriticAgent]):
        """
        Initialize critic ensemble.

        Args:
            critics: List of critic agents
        """
        self.critics = critics
        logger.info(f"Initialized CriticEnsemble with {len(critics)} critics:")
        for c in critics:
            logger.info(f"  - {c.name} ({c.perspective})")

    async def evaluate_all(
        self, actor_response: AgentResponse, context: Dict[str, Any]
    ) -> List[Feedback]:
        """
        Evaluate actor response with all critics.

        Args:
            actor_response: Response from actor
            context: Task context

        Returns:
            List of Feedback from each critic
        """
        feedback_list = []

        for critic in self.critics:
            # Each critic evaluates
            critic_input = {"response": actor_response, "context": context}
            critic_response = await critic.process(critic_input)

            # Parse into Feedback object
            feedback = self._parse_feedback(critic_response, critic.name)
            feedback_list.append(feedback)

        return feedback_list

    def _parse_feedback(
        self, critic_response: AgentResponse, critic_name: str
    ) -> Feedback:
        """Parse critic response into Feedback object."""
        try:
            # Try to parse JSON response
            data = json.loads(critic_response.content)
            return Feedback(
                critic_name=critic_name,
                score=data.get("score", 0.5),
                reasoning=data.get("reasoning", ""),
                suggestions=data.get("suggestions", []),
                confidence=data.get("confidence", 1.0),
            )
        except json.JSONDecodeError:
            # Fallback: Create feedback from raw text
            return Feedback(
                critic_name=critic_name,
                score=0.5,
                reasoning=critic_response.content,
                suggestions=[],
                confidence=0.5,
            )

    def __len__(self) -> int:
        """Number of critics in ensemble."""
        return len(self.critics)

    def __iter__(self):
        """Iterate over critics."""
        return iter(self.critics)
