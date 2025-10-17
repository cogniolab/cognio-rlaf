"""Actor Agent - The agent being trained."""

import logging
from typing import Any, Dict, Optional
import anthropic

from ..core.base import BaseAgent, BaseConfig, AgentRole, AgentResponse

logger = logging.getLogger(__name__)


class ActorAgent(BaseAgent):
    """
    Actor agent that performs tasks and learns from critic feedback.

    The actor is the agent being trained in RLAF. It can be:
    - An LLM (Claude, GPT-4, Qwen, Llama)
    - A code generation model
    - An ITSM triage agent
    - Any task-performing agent

    Example:
        >>> actor = ActorAgent(
        ...     name="itsm-triage",
        ...     model="claude-3-5-sonnet-20241022",
        ...     system_prompt="You are an ITSM triage agent..."
        ... )
        >>> response = await actor.process(incident_data)
    """

    def __init__(
        self,
        name: str = "actor",
        model: str = "claude-3-5-sonnet-20241022",
        system_prompt: Optional[str] = None,
        config: Optional[BaseConfig] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize actor agent.

        Args:
            name: Agent name
            model: Model to use (Claude, GPT, Qwen, etc.)
            system_prompt: System instructions
            config: Configuration
            api_key: API key for the model
        """
        super().__init__(name=name, role=AgentRole.ACTOR, config=config)

        self.model = model
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.api_key = api_key

        # Initialize model client
        if "claude" in model.lower():
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = None  # Placeholder for other models
            logger.warning(f"Model {model} not fully supported yet, using mock")

    def _default_system_prompt(self) -> str:
        """Default system prompt for actor."""
        return """You are an AI agent learning to perform tasks optimally.

You will receive:
1. A task description
2. Context and requirements
3. Feedback from critic agents

Your goal is to:
- Complete tasks accurately
- Learn from multi-perspective feedback
- Improve through reinforcement learning

Respond with clear, actionable outputs."""

    async def process(self, input_data: Any) -> AgentResponse:
        """
        Process input and generate response.

        Args:
            input_data: Task to perform

        Returns:
            AgentResponse with generated content
        """
        # Format input for the model
        user_message = self._format_input(input_data)

        # Generate response
        if self.client and "claude" in self.model.lower():
            response_content = await self._claude_generate(user_message)
        else:
            response_content = self._mock_generate(user_message)

        # Create response object
        response = AgentResponse(
            content=response_content,
            role=AgentRole.ACTOR,
            metadata={"model": self.model, "input": input_data},
        )

        self.add_to_history(response)
        return response

    def _format_input(self, input_data: Any) -> str:
        """Format input data as a prompt."""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            # Handle structured input
            if "task" in input_data:
                return input_data["task"]
            elif "incident" in input_data:
                return f"Triage this incident: {input_data['incident']}"
            else:
                return str(input_data)
        else:
            return str(input_data)

    async def _claude_generate(self, user_message: str) -> str:
        """Generate response using Claude."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Claude generation failed: {e}")
            return self._mock_generate(user_message)

    def _mock_generate(self, user_message: str) -> str:
        """Mock generation for testing without API key."""
        return f"[Mock Response] Processed: {user_message[:100]}..."
