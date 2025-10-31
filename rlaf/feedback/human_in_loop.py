"""Human-in-the-Loop Feedback Provider for RLTHF.

RLTHF (Reinforcement Learning with Targeted Human Feedback) achieves
full-human annotation-level alignment with only 6-7% of the human
annotation effort by combining AI feedback with selective human corrections.

This module provides interfaces for requesting human feedback when needed.
"""

import logging
from typing import Any, Dict, List, Optional, Protocol
from abc import ABC, abstractmethod
import asyncio

from ..core.base import AgentResponse, Feedback

logger = logging.getLogger(__name__)


class HumanFeedbackProvider(ABC):
    """
    Abstract base class for human feedback providers.

    Implementations can use various interfaces:
    - Web UI for human annotators
    - CLI prompts for interactive feedback
    - API calls to external annotation services
    - Mock providers for testing
    """

    @abstractmethod
    async def request_feedback(
        self,
        response: AgentResponse,
        context: Dict[str, Any],
        ai_feedback: List[Feedback],
    ) -> Dict[str, Any]:
        """
        Request human feedback for a response.

        Args:
            response: Actor's response to evaluate
            context: Task context
            ai_feedback: Existing AI critic feedback

        Returns:
            Dictionary with:
            {
                "score": float (0.0-1.0),
                "reasoning": str,
                "suggestions": List[str],
            }
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if human feedback provider is available."""
        pass


class CLIHumanFeedbackProvider(HumanFeedbackProvider):
    """
    CLI-based human feedback provider for interactive sessions.

    Displays the response and prompts the user for feedback via command line.
    Useful for development and testing.
    """

    def __init__(self, timeout: int = 60):
        """
        Initialize CLI provider.

        Args:
            timeout: Seconds to wait for human input before timing out
        """
        self.timeout = timeout

    async def request_feedback(
        self,
        response: AgentResponse,
        context: Dict[str, Any],
        ai_feedback: List[Feedback],
    ) -> Dict[str, Any]:
        """Request feedback via CLI."""
        print("\n" + "=" * 80)
        print("🧑 HUMAN FEEDBACK REQUESTED")
        print("=" * 80)
        print(f"\n📋 Task Context:\n{context}\n")
        print(f"🤖 Actor Response:\n{response.content}\n")
        print("💭 AI Critic Feedback:")
        for fb in ai_feedback:
            print(f"  - {fb.critic_name}: {fb.score:.2f} - {fb.reasoning}")
        print("\n" + "-" * 80)

        try:
            # Request score
            score_input = input("\n👉 Your score (0.0-1.0): ")
            score = float(score_input)
            score = max(0.0, min(1.0, score))

            # Request reasoning
            reasoning = input("👉 Your reasoning: ")

            # Request suggestions (optional)
            suggestions_input = input("👉 Suggestions (comma-separated, or press Enter): ")
            suggestions = (
                [s.strip() for s in suggestions_input.split(",") if s.strip()]
                if suggestions_input
                else []
            )

            print("\n✅ Thank you for your feedback!\n")

            return {
                "score": score,
                "reasoning": reasoning or "Human evaluation",
                "suggestions": suggestions,
            }

        except Exception as e:
            logger.error(f"CLI feedback error: {e}")
            return {
                "score": 0.5,
                "reasoning": "Human feedback failed",
                "suggestions": [],
            }

    async def is_available(self) -> bool:
        """CLI is always available."""
        return True


class WebHumanFeedbackProvider(HumanFeedbackProvider):
    """
    Web-based human feedback provider.

    Sends requests to a web service where human annotators can
    review and provide feedback asynchronously.
    """

    def __init__(
        self,
        api_endpoint: str,
        api_key: Optional[str] = None,
        timeout: int = 300,
    ):
        """
        Initialize web provider.

        Args:
            api_endpoint: URL of the feedback service
            api_key: Optional API key for authentication
            timeout: Seconds to wait for feedback
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.timeout = timeout

    async def request_feedback(
        self,
        response: AgentResponse,
        context: Dict[str, Any],
        ai_feedback: List[Feedback],
    ) -> Dict[str, Any]:
        """
        Request feedback via web API.

        Sends a request to the web service and polls for completion.
        """
        # In production, this would make HTTP requests to your feedback service
        # For now, placeholder implementation
        logger.info(f"Requesting human feedback via {self.api_endpoint}")

        # Example payload
        payload = {
            "response": response.content,
            "context": context,
            "ai_feedback": [
                {
                    "critic": fb.critic_name,
                    "score": fb.score,
                    "reasoning": fb.reasoning,
                }
                for fb in ai_feedback
            ],
        }

        # TODO: Implement actual HTTP request
        # response = await http_client.post(self.api_endpoint, json=payload)
        # feedback_id = response.json()["feedback_id"]
        # result = await self._poll_for_feedback(feedback_id)

        logger.warning("Web feedback provider not fully implemented yet")
        return {
            "score": 0.75,
            "reasoning": "Placeholder web feedback",
            "suggestions": [],
        }

    async def is_available(self) -> bool:
        """Check if web service is available."""
        # TODO: Implement health check
        return True


class MockHumanFeedbackProvider(HumanFeedbackProvider):
    """
    Mock human feedback provider for testing.

    Returns synthetic feedback without requiring actual human input.
    Useful for unit tests and automated benchmarks.
    """

    def __init__(self, default_score: float = 0.8):
        """
        Initialize mock provider.

        Args:
            default_score: Default score to return
        """
        self.default_score = default_score
        self.feedback_count = 0

    async def request_feedback(
        self,
        response: AgentResponse,
        context: Dict[str, Any],
        ai_feedback: List[Feedback],
    ) -> Dict[str, Any]:
        """Return mock feedback."""
        self.feedback_count += 1

        # Simulate some variation based on AI feedback
        avg_ai_score = sum(fb.score for fb in ai_feedback) / len(ai_feedback) if ai_feedback else 0.5
        mock_score = (avg_ai_score + self.default_score) / 2

        return {
            "score": mock_score,
            "reasoning": f"Mock human feedback #{self.feedback_count}",
            "suggestions": ["Mock suggestion: Consider edge cases"],
        }

    async def is_available(self) -> bool:
        """Mock is always available."""
        return True


class HumanFeedbackQueue:
    """
    Queue-based human feedback system for batch processing.

    Collects feedback requests and processes them in batches,
    allowing efficient use of human annotator time.
    """

    def __init__(
        self,
        provider: HumanFeedbackProvider,
        batch_size: int = 10,
        max_wait_time: int = 300,
    ):
        """
        Initialize feedback queue.

        Args:
            provider: Underlying feedback provider
            batch_size: Process feedback in batches of this size
            max_wait_time: Max seconds to wait before processing incomplete batch
        """
        self.provider = provider
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.queue: List[Dict[str, Any]] = []
        self.pending_responses: Dict[str, asyncio.Future] = {}

    async def request_feedback(
        self,
        response: AgentResponse,
        context: Dict[str, Any],
        ai_feedback: List[Feedback],
    ) -> Dict[str, Any]:
        """
        Add feedback request to queue.

        Returns immediately with a future that resolves when feedback is ready.
        """
        request_id = f"req_{len(self.queue)}"

        # Create future for this request
        future = asyncio.Future()
        self.pending_responses[request_id] = future

        # Add to queue
        self.queue.append(
            {
                "id": request_id,
                "response": response,
                "context": context,
                "ai_feedback": ai_feedback,
            }
        )

        # Process if batch is full
        if len(self.queue) >= self.batch_size:
            await self._process_batch()

        # Wait for result
        return await future

    async def _process_batch(self):
        """Process current batch of feedback requests."""
        if not self.queue:
            return

        batch = self.queue[: self.batch_size]
        self.queue = self.queue[self.batch_size :]

        logger.info(f"Processing batch of {len(batch)} human feedback requests")

        # Process each item in batch
        for item in batch:
            try:
                feedback = await self.provider.request_feedback(
                    response=item["response"],
                    context=item["context"],
                    ai_feedback=item["ai_feedback"],
                )

                # Resolve the future
                future = self.pending_responses.pop(item["id"])
                future.set_result(feedback)

            except Exception as e:
                logger.error(f"Error processing feedback request {item['id']}: {e}")
                future = self.pending_responses.pop(item["id"])
                future.set_exception(e)

    async def is_available(self) -> bool:
        """Check if provider is available."""
        return await self.provider.is_available()


# Example usage
async def example_usage():
    """Example of using human feedback providers."""
    from ..core.base import AgentRole

    # Create mock response
    response = AgentResponse(
        content="This is a test response",
        role=AgentRole.ACTOR,
        metadata={},
    )

    context = {"task": "Test task", "goal": "Demonstrate human feedback"}

    ai_feedback = [
        Feedback(
            critic_name="accuracy",
            score=0.7,
            reasoning="Response is partially correct",
            suggestions=["Add more detail"],
            confidence=0.8,
        ),
        Feedback(
            critic_name="reasoning",
            score=0.85,
            reasoning="Logic is sound",
            suggestions=[],
            confidence=0.9,
        ),
    ]

    # Use CLI provider
    # cli_provider = CLIHumanFeedbackProvider()
    # feedback = await cli_provider.request_feedback(response, context, ai_feedback)
    # print(feedback)

    # Use mock provider (for testing)
    mock_provider = MockHumanFeedbackProvider(default_score=0.8)
    feedback = await mock_provider.request_feedback(response, context, ai_feedback)
    print("Mock feedback:", feedback)


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
