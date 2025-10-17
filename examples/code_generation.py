"""
Code Generation Example - Training a code agent with RLAF

Demonstrates:
- Actor: Python code generation agent
- Critics: Code quality, correctness, efficiency
- Training: GRPO-TCR algorithm (tool-aware reasoning)
"""

import asyncio
import logging
import os

from rlaf import RLAFTrainer
from rlaf.agents import ActorAgent, CriticAgent, CriticEnsemble
from rlaf.core.trainer import TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample coding tasks
CODE_DATASET = [
    {
        "task": "Write a function to check if a number is prime",
        "expected_signature": "def is_prime(n: int) -> bool",
        "test_cases": [
            {"input": 2, "output": True},
            {"input": 4, "output": False},
            {"input": 17, "output": True},
        ],
    },
    {
        "task": "Implement binary search on a sorted array",
        "expected_signature": "def binary_search(arr: list, target: int) -> int",
        "test_cases": [
            {"input": ([1, 3, 5, 7, 9], 5), "output": 2},
            {"input": ([1, 3, 5, 7, 9], 6), "output": -1},
        ],
    },
    {
        "task": "Create a function to reverse a linked list",
        "expected_signature": "def reverse_linked_list(head: ListNode) -> ListNode",
        "test_cases": [],
    },
    {
        "task": "Write a function to find the longest palindrome substring",
        "expected_signature": "def longest_palindrome(s: str) -> str",
        "test_cases": [
            {"input": "babad", "output": "bab"},
            {"input": "cbbd", "output": "bb"},
        ],
    },
]


def create_code_actor(api_key: str) -> ActorAgent:
    """Create code generation actor."""
    system_prompt = """You are an expert Python programmer.

Generate clean, efficient, well-documented Python code.

Your code should:
1. Be syntactically correct
2. Follow PEP 8 style guidelines
3. Include docstrings
4. Handle edge cases
5. Be efficient (optimal time/space complexity)

Provide only the code implementation, no extra explanation."""

    return ActorAgent(
        name="code-generator",
        model="claude-3-5-sonnet-20241022",
        system_prompt=system_prompt,
        api_key=api_key,
    )


def create_code_critics(api_key: str) -> CriticEnsemble:
    """Create code-focused critic ensemble."""

    # Correctness critic
    correctness_critic = CriticAgent(
        name="correctness-critic",
        perspective="accuracy",
        model="claude-3-5-sonnet-20241022",
        api_key=api_key,
    )

    # Code quality critic
    quality_critic = CriticAgent(
        name="quality-critic",
        perspective="code_quality",
        model="claude-3-5-sonnet-20241022",
        api_key=api_key,
    )

    # Efficiency critic
    efficiency_critic = CriticAgent(
        name="efficiency-critic",
        perspective="speed",
        model="claude-3-5-sonnet-20241022",
        api_key=api_key,
    )

    return CriticEnsemble([correctness_critic, quality_critic, efficiency_critic])


async def main():
    """Run code generation training example."""

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("No API key found, using mock mode")
        api_key = "mock-key"

    logger.info("=" * 60)
    logger.info("RLAF Code Generation Training Example")
    logger.info("=" * 60)

    # Create actor and critics
    actor = create_code_actor(api_key)
    critics = create_code_critics(api_key)

    # Configure training with GRPO-TCR (tool-aware)
    config = TrainingConfig(
        algorithm="grpo-tcr",  # Tool-call reasoning
        max_iterations=4,
        eval_every=2,
        batch_size=2,
        tool_call_reasoning=True,
        deliberative_mode=True,
        reward_aggregation="weighted_average",
    )

    # Initialize trainer
    trainer = RLAFTrainer(actor=actor, critics=critics, config=config)

    logger.info(f"\nDataset: {len(CODE_DATASET)} coding tasks")
    logger.info(f"Algorithm: {config.algorithm}")
    logger.info(f"Critics: {len(critics)} perspectives\n")

    # Run training
    results = await trainer.train(CODE_DATASET)

    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("Training Results")
    logger.info("=" * 60)

    logger.info(f"\nLosses: {results['losses']}")
    logger.info(f"Rewards: {results['rewards'][:5]}...")  # Show first 5

    # Test on new task
    logger.info("\n" + "=" * 60)
    logger.info("Testing on New Task")
    logger.info("=" * 60)

    test_task = {
        "task": "Write a function to merge two sorted linked lists",
        "expected_signature": "def merge_sorted_lists(l1: ListNode, l2: ListNode) -> ListNode",
    }

    logger.info(f"\nTask: {test_task['task']}")

    response = await actor.process(test_task)
    logger.info(f"\nGenerated Code:\n{response.content}")

    # Critic evaluation
    logger.info("\nCritic Evaluations:")
    for critic in critics:
        critic_input = {"response": response, "context": test_task}
        critic_response = await critic.process(critic_input)
        logger.info(f"\n{critic.name}:\n{critic_response.content}")


if __name__ == "__main__":
    asyncio.run(main())
