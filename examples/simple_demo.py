"""
Simple RLAF Demo - Minimal example to get started

Shows the basic RLAF workflow in ~50 lines of code.
"""

import asyncio
import logging

from rlaf import RLAFTrainer
from rlaf.agents import ActorAgent, CriticAgent, CriticEnsemble
from rlaf.core.trainer import TrainingConfig

logging.basicConfig(level=logging.INFO)


async def main():
    """Minimal RLAF example."""

    # 1. Create actor (agent being trained)
    actor = ActorAgent(
        name="qa-agent",
        model="claude-3-5-sonnet-20241022",
        system_prompt="You are a helpful Q&A assistant.",
        api_key="mock-key",  # Set ANTHROPIC_API_KEY env var for real usage
    )

    # 2. Create critic ensemble
    critics = CriticEnsemble(
        [
            CriticAgent("accuracy-critic", "accuracy", api_key="mock-key"),
            CriticAgent("helpfulness-critic", "reasoning", api_key="mock-key"),
        ]
    )

    # 3. Configure training
    config = TrainingConfig(
        algorithm="arpo",  # or "grpo-tcr", "ppo", "dpo"
        max_iterations=3,
        batch_size=2,
    )

    # 4. Create trainer
    trainer = RLAFTrainer(actor=actor, critics=critics, config=config)

    # 5. Prepare dataset
    dataset = [
        {"task": "What is the capital of France?"},
        {"task": "Explain quantum computing simply"},
        {"task": "How do I make a cup of tea?"},
    ]

    # 6. Train
    print("\n🚀 Starting RLAF training...\n")
    results = await trainer.train(dataset)

    # 7. Results
    print("\n✅ Training complete!")
    print(f"Losses: {results['losses']}")
    print(f"Avg Reward: {sum(results['rewards']) / len(results['rewards']):.3f}")

    # 8. Test
    test_response = await actor.process("What is machine learning?")
    print(f"\n📝 Test response:\n{test_response.content}")


if __name__ == "__main__":
    asyncio.run(main())
