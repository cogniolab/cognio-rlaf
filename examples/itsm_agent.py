"""
ITSM Agent Example - Incident triage with RLAF

Demonstrates:
- Actor: ITSM triage agent
- Critics: Accuracy, policy compliance, speed
- Training: ARPO algorithm with multi-critic feedback
"""

import asyncio
import logging
from typing import Dict, Any, List
import os

from rlaf import RLAFTrainer
from rlaf.agents import ActorAgent, CriticAgent, CriticEnsemble
from rlaf.core.trainer import TrainingConfig
from rlaf.core.base import AgentRole

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample ITSM incidents for training
ITSM_DATASET = [
    {
        "incident_number": "INC0001",
        "description": "Cannot access email, getting authentication error",
        "priority": "3",
        "category": "Access",
        "expected_assignment": "Email Support",
        "expected_priority": "3",
    },
    {
        "incident_number": "INC0002",
        "description": "Production database server down, all services affected",
        "priority": "1",
        "category": "Infrastructure",
        "expected_assignment": "Database Team",
        "expected_priority": "1",
    },
    {
        "incident_number": "INC0003",
        "description": "Laptop running slow, need performance optimization",
        "priority": "4",
        "category": "Hardware",
        "expected_assignment": "Desktop Support",
        "expected_priority": "4",
    },
    {
        "incident_number": "INC0004",
        "description": "VPN connection keeps dropping every 5 minutes",
        "priority": "2",
        "category": "Network",
        "expected_assignment": "Network Team",
        "expected_priority": "2",
    },
    {
        "incident_number": "INC0005",
        "description": "Need access to shared folder for marketing materials",
        "priority": "3",
        "category": "Access",
        "expected_assignment": "Access Management",
        "expected_priority": "3",
    },
]


def create_itsm_actor(api_key: str) -> ActorAgent:
    """Create ITSM triage actor agent."""
    system_prompt = """You are an expert ITSM triage agent.

Your task is to analyze IT incidents and provide:
1. Recommended assignment group
2. Suggested priority (1-4, where 1=Critical, 4=Low)
3. Brief reasoning for your decision

Guidelines:
- Priority 1: Production outages, critical services down
- Priority 2: Significant impact, multiple users affected
- Priority 3: Individual user issues, workarounds available
- Priority 4: Requests, minor issues, cosmetic problems

Respond in JSON format:
{
    "assignment_group": "<team name>",
    "priority": <1-4>,
    "reasoning": "<brief explanation>"
}"""

    return ActorAgent(
        name="itsm-triage-agent",
        model="claude-3-5-sonnet-20241022",
        system_prompt=system_prompt,
        api_key=api_key,
    )


def create_itsm_critics(api_key: str) -> CriticEnsemble:
    """Create multi-perspective critic ensemble for ITSM."""

    # Accuracy critic: Is the triage correct?
    accuracy_critic = CriticAgent(
        name="accuracy-critic",
        perspective="accuracy",
        model="claude-3-5-sonnet-20241022",
        api_key=api_key,
    )

    # Policy critic: Does it follow SLA/policy?
    policy_critic = CriticAgent(
        name="policy-critic",
        perspective="policy",
        model="claude-3-5-sonnet-20241022",
        api_key=api_key,
    )

    # Speed critic: Is the response efficient?
    speed_critic = CriticAgent(
        name="speed-critic",
        perspective="speed",
        model="claude-3-5-sonnet-20241022",
        api_key=api_key,
    )

    return CriticEnsemble([accuracy_critic, policy_critic, speed_critic])


async def main():
    """Run ITSM agent training example."""

    # Get API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("No API key found, using mock mode")
        api_key = "mock-key"

    logger.info("=" * 60)
    logger.info("RLAF ITSM Agent Training Example")
    logger.info("=" * 60)

    # Create actor and critics
    actor = create_itsm_actor(api_key)
    critics = create_itsm_critics(api_key)

    # Configure training
    config = TrainingConfig(
        algorithm="arpo",  # Using ARPO for adaptive training
        max_iterations=5,  # Short demo
        eval_every=2,
        batch_size=2,
        entropy_threshold=0.7,
        adaptive_rollout=True,
        reward_aggregation="weighted_average",
    )

    # Initialize trainer
    trainer = RLAFTrainer(actor=actor, critics=critics, config=config)

    logger.info(f"\nDataset: {len(ITSM_DATASET)} incidents")
    logger.info(f"Algorithm: {config.algorithm}")
    logger.info(f"Critics: {len(critics)} perspectives\n")

    # Run training
    results = await trainer.train(ITSM_DATASET)

    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("Training Results")
    logger.info("=" * 60)

    logger.info(f"\nLosses: {results['losses']}")
    logger.info(f"Rewards: {results['rewards']}")
    logger.info(f"\nMetrics: {results['metrics']}")

    # Test on a new incident
    logger.info("\n" + "=" * 60)
    logger.info("Testing on New Incident")
    logger.info("=" * 60)

    test_incident = {
        "incident_number": "INC9999",
        "description": "Cannot print to network printer, urgent report needed",
        "priority": "unknown",
        "category": "Hardware",
    }

    logger.info(f"\nTest Incident: {test_incident['description']}")

    response = await actor.process(test_incident)
    logger.info(f"\nActor Response:\n{response.content}")

    # Get critic feedback
    logger.info("\nCritic Feedback:")
    for critic in critics:
        critic_input = {"response": response, "context": test_incident}
        critic_response = await critic.process(critic_input)
        logger.info(f"\n{critic.name}:\n{critic_response.content}")


if __name__ == "__main__":
    asyncio.run(main())
