"""Feedback Collector - Aggregates multi-critic feedback."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..core.base import Feedback

logger = logging.getLogger(__name__)


@dataclass
class AggregatedFeedback:
    """Aggregated feedback from multiple critics."""

    feedbacks: List[Feedback]
    aggregated_score: float
    aggregation_method: str
    consensus_level: float  # 0.0 to 1.0, measures critic agreement
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FeedbackCollector:
    """
    Collects and aggregates feedback from multiple critics.

    Supports different aggregation strategies:
    - weighted_average: Confidence-weighted average of scores
    - voting: Majority vote on quality thresholds
    - debate: Critics "debate" and highest confidence wins
    - consensus: Only high-agreement feedback

    Example:
        >>> collector = FeedbackCollector(strategy="weighted_average")
        >>> aggregated = collector.aggregate([feedback1, feedback2, feedback3])
        >>> print(f"Score: {aggregated.aggregated_score}")
    """

    def __init__(
        self,
        strategy: str = "weighted_average",
        quality_threshold: float = 0.5,
        consensus_threshold: float = 0.7,
    ):
        """
        Initialize feedback collector.

        Args:
            strategy: Aggregation strategy (weighted_average, voting, debate, consensus)
            quality_threshold: Threshold for voting strategy (0.0-1.0)
            consensus_threshold: Minimum agreement level for consensus strategy
        """
        valid_strategies = ["weighted_average", "voting", "debate", "consensus"]
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")

        self.strategy = strategy
        self.quality_threshold = quality_threshold
        self.consensus_threshold = consensus_threshold

        logger.info(f"Initialized FeedbackCollector with strategy: {strategy}")

    def aggregate(self, feedbacks: List[Feedback]) -> AggregatedFeedback:
        """
        Aggregate multiple critic feedbacks.

        Args:
            feedbacks: List of Feedback from critics

        Returns:
            AggregatedFeedback with aggregated score and metadata
        """
        if not feedbacks:
            logger.warning("No feedbacks to aggregate")
            return AggregatedFeedback(
                feedbacks=[],
                aggregated_score=0.0,
                aggregation_method=self.strategy,
                consensus_level=0.0,
            )

        # Calculate consensus level
        consensus = self._calculate_consensus(feedbacks)

        # Aggregate based on strategy
        if self.strategy == "weighted_average":
            score = self._weighted_average(feedbacks)
        elif self.strategy == "voting":
            score = self._voting(feedbacks)
        elif self.strategy == "debate":
            score = self._debate(feedbacks)
        elif self.strategy == "consensus":
            score = self._consensus(feedbacks, consensus)
        else:
            score = 0.0

        return AggregatedFeedback(
            feedbacks=feedbacks,
            aggregated_score=score,
            aggregation_method=self.strategy,
            consensus_level=consensus,
            metadata=self._extract_metadata(feedbacks),
        )

    def _weighted_average(self, feedbacks: List[Feedback]) -> float:
        """
        Confidence-weighted average of scores.

        High-confidence critics have more influence.
        """
        total_score = sum(f.score * f.confidence for f in feedbacks)
        total_confidence = sum(f.confidence for f in feedbacks)

        if total_confidence == 0:
            logger.warning("Total confidence is zero, returning 0.0")
            return 0.0

        weighted_avg = total_score / total_confidence
        logger.debug(f"Weighted average: {weighted_avg:.3f}")
        return weighted_avg

    def _voting(self, feedbacks: List[Feedback]) -> float:
        """
        Majority vote on quality threshold.

        Each critic votes "pass" (score > threshold) or "fail".
        Returns percentage of "pass" votes.
        """
        votes = [1 if f.score > self.quality_threshold else 0 for f in feedbacks]
        vote_result = sum(votes) / len(votes)

        logger.debug(
            f"Voting result: {vote_result:.3f} ({sum(votes)}/{len(votes)} pass)"
        )
        return vote_result

    def _debate(self, feedbacks: List[Feedback]) -> float:
        """
        Debate strategy: highest-confidence critic wins.

        Simulates critics debating, with most confident critic
        having the final say.
        """
        if not feedbacks:
            return 0.0

        most_confident = max(feedbacks, key=lambda f: f.confidence)
        logger.debug(
            f"Debate winner: {most_confident.critic_name} "
            f"(confidence: {most_confident.confidence:.3f}, "
            f"score: {most_confident.score:.3f})"
        )
        return most_confident.score

    def _consensus(self, feedbacks: List[Feedback], consensus_level: float) -> float:
        """
        Consensus strategy: Only accept high-agreement feedback.

        If critics disagree too much, return conservative score.
        """
        if consensus_level >= self.consensus_threshold:
            # High agreement: Use weighted average
            return self._weighted_average(feedbacks)
        else:
            # Low agreement: Be conservative
            logger.warning(
                f"Low consensus ({consensus_level:.3f}), returning conservative score"
            )
            return min(f.score for f in feedbacks)

    def _calculate_consensus(self, feedbacks: List[Feedback]) -> float:
        """
        Calculate consensus level among critics.

        Measures how much critics agree (0.0 = complete disagreement, 1.0 = perfect agreement).
        Uses coefficient of variation (std/mean) inverted to 0-1 scale.
        """
        if len(feedbacks) < 2:
            return 1.0  # Single critic = perfect "consensus"

        scores = [f.score for f in feedbacks]
        mean_score = sum(scores) / len(scores)

        if mean_score == 0:
            return 0.0

        # Calculate standard deviation
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = variance**0.5

        # Coefficient of variation
        cv = std_dev / mean_score if mean_score > 0 else 1.0

        # Convert to consensus score (lower CV = higher consensus)
        # CV of 0 = consensus of 1.0, CV of 1 = consensus of 0.0
        consensus = max(0.0, 1.0 - cv)

        logger.debug(
            f"Consensus calculation: mean={mean_score:.3f}, "
            f"std={std_dev:.3f}, cv={cv:.3f}, consensus={consensus:.3f}"
        )
        return consensus

    def _extract_metadata(self, feedbacks: List[Feedback]) -> Dict[str, Any]:
        """Extract useful metadata from feedbacks."""
        all_suggestions = []
        critic_scores = {}

        for f in feedbacks:
            all_suggestions.extend(f.suggestions)
            critic_scores[f.critic_name] = f.score

        return {
            "num_critics": len(feedbacks),
            "all_suggestions": all_suggestions,
            "critic_scores": critic_scores,
            "avg_confidence": sum(f.confidence for f in feedbacks) / len(feedbacks)
            if feedbacks
            else 0.0,
        }

    def get_top_suggestions(
        self, aggregated: AggregatedFeedback, top_k: int = 3
    ) -> List[str]:
        """
        Get top-k most important suggestions.

        Prioritizes suggestions from high-confidence critics.
        """
        # Create (suggestion, confidence) pairs
        suggestion_confidence_pairs = []

        for feedback in aggregated.feedbacks:
            for suggestion in feedback.suggestions:
                suggestion_confidence_pairs.append((suggestion, feedback.confidence))

        # Sort by confidence
        sorted_suggestions = sorted(
            suggestion_confidence_pairs, key=lambda x: x[1], reverse=True
        )

        # Return top-k unique suggestions
        seen = set()
        top_suggestions = []

        for suggestion, _ in sorted_suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                top_suggestions.append(suggestion)
                if len(top_suggestions) >= top_k:
                    break

        return top_suggestions
