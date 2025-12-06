"""
Sentiment Analyzer
==================
Sentiment analysis for news, social media, and market commentary.

Features:
- Multi-source sentiment aggregation
- Time-series sentiment tracking
- Sentiment-based signals
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SentimentSource(Enum):
    """Sentiment data sources."""
    NEWS = "news"
    TWITTER = "twitter"
    REDDIT = "reddit"
    ANALYST = "analyst"
    REPORT = "report"
    CUSTOM = "custom"


class SentimentLabel(Enum):
    """Sentiment labels."""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    source: SentimentSource
    text: str
    label: SentimentLabel
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    commodities: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source.value,
            "label": self.label.value,
            "score": self.score,
            "confidence": self.confidence,
            "commodities": self.commodities,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis."""
    # Analysis settings
    use_llm: bool = False  # Use LLM for enhanced analysis
    llm_provider: str = "openai"

    # Commodity focus
    commodities: list[str] = field(default_factory=lambda: ["oil", "crude", "wti", "brent"])

    # Aggregation settings
    aggregation_window_hours: int = 24
    min_confidence: float = 0.5

    # Signal thresholds
    bullish_threshold: float = 0.3
    bearish_threshold: float = -0.3


class SentimentAnalyzer:
    """
    Multi-source sentiment analyzer for oil markets.

    Analyzes text from various sources and produces sentiment scores.
    """

    def __init__(self, config: SentimentConfig | None = None):
        self.config = config or SentimentConfig()

        # Sentiment history
        self._history: list[SentimentResult] = []

        # Keyword dictionaries for rule-based analysis
        self._bullish_keywords = {
            "strong": [
                "surge", "soar", "skyrocket", "breakout", "rally", "boom",
                "supply cut", "shortage", "sanctions", "disruption",
            ],
            "moderate": [
                "rise", "gain", "increase", "higher", "up", "advance",
                "bullish", "optimistic", "positive", "growth", "demand",
            ],
        }

        self._bearish_keywords = {
            "strong": [
                "crash", "plunge", "collapse", "plummet", "crisis",
                "glut", "oversupply", "recession", "demand destruction",
            ],
            "moderate": [
                "fall", "drop", "decline", "lower", "down", "weak",
                "bearish", "pessimistic", "negative", "surplus",
            ],
        }

        # Negation words
        self._negations = ["not", "no", "never", "neither", "nobody", "nothing", "nowhere", "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't"]

    def analyze(
        self,
        text: str,
        source: SentimentSource = SentimentSource.CUSTOM,
        metadata: dict | None = None,
    ) -> SentimentResult:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze
            source: Source of the text
            metadata: Additional metadata

        Returns:
            SentimentResult with analysis
        """
        # Preprocess text
        text_lower = text.lower()

        # Detect commodities
        commodities = self._detect_commodities(text_lower)

        # Calculate sentiment score
        score, confidence = self._calculate_sentiment(text_lower)

        # Determine label
        label = self._score_to_label(score)

        result = SentimentResult(
            source=source,
            text=text[:500],  # Truncate for storage
            label=label,
            score=score,
            confidence=confidence,
            commodities=commodities,
            metadata=metadata or {},
        )

        # Store in history
        self._history.append(result)

        return result

    def analyze_batch(
        self,
        texts: list[str],
        source: SentimentSource = SentimentSource.CUSTOM,
    ) -> list[SentimentResult]:
        """Analyze multiple texts."""
        return [self.analyze(text, source) for text in texts]

    def get_aggregate_sentiment(
        self,
        hours: int | None = None,
        source: SentimentSource | None = None,
        commodity: str | None = None,
    ) -> dict[str, Any]:
        """
        Get aggregate sentiment over a time window.

        Args:
            hours: Time window in hours (default: config.aggregation_window_hours)
            source: Filter by source
            commodity: Filter by commodity

        Returns:
            Aggregated sentiment metrics
        """
        hours = hours or self.config.aggregation_window_hours
        cutoff = datetime.now() - timedelta(hours=hours)

        # Filter history
        results = [r for r in self._history if r.timestamp >= cutoff]

        if source:
            results = [r for r in results if r.source == source]

        if commodity:
            results = [r for r in results if commodity.lower() in [c.lower() for c in r.commodities]]

        if not results:
            return {
                "count": 0,
                "avg_score": 0,
                "avg_confidence": 0,
                "label": SentimentLabel.NEUTRAL.value,
                "signal": "neutral",
            }

        # Calculate aggregates
        scores = [r.score for r in results]
        confidences = [r.confidence for r in results]

        avg_score = sum(scores) / len(scores)
        avg_confidence = sum(confidences) / len(confidences)

        # Weighted average by confidence
        weighted_score = sum(s * c for s, c in zip(scores, confidences)) / sum(confidences)

        # Determine overall label
        label = self._score_to_label(weighted_score)

        # Generate trading signal
        if weighted_score >= self.config.bullish_threshold and avg_confidence >= self.config.min_confidence:
            signal = "bullish"
        elif weighted_score <= self.config.bearish_threshold and avg_confidence >= self.config.min_confidence:
            signal = "bearish"
        else:
            signal = "neutral"

        # Count by label
        label_counts = {}
        for r in results:
            lbl = r.label.value
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        return {
            "count": len(results),
            "avg_score": round(avg_score, 3),
            "weighted_score": round(weighted_score, 3),
            "avg_confidence": round(avg_confidence, 3),
            "label": label.value,
            "signal": signal,
            "label_distribution": label_counts,
            "period_hours": hours,
            "generated_at": datetime.now().isoformat(),
        }

    def get_sentiment_timeseries(
        self,
        hours: int = 24,
        interval_hours: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Get sentiment as a time series.

        Args:
            hours: Total lookback hours
            interval_hours: Interval for each data point

        Returns:
            List of sentiment data points
        """
        now = datetime.now()
        timeseries = []

        for i in range(hours // interval_hours):
            end_time = now - timedelta(hours=i * interval_hours)
            start_time = end_time - timedelta(hours=interval_hours)

            # Filter to this interval
            interval_results = [
                r for r in self._history
                if start_time <= r.timestamp < end_time
            ]

            if interval_results:
                scores = [r.score for r in interval_results]
                avg_score = sum(scores) / len(scores)
            else:
                avg_score = 0

            timeseries.append({
                "timestamp": start_time.isoformat(),
                "score": round(avg_score, 3),
                "count": len(interval_results),
            })

        # Reverse to chronological order
        return list(reversed(timeseries))

    def _detect_commodities(self, text: str) -> list[str]:
        """Detect mentioned commodities."""
        commodities = []

        commodity_patterns = {
            "WTI": [r"\bwti\b", r"west texas", r"\bcl1\b"],
            "Brent": [r"\bbrent\b", r"\bco1\b", r"north sea"],
            "Crude Oil": [r"crude oil", r"crude", r"oil price"],
            "Gasoline": [r"gasoline", r"petrol", r"\brbob\b"],
            "Heating Oil": [r"heating oil", r"distillate", r"\bho1\b"],
            "Natural Gas": [r"natural gas", r"\bng1\b", r"natgas"],
        }

        for commodity, patterns in commodity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    commodities.append(commodity)
                    break

        return commodities

    def _calculate_sentiment(self, text: str) -> tuple:
        """Calculate sentiment score and confidence."""
        # Tokenize
        words = text.split()

        # Count sentiment keywords
        bullish_strong = 0
        bullish_moderate = 0
        bearish_strong = 0
        bearish_moderate = 0

        # Check for negation context
        negation_window = 3  # Words after negation to consider
        negation_indices = set()

        for i, word in enumerate(words):
            if word in self._negations:
                for j in range(i + 1, min(i + negation_window + 1, len(words))):
                    negation_indices.add(j)

        # Count keywords
        for i, word in enumerate(words):
            is_negated = i in negation_indices

            for kw in self._bullish_keywords["strong"]:
                if kw in word or word in kw:
                    if is_negated:
                        bearish_moderate += 1
                    else:
                        bullish_strong += 1
                    break

            for kw in self._bullish_keywords["moderate"]:
                if kw in word or word in kw:
                    if is_negated:
                        bearish_moderate += 0.5
                    else:
                        bullish_moderate += 1
                    break

            for kw in self._bearish_keywords["strong"]:
                if kw in word or word in kw:
                    if is_negated:
                        bullish_moderate += 1
                    else:
                        bearish_strong += 1
                    break

            for kw in self._bearish_keywords["moderate"]:
                if kw in word or word in kw:
                    if is_negated:
                        bullish_moderate += 0.5
                    else:
                        bearish_moderate += 1
                    break

        # Calculate score
        bullish_total = bullish_strong * 2 + bullish_moderate
        bearish_total = bearish_strong * 2 + bearish_moderate

        total = bullish_total + bearish_total

        if total == 0:
            score = 0
            confidence = 0.3
        else:
            score = (bullish_total - bearish_total) / total
            # Confidence based on keyword density
            word_count = len(words)
            keyword_density = total / max(word_count, 1)
            confidence = min(0.9, 0.4 + keyword_density * 5)

        return round(score, 3), round(confidence, 3)

    def _score_to_label(self, score: float) -> SentimentLabel:
        """Convert score to sentiment label."""
        if score >= 0.6:
            return SentimentLabel.VERY_BULLISH
        elif score >= 0.2:
            return SentimentLabel.BULLISH
        elif score <= -0.6:
            return SentimentLabel.VERY_BEARISH
        elif score <= -0.2:
            return SentimentLabel.BEARISH
        else:
            return SentimentLabel.NEUTRAL

    def clear_history(self, before: datetime | None = None):
        """Clear sentiment history."""
        if before:
            self._history = [r for r in self._history if r.timestamp >= before]
        else:
            self._history.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get analyzer statistics."""
        if not self._history:
            return {"count": 0}

        return {
            "total_analyzed": len(self._history),
            "sources": list({r.source.value for r in self._history}),
            "avg_score": round(sum(r.score for r in self._history) / len(self._history), 3),
            "oldest": self._history[0].timestamp.isoformat() if self._history else None,
            "newest": self._history[-1].timestamp.isoformat() if self._history else None,
        }
