"""
Research Module
===============
Advanced analytics, AI-powered research, and alternative data sources.

This module provides:
- LLM-powered news analysis and summarization
- Sentiment scoring for news and social media
- Alternative data integrations (satellite, shipping, positioning)
- Cross-asset correlation analysis
- Regime detection using Hidden Markov Models
- Factor decomposition

Phase 8 Implementation - Advanced Analytics & AI
"""

from .alt_data import (
    AlternativeDataProvider,
    PositioningData,
    SatelliteData,
    ShippingData,
)
from .correlations import (
    CorrelationAnalyzer,
    CorrelationRegime,
    CrossAssetCorrelation,
    RollingCorrelation,
)
from .factors import (
    FactorConfig,
    FactorDecomposition,
    FactorModel,
    RiskFactor,
)
from .llm import (
    AnalysisConfig,
    GrokAI,
    GrokConfig,
    NewsAnalyzer,
    NewsArticle,
    SentimentAnalyzer,
    SentimentResult,
    Tweet,
    WordCloudData,
    get_tweet_wordcloud,
)
from .regimes import (
    MarketRegime,
    RegimeConfig,
    RegimeDetector,
    RegimeTransition,
)

__all__ = [
    # LLM
    "NewsAnalyzer",
    "SentimentAnalyzer",
    "NewsArticle",
    "SentimentResult",
    "AnalysisConfig",
    # GrokAI
    "GrokAI",
    "GrokConfig",
    "Tweet",
    "WordCloudData",
    "get_tweet_wordcloud",
    # Correlations
    "CorrelationAnalyzer",
    "CrossAssetCorrelation",
    "RollingCorrelation",
    "CorrelationRegime",
    # Regimes
    "RegimeDetector",
    "MarketRegime",
    "RegimeConfig",
    "RegimeTransition",
    # Factors
    "FactorModel",
    "FactorDecomposition",
    "FactorConfig",
    "RiskFactor",
    # Alternative Data
    "AlternativeDataProvider",
    "SatelliteData",
    "ShippingData",
    "PositioningData",
]
