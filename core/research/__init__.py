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

from .llm import (
    NewsAnalyzer,
    SentimentAnalyzer,
    NewsArticle,
    SentimentResult,
    AnalysisConfig,
)

from .correlations import (
    CorrelationAnalyzer,
    CrossAssetCorrelation,
    RollingCorrelation,
    CorrelationRegime,
)

from .regimes import (
    RegimeDetector,
    MarketRegime,
    RegimeConfig,
    RegimeTransition,
)

from .factors import (
    FactorModel,
    FactorDecomposition,
    FactorConfig,
    RiskFactor,
)

from .alt_data import (
    AlternativeDataProvider,
    SatelliteData,
    ShippingData,
    PositioningData,
)

__all__ = [
    # LLM
    "NewsAnalyzer",
    "SentimentAnalyzer",
    "NewsArticle",
    "SentimentResult",
    "AnalysisConfig",
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
