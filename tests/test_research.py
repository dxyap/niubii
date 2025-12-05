"""
Tests for Research Module
=========================
Tests for LLM analysis, correlations, regimes, factors, and alternative data.
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from core.research import (
    NewsAnalyzer,
    SentimentAnalyzer,
    CorrelationAnalyzer,
    RegimeDetector,
    FactorModel,
    AlternativeDataProvider,
)
from core.research.llm import AnalysisConfig, SentimentConfig
from core.research.correlations import CorrelationMethod
from core.research.regimes import MarketRegime, RegimeConfig
from core.research.factors import RiskFactor, FactorConfig


class TestNewsAnalyzer:
    """Tests for news analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create news analyzer for tests."""
        config = AnalysisConfig(use_llm=False)  # Use rule-based for testing
        return NewsAnalyzer(config)
    
    def test_analyzer_creation(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
    
    def test_analyze_bullish_article(self, analyzer):
        """Test analyzing a bullish article."""
        text = "Oil prices surge as OPEC announces production cuts. Strong demand outlook expected."
        
        summary = analyzer.analyze_article(text)
        
        assert summary is not None
        assert summary.summary  # Has a summary
        assert summary.impact_direction in ["BULLISH", "NEUTRAL"]
    
    def test_analyze_bearish_article(self, analyzer):
        """Test analyzing a bearish article."""
        text = "Oil prices collapse amid oversupply concerns. Weak demand weighs on market."
        
        summary = analyzer.analyze_article(text)
        
        assert summary is not None
        assert summary.impact_direction in ["BEARISH", "NEUTRAL"]
    
    def test_key_points_extraction(self, analyzer):
        """Test key points extraction."""
        text = """
        OPEC+ agrees to extend production cuts through Q2 2024.
        Saudi Arabia to maintain voluntary 1 million bpd cut.
        China demand outlook improves as economy recovers.
        """
        
        summary = analyzer.analyze_article(text)
        
        assert len(summary.key_points) > 0
    
    def test_commodity_detection(self, analyzer):
        """Test commodity entity detection."""
        text = "WTI crude and Brent oil prices both rallied on supply concerns."
        
        summary = analyzer.analyze_article(text)
        
        assert len(summary.commodities) > 0


class TestSentimentAnalyzer:
    """Tests for sentiment analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create sentiment analyzer for tests."""
        return SentimentAnalyzer()
    
    def test_analyzer_creation(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
    
    def test_positive_sentiment(self, analyzer):
        """Test positive sentiment detection."""
        text = "Oil prices surge on strong demand and bullish outlook"
        
        result = analyzer.analyze_text(text)
        
        assert result.score > 0
    
    def test_negative_sentiment(self, analyzer):
        """Test negative sentiment detection."""
        text = "Oil prices collapse amid weak demand and bearish outlook"
        
        result = analyzer.analyze_text(text)
        
        assert result.score < 0
    
    def test_neutral_sentiment(self, analyzer):
        """Test neutral sentiment detection."""
        text = "Oil traded in a narrow range today"
        
        result = analyzer.analyze_text(text)
        
        assert -0.3 <= result.score <= 0.3
    
    def test_aggregate_sentiment(self, analyzer):
        """Test aggregate sentiment calculation."""
        texts = [
            "Oil prices surge on bullish outlook",
            "Strong demand supports prices",
            "Market remains cautious",
        ]
        
        aggregate = analyzer.get_aggregate_sentiment(texts)
        
        assert "avg_score" in aggregate
        assert "sentiment_distribution" in aggregate


class TestCorrelationAnalyzer:
    """Tests for correlation analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create correlation analyzer for tests."""
        return CorrelationAnalyzer()
    
    def test_analyzer_creation(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
    
    def test_pairwise_correlation(self, analyzer):
        """Test pairwise correlation calculation."""
        correlation = analyzer.calculate_correlation("Brent", "WTI")
        
        assert correlation is not None
        assert -1 <= correlation.correlation <= 1
    
    def test_correlation_matrix(self, analyzer):
        """Test correlation matrix calculation."""
        assets = ["Brent", "WTI", "Dollar"]
        
        matrix = analyzer.calculate_correlation_matrix(assets)
        
        assert not matrix.empty
        assert matrix.shape == (len(assets), len(assets))
        # Diagonal should be 1
        for asset in assets:
            if asset in matrix.index and asset in matrix.columns:
                assert abs(matrix.loc[asset, asset] - 1.0) < 0.01
    
    def test_rolling_correlation(self, analyzer):
        """Test rolling correlation calculation."""
        rolling = analyzer.calculate_rolling_correlation(
            "Brent", "WTI",
            window=21,
            days=90,
        )
        
        assert len(rolling) > 0
        for r in rolling:
            assert -1 <= r.correlation <= 1
    
    def test_regime_detection(self, analyzer):
        """Test correlation regime detection."""
        regime = analyzer.detect_regime("Brent", "WTI")
        
        assert "regime" in regime
        assert "current_correlation" in regime


class TestRegimeDetector:
    """Tests for regime detector."""
    
    @pytest.fixture
    def detector(self):
        """Create regime detector for tests."""
        return RegimeDetector()
    
    def test_detector_creation(self, detector):
        """Test detector initialization."""
        assert detector is not None
    
    def test_current_regime(self, detector):
        """Test getting current regime."""
        regime = detector.get_current_regime()
        
        assert "regime" in regime
        assert "confidence" in regime
        assert regime["confidence"] > 0
    
    def test_volatility_regime(self, detector):
        """Test volatility regime detection."""
        vol_regime = detector.get_volatility_regime()
        
        assert "regime" in vol_regime
        assert "current_volatility" in vol_regime
    
    def test_regime_history(self, detector):
        """Test regime history retrieval."""
        history = detector.get_regime_history(days=30)
        
        assert len(history) > 0
        for h in history:
            assert "date" in h
            assert "regime" in h
    
    def test_regime_transitions(self, detector):
        """Test regime transition detection."""
        transitions = detector.get_regime_transitions(limit=5)
        
        # May or may not have transitions
        assert isinstance(transitions, list)


class TestFactorModel:
    """Tests for factor model."""
    
    @pytest.fixture
    def model(self):
        """Create factor model for tests."""
        return FactorModel()
    
    def test_model_creation(self, model):
        """Test model initialization."""
        assert model is not None
    
    def test_factor_returns(self, model):
        """Test getting factor returns."""
        factors = model.get_factor_returns(days=30)
        
        assert not factors.empty
        assert len(factors.columns) > 0
    
    def test_decompose_returns(self, model):
        """Test return decomposition."""
        decomposition = model.decompose_returns("Brent", days=60)
        
        assert decomposition is not None
        assert decomposition.r_squared >= 0
        assert decomposition.r_squared <= 1
        assert len(decomposition.factor_exposures) > 0
    
    def test_factor_exposures(self, model):
        """Test factor exposure calculation."""
        decomposition = model.decompose_returns("WTI", days=30)
        
        exposures = decomposition.factor_exposures
        
        assert "market" in exposures or len(exposures) > 0


class TestAlternativeDataProvider:
    """Tests for alternative data provider."""
    
    @pytest.fixture
    def provider(self):
        """Create alternative data provider for tests."""
        return AlternativeDataProvider()
    
    def test_provider_creation(self, provider):
        """Test provider initialization."""
        assert provider is not None
        assert provider.satellite is not None
        assert provider.shipping is not None
        assert provider.positioning is not None
    
    def test_satellite_observations(self, provider):
        """Test satellite data retrieval."""
        obs = provider.satellite.get_latest_observations()
        
        assert "locations" in obs
        assert len(obs["locations"]) > 0
    
    def test_satellite_signal(self, provider):
        """Test satellite-based signal."""
        signal = provider.satellite.calculate_storage_signal()
        
        assert "signal" in signal
        assert signal["signal"] in ["bullish", "bearish", "neutral"]
        assert "confidence" in signal
    
    def test_shipping_fleet(self, provider):
        """Test shipping fleet data."""
        fleet = provider.shipping.get_fleet_overview()
        
        assert "fleet_by_type" in fleet
    
    def test_shipping_flows(self, provider):
        """Test trade flows data."""
        flows = provider.shipping.get_trade_flows()
        
        assert "flows" in flows
        assert "total_observed_mb_d" in flows
    
    def test_shipping_signal(self, provider):
        """Test shipping-based signal."""
        signal = provider.shipping.calculate_shipping_signal()
        
        assert "signal" in signal
        assert "confidence" in signal
    
    def test_positioning_cot(self, provider):
        """Test COT data retrieval."""
        cot = provider.positioning.get_latest_cot()
        
        assert "data" in cot
    
    def test_positioning_managed_money(self, provider):
        """Test managed money positions."""
        positions = provider.positioning.get_managed_money_positions()
        
        assert "positions" in positions
    
    def test_positioning_signal(self, provider):
        """Test positioning-based signal."""
        signal = provider.positioning.calculate_positioning_signal()
        
        assert "signal" in signal
        assert "confidence" in signal
    
    def test_aggregate_signal(self, provider):
        """Test aggregate alternative data signal."""
        aggregate = provider.get_aggregate_signal()
        
        assert "overall_signal" in aggregate
        assert "overall_confidence" in aggregate
        assert "signals" in aggregate
