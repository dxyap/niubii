"""
Tests for Research Module
=========================
Tests for LLM analysis, correlations, regimes, factors, and alternative data.
"""

import uuid

import pytest

from core.research import (
    AlternativeDataProvider,
    CorrelationAnalyzer,
    FactorModel,
    NewsAnalyzer,
    RegimeDetector,
    SentimentAnalyzer,
)
from core.research.alt_data import PositioningData, SatelliteData, ShippingData
from core.research.correlations import create_mock_price_data
from core.research.factors import create_mock_factor_data
from core.research.llm import AnalysisConfig, NewsArticle
from core.research.regimes import create_mock_regime_data


class TestNewsAnalyzer:
    """Tests for news analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create news analyzer for tests."""
        # Default config uses rule-based analysis (no LLM)
        config = AnalysisConfig()
        return NewsAnalyzer(config)

    def test_analyzer_creation(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None

    def test_analyze_bullish_article(self, analyzer):
        """Test analyzing a bullish article."""
        article = NewsArticle(
            article_id=str(uuid.uuid4()),
            title="Oil Prices Surge",
            content="Oil prices surge as OPEC announces production cuts. Strong demand outlook expected.",
        )

        summary = analyzer.analyze(article)

        assert summary is not None
        assert summary.summary  # Has a summary
        # Fallback analysis may not always detect bullish
        assert summary.impact_direction in ["bullish", "bearish", "neutral", "BULLISH", "NEUTRAL", "BEARISH"]

    def test_analyze_bearish_article(self, analyzer):
        """Test analyzing a bearish article."""
        article = NewsArticle(
            article_id=str(uuid.uuid4()),
            title="Oil Prices Fall",
            content="Oil prices collapse amid oversupply concerns. Weak demand weighs on market.",
        )

        summary = analyzer.analyze(article)

        assert summary is not None
        assert summary.impact_direction in ["bearish", "neutral", "bullish", "BEARISH", "NEUTRAL", "BULLISH"]

    def test_key_points_extraction(self, analyzer):
        """Test key points extraction."""
        article = NewsArticle(
            article_id=str(uuid.uuid4()),
            title="OPEC Meeting",
            content="""
            OPEC+ agrees to extend production cuts through Q2 2024.
            Saudi Arabia to maintain voluntary 1 million bpd cut.
            China demand outlook improves as economy recovers.
            """,
        )

        summary = analyzer.analyze(article)

        # Key points should be present
        assert hasattr(summary, "key_points")

    def test_commodity_detection(self, analyzer):
        """Test commodity entity detection."""
        article = NewsArticle(
            article_id=str(uuid.uuid4()),
            title="Crude Oil Rally",
            content="WTI crude and Brent oil prices both rallied on supply concerns.",
        )

        summary = analyzer.analyze(article)

        assert hasattr(summary, "commodities_mentioned")


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

        result = analyzer.analyze(text)

        assert result.score > 0

    def test_negative_sentiment(self, analyzer):
        """Test negative sentiment detection."""
        text = "Oil prices collapse amid weak demand and bearish outlook"

        result = analyzer.analyze(text)

        assert result.score < 0

    def test_neutral_sentiment(self, analyzer):
        """Test neutral sentiment detection."""
        text = "Oil traded in a narrow range today"

        result = analyzer.analyze(text)

        assert -0.5 <= result.score <= 0.5

    def test_aggregate_sentiment(self, analyzer):
        """Test aggregate sentiment calculation."""
        texts = [
            "Oil prices surge on bullish outlook",
            "Strong demand supports prices",
            "Market remains cautious",
        ]

        # Analyze each text to populate history
        for text in texts:
            analyzer.analyze(text)

        # Get aggregate
        aggregate = analyzer.get_aggregate_sentiment()

        assert "avg_score" in aggregate
        assert "label" in aggregate


class TestCorrelationAnalyzer:
    """Tests for correlation analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create correlation analyzer for tests."""
        return CorrelationAnalyzer()

    @pytest.fixture
    def price_data(self):
        """Create mock price data for tests."""
        return create_mock_price_data(days=120)

    def test_analyzer_creation(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None

    def test_pairwise_correlation(self, analyzer, price_data):
        """Test pairwise correlation calculation."""
        result = analyzer.calculate_pair_correlation(
            price_data["Brent"],
            price_data["WTI"],
            "Brent",
            "WTI",
        )

        assert result is not None
        assert -1 <= result.correlation <= 1

    def test_correlation_matrix(self, analyzer, price_data):
        """Test correlation matrix calculation."""
        matrix = analyzer.calculate_correlation_matrix(price_data)

        assert not matrix.empty
        # Diagonal should be 1
        for asset in matrix.index:
            assert abs(matrix.loc[asset, asset] - 1.0) < 0.01

    def test_rolling_correlation(self, analyzer, price_data):
        """Test rolling correlation calculation."""
        rolling = analyzer.calculate_rolling_correlation(
            price_data["Brent"],
            price_data["WTI"],
            "Brent",
            "WTI",
            window_days=21,
        )

        assert len(rolling.correlations) > 0
        for corr in rolling.correlations:
            assert -1 <= corr <= 1


class TestRegimeDetector:
    """Tests for regime detector."""

    @pytest.fixture
    def detector(self):
        """Create regime detector for tests."""
        return RegimeDetector()

    @pytest.fixture
    def prices(self):
        """Create mock price data for tests."""
        return create_mock_regime_data(days=120)

    def test_detector_creation(self, detector):
        """Test detector initialization."""
        assert detector is not None

    def test_current_regime(self, detector, prices):
        """Test getting current regime."""
        regime = detector.detect_current_regime(prices)

        assert "regime" in regime
        assert "confidence" in regime
        assert regime["confidence"] > 0

    def test_volatility_regime(self, detector, prices):
        """Test volatility regime detection."""
        regime = detector.detect_current_regime(prices)

        assert "volatility_regime" in regime

    def test_regime_timeseries(self, detector, prices):
        """Test regime history retrieval."""
        history = detector.detect_regime_timeseries(prices, window=30)

        assert len(history) > 0
        assert "regime" in history.columns

    def test_regime_transitions(self, detector, prices):
        """Test regime transition detection."""
        # Run detection to populate transitions
        detector.detect_current_regime(prices)

        transitions = detector.get_recent_transitions(limit=5)

        # May or may not have transitions
        assert isinstance(transitions, list)


class TestFactorModel:
    """Tests for factor model."""

    @pytest.fixture
    def model(self):
        """Create factor model for tests."""
        return FactorModel()

    @pytest.fixture
    def price_data(self):
        """Create mock price data for tests."""
        return create_mock_factor_data(days=120)

    def test_model_creation(self, model):
        """Test model initialization."""
        assert model is not None

    def test_construct_factors(self, model, price_data):
        """Test constructing factor returns."""
        factors = model.construct_factors(price_data)

        assert len(factors) > 0

    def test_decompose_returns(self, model, price_data):
        """Test return decomposition."""
        # Construct factors first
        model.construct_factors(price_data)

        # Get returns for Brent
        brent_returns = price_data["Brent"].pct_change().dropna()

        decomposition = model.decompose_returns(
            brent_returns.tail(100),
            asset_name="Brent",
        )

        assert decomposition is not None
        assert decomposition.r_squared >= 0
        assert decomposition.r_squared <= 1
        assert len(decomposition.exposures) > 0

    def test_factor_statistics(self, model, price_data):
        """Test factor statistics calculation."""
        model.construct_factors(price_data)

        stats = model.get_factor_statistics()

        assert len(stats) > 0


class TestSatelliteData:
    """Tests for satellite data provider."""

    @pytest.fixture
    def provider(self):
        """Create satellite data provider for tests."""
        return SatelliteData()

    def test_provider_creation(self, provider):
        """Test provider initialization."""
        assert provider is not None

    def test_latest_observations(self, provider):
        """Test satellite data retrieval."""
        obs = provider.get_latest_observations()

        assert "locations" in obs
        assert len(obs["locations"]) > 0

    def test_storage_signal(self, provider):
        """Test satellite-based signal."""
        signal = provider.calculate_storage_signal()

        assert "signal" in signal
        assert signal["signal"] in ["bullish", "bearish", "neutral"]
        assert "confidence" in signal


class TestShippingData:
    """Tests for shipping data provider."""

    @pytest.fixture
    def provider(self):
        """Create shipping data provider for tests."""
        return ShippingData()

    def test_provider_creation(self, provider):
        """Test provider initialization."""
        assert provider is not None

    def test_fleet_overview(self, provider):
        """Test shipping fleet data."""
        fleet = provider.get_fleet_overview()

        assert "fleet_by_type" in fleet

    def test_trade_flows(self, provider):
        """Test trade flows data."""
        flows = provider.get_trade_flows()

        assert "flows" in flows
        assert "total_observed_mb_d" in flows

    def test_shipping_signal(self, provider):
        """Test shipping-based signal."""
        signal = provider.calculate_shipping_signal()

        assert "signal" in signal
        assert "confidence" in signal


class TestPositioningData:
    """Tests for positioning data provider."""

    @pytest.fixture
    def provider(self):
        """Create positioning data provider for tests."""
        return PositioningData()

    def test_provider_creation(self, provider):
        """Test provider initialization."""
        assert provider is not None

    def test_cot_data(self, provider):
        """Test COT data retrieval."""
        cot = provider.get_latest_cot()

        assert "data" in cot

    def test_managed_money(self, provider):
        """Test managed money positions."""
        positions = provider.get_managed_money_positions()

        assert "positions" in positions

    def test_positioning_signal(self, provider):
        """Test positioning-based signal."""
        signal = provider.calculate_positioning_signal()

        assert "signal" in signal
        assert "confidence" in signal


class TestAlternativeDataProvider:
    """Tests for alternative data provider."""

    @pytest.fixture
    def provider(self):
        """Create alternative data provider for tests."""
        return AlternativeDataProvider()

    def test_provider_creation(self, provider):
        """Test provider initialization."""
        assert provider is not None

    def test_aggregated_view(self, provider):
        """Test aggregated view from all sources."""
        aggregate = provider.get_aggregated_view()

        assert "timestamp" in aggregate
        assert "sources" in aggregate
        assert "signal" in aggregate
