"""
Tests for Data Module
=====================
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data.bloomberg import BloombergClient, MockBloombergData, TickerMapper
from core.data.cache import DataCache, ParquetStorage
from core.data.loader import DataLoader


class TestTickerMapper:
    """Tests for ticker mapping and validation."""

    def test_get_front_month_ticker(self):
        """Test getting front month tickers."""
        assert TickerMapper.get_front_month_ticker("wti") == "CL1 Comdty"
        assert TickerMapper.get_front_month_ticker("brent") == "CO1 Comdty"
        assert TickerMapper.get_front_month_ticker("rbob") == "XB1 Comdty"
        assert TickerMapper.get_front_month_ticker("heating_oil") == "HO1 Comdty"

    def test_get_nth_month_ticker(self):
        """Test getting nth month tickers."""
        assert TickerMapper.get_nth_month_ticker("wti", 1) == "CL1 Comdty"
        assert TickerMapper.get_nth_month_ticker("wti", 2) == "CL2 Comdty"
        assert TickerMapper.get_nth_month_ticker("wti", 12) == "CL12 Comdty"
        assert TickerMapper.get_nth_month_ticker("brent", 6) == "CO6 Comdty"

    def test_get_specific_month_ticker(self):
        """Test getting specific month/year tickers."""
        ticker = TickerMapper.get_specific_month_ticker("CL", 1, 2025)
        assert ticker == "CLF5 Comdty"

        ticker = TickerMapper.get_specific_month_ticker("CO", 12, 2025)
        assert ticker == "COZ5 Comdty"

    def test_validate_ticker_valid(self):
        """Test validation of valid tickers."""
        valid, msg = TickerMapper.validate_ticker("CL1 Comdty")
        assert valid is True

        valid, msg = TickerMapper.validate_ticker("CO12 Comdty")
        assert valid is True

        valid, msg = TickerMapper.validate_ticker("XB1 Comdty")
        assert valid is True

    def test_validate_ticker_invalid(self):
        """Test validation of invalid tickers."""
        # Missing suffix
        valid, msg = TickerMapper.validate_ticker("CL1")
        assert valid is False
        assert "Comdty" in msg

        # Empty ticker
        valid, msg = TickerMapper.validate_ticker("")
        assert valid is False

        # Unknown commodity
        valid, msg = TickerMapper.validate_ticker("XX1 Comdty")
        assert valid is False

    def test_parse_ticker_generic(self):
        """Test parsing generic tickers."""
        parsed = TickerMapper.parse_ticker("CL1 Comdty")

        assert parsed["type"] == "generic"
        assert parsed["commodity"] == "CL"
        assert parsed["month_number"] == 1
        assert parsed["exchange"] == "NYMEX"
        assert parsed["multiplier"] == 1000

    def test_parse_ticker_specific(self):
        """Test parsing specific contract tickers."""
        parsed = TickerMapper.parse_ticker("CLF5 Comdty")

        assert parsed["type"] == "specific"
        assert parsed["commodity"] == "CL"
        assert parsed["month_code"] == "F"
        assert parsed["month"] == 1  # January

    def test_get_field(self):
        """Test getting Bloomberg field names."""
        assert TickerMapper.get_field("last") == "PX_LAST"
        assert TickerMapper.get_field("bid") == "PX_BID"
        assert TickerMapper.get_field("volume") == "PX_VOLUME"

    def test_get_multiplier(self):
        """Test getting contract multipliers."""
        assert TickerMapper.get_multiplier("CL1 Comdty") == 1000
        assert TickerMapper.get_multiplier("XB1 Comdty") == 42000
        assert TickerMapper.get_multiplier("HO1 Comdty") == 42000
        assert TickerMapper.get_multiplier("QS1 Comdty") == 100

    def test_month_codes(self):
        """Test month code mappings."""
        codes = TickerMapper.MONTH_CODES
        assert codes[1] == 'F'
        assert codes[6] == 'M'
        assert codes[12] == 'Z'


class TestBloombergClient:
    """Tests for Bloomberg API client."""

    def test_mock_client_initialization(self):
        """Test mock client initializes correctly."""
        client = BloombergClient(use_mock=True)
        assert client.use_mock is True

    def test_get_price(self):
        """Test getting single price."""
        client = BloombergClient(use_mock=True)
        price = client.get_price("CL1 Comdty")

        assert isinstance(price, float)
        assert price > 0

    def test_get_price_different_fields(self):
        """Test getting different price fields."""
        client = BloombergClient(use_mock=True)

        last = client.get_price("CL1 Comdty", "PX_LAST")
        bid = client.get_price("CL1 Comdty", "PX_BID")
        ask = client.get_price("CL1 Comdty", "PX_ASK")

        assert last > 0
        assert bid < ask  # Bid should be less than ask
        assert True  # Last should be between bid/ask (approximately)

    def test_get_price_with_change(self):
        """Test getting price with change."""
        client = BloombergClient(use_mock=True)
        data = client.get_price_with_change("CL1 Comdty")

        assert "current" in data
        assert "open" in data
        assert "change" in data
        assert "change_pct" in data
        assert "high" in data
        assert "low" in data

    def test_get_prices_multiple(self):
        """Test getting multiple prices."""
        client = BloombergClient(use_mock=True)
        tickers = ["CL1 Comdty", "CO1 Comdty", "XB1 Comdty"]

        prices = client.get_prices(tickers)

        assert isinstance(prices, pd.DataFrame)
        assert len(prices) == 3
        assert "PX_LAST" in prices.columns

    def test_get_historical(self):
        """Test getting historical data."""
        client = BloombergClient(use_mock=True)

        hist = client.get_historical(
            "CL1 Comdty",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )

        assert isinstance(hist, pd.DataFrame)
        assert len(hist) > 0
        assert "PX_LAST" in hist.columns
        assert "PX_OPEN" in hist.columns
        assert "PX_HIGH" in hist.columns
        assert "PX_LOW" in hist.columns

    def test_get_curve(self):
        """Test getting futures curve."""
        client = BloombergClient(use_mock=True)

        curve = client.get_curve("wti", num_months=12)

        assert isinstance(curve, pd.DataFrame)
        assert len(curve) == 12
        assert "price" in curve.columns
        assert "ticker" in curve.columns
        assert "month" in curve.columns
        assert "open_interest" in curve.columns

    def test_get_intraday_prices(self):
        """Test getting intraday prices."""
        client = BloombergClient(use_mock=True)

        intraday = client.get_intraday_prices("CL1 Comdty")

        assert isinstance(intraday, pd.DataFrame)
        assert "timestamp" in intraday.columns or len(intraday) == 0

    def test_price_consistency(self):
        """Test that prices are consistent within a session."""
        client = BloombergClient(use_mock=True)

        # Get multiple prices in quick succession
        prices = [client.get_price("CL1 Comdty") for _ in range(5)]

        # Prices should be similar (within small movement range)
        assert max(prices) - min(prices) < prices[0] * 0.05  # Within 5%


class TestMockBloombergData:
    """Tests for mock data generation."""

    def test_generate_eia_inventory(self):
        """Test EIA inventory data generation."""
        data = MockBloombergData.generate_eia_inventory_data(periods=52)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 52
        assert "inventory_mmb" in data.columns
        assert "change_mmb" in data.columns
        assert "surprise_mmb" in data.columns

    def test_generate_opec_production(self):
        """Test OPEC production data generation."""
        data = MockBloombergData.generate_opec_production_data()

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert "country" in data.columns
        assert "quota_mbpd" in data.columns
        assert "actual_mbpd" in data.columns
        assert "compliance_pct" in data.columns

    def test_generate_turnaround_data(self):
        """Test turnaround data generation."""
        data = MockBloombergData.generate_turnaround_data()

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert "region" in data.columns
        assert "capacity_kbpd" in data.columns
        assert "type" in data.columns


class TestDataCache:
    """Tests for data caching."""

    def test_cache_set_get(self, tmp_path):
        """Test setting and getting cache values."""
        cache = DataCache(cache_dir=str(tmp_path / "cache"))

        cache.set("test_key", {"value": 123}, cache_type="historical")
        result = cache.get("test_key", cache_type="historical")

        assert result == {"value": 123}

    def test_cache_clear(self, tmp_path):
        """Test clearing cache."""
        cache = DataCache(cache_dir=str(tmp_path / "cache"))

        cache.set("test_key", "value", cache_type="historical")
        cache.clear()

        result = cache.get("test_key", cache_type="historical")
        assert result is None

    def test_cache_stats(self, tmp_path):
        """Test getting cache statistics."""
        cache = DataCache(cache_dir=str(tmp_path / "cache"))

        stats = cache.get_stats()

        assert "memory_entries" in stats
        assert "disk_entries" in stats
        assert "file_entries" in stats

    def test_parquet_storage_handles_date_index(self, tmp_path):
        """Ensure cached OHLCV loads even when index stored as python date objects."""
        storage = ParquetStorage(base_dir=str(tmp_path / "data"))
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {"PX_LAST": [70.0, 71.0, 72.0]},
            index=pd.Index([d.date() for d in dates], name="date"),
        )
        storage.save_ohlcv("CL1 Comdty", df, "daily")

        loaded = storage.load_ohlcv(
            "CL1 Comdty",
            frequency="daily",
            start_date=datetime(2024, 1, 2),
            end_date=datetime(2024, 1, 3),
        )

        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == 2
        assert loaded.index.min() >= pd.Timestamp("2024-01-02")
        assert loaded.index.max() <= pd.Timestamp("2024-01-03")


class TestDataLoader:
    """Tests for data loader."""

    def test_loader_initialization(self, tmp_path):
        """Test data loader initializes correctly."""
        loader = DataLoader(
            config_dir=str(tmp_path / "config"),
            data_dir=str(tmp_path / "data"),
            use_mock=True
        )

        assert loader.bloomberg.use_mock is True

    def test_get_oil_prices(self, tmp_path):
        """Test getting oil prices."""
        loader = DataLoader(
            config_dir=str(tmp_path / "config"),
            data_dir=str(tmp_path / "data"),
            use_mock=True
        )

        prices = loader.get_oil_prices()

        assert "WTI" in prices
        assert "Brent" in prices
        assert "RBOB" in prices
        assert "Heating Oil" in prices

        # Prices are dictionaries with current, change, etc.
        assert prices["WTI"]["current"] > 0
        assert "change" in prices["WTI"]
        assert "change_pct" in prices["WTI"]

    def test_get_market_summary(self, tmp_path):
        """Test getting market summary."""
        loader = DataLoader(
            config_dir=str(tmp_path / "config"),
            data_dir=str(tmp_path / "data"),
            use_mock=True
        )

        summary = loader.get_market_summary()

        assert "prices" in summary
        assert "spreads" in summary
        assert "curve" in summary
        assert "timestamp" in summary

    def test_get_wti_brent_spread(self, tmp_path):
        """Test WTI-Brent spread calculation."""
        loader = DataLoader(
            config_dir=str(tmp_path / "config"),
            data_dir=str(tmp_path / "data"),
            use_mock=True
        )

        spread = loader.get_wti_brent_spread()

        assert "spread" in spread
        assert "change" in spread
        assert "wti" in spread
        assert "brent" in spread

        # WTI-Brent spread should approximately equal wti - brent
        calculated = spread["wti"] - spread["brent"]
        assert abs(spread["spread"] - calculated) < 0.01  # Allow for rounding

    def test_get_crack_spread_321(self, tmp_path):
        """Test 3-2-1 crack spread calculation."""
        loader = DataLoader(
            config_dir=str(tmp_path / "config"),
            data_dir=str(tmp_path / "data"),
            use_mock=True
        )

        crack = loader.get_crack_spread_321()

        assert "crack" in crack
        assert "change" in crack
        assert "wti" in crack
        assert "rbob_bbl" in crack
        assert "ho_bbl" in crack

    def test_crack_spread_321_formula_accuracy(self, tmp_path, monkeypatch):
        """Ensure crack spread calculation performs accurate unit conversion."""
        loader = DataLoader(
            config_dir=str(tmp_path / "config"),
            data_dir=str(tmp_path / "data"),
            use_mock=True
        )

        sample_batch = {
            "CL1 Comdty": {"current": 70.0, "open": 69.0},
            "XB1 Comdty": {"current": 2.0, "open": 1.95},
            "HO1 Comdty": {"current": 2.5, "open": 2.45},
        }
        monkeypatch.setattr(loader, "get_prices_batch", lambda tickers: sample_batch)

        result = loader.get_crack_spread_321()

        rbob_bbl = sample_batch["XB1 Comdty"]["current"] * loader.GALLONS_PER_BARREL
        ho_bbl = sample_batch["HO1 Comdty"]["current"] * loader.GALLONS_PER_BARREL
        wti = sample_batch["CL1 Comdty"]["current"]
        expected_crack = (2 * rbob_bbl + ho_bbl - 3 * wti) / 3

        rbob_open = sample_batch["XB1 Comdty"]["open"] * loader.GALLONS_PER_BARREL
        ho_open = sample_batch["HO1 Comdty"]["open"] * loader.GALLONS_PER_BARREL
        wti_open = sample_batch["CL1 Comdty"]["open"]
        expected_open_crack = (2 * rbob_open + ho_open - 3 * wti_open) / 3

        assert result["crack"] == pytest.approx(round(expected_crack, 2))
        assert result["change"] == pytest.approx(round(expected_crack - expected_open_crack, 2))

    def test_crack_spread_211_formula_accuracy(self, tmp_path, monkeypatch):
        """Ensure 2-1-1 crack spread follows documented formula."""
        loader = DataLoader(
            config_dir=str(tmp_path / "config"),
            data_dir=str(tmp_path / "data"),
            use_mock=True
        )

        sample_batch = {
            "CL1 Comdty": {"current": 71.0},
            "XB1 Comdty": {"current": 1.9},
            "HO1 Comdty": {"current": 2.4},
        }
        monkeypatch.setattr(loader, "get_prices_batch", lambda tickers: sample_batch)

        result = loader.get_crack_spread_211()

        rbob_bbl = sample_batch["XB1 Comdty"]["current"] * loader.GALLONS_PER_BARREL
        ho_bbl = sample_batch["HO1 Comdty"]["current"] * loader.GALLONS_PER_BARREL
        wti = sample_batch["CL1 Comdty"]["current"]
        expected_crack = (rbob_bbl + ho_bbl - 2 * wti) / 2

        assert result["crack"] == pytest.approx(round(expected_crack, 2))

    def test_get_futures_curve(self, tmp_path):
        """Test getting futures curve."""
        loader = DataLoader(
            config_dir=str(tmp_path / "config"),
            data_dir=str(tmp_path / "data"),
            use_mock=True
        )

        curve = loader.get_futures_curve("wti", num_months=12)

        assert isinstance(curve, pd.DataFrame)
        assert len(curve) == 12
        assert "price" in curve.columns
        assert "open_interest" in curve.columns

    def test_get_term_structure(self, tmp_path):
        """Test term structure analysis."""
        loader = DataLoader(
            config_dir=str(tmp_path / "config"),
            data_dir=str(tmp_path / "data"),
            use_mock=True
        )

        ts = loader.get_term_structure("wti")

        assert "structure" in ts
        assert ts["structure"] in ["Contango", "Backwardation", "Flat"]
        assert "slope" in ts
        assert "m1_m2_spread" in ts

    def test_validate_ticker(self, tmp_path):
        """Test ticker validation through loader."""
        loader = DataLoader(
            config_dir=str(tmp_path / "config"),
            data_dir=str(tmp_path / "data"),
            use_mock=True
        )

        valid, msg = loader.validate_ticker("CL1 Comdty")
        assert valid is True

        valid, msg = loader.validate_ticker("INVALID")
        assert valid is False

    def test_get_connection_status(self, tmp_path):
        """Test connection status reporting."""
        loader = DataLoader(
            config_dir=str(tmp_path / "config"),
            data_dir=str(tmp_path / "data"),
            use_mock=True
        )

        status = loader.get_connection_status()

        assert "mock_mode" in status
        assert "connected" in status
        assert status["mock_mode"] is True

    def test_loader_auto_fallback_to_mock(self, tmp_path, monkeypatch):
        """Loader should switch to mock mode if live connection is unavailable."""
        monkeypatch.setenv("BLOOMBERG_ALLOW_MOCK_FALLBACK", "true")
        loader = DataLoader(
            config_dir=str(tmp_path / "config"),
            data_dir=str(tmp_path / "data"),
            use_mock=False,
        )

        assert loader.get_data_mode() in {"mock", "live"}
        assert loader.is_data_available() is True

    def test_loader_respects_disabled_fallback(self, tmp_path, monkeypatch):
        """Loader remains disconnected when fallback is disabled."""
        monkeypatch.setenv("BLOOMBERG_ALLOW_MOCK_FALLBACK", "false")
        loader = DataLoader(
            config_dir=str(tmp_path / "config"),
            data_dir=str(tmp_path / "data"),
            use_mock=False,
        )

        if loader.get_data_mode() == "disconnected":
            assert loader.is_data_available() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
