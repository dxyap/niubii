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

from core.data.bloomberg import BloombergClient, TickerMapper
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
    """
    Tests for Bloomberg API client.
    
    Note: These tests require a live Bloomberg connection or use unittest.mock 
    to simulate responses. Tests are skipped if no Bloomberg connection is available.
    """

    @pytest.fixture
    def client(self):
        """Create a Bloomberg client - skip if not connected."""
        client = BloombergClient()
        if not client.connected:
            pytest.skip("Bloomberg Terminal not connected")
        return client

    def test_client_initialization(self):
        """Test client initialization."""
        client = BloombergClient()
        # Client should either be connected or have a connection error
        assert hasattr(client, 'connected')
        assert hasattr(client, '_connection_error')

    def test_get_price(self, client):
        """Test getting single price."""
        price = client.get_price("CL1 Comdty")

        assert isinstance(price, float)
        assert price > 0

    def test_get_price_different_fields(self, client):
        """Test getting different price fields."""
        last = client.get_price("CL1 Comdty", "PX_LAST")
        
        assert last > 0

    def test_get_price_with_change(self, client):
        """Test getting price with change."""
        data = client.get_price_with_change("CL1 Comdty")

        assert "current" in data
        assert "open" in data
        assert "change" in data
        assert "change_pct" in data
        assert "high" in data
        assert "low" in data

    def test_get_prices_multiple(self, client):
        """Test getting multiple prices."""
        tickers = ["CL1 Comdty", "CO1 Comdty", "XB1 Comdty"]

        prices = client.get_prices(tickers)

        assert isinstance(prices, pd.DataFrame)
        assert len(prices) == 3
        assert "PX_LAST" in prices.columns

    def test_get_historical(self, client):
        """Test getting historical data."""
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

    def test_get_curve(self, client):
        """Test getting futures curve."""
        curve = client.get_curve("wti", num_months=12)

        assert isinstance(curve, pd.DataFrame)
        assert len(curve) == 12
        assert "price" in curve.columns
        assert "ticker" in curve.columns
        assert "month" in curve.columns

    def test_get_intraday_prices(self, client):
        """Test getting intraday prices."""
        intraday = client.get_intraday_prices("CL1 Comdty")

        assert isinstance(intraday, pd.DataFrame)
        assert "timestamp" in intraday.columns or len(intraday) == 0


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
    """
    Tests for data loader.
    
    Note: Tests that require live data will skip if Bloomberg is not connected.
    Tests that test formulas use monkeypatch to provide sample data.
    """

    @pytest.fixture
    def loader(self, tmp_path):
        """Create a data loader."""
        return DataLoader(
            config_dir=str(tmp_path / "config"),
            data_dir=str(tmp_path / "data"),
        )

    @pytest.fixture
    def connected_loader(self, loader):
        """Get loader only if Bloomberg is connected."""
        if not loader.is_data_available():
            pytest.skip("Bloomberg Terminal not connected")
        return loader

    def test_loader_initialization(self, loader):
        """Test data loader initializes correctly."""
        assert hasattr(loader, 'bloomberg')
        assert hasattr(loader, '_data_mode')

    def test_get_oil_prices(self, connected_loader):
        """Test getting oil prices."""
        prices = connected_loader.get_oil_prices()

        assert "WTI" in prices
        assert "Brent" in prices
        assert "RBOB" in prices
        assert "Heating Oil" in prices

        # Prices are dictionaries with current, change, etc.
        assert prices["WTI"]["current"] > 0
        assert "change" in prices["WTI"]
        assert "change_pct" in prices["WTI"]

    def test_get_market_summary(self, connected_loader):
        """Test getting market summary."""
        summary = connected_loader.get_market_summary()

        assert "prices" in summary
        assert "spreads" in summary
        assert "curve" in summary
        assert "timestamp" in summary

    def test_get_wti_brent_spread(self, connected_loader):
        """Test WTI-Brent spread calculation."""
        spread = connected_loader.get_wti_brent_spread()

        assert "spread" in spread
        assert "change" in spread
        assert "wti" in spread
        assert "brent" in spread

        # WTI-Brent spread should approximately equal wti - brent
        calculated = spread["wti"] - spread["brent"]
        assert abs(spread["spread"] - calculated) < 0.01  # Allow for rounding

    def test_get_crack_spread_321(self, connected_loader):
        """Test 3-2-1 crack spread calculation."""
        crack = connected_loader.get_crack_spread_321()

        assert "crack" in crack
        assert "change" in crack
        assert "wti" in crack
        assert "rbob_bbl" in crack
        assert "ho_bbl" in crack

    def test_crack_spread_321_formula_accuracy(self, loader, monkeypatch):
        """Ensure crack spread calculation performs accurate unit conversion."""
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

    def test_crack_spread_211_formula_accuracy(self, loader, monkeypatch):
        """Ensure 2-1-1 crack spread follows documented formula."""
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

    def test_get_futures_curve(self, connected_loader):
        """Test getting futures curve."""
        curve = connected_loader.get_futures_curve("wti", num_months=12)

        assert isinstance(curve, pd.DataFrame)
        assert len(curve) == 12
        assert "price" in curve.columns

    def test_get_term_structure(self, connected_loader):
        """Test term structure analysis."""
        ts = connected_loader.get_term_structure("wti")

        assert "structure" in ts
        assert ts["structure"] in ["Contango", "Backwardation", "Flat"]
        assert "slope" in ts
        assert "m1_m2_spread" in ts

    def test_validate_ticker(self, loader):
        """Test ticker validation through loader."""
        valid, msg = loader.validate_ticker("CL1 Comdty")
        assert valid is True

        valid, msg = loader.validate_ticker("INVALID")
        assert valid is False

    def test_get_connection_status(self, tmp_path):
        """Test connection status reporting."""
        loader = DataLoader(
            config_dir=str(tmp_path / "config"),
            data_dir=str(tmp_path / "data"),
        )

        status = loader.get_connection_status()

        assert "connected" in status
        assert "data_mode" in status

    def test_loader_connection_status(self, tmp_path):
        """Loader should report correct connection status."""
        loader = DataLoader(
            config_dir=str(tmp_path / "config"),
            data_dir=str(tmp_path / "data"),
        )

        # Should be either live (if Bloomberg connected) or disconnected
        assert loader.get_data_mode() in {"live", "disconnected"}
        
        if loader.get_data_mode() == "live":
            assert loader.is_data_available() is True
        else:
            assert loader.is_data_available() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
