"""
Tests for Data Module
=====================
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data.bloomberg import BloombergClient, MockBloombergData
from core.data.cache import DataCache, ParquetStorage
from core.data.loader import DataLoader


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
    
    def test_get_curve(self):
        """Test getting futures curve."""
        client = BloombergClient(use_mock=True)
        
        curve = client.get_curve("wti", num_months=12)
        
        assert isinstance(curve, pd.DataFrame)
        assert len(curve) == 12
        assert "price" in curve.columns


class TestMockBloombergData:
    """Tests for mock data generation."""
    
    def test_generate_eia_inventory(self):
        """Test EIA inventory data generation."""
        data = MockBloombergData.generate_eia_inventory_data(periods=52)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 52
        assert "inventory_mmb" in data.columns
        assert "change_mmb" in data.columns
    
    def test_generate_opec_production(self):
        """Test OPEC production data generation."""
        data = MockBloombergData.generate_opec_production_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert "country" in data.columns
        assert "quota_mbpd" in data.columns
    
    def test_generate_turnaround_data(self):
        """Test turnaround data generation."""
        data = MockBloombergData.generate_turnaround_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert "region" in data.columns
        assert "capacity_kbpd" in data.columns


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
        assert prices["WTI"] > 0
    
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
