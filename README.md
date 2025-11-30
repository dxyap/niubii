# 🛢️ Quantitative Oil Trading Dashboard

A lightweight, local-first quantitative trading dashboard for oil markets. Built with Python and Streamlit, featuring real-time market analysis, signal generation, risk management, and trade tracking.

## Features

### Market Insights
- Real-time price monitoring for WTI, Brent, RBOB, and Heating Oil
- Futures curve analysis with term structure visualization
- Crack spread monitoring (3-2-1, 2-1-1, component cracks)
- EIA inventory analytics with surprise calculations
- OPEC production monitoring and compliance tracking

### Signal Generation
- **Technical signals**: MA crossovers, RSI, Bollinger Bands, momentum
- **Fundamental signals**: Inventory surprises, OPEC compliance, term structure
- Signal aggregation with confidence scoring

### Risk Management
- Portfolio VaR (parametric, historical, Monte Carlo)
- Position and exposure limits
- Concentration monitoring
- Stress testing with historical scenarios

### Trading
- Manual trade entry with pre-trade risk checks
- Position monitor with live P&L
- Trade blotter with history and statistics

## Quick Start

### Prerequisites
- Python 3.10+
- Bloomberg Terminal (optional - realistic mock data available)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment (copy and edit)
cp .env.example .env

# Run the dashboard
streamlit run app/main.py
```

Open in browser at `http://localhost:8501`

## Project Structure

```
├── app/                      # Streamlit application
│   ├── main.py              # Main dashboard entry
│   ├── pages/               # Dashboard pages
│   │   ├── 1_📈_Market_Insights.py
│   │   ├── 2_📡_Signals.py
│   │   ├── 3_🛡️_Risk.py
│   │   ├── 4_💼_Trade_Entry.py
│   │   ├── 5_📋_Blotter.py
│   │   └── 6_📊_Analytics.py
│   ├── components/          # Reusable UI components
│   └── shared_state.py      # Session state management
│
├── core/                    # Core business logic
│   ├── data/               # Data loading & caching
│   │   ├── bloomberg.py    # Bloomberg API wrapper with TickerMapper
│   │   ├── cache.py        # Multi-layer caching
│   │   └── loader.py       # Unified data interface
│   ├── analytics/          # Market analytics
│   │   ├── curves.py       # Term structure analysis
│   │   ├── spreads.py      # Spread calculations
│   │   └── fundamentals.py # Fundamental analysis
│   ├── signals/            # Signal generation
│   │   ├── technical.py    # Technical signals
│   │   ├── fundamental.py  # Fundamental signals
│   │   └── aggregator.py   # Signal combination
│   ├── risk/               # Risk management
│   │   ├── var.py          # VaR calculations
│   │   ├── limits.py       # Position limits
│   │   └── monitor.py      # Risk monitoring
│   └── trading/            # Trading operations
│       ├── blotter.py      # Trade recording
│       ├── positions.py    # Position management
│       └── pnl.py          # P&L calculations
│
├── config/                  # Configuration files
│   ├── instruments.yaml    # Instrument definitions
│   ├── risk_limits.yaml    # Risk parameters
│   └── bloomberg_tickers.yaml  # Bloomberg ticker mappings
│
└── tests/                   # Test suite (64 tests)
```

## Bloomberg Integration

### Live Mode (Default)
The dashboard is configured to use live Bloomberg data by default. When a Bloomberg Terminal is not available, it automatically falls back to the sophisticated price simulator.

**Default behavior:**
- Attempts to connect to Bloomberg Terminal via BLPAPI
- Falls back to simulation if Bloomberg is unavailable
- Real-time subscription service for streaming updates

```python
# Default: Uses environment configuration
from core.data import DataLoader
loader = DataLoader()  # Reads BLOOMBERG_USE_MOCK from .env (default: false for live)

# Explicitly use live data
loader = DataLoader(use_mock=False)  # Connect to Bloomberg

# Explicitly use simulation
loader = DataLoader(use_mock=True)   # Use price simulator
```

### Mock Mode (Fallback/Development)
The dashboard includes a sophisticated price simulator that generates realistic market data:
- Tick-by-tick price updates with GARCH-like volatility clustering
- Proper term structure (contango/backwardation)
- Realistic bid/ask spreads
- Session-consistent prices with change tracking

### Real Bloomberg Connection
To connect to a real Bloomberg Terminal:

1. Install Bloomberg API (blpapi):
```bash
pip install blpapi
```

2. Configure environment:
```bash
# In .env file
BLOOMBERG_USE_MOCK=false
BLOOMBERG_HOST=localhost
BLOOMBERG_PORT=8194
```

3. Initialize with real API:
```python
from core.data import DataLoader
loader = DataLoader(use_mock=False)
```

### Ticker Mapping

The system includes a comprehensive `TickerMapper` utility for consistent ticker handling:

```python
from core.data import TickerMapper

# Get front month ticker
ticker = TickerMapper.get_front_month_ticker("wti")  # "CL1 Comdty"

# Get specific contract
ticker = TickerMapper.get_specific_month_ticker("CL", 1, 2025)  # "CLF5 Comdty"

# Validate ticker
valid, msg = TickerMapper.validate_ticker("CL1 Comdty")  # True, "Valid"

# Get contract multiplier
multiplier = TickerMapper.get_multiplier("CL1 Comdty")  # 1000 barrels
```

## Configuration

### Environment Variables (`.env`)

```bash
# Bloomberg Configuration
BLOOMBERG_USE_MOCK=true        # Use mock data (true/false)
BLOOMBERG_HOST=localhost       # Bloomberg API host
BLOOMBERG_PORT=8194            # Bloomberg API port

# Risk Limits
MAX_VAR_LIMIT=375000           # Maximum 1-day VaR
MAX_GROSS_EXPOSURE=20000000    # Maximum gross exposure

# Dashboard
AUTO_REFRESH_INTERVAL=5        # Auto-refresh interval (seconds)
```

### Risk Limits (`config/risk_limits.yaml`)

```yaml
portfolio_limits:
  max_var_95_1d: 375000      # $375K max 1-day VaR
  max_drawdown_daily: 0.05   # 5% daily drawdown limit
  max_gross_exposure: 20000000

position_limits:
  WTI_CL:
    max_contracts: 100
    max_notional: 8000000
```

### Instruments (`config/instruments.yaml`)

```yaml
futures:
  wti:
    name: "WTI Crude Oil"
    bloomberg_ticker: "CL1 Comdty"
    exchange: "NYMEX"
    contract_size: 1000
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=core --cov-report=html

# Run specific test module
pytest tests/test_data.py -v
```

## Status & Roadmap

| Component | Status |
|-----------|--------|
| Data Infrastructure | ✅ Complete |
| Market Analytics | ✅ Complete |
| Signal Engine | ✅ Complete |
| Risk Management | ✅ Complete |
| Trading Module | ✅ Complete |
| Dashboard UI | ✅ Complete |
| Test Suite | ✅ 64 tests |
| **Live Price Simulation** | ✅ Complete |
| **Auto-Refresh (5s)** | ✅ Complete |
| **Bloomberg Integration** | ✅ Complete |
| **Ticker Validation** | ✅ Complete |
| **Live Data Mode** | ✅ Complete |
| **Real-time Subscriptions** | ✅ Complete |
| ML Integration | 🔲 Planned |
| Backtesting | 🔲 Planned |

### Phase 3 Complete ✅
- **Live Bloomberg Data**: Dashboard defaults to live Bloomberg data (falls back to simulation if unavailable)
- **Real-time Subscription Service**: Subscribe to core oil market tickers for streaming updates
- **Environment-based Configuration**: Control data mode via `.env` file (`BLOOMBERG_USE_MOCK=false` for live data)
- **Real Bloomberg API Support**: Connect to Bloomberg Terminal when available
- **Ticker Mapper Utility**: Comprehensive ticker validation and mapping
- **Enhanced Mock Data**: GARCH-like volatility, proper term structure
- **Expanded Test Suite**: 64 tests covering all modules
- **Comprehensive Documentation**: Full ticker reference in YAML

### Planned Features
- ML signal models (XGBoost/LightGBM)
- Backtesting engine with vectorbt
- Multi-channel alerts (Email/SMS/Telegram)
- LLM news summarization

## API Reference

### DataLoader
```python
from core.data import DataLoader

loader = DataLoader(use_mock=True)

# Prices
loader.get_price("CL1 Comdty")
loader.get_oil_prices()
loader.get_price_with_change("CL1 Comdty")

# Curves
loader.get_futures_curve("wti", num_months=12)
loader.get_term_structure("wti")

# Spreads
loader.get_wti_brent_spread()
loader.get_crack_spread_321()

# Historical
loader.get_historical("CL1 Comdty", start_date, end_date)
```

### TickerMapper
```python
from core.data import TickerMapper

# Ticker generation
TickerMapper.get_front_month_ticker("wti")  # "CL1 Comdty"
TickerMapper.get_nth_month_ticker("wti", 3)  # "CL3 Comdty"
TickerMapper.get_specific_month_ticker("CL", 1, 2025)  # "CLF5 Comdty"

# Validation
valid, msg = TickerMapper.validate_ticker("CL1 Comdty")

# Parsing
info = TickerMapper.parse_ticker("CL1 Comdty")
# {'ticker': 'CL1 Comdty', 'commodity': 'CL', 'type': 'generic', ...}

# Contract info
TickerMapper.get_multiplier("CL1 Comdty")  # 1000
TickerMapper.get_field("last")  # "PX_LAST"
```

## Design Philosophy

**Lightweight & Local-First:**
- Everything runs on a single machine
- SQLite for transactions, Parquet for analytics
- In-memory caching for real-time data
- Scale to cloud only when needed

**Bloomberg Fallback:**
- Seamless mock mode when Bloomberg unavailable
- Realistic price simulation for development
- Same API interface for mock and real data

## Disclaimer

This software is for informational and educational purposes only. It does not constitute investment advice. Trading commodities involves substantial risk of loss.

## License

MIT License
