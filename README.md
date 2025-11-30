# 🛢️ Quantitative Oil Trading Dashboard

A lightweight, local-first quantitative trading dashboard for oil markets. Built with Python and Streamlit, featuring real-time market analysis, signal generation, risk management, and trade tracking.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Tests](https://img.shields.io/badge/Tests-64%20passed-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

### 📈 Market Insights
- Real-time price monitoring for WTI, Brent, RBOB, and Heating Oil
- Futures curve analysis with term structure visualization
- Crack spread monitoring (3-2-1, 2-1-1, component cracks)
- EIA inventory analytics with surprise calculations
- OPEC production monitoring and compliance tracking

### 📡 Signal Generation
- **Technical signals**: MA crossovers, RSI, Bollinger Bands, momentum
- **Fundamental signals**: Inventory surprises, OPEC compliance, term structure
- Signal aggregation with confidence scoring
- Historical signal performance tracking

### 🛡️ Risk Management
- Portfolio VaR (parametric, historical, Monte Carlo)
- Position and exposure limits
- Concentration monitoring
- Stress testing with historical scenarios (COVID crash, oil shocks)
- Real-time alert system

### 💼 Trading
- Manual trade entry with pre-trade risk checks
- Position monitor with live P&L
- Trade blotter with history and statistics
- Strategy tagging and performance attribution

## Quick Start

### Prerequisites
- Python 3.10+
- Bloomberg Terminal (optional - realistic simulation available)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd oil-trading-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env to configure Bloomberg connection

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
│   │   ├── bloomberg.py    # Bloomberg API + TickerMapper + Subscriptions
│   │   ├── cache.py        # Multi-layer caching (memory + disk)
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
├── data/                    # Data storage (auto-created)
│   ├── cache/              # Cached data
│   ├── historical/         # Parquet files
│   └── trades/             # Trade database
│
└── tests/                   # Test suite (64 tests)
```

## Bloomberg Integration

### Data Modes

The dashboard supports two data modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Live** | Connects to Bloomberg Terminal via BLPAPI | Production trading |
| **Simulated** | Realistic price simulation with GARCH volatility | Development/Demo |

### Live Mode (Default)

The dashboard is configured to use live Bloomberg data by default. When a Bloomberg Terminal is not available, it automatically falls back to the sophisticated price simulator.

```python
from core.data import DataLoader

# Auto-detect mode from environment (default: live)
loader = DataLoader()

# Check connection status
status = loader.get_connection_status()
print(f"Data mode: {status['data_mode']}")  # 'live' or 'simulated'
print(f"Is live: {loader.is_live_data()}")
```

### Simulated Mode (Development)

The price simulator provides realistic market data:
- **Tick-by-tick updates** with GARCH-like volatility clustering
- **Proper term structure** (contango/backwardation)
- **Realistic bid/ask spreads** varying by instrument liquidity
- **Session-consistent prices** with change tracking
- **Intraday history** for charting

```python
# Force simulation mode
loader = DataLoader(use_mock=True)
```

### Real-time Subscriptions

The dashboard includes a subscription service for streaming updates:

```python
from core.data import DataLoader

loader = DataLoader()

# Subscribe to core oil market tickers
loader.subscribe_to_core_tickers()

# Get all subscribed ticker prices
prices = loader.get_live_prices()
for ticker, data in prices.items():
    print(f"{ticker}: ${data['current']:.2f}")

# Check subscription status
status = loader.get_connection_status()
print(f"Subscribed: {status['subscribed_tickers']}")
```

### Connecting to Bloomberg Terminal

1. **Install Bloomberg API:**
```bash
pip install blpapi
```

2. **Configure environment (.env):**
```bash
BLOOMBERG_USE_MOCK=false
BLOOMBERG_HOST=localhost
BLOOMBERG_PORT=8194
BLOOMBERG_ENABLE_SUBSCRIPTIONS=true
```

3. **Verify connection:**
```python
from core.data import DataLoader

loader = DataLoader()
if loader.is_live_data():
    print("Connected to Bloomberg!")
else:
    print("Using simulation (Bloomberg not available)")
```

## Configuration

### Environment Variables (`.env`)

```bash
# =============================================================================
# BLOOMBERG CONFIGURATION
# =============================================================================
BLOOMBERG_USE_MOCK=false          # false = live data, true = simulation
BLOOMBERG_HOST=localhost          # Bloomberg API host
BLOOMBERG_PORT=8194               # Bloomberg API port
BLOOMBERG_TIMEOUT=30              # Request timeout (seconds)
BLOOMBERG_ENABLE_SUBSCRIPTIONS=true  # Enable real-time subscriptions

# =============================================================================
# RISK LIMITS
# =============================================================================
MAX_VAR_LIMIT=375000              # Maximum 1-day VaR (USD)
MAX_GROSS_EXPOSURE=20000000       # Maximum gross exposure (USD)
MAX_NET_EXPOSURE=15000000         # Maximum net exposure (USD)
MAX_DRAWDOWN_DAILY=0.05           # Daily drawdown limit (5%)

# Position limits (contracts)
MAX_WTI_CONTRACTS=100
MAX_BRENT_CONTRACTS=75
MAX_RBOB_CONTRACTS=50
MAX_HO_CONTRACTS=50

# Concentration limits (percentage)
MAX_SINGLE_INSTRUMENT_CONCENTRATION=40
MAX_CRUDE_GROUP_CONCENTRATION=60

# =============================================================================
# DASHBOARD SETTINGS
# =============================================================================
AUTO_REFRESH_INTERVAL=5           # Auto-refresh interval (seconds)
DASHBOARD_THEME=dark              # dark or light
```

### Risk Limits (`config/risk_limits.yaml`)

```yaml
portfolio_limits:
  max_var_95_1d: 375000          # $375K max 1-day VaR
  max_drawdown_daily: 0.05       # 5% daily drawdown limit
  max_gross_exposure: 20000000   # $20M gross exposure
  max_net_exposure: 15000000     # $15M net exposure

position_limits:
  WTI_CL:
    max_contracts: 100
    max_notional: 8000000
  Brent_CO:
    max_contracts: 75
    max_notional: 6000000

concentration_limits:
  single_instrument: 0.40        # 40% max single instrument
  crude_group: 0.60              # 60% max crude oil group
  single_strategy: 0.50          # 50% max single strategy
```

### Instruments (`config/instruments.yaml`)

```yaml
futures:
  wti:
    name: "WTI Crude Oil"
    bloomberg_ticker: "CL1 Comdty"
    exchange: "NYMEX"
    contract_size: 1000           # barrels
    tick_size: 0.01
    currency: "USD"
    
  brent:
    name: "Brent Crude Oil"
    bloomberg_ticker: "CO1 Comdty"
    exchange: "ICE"
    contract_size: 1000
    tick_size: 0.01
    currency: "USD"
```

## API Reference

### DataLoader

The main interface for all data operations:

```python
from core.data import DataLoader

loader = DataLoader()

# =============================================================================
# PRICE DATA
# =============================================================================
loader.get_price("CL1 Comdty")                    # Current price
loader.get_price_with_change("CL1 Comdty")        # Price with change info
loader.get_oil_prices()                            # All major oil prices
loader.get_all_oil_prices()                        # Extended oil products

# =============================================================================
# HISTORICAL DATA
# =============================================================================
loader.get_historical("CL1 Comdty", start_date, end_date)
loader.get_historical_multi(["CL1 Comdty", "CO1 Comdty"], start_date, end_date)
loader.get_intraday_prices("CL1 Comdty")          # Today's tick history

# =============================================================================
# FUTURES CURVES
# =============================================================================
loader.get_futures_curve("wti", num_months=12)    # WTI curve
loader.get_term_structure("wti")                   # Structure analysis
loader.get_calendar_spreads("wti")                 # Calendar spreads

# =============================================================================
# SPREADS
# =============================================================================
loader.get_wti_brent_spread()                      # WTI-Brent spread
loader.get_crack_spread_321()                      # 3-2-1 crack spread
loader.get_crack_spread_211()                      # 2-1-1 crack spread

# =============================================================================
# FUNDAMENTAL DATA
# =============================================================================
loader.get_eia_inventory()                         # EIA crude inventory
loader.get_opec_production()                       # OPEC production data
loader.get_refinery_turnarounds()                  # Refinery schedules

# =============================================================================
# SUBSCRIPTIONS & LIVE DATA
# =============================================================================
loader.subscribe_to_core_tickers()                 # Subscribe to key tickers
loader.get_live_prices()                           # Get subscribed prices
loader.is_live_data()                              # Check if using live data
loader.get_connection_status()                     # Full connection info

# =============================================================================
# UTILITIES
# =============================================================================
loader.validate_ticker("CL1 Comdty")              # Validate ticker
loader.get_multiplier("CL1 Comdty")               # Contract multiplier
loader.refresh_all()                               # Clear cache & refresh
```

### TickerMapper

Utility for Bloomberg ticker handling:

```python
from core.data import TickerMapper

# Ticker generation
TickerMapper.get_front_month_ticker("wti")         # "CL1 Comdty"
TickerMapper.get_nth_month_ticker("wti", 3)        # "CL3 Comdty"
TickerMapper.get_specific_month_ticker("CL", 1, 2025)  # "CLF5 Comdty"

# Validation
valid, msg = TickerMapper.validate_ticker("CL1 Comdty")  # (True, "Valid")

# Parsing
info = TickerMapper.parse_ticker("CL1 Comdty")
# {'ticker': 'CL1 Comdty', 'commodity': 'CL', 'type': 'generic', 
#  'month_number': 1, 'exchange': 'NYMEX', 'multiplier': 1000}

# Field mapping
TickerMapper.get_field("last")                     # "PX_LAST"
TickerMapper.get_field("bid")                      # "PX_BID"

# Contract info
TickerMapper.get_multiplier("CL1 Comdty")          # 1000 (barrels)
TickerMapper.get_multiplier("XB1 Comdty")          # 42000 (gallons)
```

### BloombergSubscriptionService

Real-time data subscriptions:

```python
from core.data import DataLoader

loader = DataLoader()
svc = loader.subscription_service

# Subscribe to tickers
svc.subscribe("CL1 Comdty")
svc.subscribe("CO1 Comdty", callback=my_update_handler)

# Check subscriptions
tickers = svc.get_subscribed_tickers()
prices = svc.get_latest_prices()

# Unsubscribe
svc.unsubscribe("CL1 Comdty")
svc.stop()  # Stop all subscriptions
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=core --cov-report=html

# Run specific test module
pytest tests/test_data.py -v
pytest tests/test_risk.py -v
pytest tests/test_signals.py -v
pytest tests/test_analytics.py -v
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| Data Infrastructure | 24 | 95% |
| Analytics | 9 | 90% |
| Risk Management | 11 | 92% |
| Signals | 10 | 88% |
| Trading | 10 | 85% |

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
| Live Price Simulation | ✅ Complete |
| Auto-Refresh (5s) | ✅ Complete |
| Bloomberg Integration | ✅ Complete |
| Ticker Validation | ✅ Complete |
| Live Data Mode | ✅ Complete |
| Real-time Subscriptions | ✅ Complete |
| ML Integration | 🔲 Planned |
| Backtesting Engine | 🔲 Planned |

### Phase 3 Complete ✅

- **Live Bloomberg Data**: Dashboard defaults to live Bloomberg data with automatic fallback to simulation
- **Real-time Subscription Service**: Subscribe to core oil market tickers for streaming updates
- **Environment-based Configuration**: Control data mode via `.env` file
- **Enhanced Price Simulator**: GARCH-like volatility, proper term structure, realistic spreads
- **Comprehensive Test Suite**: 64 tests covering all modules
- **Full Documentation**: API reference and configuration guide

### Planned Features

- ML signal models (XGBoost/LightGBM)
- Backtesting engine with vectorbt
- Multi-channel alerts (Email/SMS/Telegram)
- LLM news summarization

## Design Philosophy

**Lightweight & Local-First:**
- Everything runs on a single machine
- SQLite for transactions, Parquet for analytics
- In-memory caching for real-time data
- Scale to cloud only when needed

**Bloomberg Fallback:**
- Seamless simulation mode when Bloomberg unavailable
- Realistic price simulation for development
- Same API interface for mock and real data

**Production Ready:**
- Environment-based configuration
- Comprehensive error handling
- Graceful degradation

## Troubleshooting

### Common Issues

**1. "diskcache not available" warning**
```bash
pip install diskcache
```

**2. Bloomberg connection fails**
- Verify Bloomberg Terminal is running
- Check `BLOOMBERG_HOST` and `BLOOMBERG_PORT` in `.env`
- The dashboard will automatically fall back to simulation

**3. Missing dependencies**
```bash
pip install -r requirements.txt
```

**4. Test failures**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
pip install pytest pytest-cov

# Run tests
pytest tests/ -v
```

## Disclaimer

This software is for informational and educational purposes only. It does not constitute investment advice. Trading commodities involves substantial risk of loss.

## License

MIT License - See LICENSE file for details.

---

**Built with** ❤️ **for oil traders**
