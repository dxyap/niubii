# 🛢️ Quantitative Oil Trading Dashboard

A lightweight, local-first quantitative trading dashboard for oil markets. Built with Python and Streamlit, featuring real-time market analysis, signal generation, risk management, and trade tracking.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Tests](https://img.shields.io/badge/Tests-90%20passed-green.svg)
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
- **ML signals**: XGBoost/LightGBM models with 60+ engineered features
- Signal aggregation with confidence scoring and configurable weights
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
│   │   ├── 6_📊_Analytics.py
│   │   └── 7_🤖_ML_Signals.py    # NEW: ML-powered signals
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
│   │   └── aggregator.py   # Signal combination + MLSignalGenerator
│   ├── risk/               # Risk management
│   │   ├── var.py          # VaR calculations
│   │   ├── limits.py       # Position limits
│   │   └── monitor.py      # Risk monitoring
│   ├── trading/            # Trading operations
│   │   ├── blotter.py      # Trade recording
│   │   ├── positions.py    # Position management
│   │   └── pnl.py          # P&L calculations
│   └── ml/                  # NEW: Machine Learning (Phase 4)
│       ├── features.py      # Feature engineering pipeline
│       ├── models/
│       │   ├── gradient_boost.py  # XGBoost/LightGBM
│       │   └── ensemble.py        # Model ensembling
│       ├── training.py      # Training pipeline
│       ├── prediction.py    # Inference service
│       └── monitoring.py    # Model monitoring & drift detection
│
├── config/                  # Configuration files
│   ├── instruments.yaml    # Instrument definitions
│   ├── risk_limits.yaml    # Risk parameters
│   └── bloomberg_tickers.yaml  # Bloomberg ticker mappings
│
├── models/                  # Trained ML models (auto-created)
│
├── data/                    # Data storage (auto-created)
│   ├── cache/              # Cached data
│   ├── historical/         # Parquet files
│   └── trades/             # Trade database
│
└── tests/                   # Test suite (90 tests)
```

## Bloomberg Integration

### Data Requirements

**This dashboard requires a Bloomberg Terminal connection for live data.** Without Bloomberg, the dashboard will display "Disconnected" status and show "Data Unavailable" for all market data.

| Mode | Description | Use Case |
|------|-------------|----------|
| **Live** | Connects to Bloomberg Terminal via BLPAPI | Production trading |
| **Disconnected** | No data source available | Shows error messages |
| **Mock** | Simulated prices (development only) | Development/Testing |

### Live Mode (Default)

The dashboard defaults to live Bloomberg data. If Bloomberg is not connected, you will see:
- Red "Disconnected" indicator in the sidebar
- Error message explaining the connection failure
- "N/A" or "Data Unavailable" for all price data

```python
from core.data import DataLoader

# Default: requires Bloomberg Terminal
loader = DataLoader()

# Check connection status
status = loader.get_connection_status()
print(f"Data mode: {status['data_mode']}")  # 'live', 'mock', or 'disconnected'
print(f"Connected: {status['connected']}")
print(f"Error: {status['connection_error']}")
```

### Mock Mode (Development Only)

⚠️ **Warning**: Mock mode displays **simulated data, NOT real market data**. Only use for development/testing.

```python
# Force mock mode for development
loader = DataLoader(use_mock=True)

# Or set environment variable
# BLOOMBERG_USE_MOCK=true
```

Mock mode features (for development testing only):
- Simulated tick-by-tick updates
- Simulated term structure
- Simulated bid/ask spreads

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

## Performance Optimizations

The dashboard is optimized for fast data loading and responsive UI with the following techniques:

### Batch API Calls

All price fetches are batched to minimize API round-trips:

```python
# Before: 12 sequential API calls for futures curve
for i in range(1, 13):
    get_price(f"CL{i} Comdty")  # Slow!

# After: Single batch call
get_prices(["CL1 Comdty", "CL2 Comdty", ..., "CL12 Comdty"])  # Fast!
```

**Impact**: Futures curve loads ~10x faster (1 call vs 12 calls)

### Streamlit Caching

Expensive operations use Streamlit's caching with appropriate TTLs:

| Data Type | Cache TTL | Reason |
|-----------|-----------|--------|
| Historical data | 5 minutes | Doesn't change frequently |
| Futures curve | 1 minute | Updates throughout trading day |
| Real-time prices | No cache | Always fresh |

```python
@st.cache_data(ttl=300)  # 5 minutes
def get_historical_data_cached(lookback_days: int = 90):
    return data_loader.get_historical("CL1 Comdty", ...)
```

### Thread-Safe TTL Cache

Real-time price data uses an efficient in-memory TTL cache:

```python
from core.data import TTLCache

cache = TTLCache(max_size=1000, default_ttl=5.0)
cache.set("CL1 Comdty", 72.50)
price = cache.get("CL1 Comdty")  # Fast lookup, auto-expires
```

### Non-Blocking Auto-Refresh

Auto-refresh uses `streamlit-autorefresh` instead of `time.sleep()`:

```python
# Before: Blocking sleep (freezes UI)
time.sleep(5)
st.rerun()

# After: Non-blocking with streamlit-autorefresh
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=5000)  # Milliseconds
```

### Lazy Loading

Expensive data is loaded only when needed:

```python
class DashboardData:
    @property
    def wti_history(self):
        # Only fetches when first accessed
        if self._wti_history is self._NOT_LOADED:
            self._wti_history = self.data_loader.get_historical(...)
        return self._wti_history
```

### Configuration

Adjust performance settings in `.env`:

```bash
# Auto-refresh interval (seconds) - increase to reduce load
AUTO_REFRESH_INTERVAL=10

# Enable/disable real-time subscriptions
BLOOMBERG_ENABLE_SUBSCRIPTIONS=true
```

## Configuration

### Environment Variables (`.env`)

```bash
# =============================================================================
# BLOOMBERG CONFIGURATION
# =============================================================================
# IMPORTANT: Dashboard requires Bloomberg Terminal by default
BLOOMBERG_USE_MOCK=false          # false = live data (default), true = mock (dev only)
BLOOMBERG_HOST=localhost          # Bloomberg API host
BLOOMBERG_PORT=8194               # Bloomberg API port
BLOOMBERG_TIMEOUT=30              # Request timeout (seconds)
BLOOMBERG_ENABLE_SUBSCRIPTIONS=true  # Enable real-time subscriptions

# If Bloomberg is unavailable, set BLOOMBERG_USE_MOCK=true for development
# WARNING: Mock mode displays simulated data, NOT real market prices

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

### ML Module

Machine learning for trading signals:

```python
from core.ml import FeatureEngineer, FeatureConfig
from core.ml import ModelTrainer, TrainingConfig
from core.ml import PredictionService
from core.ml.models import GradientBoostModel, EnsembleModel

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
config = FeatureConfig(target_horizon=5)  # 5-day prediction
engineer = FeatureEngineer(config)

# Create 60+ features from OHLCV data
features = engineer.create_features(historical_df)
print(f"Created {len(engineer.feature_names)} features")

# =============================================================================
# MODEL TRAINING
# =============================================================================
trainer = ModelTrainer(TrainingConfig(use_ensemble=True))

# Train with walk-forward validation
results = trainer.walk_forward_train(historical_df)
print(f"Test Accuracy: {results['avg_metrics']['accuracy']:.2%}")

# Save model
trainer.save_model("models/my_model.pkl")

# =============================================================================
# PREDICTIONS
# =============================================================================
service = PredictionService("models/my_model.pkl")

# Generate ML signal
signal = service.predict(recent_data)
print(f"Signal: {signal['signal']} (Confidence: {signal['confidence']:.1%})")

# =============================================================================
# SIGNAL AGGREGATION (with ML)
# =============================================================================
from core.signals import SignalAggregator, MLSignalGenerator

aggregator = SignalAggregator()
ml_gen = MLSignalGenerator()

# Get ML signal
ml_signal = ml_gen.generate_signal(historical_data)

# Aggregate with technical and fundamental signals
composite = aggregator.aggregate_signals(
    technical_signal={"signal": "LONG", "confidence": 70},
    fundamental_signal={"signal": "LONG", "confidence": 60},
    ml_signal=ml_signal,
    current_price=77.50
)
print(f"Composite: {composite.direction} (Confidence: {composite.confidence}%)")
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
| ML (Feature Engineering) | 25 | 90% |
| Trading | 10 | 85% |

## Status & Roadmap

### Current Status

| Component | Status | Phase |
|-----------|--------|-------|
| Data Infrastructure | ✅ Complete | 1 |
| Market Analytics | ✅ Complete | 1 |
| Signal Engine | ✅ Complete | 2 |
| Risk Management | ✅ Complete | 2 |
| Trading Module | ✅ Complete | 2 |
| Dashboard UI | ✅ Complete | 2 |
| Test Suite | ✅ 64 tests | 2 |
| Live Price Simulation | ✅ Complete | 3 |
| Auto-Refresh (5s) | ✅ Complete | 3 |
| Bloomberg Integration | ✅ Complete | 3 |
| Ticker Validation | ✅ Complete | 3 |
| Live Data Mode | ✅ Complete | 3 |
| Real-time Subscriptions | ✅ Complete | 3 |
| Feature Engineering | ✅ Complete | 4 |
| ML Models (XGBoost/LightGBM) | ✅ Complete | 4 |
| Model Training Pipeline | ✅ Complete | 4 |
| ML Signal Integration | ✅ Complete | 4 |
| Model Monitoring | ✅ Complete | 4 |
| Backtesting Engine | 🔲 Planned | 5 |
| Execution & Automation | 🔲 Planned | 6 |
| Multi-channel Alerts | 🔲 Planned | 7 |
| Advanced Analytics & AI | 🔲 Planned | 8 |
| Production Hardening | 🔲 Planned | 9 |

---

## Development Phases

### ✅ Phase 1: Foundation (Complete)

**Data Infrastructure & Market Analytics**

- [x] Bloomberg API integration with `blpapi`
- [x] Multi-layer caching (memory + disk with `diskcache`)
- [x] Parquet storage for historical data
- [x] Unified `DataLoader` interface
- [x] Ticker mapping and validation (`TickerMapper`)
- [x] Futures curve analysis (contango/backwardation)
- [x] Spread calculations (WTI-Brent, crack spreads)
- [x] Fundamental data (EIA inventory, OPEC production)

### ✅ Phase 2: Core Trading Features (Complete)

**Signals, Risk, and Trading**

- [x] Technical signal generation (MA crossovers, RSI, Bollinger Bands)
- [x] Fundamental signal generation (inventory surprises, term structure)
- [x] Signal aggregation with weighted confidence scoring
- [x] VaR calculations (parametric, historical, Monte Carlo)
- [x] Expected Shortfall (CVaR)
- [x] Position and exposure limits
- [x] Stress testing with historical scenarios
- [x] Trade blotter with SQLite persistence
- [x] Position management and live P&L
- [x] Streamlit dashboard with 6 pages

### ✅ Phase 3: Live Data Integration (Complete)

**Bloomberg Live Mode & Enhanced Simulation**

- [x] Live Bloomberg data as default mode
- [x] Real-time subscription service for streaming updates
- [x] Environment-based configuration (`.env`)
- [x] Enhanced price simulator with GARCH-like volatility
- [x] Proper term structure simulation
- [x] Realistic bid/ask spreads
- [x] Comprehensive test suite (64 tests)
- [x] Full API documentation

---

## 🔮 Future Phases

### ✅ Phase 4: Machine Learning Integration (Complete)

**ML-Powered Signal Generation**

Machine learning models for enhanced signal quality and prediction accuracy.

| Feature | Description | Status |
|---------|-------------|--------|
| Feature Engineering | 60+ ML features from price, volume, and fundamental data | ✅ Complete |
| XGBoost/LightGBM Models | Gradient boosting for direction prediction | ✅ Complete |
| Ensemble Methods | Combine multiple models with weighted averaging | ✅ Complete |
| Model Training Pipeline | Walk-forward validation, hyperparameter config | ✅ Complete |
| Prediction Service | Real-time ML signal generation | ✅ Complete |
| Model Monitoring | Performance tracking and drift detection | ✅ Complete |
| ML Dashboard Page | Training UI and signal visualization | ✅ Complete |
| Signal Integration | ML signals in aggregator with configurable weights | ✅ Complete |

**Implementation:**
```
core/
├── ml/
│   ├── __init__.py           # Module exports
│   ├── features.py           # Feature engineering (60+ features)
│   ├── models/
│   │   ├── gradient_boost.py # XGBoost/LightGBM wrapper
│   │   └── ensemble.py       # Model ensembling
│   ├── training.py           # Training pipeline with walk-forward
│   ├── prediction.py         # Inference service
│   └── monitoring.py         # Performance & drift detection
app/pages/
├── 7_🤖_ML_Signals.py        # ML signals dashboard
```

**Features Include:**
- **Price Features**: Lagged prices, overnight gaps, range position
- **Return Features**: Multi-horizon returns with z-scores
- **Moving Averages**: 5/10/20/50/100/200 MA ratios and crossovers
- **Volatility**: Realized vol, Parkinson vol, ATR, vol ratios
- **Momentum**: RSI, MACD, Stochastic, Williams %R, ROC
- **Volume**: Volume MAs, OBV, volume-price trend
- **Open Interest**: OI change, MA ratio, price divergence
- **Bollinger Bands**: Position, width, distance from bands
- **Calendar**: Day of week, month, quarter-end effects

**ML Dependencies (Added to requirements.txt):**
```
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.1.0
```

---

### 🔲 Phase 5: Backtesting Engine

**Historical Strategy Testing & Optimization**

Build a robust backtesting framework for strategy development and validation.

| Feature | Description | Priority |
|---------|-------------|----------|
| Event-Driven Backtest | Tick-by-tick or bar-by-bar simulation engine | High |
| Strategy Framework | Define strategies as composable classes | High |
| Transaction Costs | Realistic slippage, commissions, and market impact | High |
| Walk-Forward Optimization | Rolling window parameter optimization | Medium |
| Performance Metrics | Sharpe, Sortino, Calmar, max drawdown, etc. | High |
| Monte Carlo Analysis | Bootstrap resampling for robustness testing | Medium |
| Strategy Comparison | Side-by-side strategy evaluation | Medium |

**Implementation Plan:**
```
core/
├── backtest/
│   ├── __init__.py
│   ├── engine.py            # Main backtesting engine
│   ├── strategy.py          # Strategy base class and examples
│   ├── execution.py         # Order execution simulation
│   ├── costs.py             # Transaction cost models
│   ├── metrics.py           # Performance metrics
│   ├── optimization.py      # Parameter optimization
│   └── reporting.py         # Backtest reports
app/pages/
├── 7_🔬_Backtest.py        # Backtest configuration UI
├── 8_📈_Strategy_Builder.py # Visual strategy builder
```

**New Dependencies:**
```
vectorbt>=0.26.0       # Vectorized backtesting
empyrical>=0.5.5       # Performance metrics
pyfolio>=0.9.2         # Portfolio analysis
```

---

### 🔲 Phase 6: Execution & Automation

**Order Management & Automated Trading**

Connect signals to execution with an order management system.

| Feature | Description | Priority |
|---------|-------------|----------|
| Order Management System | Track orders through lifecycle (new→filled→settled) | High |
| Paper Trading Mode | Simulate execution without real orders | High |
| Position Sizing | Kelly criterion, volatility targeting, risk parity | High |
| Execution Algorithms | TWAP, VWAP, implementation shortfall | Medium |
| Broker Integration | Connect to Interactive Brokers, CQG, or TT | Medium |
| Smart Order Routing | Optimal venue selection | Low |
| Auto-Execution Rules | Trigger orders based on signals + conditions | Medium |

**Implementation Plan:**
```
core/
├── execution/
│   ├── __init__.py
│   ├── oms.py               # Order management system
│   ├── paper_trading.py     # Paper trading engine
│   ├── sizing.py            # Position sizing algorithms
│   ├── algorithms.py        # Execution algorithms (TWAP, VWAP)
│   ├── brokers/
│   │   ├── base.py          # Broker interface
│   │   ├── ib.py            # Interactive Brokers
│   │   └── simulator.py     # Simulated broker
│   └── routing.py           # Order routing logic
app/pages/
├── 9_🤖_Automation.py      # Automation rules UI
```

**New Dependencies:**
```
ib_insync>=0.9.86      # Interactive Brokers API
```

---

### 🔲 Phase 7: Alerts & Notifications

**Multi-Channel Alert System**

Proactive notifications for trading signals, risk breaches, and market events.

| Feature | Description | Priority |
|---------|-------------|----------|
| Alert Rules Engine | Configurable conditions and triggers | High |
| Email Notifications | SMTP-based email alerts | High |
| Telegram Bot | Real-time Telegram notifications | High |
| Slack Integration | Slack channel alerts | Medium |
| SMS Alerts | Critical alerts via SMS (Twilio) | Medium |
| Scheduled Reports | Daily/weekly P&L and risk summaries | High |
| Alert History | Track and audit all alerts | Medium |
| Alert Escalation | Escalate unacknowledged critical alerts | Low |

**Implementation Plan:**
```
core/
├── alerts/
│   ├── __init__.py
│   ├── rules.py             # Alert rule definitions
│   ├── engine.py            # Alert evaluation engine
│   ├── channels/
│   │   ├── email.py         # Email notifications
│   │   ├── telegram.py      # Telegram bot
│   │   ├── slack.py         # Slack integration
│   │   └── sms.py           # SMS via Twilio
│   ├── scheduler.py         # Scheduled reports
│   └── history.py           # Alert audit log
config/
├── alerts.yaml              # Alert configurations
```

**New Dependencies:**
```
python-telegram-bot>=20.6
slack-sdk>=3.23.0
twilio>=8.10.0
jinja2>=3.1.2          # Report templates
```

---

### 🔲 Phase 8: Advanced Analytics & AI

**Research Tools & Alternative Data**

Advanced analytics, AI-powered research, and alternative data sources.

| Feature | Description | Priority |
|---------|-------------|----------|
| LLM News Analysis | Summarize and sentiment-score news with GPT/Claude | High |
| Alternative Data | Satellite imagery, shipping data, refinery schedules | Medium |
| Cross-Asset Correlations | Oil vs. equities, FX, rates correlations | Medium |
| Regime Detection | Hidden Markov Models for market regime identification | Medium |
| Scenario Analysis | What-if analysis for portfolio changes | High |
| Factor Analysis | Decompose returns into risk factors | Medium |
| Research Notebooks | Jupyter integration for ad-hoc analysis | Medium |

**Implementation Plan:**
```
core/
├── research/
│   ├── __init__.py
│   ├── llm/
│   │   ├── news_analyzer.py    # LLM news summarization
│   │   ├── sentiment.py        # Sentiment scoring
│   │   └── embeddings.py       # Document embeddings
│   ├── alt_data/
│   │   ├── satellite.py        # Satellite imagery analysis
│   │   ├── shipping.py         # Tanker tracking
│   │   └── positioning.py      # COT/positioning data
│   ├── correlations.py      # Cross-asset analysis
│   ├── regimes.py           # Regime detection
│   └── factors.py           # Factor models
app/pages/
├── 10_🔍_Research.py       # Research dashboard
├── 11_📰_News.py           # News & sentiment feed
notebooks/
├── research_template.ipynb
```

**New Dependencies:**
```
openai>=1.3.0          # GPT API
anthropic>=0.7.0       # Claude API
langchain>=0.0.340     # LLM orchestration
hmmlearn>=0.3.0        # Hidden Markov Models
statsmodels>=0.14.0    # Statistical models
```

---

### 🔲 Phase 9: Production Hardening

**Enterprise-Ready Deployment**

Prepare the system for production deployment with security, reliability, and compliance.

| Feature | Description | Priority |
|---------|-------------|----------|
| Authentication | User authentication (OAuth2, SSO) | High |
| Role-Based Access | Permission levels for traders, risk, admins | High |
| Audit Logging | Complete audit trail of all actions | High |
| Database Migration | Alembic migrations for schema changes | Medium |
| High Availability | Redis for state, PostgreSQL for persistence | Medium |
| Containerization | Docker + Docker Compose deployment | High |
| Kubernetes | K8s manifests for cloud deployment | Low |
| Monitoring | Prometheus metrics + Grafana dashboards | Medium |
| Disaster Recovery | Backup and restore procedures | Medium |
| Compliance Reports | MiFID II, Dodd-Frank reporting templates | Low |

**Implementation Plan:**
```
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-compose.prod.yml
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
├── migrations/
│   └── versions/
core/
├── auth/
│   ├── __init__.py
│   ├── authentication.py    # Auth providers
│   ├── authorization.py     # RBAC
│   └── audit.py             # Audit logging
├── monitoring/
│   ├── metrics.py           # Prometheus metrics
│   └── health.py            # Health checks
config/
├── logging.yaml             # Structured logging config
```

**New Dependencies:**
```
redis>=5.0.0
psycopg2-binary>=2.9.9
alembic>=1.12.0
python-jose>=3.3.0     # JWT handling
passlib>=1.7.4         # Password hashing
prometheus-client>=0.18.0
```

---

## Prioritized Roadmap

```
Q1 2025: Phase 4 - ML Integration
├── Feature engineering pipeline
├── XGBoost/LightGBM models for direction prediction
├── Model monitoring and drift detection
└── Integration with signal aggregator

Q2 2025: Phase 5 - Backtesting Engine
├── Event-driven backtest framework
├── Strategy definition DSL
├── Walk-forward optimization
└── Performance reporting

Q3 2025: Phase 6 & 7 - Execution & Alerts
├── Paper trading mode
├── Position sizing algorithms
├── Multi-channel alert system
├── Scheduled reporting

Q4 2025: Phase 8 & 9 - Advanced Analytics & Production
├── LLM news analysis
├── Cross-asset correlations
├── Docker deployment
├── Authentication & audit logging
```

---

## Contributing to Future Phases

We welcome contributions to any of the planned phases. To contribute:

1. Check the phase you want to work on
2. Open an issue to discuss your approach
3. Follow the implementation plan structure
4. Include tests for all new functionality
5. Update documentation

### Development Setup for Contributors

```bash
# Clone and setup
git clone <repository-url>
cd oil-trading-dashboard
python -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements.txt
pip install -r requirements-research.txt  # For ML/research work

# Run tests
pytest tests/ -v --cov=core

# Start dashboard
streamlit run app/main.py
```

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

**1. Dashboard shows "Disconnected" status**

The dashboard requires a Bloomberg Terminal connection by default. If disconnected:
- Verify Bloomberg Terminal is running on localhost:8194
- Check `BLOOMBERG_HOST` and `BLOOMBERG_PORT` in `.env`
- Install the Bloomberg API: `pip install blpapi`
- For development without Bloomberg, set `BLOOMBERG_USE_MOCK=true` in `.env`

**2. "Data Unavailable" messages**

This means the required data cannot be retrieved from Bloomberg:
- Check your Bloomberg Terminal connection
- Verify you have the required Bloomberg data subscriptions
- Check the connection error message in the sidebar

**3. "diskcache not available" warning**
```bash
pip install diskcache
```

**4. Missing dependencies**
```bash
pip install -r requirements.txt
```

**5. Test failures**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
pip install pytest pytest-cov

# Run tests
pytest tests/ -v
```

**6. How to run in development mode (without Bloomberg)**
```bash
# Set environment variable
export BLOOMBERG_USE_MOCK=true

# Or add to .env file
echo "BLOOMBERG_USE_MOCK=true" >> .env

# Run the dashboard
streamlit run app/main.py
```

⚠️ Note: Development mode shows simulated data, NOT real market prices.

## Disclaimer

This software is for informational and educational purposes only. It does not constitute investment advice. Trading commodities involves substantial risk of loss.

## License

MIT License - See LICENSE file for details.

---

**Built with** ❤️ **for oil traders**
