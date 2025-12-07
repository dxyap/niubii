# 🛢️ Quantitative Oil Trading Dashboard

A lightweight, **local-first** quantitative trading dashboard for oil markets. Built with Python and Streamlit, featuring real-time market analysis, signal generation, risk management, and trade tracking.

> ⚠️ **Important**: This is a **simulation and analysis tool only**. There is **no automatic execution of trades** and **no direct connection to any broker or exchange**. All trading operations are paper trading simulations for strategy testing and educational purposes.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Tests](https://img.shields.io/badge/Tests-255%20passed-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Local Only](https://img.shields.io/badge/Runs-Locally-brightgreen.svg)

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

### 🤖 Execution & Automation (Simulation Only)
- **Order Management System**: Full order lifecycle tracking (created→submitted→filled)
- **Paper Trading Mode**: Simulated execution for strategy testing - **no real trades**
- **Position Sizing**: Kelly criterion, volatility targeting, risk parity, ATR/VaR-based
- **Execution Algorithms**: TWAP, VWAP, POV, Implementation Shortfall (simulation)
- **Simulated Broker**: Realistic fills and slippage for testing purposes
- **Automation Rules**: Signal-to-order conversion for paper trading only

> 🔒 **No Live Trading**: All execution is simulated. There is no connection to real brokers or exchanges.

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
│   │   ├── 7_🤖_ML_Signals.py    # ML-powered signals
│   │   ├── 8_🔬_Backtest.py      # Strategy backtesting
│   │   └── 9_🤖_Automation.py    # Execution & automation
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
│   ├── ml/                  # Machine Learning (Phase 4)
│   │   ├── features.py      # Feature engineering pipeline
│   │   ├── models/
│   │   │   ├── gradient_boost.py  # XGBoost/LightGBM
│   │   │   └── ensemble.py        # Model ensembling
│   │   ├── training.py      # Training pipeline
│   │   ├── prediction.py    # Inference service
│   │   └── monitoring.py    # Model monitoring & drift detection
│   ├── backtest/            # Backtesting Engine (Phase 5)
│   │   ├── engine.py        # Main backtesting engine
│   │   ├── strategy.py      # Strategy framework & examples
│   │   ├── execution.py     # Order execution simulation
│   │   ├── costs.py         # Transaction cost models
│   │   ├── metrics.py       # Performance metrics
│   │   ├── optimization.py  # Walk-forward optimization
│   │   └── reporting.py     # Reports & visualization
│   ├── execution/            # Execution & Automation (Simulation Only)
│   │   ├── oms.py            # Order Management System
│   │   ├── sizing.py         # Position sizing algorithms
│   │   ├── algorithms.py     # TWAP, VWAP, POV, IS algorithms
│   │   ├── paper_trading.py  # Paper trading engine (no real execution)
│   │   ├── automation.py     # Automation rules engine (simulation only)
│   │   └── brokers/          # Simulated broker only
│   │       ├── base.py       # Broker interface (abstract)
│   │       └── simulator.py  # Simulated broker (no real connections)
│   ├── alerts/               # Alerts & Notifications (Phase 7)
│   │   ├── rules.py          # Alert rule definitions
│   │   ├── engine.py         # Alert evaluation engine
│   │   ├── channels/         # Notification channels
│   │   │   ├── email.py      # Email (SMTP)
│   │   │   ├── telegram.py   # Telegram Bot
│   │   │   ├── slack.py      # Slack webhooks
│   │   │   └── sms.py        # SMS (Twilio)
│   │   ├── scheduler.py      # Scheduled reports
│   │   └── history.py        # Alert history (SQLite)
│   ├── research/             # Advanced Analytics & AI (Phase 8)
│   │   ├── llm/              # LLM integration
│   │   │   ├── news_analyzer.py  # News summarization
│   │   │   └── sentiment.py      # Sentiment scoring
│   │   ├── correlations.py   # Cross-asset correlations
│   │   ├── regimes.py        # Market regime detection
│   │   ├── factors.py        # Factor decomposition
│   │   └── alt_data/         # Alternative data
│   │       ├── satellite.py  # Storage tank levels
│   │       ├── shipping.py   # Tanker tracking
│   │       └── positioning.py # COT/positioning data
│   └── infrastructure/       # Production Hardening (Phase 9)
│       ├── auth.py           # Authentication
│       ├── rbac.py           # Role-based access control
│       ├── audit.py          # Audit logging
│       └── monitoring.py     # Health checks & metrics
│
├── config/                  # Configuration files
│   ├── instruments.yaml    # Instrument definitions
│   ├── risk_limits.yaml    # Risk parameters
│   ├── execution.yaml      # Execution & automation config
│   ├── alerts.yaml         # Alert configurations
│   └── bloomberg_tickers.yaml  # Bloomberg ticker mappings
│
├── migrations/               # Database migrations
│   └── versions/            # Alembic migration files
│
├── alembic.ini              # Alembic configuration
│
├── models/                  # Trained ML models (auto-created)
│
├── data/                    # Data storage (auto-created)
│   ├── cache/              # Cached data
│   ├── historical/         # Parquet files
│   └── trades/             # Trade database
│
└── tests/                   # Test suite (200+ tests)
    ├── test_alerts.py       # Alerts module tests
    ├── test_research.py     # Research module tests
    └── test_infrastructure.py # Infrastructure tests
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
st_autorefresh(interval=60000)  # Milliseconds (60s)
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
# Auto-refresh interval (seconds) - 60s default reduces churn
AUTO_REFRESH_INTERVAL=60

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
AUTO_REFRESH_INTERVAL=60          # Auto-refresh interval (60 seconds)
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

### Backtest Module

Comprehensive strategy backtesting:

```python
from core.backtest import (
    # Engine
    BacktestEngine, BacktestConfig, run_backtest,
    # Strategies
    MACrossoverStrategy, RSIMeanReversionStrategy,
    BollingerBandStrategy, MomentumStrategy,
    BuyAndHoldStrategy, StrategyConfig,
    # Costs
    SimpleCostModel, CostModelConfig,
    # Metrics
    MetricsCalculator, PerformanceMetrics,
    # Optimization
    StrategyOptimizer, OptimizationConfig,
    # Reporting
    generate_summary_report, create_equity_chart,
)

# =============================================================================
# SIMPLE BACKTEST
# =============================================================================
strategy = MACrossoverStrategy(fast_period=10, slow_period=30)
result = run_backtest(strategy, historical_data, initial_capital=1_000_000)

print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
print(f"Return: {result.metrics.total_return_pct:.2f}%")
print(f"Max DD: {result.metrics.max_drawdown:.2f}%")

# =============================================================================
# CUSTOM STRATEGY
# =============================================================================
from core.backtest import Strategy, Signal, Position

class MyStrategy(Strategy):
    def generate_signal(self, timestamp, data, position):
        prices = data["PX_LAST"]
        ma = prices.rolling(20).mean().iloc[-1]
        
        if prices.iloc[-1] > ma:
            return Signal.LONG
        elif prices.iloc[-1] < ma:
            return Signal.SHORT
        return Signal.HOLD

# =============================================================================
# WITH TRANSACTION COSTS
# =============================================================================
cost_config = CostModelConfig(
    commission_per_contract=2.50,
    slippage_ticks=1.0,
    contract_multiplier=1000,
)
cost_model = SimpleCostModel(cost_config)

config = BacktestConfig(
    initial_capital=1_000_000,
    commission_per_contract=2.50,
    slippage_pct=0.01,
)

engine = BacktestEngine(config, cost_model)
result = engine.run(strategy, data, "CL1")

# =============================================================================
# WALK-FORWARD OPTIMIZATION
# =============================================================================
optimizer = StrategyOptimizer(
    strategy_class=MACrossoverStrategy,
    param_grid={
        "fast_period": [5, 10, 15, 20],
        "slow_period": [20, 30, 40, 50],
    },
    config=OptimizationConfig(
        target_metric="sharpe_ratio",
        num_folds=5,
    )
)

opt_result = optimizer.walk_forward_optimize(data)
print(f"Best params: {opt_result.best_params}")
print(f"OOS Sharpe: {opt_result.oos_metrics.sharpe_ratio:.2f}")

# =============================================================================
# COMPARE STRATEGIES
# =============================================================================
strategies = [
    BuyAndHoldStrategy(),
    MACrossoverStrategy(10, 30),
    RSIMeanReversionStrategy(14),
]

engine = BacktestEngine()
results = engine.run_multiple(strategies, data)
comparison = engine.compare_strategies(results)
print(comparison)
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
| Backtesting | 25 | 90% |
| Execution & Automation | 47 | 92% |
| Alerts & Notifications | 20 | 88% |
| Research & Analytics | 25 | 85% |
| Infrastructure (Auth/Audit) | 30 | 90% |

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
| Test Suite | ✅ 90+ tests | 2 |
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
| Backtesting Engine | ✅ Complete | 5 |
| Strategy Framework | ✅ Complete | 5 |
| Walk-Forward Optimization | ✅ Complete | 5 |
| Performance Metrics | ✅ Complete | 5 |
| Order Management System | ✅ Complete | 6 |
| Paper Trading (Simulation) | ✅ Complete | 6 |
| Position Sizing Algorithms | ✅ Complete | 6 |
| Execution Algorithms (Simulated) | ✅ Complete | 6 |
| Automation Rules (Paper Trading) | ✅ Complete | 6 |
| Multi-channel Alerts | ✅ Complete | 7 |
| Advanced Analytics & AI | ✅ Complete | 8 |
| Production Hardening | ✅ Complete | 9 |

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

### ✅ Phase 5: Backtesting Engine (Complete)

**Historical Strategy Testing & Optimization**

A comprehensive backtesting framework for strategy development and validation.

| Feature | Description | Status |
|---------|-------------|--------|
| Event-Driven Backtest | Bar-by-bar simulation engine | ✅ Complete |
| Strategy Framework | Define strategies as composable classes | ✅ Complete |
| Transaction Costs | Realistic slippage, commissions, and market impact | ✅ Complete |
| Walk-Forward Optimization | Rolling window parameter optimization | ✅ Complete |
| Performance Metrics | Sharpe, Sortino, Calmar, max drawdown, etc. | ✅ Complete |
| Monte Carlo Analysis | Bootstrap resampling for robustness testing | ✅ Complete |
| Strategy Comparison | Side-by-side strategy evaluation | ✅ Complete |
| Backtest Dashboard | Interactive UI for running backtests | ✅ Complete |

**Implementation:**
```
core/
├── backtest/
│   ├── __init__.py           # Module exports
│   ├── engine.py             # Main backtesting engine
│   ├── strategy.py           # Strategy base class and examples
│   ├── execution.py          # Order execution simulation
│   ├── costs.py              # Transaction cost models
│   ├── metrics.py            # Performance metrics (20+ metrics)
│   ├── optimization.py       # Parameter & walk-forward optimization
│   └── reporting.py          # Charts and reports
app/pages/
├── 8_🔬_Backtest.py          # Backtest configuration UI
```

**Built-in Strategies:**
- **BuyAndHoldStrategy**: Simple benchmark
- **MACrossoverStrategy**: Moving average crossover
- **RSIMeanReversionStrategy**: RSI-based mean reversion
- **BollingerBandStrategy**: Bollinger Band breakouts
- **MomentumStrategy**: Price momentum/breakout
- **CalendarSpreadStrategy**: Spread trading
- **CompositeStrategy**: Combine multiple strategies

**Cost Models:**
- **SimpleCostModel**: Fixed commissions and slippage
- **VolatilityAdjustedCostModel**: Vol-scaled slippage
- **MarketImpactCostModel**: Square-root impact model
- **TieredCommissionModel**: Volume-based tiers

**Usage Example:**
```python
from core.backtest import (
    BacktestEngine, BacktestConfig,
    MACrossoverStrategy, StrategyConfig,
    run_backtest, generate_summary_report
)

# Create strategy
strategy = MACrossoverStrategy(fast_period=10, slow_period=30)

# Run backtest
result = run_backtest(
    strategy, 
    historical_data,
    initial_capital=1_000_000
)

# View results
print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
print(f"Return: {result.metrics.total_return_pct:.2f}%")
print(f"Max DD: {result.metrics.max_drawdown:.2f}%")

# Generate report
report = generate_summary_report(result)
print(report)
```

**Walk-Forward Optimization:**
```python
from core.backtest import StrategyOptimizer, OptimizationConfig

optimizer = StrategyOptimizer(
    strategy_class=MACrossoverStrategy,
    param_grid={
        "fast_period": [5, 10, 15, 20],
        "slow_period": [20, 30, 40, 50],
    },
    config=OptimizationConfig(
        target_metric="sharpe_ratio",
        num_folds=5,
        in_sample_pct=0.7,
    )
)

result = optimizer.walk_forward_optimize(historical_data)
print(f"Best params: {result.best_params}")
print(f"OOS Sharpe: {result.oos_metrics.sharpe_ratio:.2f}")
```

---

### ✅ Phase 6: Execution & Automation (Complete - Simulation Only)

**Order Management & Paper Trading**

> ⚠️ **No Live Trading**: All execution is **simulated**. There is **no connection to real brokers or exchanges**. This is for strategy testing and educational purposes only.

Full execution infrastructure for signal-to-order conversion and **paper trading simulation**.

| Feature | Description | Status |
|---------|-------------|--------|
| Order Management System | Complete order lifecycle with SQLite persistence | ✅ Complete |
| Paper Trading Mode | Simulated execution with realistic fills | ✅ Complete |
| Position Sizing | Kelly, volatility targeting, risk parity, ATR, VaR | ✅ Complete |
| Execution Algorithms | TWAP, VWAP, POV, Implementation Shortfall (simulated) | ✅ Complete |
| Simulated Broker | Abstract broker with simulation implementation only | ✅ Complete |
| Automation Rules | Signal-based rules for paper trading only | ✅ Complete |
| Dashboard Page | Full automation UI with paper trading | ✅ Complete |

**Implementation:**
```
core/
├── execution/
│   ├── __init__.py           # Module exports
│   ├── oms.py                # Order Management System
│   ├── sizing.py             # Position sizing algorithms
│   ├── algorithms.py         # Execution algorithms (TWAP, VWAP, POV, IS)
│   ├── paper_trading.py      # Paper trading engine
│   ├── automation.py         # Automation rules engine
│   └── brokers/
│       ├── base.py           # Abstract broker interface
│       └── simulator.py      # Simulated broker
app/pages/
├── 9_🤖_Automation.py       # Automation dashboard
config/
├── execution.yaml            # Execution configuration
```

**Position Sizing Algorithms:**

```python
from core.execution import (
    PositionSizer, SizingConfig, SizingMethod,
    KellyCriterion, VolatilityTargeting, RiskParity,
    calculate_optimal_size
)

# Volatility targeting
config = SizingConfig(
    method=SizingMethod.VOLATILITY_TARGET,
    account_value=1_000_000,
    target_volatility=0.15,  # 15% annual target
)

result = calculate_optimal_size(
    price=75.0,
    volatility=0.25,  # 25% asset volatility
    account_value=1_000_000,
)

print(f"Recommended: {result.contracts} contracts")
print(f"Notional: ${result.notional_value:,.0f}")
print(f"Rationale: {result.rationale}")
```

**Execution Algorithms:**

```python
from core.execution import (
    TWAPAlgorithm, VWAPAlgorithm, AlgorithmConfig,
    Order, OrderSide
)

# Create parent order
order = Order(
    order_id="ORD-001",
    symbol="CL1",
    side=OrderSide.BUY,
    quantity=20,
)

# Generate TWAP schedule
config = AlgorithmConfig(
    duration_minutes=60,
    num_slices=12,
    randomize_timing=True,
)

algo = TWAPAlgorithm(config)
slices = algo.generate_schedule(order, current_price=75.0)

for s in slices:
    print(f"Slice {s.sequence}: {s.quantity} contracts at {s.scheduled_time}")
```

**Paper Trading:**

```python
from core.execution import PaperTradingEngine, PaperTradingConfig

# Start paper trading session
config = PaperTradingConfig(
    initial_capital=1_000_000,
    slippage_bps=1.0,
    commission_per_contract=2.50,
)

engine = PaperTradingEngine(config)
engine.start_session()

# Update prices
engine.update_prices({"CL1": 75.0, "CO1": 78.0})

# Submit order
order = engine.submit_order(
    symbol="CL1",
    side="BUY",
    quantity=5,
    order_type="MARKET",
    strategy="momentum",
)

# Check P&L
summary = engine.get_pnl_summary()
print(f"NAV: ${summary['current_nav']:,.0f}")
print(f"Return: {summary['return_pct']:.2f}%")

# Stop session
session = engine.stop_session()
print(f"Session Sharpe: {session.sharpe_ratio:.2f}")
```

**Automation Rules:**

```python
from core.execution import (
    AutomationEngine, RuleConfig, RuleCondition, RuleAction,
    ConditionType, ActionType, SizingMethod, create_signal_rule
)

engine = AutomationEngine()

# Create rule: Enter long on high-confidence bullish signal
rule = create_signal_rule(
    name="Long on Strong Signal",
    symbol="CL1",
    direction="LONG",
    min_confidence=65,
    sizing_method=SizingMethod.VOLATILITY_TARGET,
    risk_pct=0.02,
)
engine.add_rule(rule)

# Evaluate rules against current context
context = {
    "signal": {"direction": "LONG", "confidence": 72},
    "position": {"quantity": 0},
    "price": 75.0,
    "volatility": 0.25,
    "account_value": 1_000_000,
}

triggered = engine.evaluate_rules(context, execute=True)
print(f"Triggered {len(triggered)} rules")
```

---

### ✅ Phase 7: Alerts & Notifications (Complete)

**Multi-Channel Alert System**

Proactive notifications for trading signals, risk breaches, and market events.

| Feature | Description | Status |
|---------|-------------|--------|
| Alert Rules Engine | Configurable conditions and triggers | ✅ Complete |
| Email Notifications | SMTP-based email alerts | ✅ Complete |
| Telegram Bot | Real-time Telegram notifications | ✅ Complete |
| Slack Integration | Slack channel alerts | ✅ Complete |
| SMS Alerts | Critical alerts via SMS (Twilio) | ✅ Complete |
| Scheduled Reports | Daily/weekly P&L and risk summaries | ✅ Complete |
| Alert History | Track and audit all alerts (SQLite) | ✅ Complete |
| Alert Escalation | Escalate unacknowledged critical alerts | ✅ Complete |
| Dashboard Page | Full alerts management UI | ✅ Complete |

**Implementation:**
```
core/
├── alerts/
│   ├── __init__.py          # Module exports
│   ├── rules.py             # Alert rule definitions & factories
│   ├── engine.py            # Alert evaluation engine with state management
│   ├── channels/
│   │   ├── base.py          # Abstract channel with rate limiting
│   │   ├── email.py         # SMTP email notifications
│   │   ├── telegram.py      # Telegram Bot API
│   │   ├── slack.py         # Slack webhook integration
│   │   └── sms.py           # Twilio SMS
│   ├── scheduler.py         # Scheduled report generation
│   └── history.py           # SQLite-based alert history & audit log
config/
├── alerts.yaml              # Alert configurations
app/pages/
├── 10_🔔_Alerts.py          # Alerts dashboard
```

**Usage:**
```python
from core.alerts import (
    AlertEngine, AlertEngineConfig,
    create_price_alert, create_risk_alert,
    EmailChannel, TelegramChannel, SlackChannel,
    ReportScheduler, AlertHistory,
)

# Create alert engine
engine = AlertEngine()

# Add price alert
rule = create_price_alert(
    rule_id="wti_breakout",
    name="WTI Breakout Alert",
    symbol="WTI",
    threshold=80.0,
    above=True,
    severity=AlertSeverity.HIGH,
)
engine.add_rule(rule)

# Register notification channels
engine.add_channel(EmailChannel(smtp_host="smtp.gmail.com", ...))
engine.add_channel(TelegramChannel(bot_token="...", chat_id="..."))
engine.add_channel(SlackChannel(webhook_url="..."))

# Evaluate alerts
triggered = engine.evaluate({"WTI": 82.50})
```

---

### ✅ Phase 8: Advanced Analytics & AI (Complete)

**Research Tools & Alternative Data**

Advanced analytics, AI-powered research, and alternative data sources.

| Feature | Description | Status |
|---------|-------------|--------|
| LLM News Analysis | Summarize and sentiment-score news with GPT/Claude | ✅ Complete |
| Sentiment Analyzer | Rule-based and LLM sentiment scoring | ✅ Complete |
| Cross-Asset Correlations | Oil vs. equities, FX, rates correlations | ✅ Complete |
| Rolling Correlations | Time-varying correlation analysis | ✅ Complete |
| Regime Detection | Market regime identification (trending, ranging, crisis) | ✅ Complete |
| Volatility Regimes | Volatility regime classification | ✅ Complete |
| Factor Analysis | Decompose returns into risk factors (10+ factors) | ✅ Complete |
| Satellite Data | Oil storage tank monitoring simulation | ✅ Complete |
| Shipping Data | Tanker tracking and trade flows | ✅ Complete |
| Positioning Data | COT reports and managed money positions | ✅ Complete |
| Research Dashboard | Full research UI with all analytics | ✅ Complete |

**Implementation:**
```
core/
├── research/
│   ├── __init__.py           # Module exports
│   ├── llm/
│   │   ├── news_analyzer.py  # LLM news summarization (OpenAI, Anthropic, rule-based)
│   │   └── sentiment.py      # Sentiment scoring with commodity detection
│   ├── correlations.py       # Cross-asset correlation analysis
│   ├── regimes.py            # Market and volatility regime detection
│   ├── factors.py            # Factor decomposition (10+ risk factors)
│   └── alt_data/
│       ├── provider.py       # Unified alternative data provider
│       ├── satellite.py      # Storage tank levels (Cushing, Rotterdam, Singapore)
│       ├── shipping.py       # Tanker tracking, trade flows, freight rates
│       └── positioning.py    # COT data, managed money positions
app/pages/
├── 11_🔍_Research.py        # Research dashboard with 5 tabs
```

**Usage:**
```python
from core.research import (
    NewsAnalyzer, SentimentAnalyzer,
    CorrelationAnalyzer, RegimeDetector, FactorModel,
    AlternativeDataProvider,
)

# News analysis
analyzer = NewsAnalyzer()
summary = analyzer.analyze_article(article_text)
print(f"Impact: {summary.impact_level} {summary.impact_direction}")
print(f"Key Points: {summary.key_points}")

# Correlation analysis
corr_analyzer = CorrelationAnalyzer()
matrix = corr_analyzer.calculate_correlation_matrix(["Brent", "WTI", "Dollar"])
rolling = corr_analyzer.calculate_rolling_correlation("Brent", "Dollar", window=63)

# Regime detection
detector = RegimeDetector()
regime = detector.get_current_regime()
print(f"Market Regime: {regime['regime']} (Confidence: {regime['confidence']}%)")

# Factor analysis
factor_model = FactorModel()
decomp = factor_model.decompose_returns("Brent", days=60)
print(f"R-squared: {decomp.r_squared:.1%}")
print(f"Factor Exposures: {decomp.factor_exposures}")

# Alternative data
alt_data = AlternativeDataProvider()
storage_signal = alt_data.satellite.calculate_storage_signal()
shipping_signal = alt_data.shipping.calculate_shipping_signal()
positioning_signal = alt_data.positioning.calculate_positioning_signal()
aggregate = alt_data.get_aggregate_signal()
```

---

### ✅ Phase 9: Infrastructure & Security (Complete)

**Security & Monitoring for Local Deployment**

Infrastructure components for secure local operation with monitoring and compliance.

| Feature | Description | Status |
|---------|-------------|--------|
| Authentication | User authentication with session management | ✅ Complete |
| Role-Based Access | 5 roles with 20+ permissions | ✅ Complete |
| Audit Logging | Complete audit trail (SQLite) | ✅ Complete |
| Database Migrations | Alembic migrations | ✅ Complete |
| Health Checks | Component health monitoring | ✅ Complete |
| Prometheus Metrics | Application and system metrics | ✅ Complete |

**Implementation:**
```
core/
├── infrastructure/
│   ├── __init__.py           # Module exports
│   ├── auth.py               # Authentication (users, sessions, tokens)
│   ├── rbac.py               # Role-Based Access Control (5 roles, 20+ permissions)
│   ├── audit.py              # Audit logging (SQLite with retention)
│   └── monitoring.py         # Health checks, Prometheus metrics

# Database migrations
├── alembic.ini
├── migrations/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       └── 20241205_000001_initial_schema.py
```

**Usage:**
```python
from core.infrastructure import (
    AuthManager, User, Role, Permission,
    RBACManager, require_permission,
    AuditLogger, AuditEventType,
    HealthChecker, MetricsCollector,
)

# Authentication
auth = AuthManager()
user = auth.create_user("trader1", "trader@example.com", "SecurePass123!")
session = auth.authenticate("trader1", "SecurePass123!")

# RBAC
rbac = RBACManager()
can_trade = rbac.check_permission(user, Permission.EXECUTE_TRADES)

# Audit logging
audit = AuditLogger()
audit.log(
    event_type=AuditEventType.ORDER_CREATED,
    action="Created buy order for 10 CL contracts",
    user_id=user.id,
    username=user.username,
)

# Health checks
health = HealthChecker()
summary = health.get_health_summary()

# Metrics
metrics = MetricsCollector()
metrics.increment("trading_orders_total")
metrics.set("trading_pnl", 50000)
output = metrics.get_prometheus_output()
```

---

## Prioritized Roadmap

```
✅ Q1 2025: Phase 4 - ML Integration (COMPLETE)
├── Feature engineering pipeline
├── XGBoost/LightGBM models for direction prediction
├── Model monitoring and drift detection
└── Integration with signal aggregator

✅ Q1 2025: Phase 5 - Backtesting Engine (COMPLETE)
├── Event-driven backtest framework
├── Strategy framework with built-in strategies
├── Walk-forward optimization
└── Performance reporting & visualization

✅ Q2 2025: Phase 6 - Execution & Automation (COMPLETE)
├── Order Management System with full lifecycle
├── Paper trading engine with P&L tracking
├── Position sizing (Kelly, vol targeting, risk parity)
├── Execution algorithms (TWAP, VWAP, POV, IS)
├── Automation rules engine
└── Broker simulation framework

✅ Q2-Q3 2025: Phase 7 - Alerts & Notifications (COMPLETE)
├── Multi-channel alert system (Email, Telegram, Slack, SMS)
├── Alert rules engine with configurable conditions
├── Scheduled reporting (daily/weekly P&L, risk)
├── Alert history and audit logging
└── Alert escalation for critical events

✅ Q3 2025: Phase 8 - Advanced Analytics & AI (COMPLETE)
├── LLM news analysis with GPT/Claude support
├── Sentiment analysis with commodity detection
├── Cross-asset correlation analysis
├── Market regime detection
├── Factor decomposition (10+ risk factors)
└── Alternative data (satellite, shipping, positioning)

✅ Q4 2025: Phase 9 - Infrastructure & Security (COMPLETE)
├── Authentication & session management
├── Role-based access control (5 roles, 20+ permissions)
├── Audit logging (SQLite with retention)
├── Health checks & Prometheus metrics
└── Database migrations with Alembic
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
- No external infrastructure required

**Simulation Only - No Live Trading:**
- All execution is paper trading simulation
- No connection to real brokers or exchanges
- Safe environment for strategy testing
- Educational and research purposes

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
