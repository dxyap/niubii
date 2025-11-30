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
- Bloomberg Terminal (optional - mock data available)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

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
│   └── components/          # Reusable UI components
│
├── core/                    # Core business logic
│   ├── data/               # Data loading & caching
│   │   ├── bloomberg.py    # Bloomberg API wrapper
│   │   ├── cache.py        # Caching layer
│   │   └── loader.py       # Data loader utilities
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
│   └── bloomberg_tickers.yaml
│
└── tests/                   # Test suite (43 tests)
```

## Configuration

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
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=core --cov-report=html
```

## Bloomberg Integration

The dashboard supports Bloomberg Desktop API for real-time data. When Bloomberg is not available, it uses realistic mock data.

```python
from core.data import DataLoader

loader = DataLoader(use_mock=False)  # Enable real Bloomberg
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
| Test Suite | ✅ 43 tests |
| ML Integration | 🔲 Planned |
| Backtesting | 🔲 Planned |

### Planned Features
- Real-time Bloomberg WebSocket streaming
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

## Disclaimer

This software is for informational and educational purposes only. It does not constitute investment advice. Trading commodities involves substantial risk of loss.

## License

MIT License
