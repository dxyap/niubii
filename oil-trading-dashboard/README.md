# 🛢️ Quantitative Oil Trading Dashboard

A comprehensive, **lightweight, local-first** quantitative trading dashboard specifically designed for oil markets. Built with Python and Streamlit, featuring real-time market analysis, signal generation, risk management, and trade tracking.

> **Current Status:** Phase 1 Complete ✅ | See [PROGRESS.md](PROGRESS.md) for detailed roadmap

![Dashboard Preview](docs/preview.png)

## 📊 Project Status

| Component | Status | Coverage |
|-----------|--------|----------|
| Data Infrastructure | ✅ Complete | Bloomberg wrapper, caching, Parquet |
| Market Analytics | ✅ Complete | Curves, spreads, fundamentals |
| Signal Engine | ✅ Complete | Technical + fundamental signals |
| Risk Management | ✅ Complete | VaR, limits, stress testing |
| Trading Module | ✅ Complete | Blotter, positions, P&L |
| Dashboard UI | ✅ Complete | 7 pages, dark theme |
| Test Suite | ✅ Complete | 43 tests passing |
| ML Integration | 🔲 Planned | XGBoost, LightGBM |
| LLM News Summary | 🔲 Planned | GPT-4/Claude |
| Backtesting | 🔲 Planned | vectorbt |

## 🎯 Features

### Market Insights
- **Real-time price monitoring** for WTI, Brent, RBOB, and Heating Oil
- **Futures curve analysis** with term structure visualization
- **Crack spread monitoring** (3-2-1, 2-1-1, component cracks)
- **EIA inventory analytics** with surprise calculations
- **OPEC production monitoring** and compliance tracking
- **Refinery turnaround calendar**

### Signal Generation
- **Technical signals**: MA crossovers, RSI, Bollinger Bands, momentum
- **Fundamental signals**: Inventory surprises, OPEC compliance, term structure
- **Signal aggregation** with confidence scoring
- **Signal performance tracking**

### Risk Management
- **Portfolio VaR** (parametric, historical, Monte Carlo)
- **Expected Shortfall (CVaR)**
- **Position and exposure limits**
- **Concentration monitoring**
- **Stress testing** with historical scenarios
- **Real-time alerts**

### Trading
- **Manual trade entry** with pre-trade risk checks
- **Position monitor** with live P&L
- **Trade blotter** with history and statistics
- **P&L attribution** by strategy

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    QUANTITATIVE OIL TRADING DASHBOARD                   │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   MARKET    │  │   SIGNAL    │  │    RISK     │  │   TRADE     │    │
│  │  INSIGHTS   │  │   ENGINE    │  │  MANAGER    │  │  BLOTTER    │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                │                │                │            │
│  ┌──────┴────────────────┴────────────────┴────────────────┴──────┐    │
│  │                     LOCAL DATA LAYER                           │    │
│  │         SQLite / DuckDB  ←→  Parquet Files  ←→  Cache          │    │
│  └────────────────────────────────┬───────────────────────────────┘    │
│                                   │                                     │
│  ┌────────────────────────────────┴───────────────────────────────┐    │
│  │                  BLOOMBERG DESKTOP API (BLPAPI)                │    │
│  │              (Mock data available for development)             │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Bloomberg Terminal (optional - mock data available)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/oil-trading-dashboard.git
cd oil-trading-dashboard
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the dashboard**
```bash
streamlit run app/main.py
```

5. **Open in browser**
Navigate to `http://localhost:8501`

## 📁 Project Structure

```
oil-trading-dashboard/
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
├── data/                    # Local data storage
│   ├── cache/              # Temporary cache
│   ├── historical/         # Parquet files
│   └── trades/             # Trade database
│
├── tests/                   # Test suite
├── research/               # Jupyter notebooks
└── requirements.txt        # Dependencies
```

## ⚙️ Configuration

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

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ -v --cov=core --cov-report=html
```

## 📊 Using the Dashboard

### 1. Market Insights
- View real-time prices and curves
- Analyze term structure and spreads
- Monitor inventory and OPEC data

### 2. Trading Signals
- Review generated signals with confidence scores
- Understand signal drivers
- Track signal performance

### 3. Risk Management
- Monitor portfolio VaR and exposure
- Check position limits
- Run stress tests

### 4. Trade Entry
- Enter trades with pre-trade risk checks
- View current positions
- Track P&L

### 5. Analytics
- Run backtests
- Analyze correlations
- Study seasonality

## 🔌 Bloomberg Integration

The dashboard supports Bloomberg Desktop API for real-time data. When Bloomberg is not available, it uses realistic mock data.

To enable Bloomberg:

```python
from core.data import DataLoader

loader = DataLoader(use_mock=False)  # Enable real Bloomberg
```

## 🛠️ Development

### Adding a New Signal

1. Create signal class in `core/signals/`
2. Implement signal generation method
3. Register in aggregator
4. Add to dashboard

### Adding a New Analytics Module

1. Create module in `core/analytics/`
2. Add visualization in `app/pages/`
3. Write tests in `tests/`

## 🚀 What's Next (Roadmap)

See [PROGRESS.md](PROGRESS.md) for the complete development roadmap.

### Immediate Priorities (Phase 2)
| Feature | Priority | Description |
|---------|----------|-------------|
| **Real Bloomberg Streaming** | 🔴 High | WebSocket price updates <1s latency |
| **Advanced Charting** | 🔴 High | TradingView-style interactive charts |
| **Keyboard Shortcuts** | 🟡 Medium | Power user hotkeys (Ctrl+1-7, F5, etc.) |
| **Custom Alerts** | 🟡 Medium | Email/SMS/Telegram notifications |

### Short-Term (Phase 3)
| Feature | Priority | Description |
|---------|----------|-------------|
| **ML Signal Models** | 🔴 High | XGBoost/LightGBM price direction |
| **Backtesting Engine** | 🔴 High | vectorbt integration with oil-specific features |
| **LLM News Summary** | 🔴 High | GPT-4/Claude daily market digest |
| **Portfolio Optimization** | 🟡 Medium | Mean-variance, risk parity |

### Future Enhancements
- [ ] Satellite data for storage monitoring (Orbital Insight)
- [ ] AIS ship tracking for tanker movements
- [ ] Voice interface for quick queries
- [ ] Multi-user with role-based permissions
- [ ] Snowflake scaling when data grows

## 💡 Design Philosophy

**Lightweight & Local-First:**
- Everything runs on a single machine (no cloud required)
- SQLite for transactions, Parquet for analytics
- In-memory caching for real-time data
- Scale to Snowflake only when needed

**Industry Standards:**
- Professional trading terminal aesthetic
- Sub-second data refresh capability
- Pre-trade risk validation
- Complete audit trail

**Extensibility:**
- Modular architecture for easy customization
- Bloomberg-ready interface (mock data for development)
- Plugin system for custom signals

## ⚠️ Disclaimer

This software is for informational and educational purposes only. It does not constitute investment advice. Trading commodities involves substantial risk of loss. Always use proper risk management and consult qualified professionals before trading.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Version:** 1.0.0  
**Author:** Your Name  
**Last Updated:** November 2024
