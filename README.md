# 🛢️ Quantitative Oil Trading Dashboard

A comprehensive, lightweight, local-first trading dashboard for oil market analysis.

## 🎉 Project Status: Phase 1 Complete ✅

The foundation has been built! See the working dashboard in `/oil-trading-dashboard/`.

### Quick Start
```bash
cd oil-trading-dashboard
pip install -r requirements.txt
streamlit run app/main.py
```

### Documentation
| Document | Description |
|----------|-------------|
| [TRADING_DASHBOARD_PLAN.md](TRADING_DASHBOARD_PLAN.md) | Original architecture & implementation plan |
| [oil-trading-dashboard/README.md](oil-trading-dashboard/README.md) | Quick start & feature guide |
| [oil-trading-dashboard/PROGRESS.md](oil-trading-dashboard/PROGRESS.md) | Detailed progress tracker |
| [oil-trading-dashboard/NEXT_STEPS.md](oil-trading-dashboard/NEXT_STEPS.md) | Phase 2 implementation roadmap |

### What's Built ✅
- **Data Infrastructure**: Bloomberg API wrapper (with mock), caching, Parquet storage
- **Market Analytics**: Futures curves, spreads, fundamentals analysis
- **Signal Engine**: Technical + fundamental signals with weighted aggregation
- **Risk Management**: VaR (parametric/historical), position limits, stress testing
- **Trading Module**: Trade blotter, position tracking, P&L calculations
- **Dashboard UI**: 7-page Streamlit app with professional dark theme
- **Test Suite**: 43 tests passing

### What's Next 🔲
| Priority | Feature | Description |
|----------|---------|-------------|
| 🔴 High | Real-Time Streaming | Bloomberg WebSocket (<1s latency) |
| 🔴 High | Advanced Charting | TradingView-style with drawing tools |
| 🔴 High | ML Signals | XGBoost/LightGBM price direction |
| 🟡 Medium | Alerts | Email/SMS/Telegram notifications |
| 🟡 Medium | Backtesting | vectorbt framework integration |
| 🟢 Lower | LLM News | GPT-4/Claude market summaries |

### Project Structure
```
oil-trading-dashboard/
├── app/                 # Streamlit dashboard
│   ├── main.py         # Main entry point
│   ├── pages/          # Dashboard pages
│   └── components/     # Reusable UI components
├── core/               # Core business logic
│   ├── data/           # Bloomberg API, caching, storage
│   ├── analytics/      # Curves, spreads, fundamentals
│   ├── signals/        # Technical, fundamental, aggregation
│   ├── risk/           # VaR, limits, monitoring
│   └── trading/        # Blotter, positions, P&L
├── config/             # YAML configuration files
├── tests/              # Unit tests (43 tests)
└── data/               # Local data storage
```

---

**Design Philosophy:** Lightweight & local-first. Scale to Snowflake only when needed.
