# 🎯 Next Steps for Industry-Grade Dashboard

## Executive Summary

The Oil Trading Dashboard has completed **Phase 1 (Foundation)**. This document outlines the specific technical tasks required to elevate it to industry-grade standards while maintaining the lightweight, local-first architecture.

---

## 🔴 HIGH PRIORITY - Week 1-2

### 1. Real-Time Data Streaming

**Current State:** 5-second cache with polling  
**Target State:** Sub-second WebSocket streaming

```python
# Implementation: core/data/streaming.py

import asyncio
import websockets
from typing import Callable, Dict
from dataclasses import dataclass

@dataclass
class PriceUpdate:
    ticker: str
    price: float
    bid: float
    ask: float
    volume: int
    timestamp: datetime

class BloombergStreamer:
    """Real-time price streaming from Bloomberg."""
    
    def __init__(self, tickers: list[str]):
        self.tickers = tickers
        self.callbacks: list[Callable] = []
        self._running = False
    
    async def subscribe(self, callback: Callable[[PriceUpdate], None]):
        """Subscribe to price updates."""
        self.callbacks.append(callback)
    
    async def start(self):
        """Start streaming prices."""
        self._running = True
        # Bloomberg BLPAPI subscription
        # Or WebSocket to internal price server
        while self._running:
            for ticker in self.tickers:
                update = await self._fetch_price(ticker)
                for callback in self.callbacks:
                    await callback(update)
            await asyncio.sleep(0.1)  # 100ms refresh
    
    async def stop(self):
        self._running = False
```

**Tasks:**
- [ ] Create `core/data/streaming.py`
- [ ] Implement Bloomberg subscription service
- [ ] Add WebSocket endpoint for frontend
- [ ] Create connection status component
- [ ] Add auto-reconnection logic
- [ ] Update dashboard to use streaming

---

### 2. Professional Charting (Lightweight Charts)

**Current State:** Basic Plotly charts  
**Target State:** TradingView-quality with drawing tools

```html
<!-- Integration in Streamlit via components -->
<script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>

<script>
const chart = LightweightCharts.createChart(document.getElementById('chart'), {
    width: 800,
    height: 400,
    layout: {
        background: { color: '#0E1117' },
        textColor: '#FAFAFA',
    },
    grid: {
        vertLines: { color: '#2D3139' },
        horzLines: { color: '#2D3139' },
    },
});

const candlestickSeries = chart.addCandlestickSeries({
    upColor: '#00D26A',
    downColor: '#FF4B4B',
    borderVisible: false,
    wickUpColor: '#00D26A',
    wickDownColor: '#FF4B4B',
});
</script>
```

**Tasks:**
- [ ] Create Streamlit custom component for Lightweight Charts
- [ ] Implement candlestick, line, area chart types
- [ ] Add technical indicator overlays (MA, BB, RSI)
- [ ] Implement crosshair with price display
- [ ] Add drawing tools (trendlines, horizontals)
- [ ] Enable chart screenshot export

---

### 3. Keyboard Shortcuts

**Current State:** Mouse-only  
**Target State:** Full keyboard navigation

```python
# Implementation: app/components/shortcuts.py

import streamlit as st
from streamlit_shortcuts import add_keyboard_shortcuts

SHORTCUTS = {
    "ctrl+1": "navigate_overview",
    "ctrl+2": "navigate_market",
    "ctrl+3": "navigate_signals",
    "ctrl+4": "navigate_risk",
    "ctrl+5": "navigate_trade",
    "ctrl+6": "navigate_blotter",
    "ctrl+7": "navigate_analytics",
    "f5": "refresh_data",
    "ctrl+n": "new_trade",
    "ctrl+s": "export_data",
    "escape": "close_modal",
    "?": "show_help",
}

def setup_shortcuts():
    """Initialize keyboard shortcuts."""
    for key, action in SHORTCUTS.items():
        add_keyboard_shortcuts({key: action})
```

**Tasks:**
- [ ] Add `streamlit-shortcuts` package
- [ ] Create shortcut handler component
- [ ] Implement page navigation shortcuts
- [ ] Add data refresh shortcut
- [ ] Create help modal with shortcut list
- [ ] Add visual feedback for shortcut activation

---

## 🟡 MEDIUM PRIORITY - Week 3-4

### 4. Alert System

**Current State:** Dashboard-only alerts  
**Target State:** Multi-channel notifications

```yaml
# config/alerts.yaml

alert_channels:
  email:
    smtp_server: "smtp.gmail.com"
    port: 587
    sender: "alerts@yourdomain.com"
    
  telegram:
    bot_token: "${TELEGRAM_BOT_TOKEN}"
    chat_id: "${TELEGRAM_CHAT_ID}"
    
  sms:
    provider: "twilio"
    account_sid: "${TWILIO_SID}"
    auth_token: "${TWILIO_TOKEN}"
    from_number: "+1234567890"

alert_rules:
  - name: "VaR Breach"
    condition: "var_utilization > 90"
    channels: ["dashboard", "email", "sms"]
    cooldown: 300  # seconds
    
  - name: "Large Price Move"
    condition: "abs(price_change_1h) > 2.0"
    channels: ["dashboard", "telegram"]
    cooldown: 60
    
  - name: "New High-Confidence Signal"
    condition: "signal_confidence > 75"
    channels: ["dashboard", "email"]
    cooldown: 3600
```

**Tasks:**
- [ ] Create `core/alerts/` module
- [ ] Implement email sender (SMTP)
- [ ] Implement Telegram bot integration
- [ ] Implement SMS via Twilio
- [ ] Create alert rule engine
- [ ] Add alert configuration UI
- [ ] Implement cooldown/throttling

---

### 5. ML Signal Integration

**Current State:** Rule-based signals only  
**Target State:** ML-enhanced signal generation

```python
# Implementation: core/signals/ml_signals.py

import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib

class MLSignalGenerator:
    """Machine learning based signal generation."""
    
    FEATURES = [
        # Price features
        "returns_1d", "returns_5d", "returns_20d",
        "volatility_20d", "rsi_14", "macd_signal",
        
        # Curve features
        "curve_slope", "m1_m2_spread", "roll_yield",
        
        # Fundamental features
        "inventory_zscore", "inventory_surprise",
        "opec_compliance", "turnaround_impact",
        
        # Sentiment features
        "news_sentiment_score",
    ]
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = StandardScaler()
        if model_path:
            self.load_model(model_path)
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the model."""
        X_scaled = self.scaler.fit_transform(X[self.FEATURES])
        
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic',
        )
        self.model.fit(X_scaled, y)
        
    def predict(self, X: pd.DataFrame) -> dict:
        """Generate signal prediction."""
        X_scaled = self.scaler.transform(X[self.FEATURES])
        
        prob = self.model.predict_proba(X_scaled)[0]
        
        if prob[1] > 0.6:
            signal = "LONG"
            confidence = prob[1] * 100
        elif prob[0] > 0.6:
            signal = "SHORT"
            confidence = prob[0] * 100
        else:
            signal = "NEUTRAL"
            confidence = 50
        
        return {
            "signal": signal,
            "confidence": confidence,
            "probabilities": {"long": prob[1], "short": prob[0]},
            "feature_importance": dict(zip(
                self.FEATURES, 
                self.model.feature_importances_
            )),
        }
    
    def save_model(self, path: str):
        joblib.dump((self.model, self.scaler), path)
    
    def load_model(self, path: str):
        self.model, self.scaler = joblib.load(path)
```

**Tasks:**
- [ ] Create `core/signals/ml_signals.py`
- [ ] Implement feature engineering pipeline
- [ ] Train initial XGBoost model on historical data
- [ ] Add model versioning and registry
- [ ] Create model retraining scheduler
- [ ] Add feature importance dashboard
- [ ] Implement model performance monitoring

---

### 6. Backtesting Framework

**Current State:** No backtesting  
**Target State:** Comprehensive strategy testing

```python
# Implementation: core/research/backtester.py

import vectorbt as vbt
import pandas as pd
from typing import Dict, Callable
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    start_date: str
    end_date: str
    initial_capital: float = 1_000_000
    commission: float = 2.50  # per contract
    slippage_pct: float = 0.01
    contract_size: int = 1000

@dataclass  
class BacktestResult:
    returns: pd.Series
    trades: pd.DataFrame
    metrics: Dict
    equity_curve: pd.Series

class OilBacktester:
    """Backtesting engine for oil trading strategies."""
    
    def __init__(self, data_loader):
        self.data = data_loader
    
    def run(
        self,
        strategy: Callable,
        config: BacktestConfig,
        params: Dict = None
    ) -> BacktestResult:
        """Run backtest with given strategy."""
        
        # Load data
        prices = self.data.get_historical(
            "CL1 Comdty",
            start_date=config.start_date,
            end_date=config.end_date
        )
        
        # Generate signals
        signals = strategy(prices, params or {})
        
        # Run vectorbt backtest
        portfolio = vbt.Portfolio.from_signals(
            prices['PX_LAST'],
            entries=signals['entries'],
            exits=signals['exits'],
            init_cash=config.initial_capital,
            fees=config.commission / (prices['PX_LAST'] * config.contract_size),
            slippage=config.slippage_pct / 100,
        )
        
        # Calculate metrics
        metrics = {
            "total_return": portfolio.total_return(),
            "sharpe_ratio": portfolio.sharpe_ratio(),
            "sortino_ratio": portfolio.sortino_ratio(),
            "max_drawdown": portfolio.max_drawdown(),
            "win_rate": portfolio.trades.win_rate(),
            "profit_factor": portfolio.trades.profit_factor(),
            "total_trades": portfolio.trades.count(),
            "avg_trade_pnl": portfolio.trades.pnl.mean(),
        }
        
        return BacktestResult(
            returns=portfolio.returns(),
            trades=portfolio.trades.records_readable,
            metrics=metrics,
            equity_curve=portfolio.value(),
        )
    
    def optimize(
        self,
        strategy: Callable,
        config: BacktestConfig,
        param_grid: Dict,
        metric: str = "sharpe_ratio"
    ) -> Dict:
        """Grid search optimization."""
        
        results = []
        for params in self._generate_param_combinations(param_grid):
            result = self.run(strategy, config, params)
            results.append({
                "params": params,
                "metric": result.metrics[metric],
                "result": result,
            })
        
        # Return best
        best = max(results, key=lambda x: x["metric"])
        return best
```

**Tasks:**
- [ ] Create `core/research/backtester.py`
- [ ] Implement vectorbt integration
- [ ] Add oil-specific features (rolls, spreads)
- [ ] Create strategy template library
- [ ] Add optimization engine
- [ ] Create backtest results dashboard
- [ ] Add tear sheet generation (pyfolio)

---

## 🟢 LOWER PRIORITY - Month 2+

### 7. LLM News Summarization

```python
# core/analytics/news.py

from openai import OpenAI
from anthropic import Anthropic

class NewsAnalyzer:
    def __init__(self, provider: str = "anthropic"):
        if provider == "openai":
            self.client = OpenAI()
        else:
            self.client = Anthropic()
    
    def summarize_news(self, articles: list[str]) -> dict:
        """Summarize oil market news."""
        prompt = f"""
        Analyze these oil market news articles and provide:
        1. 3-5 bullet point summary
        2. Bullish/Bearish/Neutral sentiment
        3. Key price drivers identified
        4. Potential market impact
        
        Articles:
        {chr(10).join(articles)}
        """
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_response(response.content)
```

### 8. Portfolio Optimization

```python
# core/analytics/portfolio.py

import cvxpy as cp
import numpy as np

class PortfolioOptimizer:
    def optimize_mean_variance(
        self,
        returns: pd.DataFrame,
        risk_aversion: float = 1.0
    ) -> np.ndarray:
        """Mean-variance optimization."""
        mu = returns.mean().values
        sigma = returns.cov().values
        n = len(mu)
        
        w = cp.Variable(n)
        ret = mu @ w
        risk = cp.quad_form(w, sigma)
        
        objective = cp.Maximize(ret - risk_aversion * risk)
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= 0.4,  # Max 40% per position
        ]
        
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        return w.value
```

---

## 📋 Implementation Checklist

### Week 1
- [ ] Set up real-time streaming infrastructure
- [ ] Create Lightweight Charts component
- [ ] Implement basic keyboard shortcuts

### Week 2
- [ ] Complete streaming integration
- [ ] Add chart drawing tools
- [ ] Create shortcut help modal

### Week 3
- [ ] Implement alert system core
- [ ] Add email notifications
- [ ] Start ML model development

### Week 4
- [ ] Add Telegram/SMS alerts
- [ ] Complete ML signal integration
- [ ] Begin backtesting framework

### Month 2
- [ ] Complete backtesting with optimization
- [ ] Add LLM news summarization
- [ ] Implement portfolio optimization
- [ ] Performance tuning and optimization

---

## 🎯 Success Criteria

| Metric | Current | Phase 2 Target | Industry Standard |
|--------|---------|----------------|-------------------|
| Price Latency | 5000ms | <500ms | <100ms |
| Page Load | ~2s | <1s | <500ms |
| Signal Generation | ~1s | <200ms | <100ms |
| Test Coverage | ~70% | >85% | >95% |
| Uptime | N/A | 99% | 99.9% |

---

**Document Version:** 1.0  
**Last Updated:** November 30, 2024  
**Next Review:** December 7, 2024
