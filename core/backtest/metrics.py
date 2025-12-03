"""
Performance Metrics
====================
Comprehensive performance and risk metrics for backtesting and portfolio analysis.

Provides industry-standard metrics including:
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Win/loss statistics
- Risk metrics (VaR, CVaR)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime


# Trading days per year (standard assumption)
TRADING_DAYS_PER_YEAR = 252


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""
    
    # Return metrics
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    cagr: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    annualized_volatility: float = 0.0
    downside_volatility: float = 0.0
    
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    
    # Drawdown metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    avg_drawdown: float = 0.0
    recovery_time: int = 0
    
    # Win/loss metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade: float = 0.0
    expectancy: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    
    # Other metrics
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    trading_days: int = 0
    exposure_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "annualized_return": self.annualized_return,
            "cagr": self.cagr,
            "volatility": self.volatility,
            "annualized_volatility": self.annualized_volatility,
            "downside_volatility": self.downside_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "avg_drawdown": self.avg_drawdown,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "expectancy": self.expectancy,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "trading_days": self.trading_days,
            "exposure_time": self.exposure_time,
        }


class MetricsCalculator:
    """
    Calculate comprehensive performance metrics from equity curves and trade data.
    
    Supports both portfolio-level metrics (from equity curve) and 
    trade-level metrics (from individual trades).
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculations
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    
    def calculate_all(
        self,
        equity_curve: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        benchmark: Optional[pd.Series] = None
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.
        
        Args:
            equity_curve: Series of portfolio values indexed by date
            trades: DataFrame with trade P&L data
            benchmark: Optional benchmark equity curve for comparison
            
        Returns:
            PerformanceMetrics with all calculated metrics
        """
        if equity_curve is None or len(equity_curve) < 2:
            return PerformanceMetrics()
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        metrics = PerformanceMetrics()
        
        # Basic info
        metrics.start_date = equity_curve.index[0]
        metrics.end_date = equity_curve.index[-1]
        metrics.trading_days = len(equity_curve)
        
        # Return metrics
        metrics.total_return = equity_curve.iloc[-1] - equity_curve.iloc[0]
        metrics.total_return_pct = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        
        years = metrics.trading_days / TRADING_DAYS_PER_YEAR
        if years > 0:
            metrics.annualized_return = returns.mean() * TRADING_DAYS_PER_YEAR * 100
            metrics.cagr = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1) * 100
        
        # Volatility metrics
        metrics.volatility = returns.std() * 100
        metrics.annualized_volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        
        # Downside volatility (using negative returns only)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            metrics.downside_volatility = negative_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        
        # Risk-adjusted returns
        metrics.sharpe_ratio = self._calculate_sharpe(returns)
        metrics.sortino_ratio = self._calculate_sortino(returns)
        
        # Drawdown metrics
        dd_stats = self._calculate_drawdown_stats(equity_curve)
        metrics.max_drawdown = dd_stats["max_drawdown"]
        metrics.max_drawdown_duration = dd_stats["max_duration"]
        metrics.avg_drawdown = dd_stats["avg_drawdown"]
        metrics.recovery_time = dd_stats.get("recovery_time", 0)
        
        # Calmar ratio
        if metrics.max_drawdown != 0:
            metrics.calmar_ratio = metrics.cagr / abs(metrics.max_drawdown)
        
        # VaR and CVaR
        metrics.var_95 = self._calculate_var(returns, 0.95)
        metrics.var_99 = self._calculate_var(returns, 0.99)
        metrics.cvar_95 = self._calculate_cvar(returns, 0.95)
        
        # Information ratio (if benchmark provided)
        if benchmark is not None and len(benchmark) > 0:
            metrics.information_ratio = self._calculate_information_ratio(
                equity_curve, benchmark
            )
        
        # Trade metrics (if trades provided)
        if trades is not None and len(trades) > 0:
            trade_metrics = self._calculate_trade_metrics(trades)
            metrics.total_trades = trade_metrics["total_trades"]
            metrics.winning_trades = trade_metrics["winning_trades"]
            metrics.losing_trades = trade_metrics["losing_trades"]
            metrics.win_rate = trade_metrics["win_rate"]
            metrics.profit_factor = trade_metrics["profit_factor"]
            metrics.avg_win = trade_metrics["avg_win"]
            metrics.avg_loss = trade_metrics["avg_loss"]
            metrics.largest_win = trade_metrics["largest_win"]
            metrics.largest_loss = trade_metrics["largest_loss"]
            metrics.avg_trade = trade_metrics["avg_trade"]
            metrics.expectancy = trade_metrics["expectancy"]
            metrics.exposure_time = trade_metrics.get("exposure_time", 0)
        
        return metrics
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if returns.std() == 0:
            return 0.0
        
        excess_returns = returns - self.daily_rf
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        return round(sharpe, 4)
    
    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (uses downside deviation)."""
        excess_returns = returns - self.daily_rf
        
        # Downside deviation
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0 or negative_returns.std() == 0:
            return 0.0
        
        downside_std = negative_returns.std()
        sortino = excess_returns.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        return round(sortino, 4)
    
    def _calculate_drawdown_stats(self, equity_curve: pd.Series) -> Dict:
        """Calculate drawdown statistics."""
        # Calculate drawdown series
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max * 100
        
        max_dd = drawdowns.min()
        avg_dd = drawdowns[drawdowns < 0].mean() if len(drawdowns[drawdowns < 0]) > 0 else 0
        
        # Calculate max drawdown duration
        in_drawdown = drawdowns < 0
        dd_starts = (~in_drawdown).cumsum()
        
        max_duration = 0
        current_duration = 0
        
        for is_in_dd in in_drawdown:
            if is_in_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return {
            "max_drawdown": round(max_dd, 2),
            "max_duration": max_duration,
            "avg_drawdown": round(avg_dd, 2) if not np.isnan(avg_dd) else 0,
            "recovery_time": 0,  # Would need more complex logic
        }
    
    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk (parametric)."""
        if len(returns) == 0:
            return 0.0
        var = np.percentile(returns, (1 - confidence) * 100)
        return round(var * 100, 4)  # As percentage
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        if len(returns) == 0:
            return 0.0
        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var_threshold].mean()
        return round(cvar * 100, 4) if not np.isnan(cvar) else 0.0
    
    def _calculate_information_ratio(
        self, 
        equity_curve: pd.Series, 
        benchmark: pd.Series
    ) -> float:
        """Calculate Information Ratio relative to benchmark."""
        # Align the series
        aligned = pd.DataFrame({
            "portfolio": equity_curve,
            "benchmark": benchmark
        }).dropna()
        
        if len(aligned) < 2:
            return 0.0
        
        portfolio_returns = aligned["portfolio"].pct_change().dropna()
        benchmark_returns = aligned["benchmark"].pct_change().dropna()
        
        excess = portfolio_returns - benchmark_returns
        if excess.std() == 0:
            return 0.0
        
        ir = excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        return round(ir, 4)
    
    def _calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate trade-level metrics."""
        if "pnl" not in trades.columns:
            # Try to find P&L column
            for col in ["profit", "return", "P&L", "profit_loss"]:
                if col in trades.columns:
                    trades = trades.rename(columns={col: "pnl"})
                    break
        
        if "pnl" not in trades.columns:
            return {
                "total_trades": len(trades),
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "avg_trade": 0,
                "expectancy": 0,
            }
        
        pnl = trades["pnl"]
        
        winning = pnl[pnl > 0]
        losing = pnl[pnl < 0]
        
        total_wins = winning.sum() if len(winning) > 0 else 0
        total_losses = abs(losing.sum()) if len(losing) > 0 else 0
        
        metrics = {
            "total_trades": len(trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(trades) * 100 if len(trades) > 0 else 0,
            "profit_factor": total_wins / total_losses if total_losses > 0 else float('inf'),
            "avg_win": winning.mean() if len(winning) > 0 else 0,
            "avg_loss": losing.mean() if len(losing) > 0 else 0,
            "largest_win": winning.max() if len(winning) > 0 else 0,
            "largest_loss": losing.min() if len(losing) > 0 else 0,
            "avg_trade": pnl.mean() if len(pnl) > 0 else 0,
            "expectancy": pnl.mean() if len(pnl) > 0 else 0,
        }
        
        # Round numeric values
        for key in metrics:
            if isinstance(metrics[key], float):
                metrics[key] = round(metrics[key], 2)
        
        return metrics
    
    def calculate_rolling_metrics(
        self,
        equity_curve: pd.Series,
        window: int = 63  # ~3 months
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            equity_curve: Portfolio equity series
            window: Rolling window size in days
            
        Returns:
            DataFrame with rolling metrics
        """
        returns = equity_curve.pct_change().dropna()
        
        rolling = pd.DataFrame(index=returns.index)
        
        # Rolling return
        rolling["return_pct"] = returns.rolling(window).sum() * 100
        
        # Rolling volatility (annualized)
        rolling["volatility"] = returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        
        # Rolling Sharpe
        excess = returns - self.daily_rf
        rolling["sharpe"] = (
            excess.rolling(window).mean() / 
            returns.rolling(window).std() * 
            np.sqrt(TRADING_DAYS_PER_YEAR)
        )
        
        # Rolling max drawdown
        def rolling_max_dd(series):
            if len(series) < 2:
                return 0
            cum_returns = (1 + series).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            return drawdowns.min() * 100
        
        rolling["max_drawdown"] = returns.rolling(window).apply(rolling_max_dd, raw=False)
        
        return rolling


def calculate_drawdown_series(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Calculate detailed drawdown information.
    
    Returns DataFrame with:
    - drawdown: Current drawdown percentage
    - peak: Running peak value
    - drawdown_duration: Days in current drawdown
    """
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak * 100
    
    # Calculate duration in drawdown
    in_drawdown = drawdown < 0
    
    # Create groups for each drawdown period
    dd_groups = (~in_drawdown).cumsum()
    duration = in_drawdown.groupby(dd_groups).cumsum()
    
    return pd.DataFrame({
        "equity": equity_curve,
        "peak": peak,
        "drawdown": drawdown,
        "duration": duration.astype(int),
    })


def calculate_monthly_returns(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Calculate monthly return matrix.
    
    Returns DataFrame with years as rows and months as columns.
    """
    # Resample to month-end
    monthly = equity_curve.resample("M").last()
    monthly_returns = monthly.pct_change() * 100
    
    # Create matrix
    matrix = pd.DataFrame(index=monthly_returns.index.year.unique())
    
    for month in range(1, 13):
        month_name = pd.Timestamp(2000, month, 1).strftime("%b")
        month_data = monthly_returns[monthly_returns.index.month == month]
        matrix[month_name] = month_data.values[:len(matrix)]
    
    # Add yearly total
    yearly = equity_curve.resample("Y").last().pct_change() * 100
    matrix["Year"] = yearly.values[:len(matrix)]
    
    return matrix


def compare_strategies(
    equity_curves: Dict[str, pd.Series],
    risk_free_rate: float = 0.05
) -> pd.DataFrame:
    """
    Compare multiple strategies side by side.
    
    Args:
        equity_curves: Dictionary mapping strategy name to equity series
        risk_free_rate: Annual risk-free rate
        
    Returns:
        DataFrame with metrics for each strategy
    """
    calculator = MetricsCalculator(risk_free_rate)
    
    results = []
    for name, curve in equity_curves.items():
        metrics = calculator.calculate_all(curve)
        row = metrics.to_dict()
        row["strategy"] = name
        results.append(row)
    
    df = pd.DataFrame(results)
    df = df.set_index("strategy")
    
    return df
