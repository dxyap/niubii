"""
P&L Calculation
===============
Profit and loss calculations and tracking.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple


class PnLCalculator:
    """
    P&L calculation and attribution.
    
    Features:
    - Realized/unrealized P&L
    - Daily P&L tracking
    - Attribution by strategy
    - Performance metrics
    """
    
    def __init__(self):
        """Initialize P&L calculator."""
        self.daily_pnl_history: List[Dict] = []
    
    def calculate_realized_pnl(
        self,
        trades: pd.DataFrame,
        method: str = "fifo"
    ) -> Dict:
        """
        Calculate realized P&L from closed trades.
        
        Args:
            trades: DataFrame of trades
            method: 'fifo' or 'average'
            
        Returns:
            Realized P&L breakdown
        """
        if trades.empty:
            return {"total_realized_pnl": 0, "by_instrument": {}}
        
        realized_by_instrument = {}
        
        for instrument in trades["instrument"].unique():
            inst_trades = trades[trades["instrument"] == instrument].sort_values(["trade_date", "trade_time"])
            
            # FIFO calculation
            buy_queue = []  # List of (quantity, price) tuples
            realized = 0
            
            contract_type = instrument[:2]
            multiplier = 1000 if contract_type in ["CL", "CO"] else 42000
            
            for _, trade in inst_trades.iterrows():
                qty = trade["quantity"]
                price = trade["price"]
                
                if trade["side"] == "BUY":
                    buy_queue.append((qty, price))
                else:
                    # Sell - match against buys
                    sell_qty = qty
                    while sell_qty > 0 and buy_queue:
                        buy_qty, buy_price = buy_queue[0]
                        
                        if buy_qty <= sell_qty:
                            # Fully consume this buy
                            realized += buy_qty * (price - buy_price) * multiplier
                            sell_qty -= buy_qty
                            buy_queue.pop(0)
                        else:
                            # Partially consume
                            realized += sell_qty * (price - buy_price) * multiplier
                            buy_queue[0] = (buy_qty - sell_qty, buy_price)
                            sell_qty = 0
            
            realized_by_instrument[instrument] = round(realized, 2)
        
        total = sum(realized_by_instrument.values())
        
        return {
            "total_realized_pnl": round(total, 2),
            "by_instrument": realized_by_instrument,
        }
    
    def calculate_unrealized_pnl(
        self,
        positions: Dict[str, Dict],
        current_prices: Dict[str, float]
    ) -> Dict:
        """
        Calculate unrealized P&L from open positions.
        
        Args:
            positions: Open positions
            current_prices: Current market prices
            
        Returns:
            Unrealized P&L breakdown
        """
        unrealized_by_instrument = {}
        
        for instrument, position in positions.items():
            qty = position["quantity"]
            avg_entry = position["avg_entry_price"]
            multiplier = position.get("multiplier", 1000)
            
            current_price = current_prices.get(instrument, avg_entry)
            
            unrealized = qty * (current_price - avg_entry) * multiplier
            unrealized_by_instrument[instrument] = round(unrealized, 2)
        
        total = sum(unrealized_by_instrument.values())
        
        return {
            "total_unrealized_pnl": round(total, 2),
            "by_instrument": unrealized_by_instrument,
        }
    
    def calculate_total_pnl(
        self,
        trades: pd.DataFrame,
        positions: Dict[str, Dict],
        current_prices: Dict[str, float]
    ) -> Dict:
        """
        Calculate total P&L (realized + unrealized).
        
        Args:
            trades: Trade history
            positions: Open positions
            current_prices: Current prices
            
        Returns:
            Total P&L breakdown
        """
        realized = self.calculate_realized_pnl(trades)
        unrealized = self.calculate_unrealized_pnl(positions, current_prices)
        
        # Commission
        total_commission = trades["commission"].sum() if not trades.empty else 0
        
        total_pnl = realized["total_realized_pnl"] + unrealized["total_unrealized_pnl"] - total_commission
        
        return {
            "realized_pnl": realized["total_realized_pnl"],
            "unrealized_pnl": unrealized["total_unrealized_pnl"],
            "total_commission": round(total_commission, 2),
            "net_pnl": round(total_pnl, 2),
            "realized_by_instrument": realized["by_instrument"],
            "unrealized_by_instrument": unrealized["by_instrument"],
        }
    
    def calculate_pnl_by_strategy(
        self,
        trades: pd.DataFrame,
        current_prices: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Calculate P&L attribution by strategy.
        
        Args:
            trades: Trade history
            current_prices: Current prices
            
        Returns:
            DataFrame of P&L by strategy
        """
        if trades.empty:
            return pd.DataFrame()
        
        strategies = trades["strategy"].fillna("Untagged").unique()
        
        data = []
        for strategy in strategies:
            strategy_trades = trades[trades["strategy"].fillna("Untagged") == strategy]
            
            # Calculate for this strategy
            num_trades = len(strategy_trades)
            total_volume = strategy_trades["quantity"].sum()
            total_commission = strategy_trades["commission"].sum()
            
            # Simplified P&L (using average prices)
            buys = strategy_trades[strategy_trades["side"] == "BUY"]
            sells = strategy_trades[strategy_trades["side"] == "SELL"]
            
            buy_value = (buys["quantity"] * buys["price"]).sum()
            sell_value = (sells["quantity"] * sells["price"]).sum()
            
            # Rough P&L estimate
            pnl = sell_value - buy_value
            
            data.append({
                "strategy": strategy,
                "num_trades": num_trades,
                "total_volume": int(total_volume),
                "commission": round(total_commission, 2),
                "pnl": round(pnl * 1000, 2),  # Rough estimate
            })
        
        return pd.DataFrame(data).sort_values("pnl", ascending=False)
    
    def calculate_daily_pnl(
        self,
        trades: pd.DataFrame,
        positions: Dict[str, Dict],
        current_prices: Dict[str, float],
        previous_prices: Dict[str, float] = None
    ) -> Dict:
        """
        Calculate daily P&L breakdown.
        
        Args:
            trades: Today's trades
            positions: Current positions
            current_prices: Current prices
            previous_prices: Previous day's closing prices
            
        Returns:
            Daily P&L breakdown
        """
        # P&L from today's trades
        today_realized = self.calculate_realized_pnl(trades)
        
        # P&L from position movement
        position_pnl = 0
        if previous_prices:
            for instrument, position in positions.items():
                prev_price = previous_prices.get(instrument, position["avg_entry_price"])
                curr_price = current_prices.get(instrument, prev_price)
                
                multiplier = position.get("multiplier", 1000)
                pnl = position["quantity"] * (curr_price - prev_price) * multiplier
                position_pnl += pnl
        
        # Commission
        commission = trades["commission"].sum() if not trades.empty else 0
        
        total_daily_pnl = today_realized["total_realized_pnl"] + position_pnl - commission
        
        daily_record = {
            "date": datetime.now().date().isoformat(),
            "trading_pnl": round(today_realized["total_realized_pnl"], 2),
            "position_pnl": round(position_pnl, 2),
            "commission": round(commission, 2),
            "total_pnl": round(total_daily_pnl, 2),
            "num_trades": len(trades),
        }
        
        self.daily_pnl_history.append(daily_record)
        
        return daily_record
    
    def calculate_performance_metrics(
        self,
        pnl_series: pd.Series,
        risk_free_rate: float = 0.05
    ) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            pnl_series: Series of daily P&L
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Performance metrics
        """
        if pnl_series.empty or len(pnl_series) < 2:
            return {
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
            }
        
        # Convert to returns
        returns = pnl_series / abs(pnl_series.iloc[0]) if pnl_series.iloc[0] != 0 else pnl_series
        
        # Daily risk-free rate
        rf_daily = risk_free_rate / 252
        
        # Sharpe Ratio
        excess_returns = returns.mean() - rf_daily
        if returns.std() > 0:
            sharpe = (excess_returns / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = (excess_returns / downside_returns.std()) * np.sqrt(252)
        else:
            sortino = sharpe
        
        # Maximum Drawdown
        cumulative = pnl_series.cumsum()
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        # Win Rate
        wins = (pnl_series > 0).sum()
        total = len(pnl_series)
        win_rate = wins / total * 100 if total > 0 else 0
        
        # Profit Factor
        gross_profit = pnl_series[pnl_series > 0].sum()
        gross_loss = abs(pnl_series[pnl_series < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        
        return {
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "max_drawdown": round(max_drawdown, 2),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "total_pnl": round(pnl_series.sum(), 2),
            "avg_daily_pnl": round(pnl_series.mean(), 2),
            "best_day": round(pnl_series.max(), 2),
            "worst_day": round(pnl_series.min(), 2),
        }
    
    def get_pnl_summary(
        self,
        trades: pd.DataFrame,
        positions: Dict[str, Dict],
        current_prices: Dict[str, float]
    ) -> Dict:
        """
        Get comprehensive P&L summary.
        
        Args:
            trades: Trade history
            positions: Current positions
            current_prices: Current prices
            
        Returns:
            P&L summary
        """
        total_pnl = self.calculate_total_pnl(trades, positions, current_prices)
        by_strategy = self.calculate_pnl_by_strategy(trades, current_prices)
        
        return {
            "total": total_pnl,
            "by_strategy": by_strategy.to_dict("records") if not by_strategy.empty else [],
            "timestamp": datetime.now().isoformat(),
        }
