"""
Strategy Framework
==================
Base classes and example strategies for backtesting.

Provides:
- Abstract strategy interface
- Signal-based strategies
- Technical indicator strategies
- Spread trading strategies
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime


class Signal(Enum):
    """Trading signal enumeration."""
    LONG = 1
    SHORT = -1
    FLAT = 0
    HOLD = None  # No change


@dataclass
class Order:
    """Order representation for backtesting."""
    timestamp: datetime
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: int
    order_type: str = "MARKET"  # "MARKET", "LIMIT", "STOP"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"  # "DAY", "GTC", "IOC"
    
    def __post_init__(self):
        if self.side not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid side: {self.side}")
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive: {self.quantity}")


@dataclass
class Position:
    """Position representation."""
    symbol: str
    quantity: int = 0  # Positive for long, negative for short
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        return self.quantity == 0
    
    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.avg_price


@dataclass
class StrategyConfig:
    """Configuration for strategies."""
    # Position sizing
    position_size: int = 1  # Default contracts
    max_position: int = 10  # Maximum position size
    
    # Risk parameters
    stop_loss_pct: Optional[float] = None  # Stop loss percentage
    take_profit_pct: Optional[float] = None  # Take profit percentage
    max_drawdown_pct: float = 0.20  # Maximum drawdown before stopping
    
    # Execution
    use_stop_orders: bool = False
    use_limit_orders: bool = False
    
    # Extra parameters (strategy-specific)
    params: Dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    
    Strategies generate trading signals based on market data and
    manage positions according to their logic.
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        """Initialize strategy with configuration."""
        self.config = config or StrategyConfig()
        self.name = self.__class__.__name__
        self._position: Position = Position(symbol="")
        self._orders: List[Order] = []
        self._signals: List[Dict] = []
        self._initialized = False
    
    @abstractmethod
    def generate_signal(
        self,
        timestamp: datetime,
        data: pd.DataFrame,
        position: Position
    ) -> Signal:
        """
        Generate trading signal based on current data.
        
        Args:
            timestamp: Current timestamp
            data: Historical data up to current timestamp
            position: Current position
            
        Returns:
            Signal indicating desired position direction
        """
        pass
    
    def on_bar(
        self,
        timestamp: datetime,
        data: pd.DataFrame,
        position: Position
    ) -> List[Order]:
        """
        Called on each bar during backtesting.
        
        Generates signals and converts them to orders.
        
        Args:
            timestamp: Current timestamp
            data: Historical data up to current timestamp
            position: Current position
            
        Returns:
            List of orders to execute
        """
        self._position = position
        
        # Generate signal
        signal = self.generate_signal(timestamp, data, position)
        
        # Record signal
        self._signals.append({
            "timestamp": timestamp,
            "signal": signal,
            "position": position.quantity
        })
        
        # Convert signal to orders
        orders = self._signal_to_orders(timestamp, signal, position, data)
        
        return orders
    
    def _signal_to_orders(
        self,
        timestamp: datetime,
        signal: Signal,
        position: Position,
        data: pd.DataFrame
    ) -> List[Order]:
        """Convert signal to orders."""
        orders = []
        
        if signal == Signal.HOLD or signal is None:
            return orders
        
        symbol = position.symbol or "CL1"
        current_qty = position.quantity
        target_qty = signal.value * self.config.position_size
        
        # Calculate required trade
        trade_qty = target_qty - current_qty
        
        if trade_qty == 0:
            return orders
        
        side = "BUY" if trade_qty > 0 else "SELL"
        quantity = abs(trade_qty)
        
        # Respect max position
        if abs(target_qty) > self.config.max_position:
            max_trade = self.config.max_position - abs(current_qty)
            quantity = min(quantity, max_trade)
            if quantity <= 0:
                return orders
        
        orders.append(Order(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type="MARKET"
        ))
        
        return orders
    
    def on_trade(self, fill_price: float, fill_qty: int, side: str):
        """Called when an order is filled."""
        pass
    
    def on_start(self, data: pd.DataFrame):
        """Called at the start of backtesting."""
        self._initialized = True
    
    def on_end(self):
        """Called at the end of backtesting."""
        pass
    
    def get_signals_df(self) -> pd.DataFrame:
        """Get signals as DataFrame."""
        if not self._signals:
            return pd.DataFrame()
        return pd.DataFrame(self._signals)


class BuyAndHoldStrategy(Strategy):
    """Simple buy and hold strategy."""
    
    def generate_signal(
        self,
        timestamp: datetime,
        data: pd.DataFrame,
        position: Position
    ) -> Signal:
        """Always go long."""
        if position.is_flat:
            return Signal.LONG
        return Signal.HOLD


class MACrossoverStrategy(Strategy):
    """
    Moving average crossover strategy.
    
    Goes long when fast MA crosses above slow MA.
    Goes short when fast MA crosses below slow MA.
    """
    
    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        config: Optional[StrategyConfig] = None
    ):
        super().__init__(config)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.name = f"MACrossover({fast_period},{slow_period})"
    
    def generate_signal(
        self,
        timestamp: datetime,
        data: pd.DataFrame,
        position: Position
    ) -> Signal:
        """Generate signal based on MA crossover."""
        if len(data) < self.slow_period + 1:
            return Signal.HOLD
        
        # Calculate MAs
        prices = data["PX_LAST"] if "PX_LAST" in data.columns else data["close"]
        fast_ma = prices.rolling(self.fast_period).mean()
        slow_ma = prices.rolling(self.slow_period).mean()
        
        # Get current and previous values
        fast_curr = fast_ma.iloc[-1]
        slow_curr = slow_ma.iloc[-1]
        fast_prev = fast_ma.iloc[-2]
        slow_prev = slow_ma.iloc[-2]
        
        # Check for crossover
        if fast_prev <= slow_prev and fast_curr > slow_curr:
            return Signal.LONG
        elif fast_prev >= slow_prev and fast_curr < slow_curr:
            return Signal.SHORT
        
        return Signal.HOLD


class RSIMeanReversionStrategy(Strategy):
    """
    RSI-based mean reversion strategy.
    
    Buys when RSI is oversold, sells when RSI is overbought.
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        oversold_level: float = 30,
        overbought_level: float = 70,
        config: Optional[StrategyConfig] = None
    ):
        super().__init__(config)
        self.rsi_period = rsi_period
        self.oversold = oversold_level
        self.overbought = overbought_level
        self.name = f"RSIMeanReversion({rsi_period})"
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signal(
        self,
        timestamp: datetime,
        data: pd.DataFrame,
        position: Position
    ) -> Signal:
        """Generate signal based on RSI levels."""
        if len(data) < self.rsi_period + 1:
            return Signal.HOLD
        
        prices = data["PX_LAST"] if "PX_LAST" in data.columns else data["close"]
        rsi = self._calculate_rsi(prices)
        current_rsi = rsi.iloc[-1]
        
        if np.isnan(current_rsi):
            return Signal.HOLD
        
        if current_rsi < self.oversold:
            return Signal.LONG
        elif current_rsi > self.overbought:
            return Signal.SHORT
        elif position.is_long and current_rsi > 50:
            return Signal.FLAT  # Exit long
        elif position.is_short and current_rsi < 50:
            return Signal.FLAT  # Exit short
        
        return Signal.HOLD


class BollingerBandStrategy(Strategy):
    """
    Bollinger Band mean reversion strategy.
    
    Buys when price touches lower band, sells when price touches upper band.
    """
    
    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        config: Optional[StrategyConfig] = None
    ):
        super().__init__(config)
        self.period = period
        self.num_std = num_std
        self.name = f"BollingerBand({period},{num_std})"
    
    def generate_signal(
        self,
        timestamp: datetime,
        data: pd.DataFrame,
        position: Position
    ) -> Signal:
        """Generate signal based on Bollinger Bands."""
        if len(data) < self.period + 1:
            return Signal.HOLD
        
        prices = data["PX_LAST"] if "PX_LAST" in data.columns else data["close"]
        
        # Calculate bands
        ma = prices.rolling(self.period).mean()
        std = prices.rolling(self.period).std()
        upper = ma + self.num_std * std
        lower = ma - self.num_std * std
        
        current_price = prices.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        current_ma = ma.iloc[-1]
        
        # Entry signals
        if current_price <= current_lower:
            return Signal.LONG
        elif current_price >= current_upper:
            return Signal.SHORT
        
        # Exit signals (mean reversion)
        if position.is_long and current_price >= current_ma:
            return Signal.FLAT
        elif position.is_short and current_price <= current_ma:
            return Signal.FLAT
        
        return Signal.HOLD


class MomentumStrategy(Strategy):
    """
    Momentum/trend following strategy.
    
    Goes long when price is above recent high, short when below recent low.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        breakout_pct: float = 0.0,  # Percentage above/below for breakout
        config: Optional[StrategyConfig] = None
    ):
        super().__init__(config)
        self.lookback = lookback
        self.breakout_pct = breakout_pct
        self.name = f"Momentum({lookback})"
    
    def generate_signal(
        self,
        timestamp: datetime,
        data: pd.DataFrame,
        position: Position
    ) -> Signal:
        """Generate signal based on breakout."""
        if len(data) < self.lookback + 1:
            return Signal.HOLD
        
        prices = data["PX_LAST"] if "PX_LAST" in data.columns else data["close"]
        high_col = data["PX_HIGH"] if "PX_HIGH" in data.columns else prices
        low_col = data["PX_LOW"] if "PX_LOW" in data.columns else prices
        
        # Get recent range (excluding current bar)
        recent_high = high_col.iloc[-self.lookback-1:-1].max()
        recent_low = low_col.iloc[-self.lookback-1:-1].min()
        
        current_price = prices.iloc[-1]
        
        # Calculate breakout levels
        upper_level = recent_high * (1 + self.breakout_pct / 100)
        lower_level = recent_low * (1 - self.breakout_pct / 100)
        
        if current_price > upper_level:
            return Signal.LONG
        elif current_price < lower_level:
            return Signal.SHORT
        
        return Signal.HOLD


class CalendarSpreadStrategy(Strategy):
    """
    Calendar spread strategy for futures.
    
    Trades the spread between two contract months.
    Goes long when spread is below mean, short when above.
    """
    
    def __init__(
        self,
        spread_col: str = "spread",
        lookback: int = 20,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        config: Optional[StrategyConfig] = None
    ):
        super().__init__(config)
        self.spread_col = spread_col
        self.lookback = lookback
        self.entry_z = entry_zscore
        self.exit_z = exit_zscore
        self.name = f"CalendarSpread(z={entry_zscore})"
    
    def generate_signal(
        self,
        timestamp: datetime,
        data: pd.DataFrame,
        position: Position
    ) -> Signal:
        """Generate signal based on spread z-score."""
        if self.spread_col not in data.columns:
            return Signal.HOLD
        
        if len(data) < self.lookback + 1:
            return Signal.HOLD
        
        spread = data[self.spread_col]
        
        # Calculate z-score
        mean = spread.rolling(self.lookback).mean().iloc[-1]
        std = spread.rolling(self.lookback).std().iloc[-1]
        
        if std == 0 or np.isnan(std):
            return Signal.HOLD
        
        current_z = (spread.iloc[-1] - mean) / std
        
        # Entry signals
        if current_z < -self.entry_z:
            return Signal.LONG  # Spread is cheap, buy
        elif current_z > self.entry_z:
            return Signal.SHORT  # Spread is expensive, sell
        
        # Exit signals
        if position.is_long and current_z > -self.exit_z:
            return Signal.FLAT
        elif position.is_short and current_z < self.exit_z:
            return Signal.FLAT
        
        return Signal.HOLD


class CompositeStrategy(Strategy):
    """
    Combines multiple strategies with weighted voting.
    
    Aggregates signals from multiple strategies and uses
    weighted voting to determine final signal.
    """
    
    def __init__(
        self,
        strategies: List[Tuple[Strategy, float]],  # (strategy, weight) pairs
        threshold: float = 0.5,  # Threshold for signal
        config: Optional[StrategyConfig] = None
    ):
        super().__init__(config)
        self.strategies = strategies
        self.threshold = threshold
        self.name = "CompositeStrategy"
    
    def generate_signal(
        self,
        timestamp: datetime,
        data: pd.DataFrame,
        position: Position
    ) -> Signal:
        """Aggregate signals from all strategies."""
        total_weight = sum(w for _, w in self.strategies)
        if total_weight == 0:
            return Signal.HOLD
        
        weighted_signal = 0.0
        
        for strategy, weight in self.strategies:
            signal = strategy.generate_signal(timestamp, data, position)
            if signal != Signal.HOLD and signal is not None:
                weighted_signal += signal.value * weight
        
        normalized = weighted_signal / total_weight
        
        if normalized > self.threshold:
            return Signal.LONG
        elif normalized < -self.threshold:
            return Signal.SHORT
        elif abs(normalized) < self.threshold / 2:
            return Signal.FLAT
        
        return Signal.HOLD


def create_strategy_from_signals(
    signal_func: Callable[[pd.DataFrame], pd.Series],
    name: str = "CustomSignalStrategy",
    config: Optional[StrategyConfig] = None
) -> Strategy:
    """
    Create a strategy from a signal-generating function.
    
    Args:
        signal_func: Function that takes DataFrame and returns Series of signals (-1, 0, 1)
        name: Strategy name
        config: Strategy configuration
        
    Returns:
        Strategy instance
    """
    class CustomSignalStrategy(Strategy):
        def __init__(self):
            super().__init__(config)
            self.name = name
            self._signal_func = signal_func
        
        def generate_signal(
            self,
            timestamp: datetime,
            data: pd.DataFrame,
            position: Position
        ) -> Signal:
            signals = self._signal_func(data)
            if len(signals) == 0:
                return Signal.HOLD
            
            current_signal = signals.iloc[-1]
            
            if current_signal > 0:
                return Signal.LONG
            elif current_signal < 0:
                return Signal.SHORT
            elif current_signal == 0 and not position.is_flat:
                return Signal.FLAT
            
            return Signal.HOLD
    
    return CustomSignalStrategy()
