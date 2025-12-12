"""
Technical Signal Generation
===========================
Technical analysis based trading signals.
"""

from datetime import datetime

import pandas as pd

from core.indicators import (
    calculate_bollinger_bands,
    calculate_rsi,
)


class TechnicalSignals:
    """
    Technical signal generation for oil trading.

    Signal Types:
    - Trend following (MA crossovers, breakouts)
    - Mean reversion (RSI, Bollinger Bands)
    - Momentum (ROC, relative strength)
    - Term structure signals
    """

    # No initialization required - all methods are stateless

    def calculate_moving_averages(
        self,
        prices: pd.Series,
        periods: list[int] = None
    ) -> pd.DataFrame:
        """
        Calculate multiple moving averages.

        Args:
            prices: Price series
            periods: MA periods

        Returns:
            DataFrame with MAs
        """
        if periods is None:
            periods = [20, 50, 200]
        mas = pd.DataFrame(index=prices.index)
        mas["close"] = prices

        for period in periods:
            mas[f"sma_{period}"] = prices.rolling(window=period).mean()
            mas[f"ema_{period}"] = prices.ewm(span=period, adjust=False).mean()

        return mas

    def ma_crossover_signal(
        self,
        prices: pd.Series,
        fast_period: int = 20,
        slow_period: int = 50
    ) -> dict:
        """
        Generate moving average crossover signal.

        Args:
            prices: Price series
            fast_period: Fast MA period
            slow_period: Slow MA period

        Returns:
            Signal dictionary
        """
        fast_ma = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ma = prices.ewm(span=slow_period, adjust=False).mean()

        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2] if len(fast_ma) > 1 else current_fast
        prev_slow = slow_ma.iloc[-2] if len(slow_ma) > 1 else current_slow

        # Signal detection
        if prev_fast <= prev_slow and current_fast > current_slow:
            signal = "BUY"
            confidence = min((current_fast - current_slow) / current_slow * 100, 100)
        elif prev_fast >= prev_slow and current_fast < current_slow:
            signal = "SELL"
            confidence = min((current_slow - current_fast) / current_slow * 100, 100)
        else:
            signal = "HOLD"
            spread = (current_fast - current_slow) / current_slow * 100
            confidence = 50 + spread * 10  # Directional bias

        return {
            "signal": signal,
            "confidence": round(max(0, min(100, confidence)), 1),
            "fast_ma": round(current_fast, 2),
            "slow_ma": round(current_slow, 2),
            "ma_spread_pct": round((current_fast - current_slow) / current_slow * 100, 2),
            "fast_period": fast_period,
            "slow_period": slow_period,
        }

    def rsi_signal(
        self,
        prices: pd.Series,
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30
    ) -> dict:
        """
        Generate RSI-based signal.

        Args:
            prices: Price series
            period: RSI period
            overbought: Overbought threshold
            oversold: Oversold threshold

        Returns:
            Signal dictionary
        """
        rsi = calculate_rsi(prices, period)
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2] if len(rsi) > 1 else current_rsi

        # Signal generation
        if current_rsi < oversold:
            if prev_rsi < current_rsi:  # Turning up from oversold
                signal = "BUY"
                confidence = (oversold - current_rsi + 10) * 2
            else:
                signal = "WATCH_BUY"
                confidence = (oversold - current_rsi) * 2
        elif current_rsi > overbought:
            if prev_rsi > current_rsi:  # Turning down from overbought
                signal = "SELL"
                confidence = (current_rsi - overbought + 10) * 2
            else:
                signal = "WATCH_SELL"
                confidence = (current_rsi - overbought) * 2
        else:
            signal = "NEUTRAL"
            confidence = 50

        return {
            "signal": signal,
            "confidence": round(max(0, min(100, confidence)), 1),
            "rsi": round(current_rsi, 1),
            "prev_rsi": round(prev_rsi, 1),
            "overbought_level": overbought,
            "oversold_level": oversold,
            "period": period,
        }

    def bollinger_band_signal(
        self,
        prices: pd.Series,
        period: int = 20,
        num_std: float = 2.0
    ) -> dict:
        """
        Generate Bollinger Band signal.

        Args:
            prices: Price series
            period: BB period
            num_std: Number of standard deviations

        Returns:
            Signal dictionary
        """
        sma, upper_band, lower_band = calculate_bollinger_bands(prices, period, num_std)

        current_price = prices.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_sma = sma.iloc[-1]

        # Calculate %B (position within bands)
        percent_b = (current_price - current_lower) / (current_upper - current_lower) * 100

        # Bandwidth (volatility indicator)
        bandwidth = (current_upper - current_lower) / current_sma * 100

        # Signal generation
        if current_price <= current_lower:
            signal = "BUY"
            confidence = (100 - percent_b) * 0.8
        elif current_price >= current_upper:
            signal = "SELL"
            confidence = percent_b * 0.8
        elif percent_b < 20:
            signal = "WATCH_BUY"
            confidence = (20 - percent_b) * 2
        elif percent_b > 80:
            signal = "WATCH_SELL"
            confidence = (percent_b - 80) * 2
        else:
            signal = "NEUTRAL"
            confidence = 50

        return {
            "signal": signal,
            "confidence": round(max(0, min(100, confidence)), 1),
            "current_price": round(current_price, 2),
            "upper_band": round(current_upper, 2),
            "lower_band": round(current_lower, 2),
            "middle_band": round(current_sma, 2),
            "percent_b": round(percent_b, 1),
            "bandwidth": round(bandwidth, 2),
        }

    def momentum_signal(
        self,
        prices: pd.Series,
        period: int = 14
    ) -> dict:
        """
        Generate momentum-based signal.

        Args:
            prices: Price series
            period: Momentum period

        Returns:
            Signal dictionary
        """
        # Rate of change
        roc = ((prices / prices.shift(period)) - 1) * 100
        current_roc = roc.iloc[-1]

        # Momentum (price change)
        momentum = prices.diff(period)
        current_momentum = momentum.iloc[-1]

        # Signal generation
        if current_roc > 5:
            signal = "STRONG_BUY"
            confidence = min(current_roc * 10, 100)
        elif current_roc > 2:
            signal = "BUY"
            confidence = current_roc * 15
        elif current_roc < -5:
            signal = "STRONG_SELL"
            confidence = min(abs(current_roc) * 10, 100)
        elif current_roc < -2:
            signal = "SELL"
            confidence = abs(current_roc) * 15
        else:
            signal = "NEUTRAL"
            confidence = 50 + current_roc * 5

        return {
            "signal": signal,
            "confidence": round(max(0, min(100, confidence)), 1),
            "roc_pct": round(current_roc, 2),
            "momentum": round(current_momentum, 2),
            "period": period,
        }

    def breakout_signal(
        self,
        prices: pd.Series,
        lookback: int = 20
    ) -> dict:
        """
        Generate breakout signal based on price channels.

        Args:
            prices: Price series
            lookback: Lookback period for channel

        Returns:
            Signal dictionary
        """
        high_channel = prices.rolling(window=lookback).max()
        low_channel = prices.rolling(window=lookback).min()

        current_price = prices.iloc[-1]
        prev_price = prices.iloc[-2]
        current_high = high_channel.iloc[-2]  # Previous high (not including current)
        current_low = low_channel.iloc[-2]

        channel_width = (current_high - current_low) / current_low * 100

        # Breakout detection
        if current_price > current_high and prev_price <= current_high:
            signal = "BREAKOUT_BUY"
            confidence = 70 + min(channel_width, 30)
        elif current_price < current_low and prev_price >= current_low:
            signal = "BREAKOUT_SELL"
            confidence = 70 + min(channel_width, 30)
        elif current_price >= current_high * 0.98:
            signal = "APPROACHING_RESISTANCE"
            confidence = 60
        elif current_price <= current_low * 1.02:
            signal = "APPROACHING_SUPPORT"
            confidence = 60
        else:
            signal = "RANGE_BOUND"
            (current_price - current_low) / (current_high - current_low) * 100
            confidence = 50

        return {
            "signal": signal,
            "confidence": round(max(0, min(100, confidence)), 1),
            "current_price": round(current_price, 2),
            "channel_high": round(current_high, 2),
            "channel_low": round(current_low, 2),
            "channel_width_pct": round(channel_width, 2),
            "lookback": lookback,
        }

    def generate_composite_signal(
        self,
        prices: pd.Series
    ) -> dict:
        """
        Generate composite technical signal from multiple indicators.

        Args:
            prices: Price series

        Returns:
            Composite signal dictionary
        """
        # Get individual signals
        ma_signal = self.ma_crossover_signal(prices)
        rsi_signal = self.rsi_signal(prices)
        bb_signal = self.bollinger_band_signal(prices)
        mom_signal = self.momentum_signal(prices)
        breakout = self.breakout_signal(prices)

        # Score signals
        signal_scores = {
            "BUY": 1, "STRONG_BUY": 1.5, "WATCH_BUY": 0.5, "BREAKOUT_BUY": 1.5,
            "SELL": -1, "STRONG_SELL": -1.5, "WATCH_SELL": -0.5, "BREAKOUT_SELL": -1.5,
            "NEUTRAL": 0, "HOLD": 0, "RANGE_BOUND": 0,
            "APPROACHING_RESISTANCE": -0.25, "APPROACHING_SUPPORT": 0.25,
        }

        # Weighted average
        weights = {
            "ma": 0.25,
            "rsi": 0.20,
            "bb": 0.20,
            "momentum": 0.20,
            "breakout": 0.15,
        }

        total_score = (
            weights["ma"] * signal_scores.get(ma_signal["signal"], 0) * (ma_signal["confidence"] / 100) +
            weights["rsi"] * signal_scores.get(rsi_signal["signal"], 0) * (rsi_signal["confidence"] / 100) +
            weights["bb"] * signal_scores.get(bb_signal["signal"], 0) * (bb_signal["confidence"] / 100) +
            weights["momentum"] * signal_scores.get(mom_signal["signal"], 0) * (mom_signal["confidence"] / 100) +
            weights["breakout"] * signal_scores.get(breakout["signal"], 0) * (breakout["confidence"] / 100)
        )

        # Determine composite signal
        if total_score > 0.4:
            composite_signal = "LONG"
        elif total_score < -0.4:
            composite_signal = "SHORT"
        else:
            composite_signal = "NEUTRAL"

        composite_confidence = abs(total_score) * 100

        return {
            "signal": composite_signal,
            "confidence": round(min(composite_confidence, 100), 1),
            "score": round(total_score, 3),
            "components": {
                "ma_crossover": ma_signal,
                "rsi": rsi_signal,
                "bollinger": bb_signal,
                "momentum": mom_signal,
                "breakout": breakout,
            },
            "timestamp": datetime.now().isoformat(),
        }
