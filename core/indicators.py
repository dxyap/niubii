"""
Technical Indicators
====================
Centralized technical indicator calculations.

This module provides reusable implementations of common technical indicators
used across the codebase (signals, backtesting, ML features).

All functions operate on pandas Series for consistency and efficiency.
"""


import numpy as np
import pandas as pd


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.

    Args:
        prices: Price series (typically close prices)
        period: RSI lookback period (default: 14)

    Returns:
        RSI series (0-100 scale)
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Bollinger Bands consist of a middle band (SMA) and upper/lower bands
    at a specified number of standard deviations from the middle.

    Args:
        prices: Price series
        period: Moving average period (default: 20)
        num_std: Number of standard deviations (default: 2.0)

    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    middle = prices.rolling(period).mean()
    std = prices.rolling(period).std()

    upper = middle + num_std * std
    lower = middle - num_std * std

    return middle, upper, lower


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        prices: Price series
        period: Averaging period

    Returns:
        SMA series
    """
    return prices.rolling(window=period).mean()


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        prices: Price series
        period: EMA period

    Returns:
        EMA series
    """
    return prices.ewm(span=period, adjust=False).mean()


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Price series
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)

    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period (default: 14)
        d_period: %D smoothing period (default: 3)

    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d_period).mean()

    return stoch_k, stoch_d


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    ATR measures market volatility by decomposing the entire range
    of an asset price for that period.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default: 14)

    Returns:
        ATR series
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def calculate_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Williams %R.

    Williams %R is a momentum indicator that measures overbought
    and oversold levels.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (default: 14)

    Returns:
        Williams %R series (-100 to 0)
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    wr = -100 * (highest_high - close) / (highest_high - lowest_low)

    return wr


def calculate_roc(prices: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Rate of Change (ROC).

    Args:
        prices: Price series
        period: ROC period (default: 10)

    Returns:
        ROC series (percentage)
    """
    return (prices / prices.shift(period) - 1) * 100


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).

    OBV uses volume flow to predict changes in stock price.

    Args:
        close: Close prices
        volume: Volume series

    Returns:
        OBV series
    """
    direction = np.sign(close.diff())
    direction.iloc[0] = 0

    obv = (direction * volume).cumsum()

    return obv


def calculate_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume series

    Returns:
        VWAP series
    """
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()

    return vwap


def calculate_momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Price Momentum.

    Args:
        prices: Price series
        period: Momentum period (default: 10)

    Returns:
        Momentum series (price difference)
    """
    return prices - prices.shift(period)


def calculate_returns(prices: pd.Series, period: int = 1) -> pd.Series:
    """
    Calculate percentage returns.

    Args:
        prices: Price series
        period: Return period (default: 1)

    Returns:
        Returns series (as decimal, e.g., 0.01 = 1%)
    """
    return prices.pct_change(period)


def calculate_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate rolling volatility.

    Args:
        returns: Returns series
        window: Rolling window (default: 20)
        annualize: Whether to annualize (default: True)

    Returns:
        Volatility series
    """
    vol = returns.rolling(window=window).std()

    if annualize:
        vol = vol * np.sqrt(252)

    return vol


def calculate_parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate Parkinson volatility using high-low range.

    More efficient volatility estimator than close-to-close.

    Args:
        high: High prices
        low: Low prices
        window: Rolling window (default: 20)
        annualize: Whether to annualize (default: True)

    Returns:
        Parkinson volatility series
    """
    log_hl = np.log(high / low)
    factor = 1 / (4 * np.log(2))

    parkinson = np.sqrt(factor * (log_hl ** 2).rolling(window=window).mean())

    if annualize:
        parkinson = parkinson * np.sqrt(252)

    return parkinson


def get_crossover_signal(
    fast: pd.Series,
    slow: pd.Series
) -> pd.Series:
    """
    Generate crossover signals.

    Args:
        fast: Fast moving series
        slow: Slow moving series

    Returns:
        Signal series: 1 for bullish cross, -1 for bearish cross, 0 otherwise
    """
    current_above = fast > slow
    prev_above = fast.shift(1) > slow.shift(1)

    bullish = current_above & ~prev_above  # Fast crosses above slow
    bearish = ~current_above & prev_above  # Fast crosses below slow

    signal = pd.Series(0, index=fast.index)
    signal[bullish] = 1
    signal[bearish] = -1

    return signal
