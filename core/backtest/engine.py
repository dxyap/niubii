"""
Backtesting Engine
==================
Main backtesting engine for strategy simulation.

Provides:
- Event-driven backtesting loop
- Bar-by-bar simulation
- Equity curve generation
- Trade tracking
- Performance analysis
"""

import logging
from copy import deepcopy
from dataclasses import dataclass, field

import pandas as pd

from .costs import CostModel, CostModelConfig, SimpleCostModel
from .execution import ExecutionSimulator, PositionManager
from .metrics import MetricsCalculator, PerformanceMetrics
from .strategy import Strategy, StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    # Capital settings
    initial_capital: float = 1_000_000

    # Contract settings
    contract_multiplier: float = 1000  # Barrels per contract

    # Execution settings
    slippage_pct: float = 0.01  # Basis points
    commission_per_contract: float = 2.50

    # Risk settings
    max_position_size: int = 100  # Maximum contracts
    max_drawdown_pct: float = 0.20  # Stop trading if exceeded

    # Data settings
    price_col: str = "PX_LAST"  # Column name for close price
    open_col: str = "PX_OPEN"
    high_col: str = "PX_HIGH"
    low_col: str = "PX_LOW"

    # Output settings
    verbose: bool = True
    log_trades: bool = True


@dataclass
class BacktestResult:
    """Container for backtest results."""

    # Performance metrics
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    # Equity curve
    equity_curve: pd.Series = field(default_factory=pd.Series)

    # Trade data
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Order data
    orders: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Positions over time
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Strategy signals
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Daily returns
    returns: pd.Series = field(default_factory=pd.Series)

    # Drawdown series
    drawdown: pd.Series = field(default_factory=pd.Series)

    # Config used
    config: BacktestConfig | None = None
    strategy_name: str = ""

    def summary(self) -> dict:
        """Get summary of backtest results."""
        return {
            "strategy": self.strategy_name,
            "total_return_pct": self.metrics.total_return_pct,
            "cagr": self.metrics.cagr,
            "sharpe_ratio": self.metrics.sharpe_ratio,
            "sortino_ratio": self.metrics.sortino_ratio,
            "max_drawdown": self.metrics.max_drawdown,
            "calmar_ratio": self.metrics.calmar_ratio,
            "win_rate": self.metrics.win_rate,
            "profit_factor": self.metrics.profit_factor,
            "total_trades": self.metrics.total_trades,
            "trading_days": self.metrics.trading_days,
        }


class BacktestEngine:
    """
    Main backtesting engine.

    Runs strategies on historical data and generates performance metrics.
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        cost_model: CostModel | None = None
    ):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
            cost_model: Transaction cost model
        """
        self.config = config or BacktestConfig()

        # Create cost model from config if not provided
        if cost_model is None:
            cost_config = CostModelConfig(
                commission_per_contract=self.config.commission_per_contract,
                contract_multiplier=self.config.contract_multiplier,
            )
            cost_model = SimpleCostModel(cost_config)

        self.execution = ExecutionSimulator(
            cost_model=cost_model,
            slippage_pct=self.config.slippage_pct
        )

        self.position_manager = PositionManager(
            contract_multiplier=self.config.contract_multiplier
        )

        self.metrics_calculator = MetricsCalculator()

    def run(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        symbol: str = "CL1"
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            strategy: Strategy to test
            data: Historical OHLCV data with DatetimeIndex
            symbol: Symbol being traded

        Returns:
            BacktestResult with all results
        """
        logger.info(f"Starting backtest: {strategy.name} on {symbol}")

        # Reset state
        self.execution.reset()
        self.position_manager.reset()

        # Initialize tracking
        equity_curve = {}
        position_history = []
        capital = self.config.initial_capital

        # Determine column names
        price_col = self.config.price_col
        if price_col not in data.columns:
            # Try alternatives
            for alt in ["close", "Close", "CLOSE", "PX_LAST"]:
                if alt in data.columns:
                    price_col = alt
                    break

        # Notify strategy of start
        strategy.on_start(data)

        # Main simulation loop
        for i, (timestamp, row) in enumerate(data.iterrows()):
            if i == 0:
                equity_curve[timestamp] = capital
                continue

            # Get historical data up to current bar
            hist_data = data.iloc[:i+1].copy()

            # Get current position
            position = self.position_manager.get_position(symbol)

            # Generate orders from strategy
            orders = strategy.on_bar(timestamp, hist_data, position)

            # Submit orders
            for order in orders:
                self.execution.submit_order(order)

            # Process orders against current bar
            bar_data = {
                "open": row.get(self.config.open_col) or row.get("open") or row.get(price_col),
                "high": row.get(self.config.high_col) or row.get("high") or row.get(price_col),
                "low": row.get(self.config.low_col) or row.get("low") or row.get(price_col),
                "close": row.get(price_col),
            }

            fills = self.execution.process_bar(timestamp, bar_data)

            # Update positions
            current_price = bar_data["close"]
            for fill in fills:
                position = self.position_manager.update_position(fill, current_price)
                strategy.on_trade(fill.price, fill.quantity, fill.side)

                if self.config.log_trades:
                    logger.debug(
                        f"{timestamp}: {fill.side} {fill.quantity} @ {fill.price:.2f} "
                        f"(costs: ${fill.costs.total:.2f})"
                    )

            # Mark to market
            self.position_manager.mark_to_market({symbol: current_price})

            # Calculate equity
            pnl = self.position_manager.get_total_pnl()
            current_equity = capital + pnl["total"]
            equity_curve[timestamp] = current_equity

            # Track position
            position_history.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "quantity": position.quantity,
                "avg_price": position.avg_price,
                "unrealized_pnl": position.unrealized_pnl,
                "equity": current_equity,
            })

            # Check max drawdown
            peak = max(equity_curve.values())
            current_dd = (peak - current_equity) / peak
            if current_dd > self.config.max_drawdown_pct:
                logger.warning(
                    f"Max drawdown exceeded at {timestamp}: {current_dd:.2%}"
                )
                break

        # Notify strategy of end
        strategy.on_end()

        # Build results
        result = self._build_result(
            strategy=strategy,
            equity_curve=equity_curve,
            position_history=position_history,
            symbol=symbol
        )

        logger.info(
            f"Backtest complete: {strategy.name} | "
            f"Return: {result.metrics.total_return_pct:.2f}% | "
            f"Sharpe: {result.metrics.sharpe_ratio:.2f} | "
            f"MaxDD: {result.metrics.max_drawdown:.2f}%"
        )

        return result

    def _build_result(
        self,
        strategy: Strategy,
        equity_curve: dict,
        position_history: list,
        symbol: str
    ) -> BacktestResult:
        """Build backtest result from simulation data."""

        # Create equity series
        equity_series = pd.Series(equity_curve)
        equity_series.index = pd.to_datetime(equity_series.index)
        equity_series = equity_series.sort_index()

        # Get trades
        trades_df = self.position_manager.get_trades()

        # Get orders
        orders_df = self.execution.get_order_history()

        # Get signals
        signals_df = strategy.get_signals_df()

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all(
            equity_curve=equity_series,
            trades=trades_df if not trades_df.empty else None
        )

        # Calculate returns
        returns = equity_series.pct_change().dropna()

        # Calculate drawdown
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak * 100

        # Position history
        position_df = pd.DataFrame(position_history)
        if not position_df.empty:
            position_df.set_index("timestamp", inplace=True)

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_series,
            trades=trades_df,
            orders=orders_df,
            positions=position_df,
            signals=signals_df,
            returns=returns,
            drawdown=drawdown,
            config=self.config,
            strategy_name=strategy.name,
        )

    def run_multiple(
        self,
        strategies: list[Strategy],
        data: pd.DataFrame,
        symbol: str = "CL1"
    ) -> dict[str, BacktestResult]:
        """
        Run multiple strategies on the same data.

        Args:
            strategies: List of strategies to test
            data: Historical data
            symbol: Symbol being traded

        Returns:
            Dictionary mapping strategy names to results
        """
        results = {}

        for strategy in strategies:
            # Use fresh copy of strategy to avoid state issues
            result = self.run(deepcopy(strategy), data, symbol)
            results[strategy.name] = result

        return results

    def compare_strategies(
        self,
        results: dict[str, BacktestResult]
    ) -> pd.DataFrame:
        """
        Compare multiple backtest results.

        Args:
            results: Dictionary of backtest results

        Returns:
            Comparison DataFrame
        """
        comparisons = []

        for _name, result in results.items():
            row = result.summary()
            comparisons.append(row)

        df = pd.DataFrame(comparisons)
        df = df.set_index("strategy")

        # Sort by Sharpe ratio
        df = df.sort_values("sharpe_ratio", ascending=False)

        return df


def run_backtest(
    strategy: Strategy,
    data: pd.DataFrame,
    initial_capital: float = 1_000_000,
    commission: float = 2.50,
    slippage_pct: float = 0.01,
    symbol: str = "CL1"
) -> BacktestResult:
    """
    Convenience function to run a simple backtest.

    Args:
        strategy: Strategy to test
        data: Historical OHLCV data
        initial_capital: Starting capital
        commission: Commission per contract
        slippage_pct: Slippage in basis points
        symbol: Symbol being traded

    Returns:
        BacktestResult
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission_per_contract=commission,
        slippage_pct=slippage_pct,
    )

    engine = BacktestEngine(config)
    return engine.run(strategy, data, symbol)


def quick_backtest(
    signal_series: pd.Series,
    price_series: pd.Series,
    initial_capital: float = 1_000_000,
    position_size: int = 1
) -> BacktestResult:
    """
    Quick backtest from a signal series.

    Args:
        signal_series: Series of signals (-1, 0, 1)
        price_series: Series of prices
        initial_capital: Starting capital
        position_size: Position size in contracts

    Returns:
        BacktestResult
    """
    from .strategy import create_strategy_from_signals

    # Create DataFrame
    data = pd.DataFrame({
        "PX_LAST": price_series,
        "signal": signal_series
    })

    # Create strategy
    def signal_func(df):
        return df["signal"]

    config = StrategyConfig(position_size=position_size)
    strategy = create_strategy_from_signals(signal_func, "QuickBacktest", config)

    return run_backtest(strategy, data, initial_capital)
