"""
Strategy Optimization
=====================
Parameter optimization and walk-forward analysis for strategies.

Provides:
- Grid search optimization
- Walk-forward optimization
- Rolling window analysis
- Parameter sensitivity analysis
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Type
from datetime import datetime, timedelta
from itertools import product
import logging
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from .strategy import Strategy, StrategyConfig
from .engine import BacktestEngine, BacktestConfig, BacktestResult
from .metrics import MetricsCalculator, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""
    
    # Optimization target
    target_metric: str = "sharpe_ratio"  # Metric to optimize
    higher_is_better: bool = True
    
    # Walk-forward settings
    in_sample_pct: float = 0.7  # 70% in-sample
    num_folds: int = 5
    anchored: bool = False  # If True, always start from beginning
    
    # Constraints
    min_trades: int = 20  # Minimum trades for valid result
    min_sharpe: float = 0.0  # Minimum Sharpe to consider
    
    # Execution
    parallel: bool = False
    n_jobs: int = -1  # -1 = all cores


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    
    # Best parameters
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    
    # All results
    all_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Walk-forward results
    walk_forward_results: List[Dict] = field(default_factory=list)
    
    # Combined out-of-sample equity curve
    oos_equity_curve: pd.Series = field(default_factory=pd.Series)
    
    # Out-of-sample metrics
    oos_metrics: Optional[PerformanceMetrics] = None
    
    def summary(self) -> Dict:
        """Get optimization summary."""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "oos_sharpe": self.oos_metrics.sharpe_ratio if self.oos_metrics else None,
            "oos_return": self.oos_metrics.total_return_pct if self.oos_metrics else None,
            "num_combinations": len(self.all_results),
        }


class ParameterGrid:
    """
    Generates parameter combinations for grid search.
    """
    
    def __init__(self, param_space: Dict[str, List[Any]]):
        """
        Initialize parameter grid.
        
        Args:
            param_space: Dictionary mapping parameter names to lists of values
                         e.g., {"fast_period": [5, 10, 15], "slow_period": [20, 30, 40]}
        """
        self.param_space = param_space
        self._combinations = None
    
    @property
    def combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        if self._combinations is None:
            keys = list(self.param_space.keys())
            values = list(self.param_space.values())
            
            self._combinations = [
                dict(zip(keys, combo))
                for combo in product(*values)
            ]
        
        return self._combinations
    
    def __len__(self) -> int:
        return len(self.combinations)
    
    def __iter__(self):
        return iter(self.combinations)


class StrategyOptimizer:
    """
    Optimizes strategy parameters using grid search and walk-forward analysis.
    """
    
    def __init__(
        self,
        strategy_class: Type[Strategy],
        param_grid: Dict[str, List[Any]],
        config: Optional[OptimizationConfig] = None,
        backtest_config: Optional[BacktestConfig] = None
    ):
        """
        Initialize optimizer.
        
        Args:
            strategy_class: Strategy class to optimize
            param_grid: Parameter space to search
            config: Optimization configuration
            backtest_config: Backtest configuration
        """
        self.strategy_class = strategy_class
        self.param_grid = ParameterGrid(param_grid)
        self.config = config or OptimizationConfig()
        self.backtest_config = backtest_config or BacktestConfig()
        
        self.engine = BacktestEngine(self.backtest_config)
        self.metrics_calculator = MetricsCalculator()
    
    def optimize(
        self,
        data: pd.DataFrame,
        symbol: str = "CL1"
    ) -> OptimizationResult:
        """
        Run grid search optimization.
        
        Args:
            data: Historical data
            symbol: Symbol being traded
            
        Returns:
            OptimizationResult with best parameters and all results
        """
        logger.info(f"Starting optimization: {len(self.param_grid)} combinations")
        
        results = []
        
        for params in self.param_grid:
            try:
                # Create strategy with parameters
                strategy = self._create_strategy(params)
                
                # Run backtest
                result = self.engine.run(strategy, data, symbol)
                
                # Get target metric
                score = self._get_metric(result.metrics)
                
                # Check constraints
                is_valid = self._check_constraints(result.metrics)
                
                results.append({
                    **params,
                    "score": score,
                    "valid": is_valid,
                    "sharpe": result.metrics.sharpe_ratio,
                    "return_pct": result.metrics.total_return_pct,
                    "max_dd": result.metrics.max_drawdown,
                    "trades": result.metrics.total_trades,
                    "win_rate": result.metrics.win_rate,
                })
                
            except Exception as e:
                logger.warning(f"Error with params {params}: {e}")
                results.append({
                    **params,
                    "score": float('-inf'),
                    "valid": False,
                    "error": str(e),
                })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Find best parameters
        valid_results = results_df[results_df["valid"] == True]
        
        if len(valid_results) == 0:
            logger.warning("No valid results found!")
            best_params = {}
            best_score = float('-inf')
        else:
            if self.config.higher_is_better:
                best_idx = valid_results["score"].idxmax()
            else:
                best_idx = valid_results["score"].idxmin()
            
            best_row = results_df.loc[best_idx]
            best_params = {k: best_row[k] for k in self.param_grid.param_space.keys()}
            best_score = best_row["score"]
        
        logger.info(f"Best params: {best_params} with score: {best_score:.4f}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=results_df,
        )
    
    def walk_forward_optimize(
        self,
        data: pd.DataFrame,
        symbol: str = "CL1"
    ) -> OptimizationResult:
        """
        Run walk-forward optimization.
        
        Splits data into folds, optimizes on in-sample, validates on out-of-sample.
        
        Args:
            data: Historical data
            symbol: Symbol being traded
            
        Returns:
            OptimizationResult with walk-forward results
        """
        logger.info(
            f"Starting walk-forward optimization: {self.config.num_folds} folds, "
            f"{self.config.in_sample_pct:.0%} in-sample"
        )
        
        # Create folds
        folds = self._create_folds(data)
        
        walk_forward_results = []
        oos_equity_curves = []
        
        for i, (train_data, test_data) in enumerate(folds):
            logger.info(
                f"Fold {i+1}/{len(folds)}: "
                f"Train {len(train_data)} bars, Test {len(test_data)} bars"
            )
            
            # Optimize on training data
            train_result = self._optimize_single(train_data, symbol)
            
            if not train_result["best_params"]:
                logger.warning(f"No valid params found for fold {i+1}")
                continue
            
            # Create strategy with best params
            strategy = self._create_strategy(train_result["best_params"])
            
            # Test on out-of-sample data
            test_result = self.engine.run(strategy, test_data, symbol)
            
            walk_forward_results.append({
                "fold": i + 1,
                "train_start": train_data.index[0],
                "train_end": train_data.index[-1],
                "test_start": test_data.index[0],
                "test_end": test_data.index[-1],
                "best_params": train_result["best_params"],
                "is_sharpe": train_result["best_score"],
                "oos_sharpe": test_result.metrics.sharpe_ratio,
                "oos_return": test_result.metrics.total_return_pct,
                "oos_max_dd": test_result.metrics.max_drawdown,
                "oos_trades": test_result.metrics.total_trades,
            })
            
            # Store OOS equity curve
            oos_equity_curves.append(test_result.equity_curve)
        
        # Combine OOS equity curves
        if oos_equity_curves:
            oos_combined = pd.concat(oos_equity_curves)
            oos_combined = oos_combined.sort_index()
            
            # Calculate OOS metrics
            oos_metrics = self.metrics_calculator.calculate_all(oos_combined)
        else:
            oos_combined = pd.Series()
            oos_metrics = PerformanceMetrics()
        
        # Find most common best parameters
        if walk_forward_results:
            param_counts = {}
            for wf in walk_forward_results:
                params_key = str(wf["best_params"])
                param_counts[params_key] = param_counts.get(params_key, 0) + 1
            
            most_common = max(param_counts, key=param_counts.get)
            best_params = eval(most_common)
        else:
            best_params = {}
        
        return OptimizationResult(
            best_params=best_params,
            best_score=oos_metrics.sharpe_ratio,
            walk_forward_results=walk_forward_results,
            oos_equity_curve=oos_combined,
            oos_metrics=oos_metrics,
        )
    
    def _create_folds(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create train/test folds for walk-forward."""
        n = len(data)
        fold_size = n // self.config.num_folds
        
        folds = []
        
        for i in range(self.config.num_folds):
            if self.config.anchored:
                # Anchored: always start from beginning
                train_start = 0
            else:
                # Rolling: move train window
                train_start = i * fold_size
            
            # Calculate split point
            train_end = int((i + 1) * fold_size * self.config.in_sample_pct)
            if not self.config.anchored:
                train_end = train_start + int(fold_size * self.config.in_sample_pct)
            
            test_start = train_end
            test_end = (i + 1) * fold_size
            
            if test_end > n:
                test_end = n
            
            if test_start >= test_end:
                continue
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            if len(train_data) > 20 and len(test_data) > 5:
                folds.append((train_data, test_data))
        
        return folds
    
    def _optimize_single(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> Dict:
        """Optimize on a single data segment."""
        best_score = float('-inf') if self.config.higher_is_better else float('inf')
        best_params = {}
        
        for params in self.param_grid:
            try:
                strategy = self._create_strategy(params)
                result = self.engine.run(strategy, data, symbol)
                
                if not self._check_constraints(result.metrics):
                    continue
                
                score = self._get_metric(result.metrics)
                
                is_better = (
                    score > best_score if self.config.higher_is_better
                    else score < best_score
                )
                
                if is_better:
                    best_score = score
                    best_params = params.copy()
                    
            except Exception:
                continue
        
        return {"best_params": best_params, "best_score": best_score}
    
    def _create_strategy(self, params: Dict[str, Any]) -> Strategy:
        """Create strategy instance with parameters."""
        return self.strategy_class(**params)
    
    def _get_metric(self, metrics: PerformanceMetrics) -> float:
        """Get target metric value."""
        return getattr(metrics, self.config.target_metric, 0.0)
    
    def _check_constraints(self, metrics: PerformanceMetrics) -> bool:
        """Check if result meets constraints."""
        if metrics.total_trades < self.config.min_trades:
            return False
        if metrics.sharpe_ratio < self.config.min_sharpe:
            return False
        return True


def sensitivity_analysis(
    strategy_class: Type[Strategy],
    base_params: Dict[str, Any],
    param_to_vary: str,
    param_values: List[Any],
    data: pd.DataFrame,
    symbol: str = "CL1"
) -> pd.DataFrame:
    """
    Analyze sensitivity to a single parameter.
    
    Args:
        strategy_class: Strategy class
        base_params: Base parameter values
        param_to_vary: Parameter to vary
        param_values: Values to test
        data: Historical data
        symbol: Symbol
        
    Returns:
        DataFrame with results for each parameter value
    """
    engine = BacktestEngine()
    results = []
    
    for value in param_values:
        params = base_params.copy()
        params[param_to_vary] = value
        
        try:
            strategy = strategy_class(**params)
            result = engine.run(strategy, data, symbol)
            
            results.append({
                param_to_vary: value,
                "sharpe": result.metrics.sharpe_ratio,
                "return_pct": result.metrics.total_return_pct,
                "max_dd": result.metrics.max_drawdown,
                "win_rate": result.metrics.win_rate,
                "trades": result.metrics.total_trades,
            })
        except Exception as e:
            results.append({
                param_to_vary: value,
                "error": str(e),
            })
    
    return pd.DataFrame(results)


def monte_carlo_analysis(
    result: BacktestResult,
    n_simulations: int = 1000,
    block_size: int = 5
) -> Dict:
    """
    Monte Carlo analysis using block bootstrap.
    
    Resamples trade returns to estimate confidence intervals.
    
    Args:
        result: Backtest result
        n_simulations: Number of simulations
        block_size: Block size for bootstrap
        
    Returns:
        Dictionary with confidence intervals
    """
    if result.trades.empty or "pnl" not in result.trades.columns:
        return {"error": "No trade data available"}
    
    trade_pnl = result.trades["pnl"].values
    n_trades = len(trade_pnl)
    
    if n_trades < block_size:
        return {"error": "Not enough trades for analysis"}
    
    simulated_returns = []
    simulated_sharpes = []
    simulated_max_dd = []
    
    for _ in range(n_simulations):
        # Block bootstrap
        n_blocks = n_trades // block_size + 1
        blocks = []
        
        for _ in range(n_blocks):
            start = np.random.randint(0, n_trades - block_size + 1)
            blocks.extend(trade_pnl[start:start + block_size])
        
        sim_pnl = np.array(blocks[:n_trades])
        
        # Calculate metrics
        cumulative = np.cumsum(sim_pnl)
        total_return = cumulative[-1] / result.config.initial_capital * 100
        simulated_returns.append(total_return)
        
        # Sharpe (simplified)
        if np.std(sim_pnl) > 0:
            sharpe = np.mean(sim_pnl) / np.std(sim_pnl) * np.sqrt(252)
            simulated_sharpes.append(sharpe)
        
        # Max drawdown
        peak = np.maximum.accumulate(result.config.initial_capital + cumulative)
        dd = (peak - (result.config.initial_capital + cumulative)) / peak * 100
        simulated_max_dd.append(np.max(dd))
    
    return {
        "return_5th": np.percentile(simulated_returns, 5),
        "return_50th": np.percentile(simulated_returns, 50),
        "return_95th": np.percentile(simulated_returns, 95),
        "sharpe_5th": np.percentile(simulated_sharpes, 5) if simulated_sharpes else 0,
        "sharpe_50th": np.percentile(simulated_sharpes, 50) if simulated_sharpes else 0,
        "sharpe_95th": np.percentile(simulated_sharpes, 95) if simulated_sharpes else 0,
        "max_dd_5th": np.percentile(simulated_max_dd, 5),
        "max_dd_50th": np.percentile(simulated_max_dd, 50),
        "max_dd_95th": np.percentile(simulated_max_dd, 95),
    }
