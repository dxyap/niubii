"""
Value at Risk (VaR) Calculations
================================
VaR and risk metric calculations for oil portfolios.
"""


import numpy as np
import pandas as pd
from scipy import stats


class VaRCalculator:
    """
    Value at Risk calculator for oil trading portfolios.

    Methods:
    - Parametric VaR (variance-covariance)
    - Historical VaR
    - Monte Carlo VaR
    - Expected Shortfall (CVaR)
    """

    # Contract multipliers for notional calculation
    CONTRACT_SPECS = {
        "CL": {"multiplier": 1000, "currency": "USD"},  # 1000 barrels
        "CO": {"multiplier": 1000, "currency": "USD"},
        "XB": {"multiplier": 42000, "currency": "USD"},  # 42000 gallons
        "HO": {"multiplier": 42000, "currency": "USD"},
        "QS": {"multiplier": 100, "currency": "USD"},  # 100 tonnes
    }

    def __init__(self, confidence_level: float = 0.95, holding_period: int = 1):
        """
        Initialize VaR calculator.

        Args:
            confidence_level: VaR confidence level (e.g., 0.95 for 95%)
            holding_period: Holding period in days
        """
        self.confidence_level = confidence_level
        self.holding_period = holding_period

    def calculate_parametric_var(
        self,
        positions: dict[str, dict],
        returns: pd.DataFrame,
        correlation_matrix: pd.DataFrame | None = None
    ) -> dict:
        """
        Calculate parametric (variance-covariance) VaR.

        Args:
            positions: Dict of positions {ticker: {quantity, price}}
            returns: DataFrame of historical returns
            correlation_matrix: Optional correlation matrix

        Returns:
            VaR calculation results
        """
        if not positions:
            return {"var": 0, "method": "parametric"}

        tickers = list(positions.keys())

        # Calculate position values
        position_values = []
        for ticker in tickers:
            pos = positions[ticker]
            contract_type = ticker[:2]
            spec = self.CONTRACT_SPECS.get(contract_type, {"multiplier": 1000})
            value = pos["quantity"] * pos["price"] * spec["multiplier"]
            position_values.append(value)

        position_values = np.array(position_values)
        portfolio_value = np.sum(np.abs(position_values))

        # Get returns for positions
        available_returns = returns[[t for t in tickers if t in returns.columns]]

        if available_returns.empty:
            # Use assumed volatility if no historical data
            volatility = 0.02  # 2% daily volatility assumption
            var = portfolio_value * volatility * stats.norm.ppf(self.confidence_level)
        else:
            # Calculate portfolio volatility
            cov_matrix = available_returns.cov() * 252  # Annualized

            # Weight vector (normalized position values)
            weights = position_values / np.sum(np.abs(position_values))

            # Ensure dimensions match
            n_assets = len(weights)
            cov_values = cov_matrix.values if hasattr(cov_matrix, 'values') else cov_matrix

            if cov_values.shape[0] < n_assets:
                # Pad with assumed volatility
                missing = n_assets - cov_values.shape[0]
                pad_vol = np.eye(missing) * 0.02**2
                cov_values = np.pad(cov_values, ((0, missing), (0, missing)))
                cov_values[-missing:, -missing:] = pad_vol

            # Portfolio variance
            portfolio_variance = np.dot(weights.T, np.dot(cov_values[:n_assets, :n_assets], weights))
            portfolio_std = np.sqrt(portfolio_variance)

            # Daily volatility
            daily_vol = portfolio_std / np.sqrt(252)

            # VaR calculation
            z_score = stats.norm.ppf(self.confidence_level)
            var = portfolio_value * daily_vol * z_score * np.sqrt(self.holding_period)

        return {
            "var": round(abs(var), 2),
            "var_pct": round(abs(var) / portfolio_value * 100, 2) if portfolio_value > 0 else 0,
            "portfolio_value": round(portfolio_value, 2),
            "method": "parametric",
            "confidence_level": self.confidence_level,
            "holding_period": self.holding_period,
        }

    def calculate_historical_var(
        self,
        positions: dict[str, dict],
        returns: pd.DataFrame,
        lookback_days: int = 252
    ) -> dict:
        """
        Calculate historical VaR using actual return distribution.

        Args:
            positions: Dict of positions
            returns: Historical returns DataFrame
            lookback_days: Number of days to use

        Returns:
            VaR calculation results
        """
        if not positions or returns.empty:
            return {"var": 0, "method": "historical"}

        # Calculate position values
        portfolio_value = 0
        position_returns = []

        for ticker, pos in positions.items():
            contract_type = ticker[:2]
            spec = self.CONTRACT_SPECS.get(contract_type, {"multiplier": 1000})
            value = pos["quantity"] * pos["price"] * spec["multiplier"]
            portfolio_value += abs(value)

            if ticker in returns.columns:
                # Weight by position value
                weighted_returns = returns[ticker].tail(lookback_days) * (value / abs(value) if value != 0 else 1)
                position_returns.append(weighted_returns)

        if not position_returns:
            return {"var": 0, "method": "historical"}

        # Combine returns
        portfolio_returns = pd.concat(position_returns, axis=1).sum(axis=1)

        # Calculate VaR at percentile
        var_percentile = (1 - self.confidence_level) * 100
        var = np.percentile(portfolio_returns * portfolio_value, var_percentile)

        return {
            "var": round(abs(var), 2),
            "var_pct": round(abs(var) / portfolio_value * 100, 2) if portfolio_value > 0 else 0,
            "portfolio_value": round(portfolio_value, 2),
            "method": "historical",
            "confidence_level": self.confidence_level,
            "holding_period": self.holding_period,
            "lookback_days": lookback_days,
        }

    def calculate_expected_shortfall(
        self,
        positions: dict[str, dict],
        returns: pd.DataFrame,
        lookback_days: int = 252
    ) -> dict:
        """
        Calculate Expected Shortfall (CVaR) - average loss beyond VaR.

        Args:
            positions: Dict of positions
            returns: Historical returns DataFrame
            lookback_days: Number of days to use

        Returns:
            Expected Shortfall results
        """
        if not positions or returns.empty:
            return {"cvar": 0, "var": 0}

        # Calculate position values
        portfolio_value = 0
        position_returns = []

        for ticker, pos in positions.items():
            contract_type = ticker[:2]
            spec = self.CONTRACT_SPECS.get(contract_type, {"multiplier": 1000})
            value = pos["quantity"] * pos["price"] * spec["multiplier"]
            portfolio_value += abs(value)

            if ticker in returns.columns:
                weighted_returns = returns[ticker].tail(lookback_days) * np.sign(pos["quantity"])
                position_returns.append(weighted_returns)

        if not position_returns:
            return {"cvar": 0, "var": 0}

        portfolio_returns = pd.concat(position_returns, axis=1).sum(axis=1)

        # Calculate VaR
        var_percentile = (1 - self.confidence_level) * 100
        var = np.percentile(portfolio_returns, var_percentile)

        # Expected Shortfall (average of losses beyond VaR)
        losses_beyond_var = portfolio_returns[portfolio_returns <= var]
        cvar = losses_beyond_var.mean() if len(losses_beyond_var) > 0 else var

        return {
            "cvar": round(abs(cvar * portfolio_value), 2),
            "cvar_pct": round(abs(cvar) * 100, 2),
            "var": round(abs(var * portfolio_value), 2),
            "var_pct": round(abs(var) * 100, 2),
            "portfolio_value": round(portfolio_value, 2),
            "method": "expected_shortfall",
            "confidence_level": self.confidence_level,
        }

    def run_stress_test(
        self,
        positions: dict[str, dict],
        scenarios: dict[str, dict]
    ) -> pd.DataFrame:
        """
        Run stress tests on portfolio.

        Args:
            positions: Dict of positions
            scenarios: Dict of stress scenarios

        Returns:
            DataFrame of stress test results
        """
        results = []

        # Calculate base portfolio value
        base_value = 0
        for ticker, pos in positions.items():
            contract_type = ticker[:2]
            spec = self.CONTRACT_SPECS.get(contract_type, {"multiplier": 1000})
            value = pos["quantity"] * pos["price"] * spec["multiplier"]
            base_value += value

        for scenario_name, scenario in scenarios.items():
            shocked_value = 0

            for ticker, pos in positions.items():
                contract_type = ticker[:2]
                spec = self.CONTRACT_SPECS.get(contract_type, {"multiplier": 1000})

                # Apply shock factor
                shock = scenario.get("factors", {}).get("crude_oil", 0)
                if contract_type in ["XB", "HO"]:
                    shock = scenario.get("factors", {}).get("products", shock)

                shocked_price = pos["price"] * (1 + shock)
                value = pos["quantity"] * shocked_price * spec["multiplier"]
                shocked_value += value

            pnl = shocked_value - base_value

            results.append({
                "scenario": scenario_name,
                "description": scenario.get("description", ""),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / abs(base_value) * 100, 2) if base_value != 0 else 0,
                "shocked_value": round(shocked_value, 2),
            })

        return pd.DataFrame(results)

    def calculate_marginal_var(
        self,
        positions: dict[str, dict],
        returns: pd.DataFrame,
        new_position: dict
    ) -> dict:
        """
        Calculate marginal VaR contribution of a new position.

        Args:
            positions: Current positions
            returns: Historical returns
            new_position: New position to add

        Returns:
            Marginal VaR analysis
        """
        # Current VaR
        current_var = self.calculate_parametric_var(positions, returns)

        # Add new position
        new_positions = positions.copy()
        new_ticker = new_position["ticker"]
        new_positions[new_ticker] = {
            "quantity": new_position.get("quantity", 0),
            "price": new_position.get("price", 0),
        }

        # New VaR
        new_var = self.calculate_parametric_var(new_positions, returns)

        # Marginal VaR
        marginal_var = new_var["var"] - current_var["var"]

        return {
            "current_var": current_var["var"],
            "new_var": new_var["var"],
            "marginal_var": round(marginal_var, 2),
            "var_impact_pct": round(marginal_var / current_var["var"] * 100, 2) if current_var["var"] > 0 else 0,
        }
