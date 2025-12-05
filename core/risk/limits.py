"""
Risk Limits Management
======================
Position and portfolio limit checking and management.
"""

from datetime import datetime
from pathlib import Path

import yaml


class RiskLimits:
    """
    Risk limit management and checking.

    Features:
    - Position limit checking
    - VaR limit monitoring
    - Concentration limits
    - Drawdown limits
    """

    def __init__(self, config_path: str = "config/risk_limits.yaml"):
        """
        Initialize risk limits from configuration.

        Args:
            config_path: Path to risk limits configuration
        """
        self.config_path = Path(config_path)
        self.limits = self._load_limits()

        # Alert levels
        self.WARNING_LEVEL = 0.75
        self.CRITICAL_LEVEL = 0.90

    def _load_limits(self) -> dict:
        """Load limits from configuration file."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)

        # Default limits
        return {
            "portfolio_limits": {
                "max_var_95_1d": 375000,
                "max_drawdown_daily": 0.05,
                "max_drawdown_weekly": 0.08,
                "max_gross_exposure": 20000000,
                "max_net_exposure": 15000000,
            },
            "position_limits": {
                "WTI_CL": {"max_contracts": 100, "max_notional": 8000000},
                "Brent_CO": {"max_contracts": 75, "max_notional": 6000000},
                "RBOB_XB": {"max_contracts": 50, "max_notional": 4000000},
                "HeatOil_HO": {"max_contracts": 50, "max_notional": 4000000},
            },
            "concentration_limits": {
                "max_single_instrument": 0.40,
                "max_correlated_exposure": 0.60,
            },
        }

    def check_position_limit(
        self,
        ticker: str,
        current_quantity: int,
        proposed_quantity: int,
        price: float
    ) -> dict:
        """
        Check if position is within limits.

        Args:
            ticker: Instrument ticker
            current_quantity: Current position
            proposed_quantity: Proposed new position
            price: Current price

        Returns:
            Limit check result
        """
        # Map ticker to limit key
        ticker_prefix = ticker[:2] if len(ticker) >= 2 else ticker
        limit_key_map = {
            "CL": "WTI_CL",
            "CO": "Brent_CO",
            "XB": "RBOB_XB",
            "HO": "HeatOil_HO",
        }
        limit_key = limit_key_map.get(ticker_prefix, "WTI_CL")

        limits = self.limits.get("position_limits", {}).get(limit_key, {})
        max_contracts = limits.get("max_contracts", 100)
        max_notional = limits.get("max_notional", 8000000)

        # Calculate new position
        new_quantity = current_quantity + proposed_quantity

        # Contract specs for notional
        multiplier = 1000 if ticker_prefix in ["CL", "CO"] else 42000
        new_notional = abs(new_quantity) * price * multiplier

        # Check limits
        contract_util = abs(new_quantity) / max_contracts * 100
        notional_util = new_notional / max_notional * 100

        # Determine status
        max_util = max(contract_util, notional_util)
        if max_util >= 100:
            status = "BREACH"
            approved = False
        elif max_util >= self.CRITICAL_LEVEL * 100:
            status = "CRITICAL"
            approved = True
        elif max_util >= self.WARNING_LEVEL * 100:
            status = "WARNING"
            approved = True
        else:
            status = "OK"
            approved = True

        return {
            "ticker": ticker,
            "current_quantity": current_quantity,
            "proposed_quantity": proposed_quantity,
            "new_quantity": new_quantity,
            "max_contracts": max_contracts,
            "contract_utilization_pct": round(contract_util, 1),
            "new_notional": round(new_notional, 2),
            "max_notional": max_notional,
            "notional_utilization_pct": round(notional_util, 1),
            "status": status,
            "approved": approved,
        }

    def check_var_limit(
        self,
        current_var: float,
        proposed_var_impact: float = 0
    ) -> dict:
        """
        Check VaR against limit.

        Args:
            current_var: Current portfolio VaR
            proposed_var_impact: VaR impact of proposed trade

        Returns:
            VaR limit check result
        """
        max_var = self.limits.get("portfolio_limits", {}).get("max_var_95_1d", 375000)

        new_var = current_var + proposed_var_impact
        utilization = new_var / max_var * 100

        if utilization >= 100:
            status = "BREACH"
            approved = False
        elif utilization >= self.CRITICAL_LEVEL * 100:
            status = "CRITICAL"
            approved = True
        elif utilization >= self.WARNING_LEVEL * 100:
            status = "WARNING"
            approved = True
        else:
            status = "OK"
            approved = True

        return {
            "current_var": round(current_var, 2),
            "proposed_var_impact": round(proposed_var_impact, 2),
            "new_var": round(new_var, 2),
            "max_var": max_var,
            "utilization_pct": round(utilization, 1),
            "status": status,
            "approved": approved,
        }

    def check_drawdown_limit(
        self,
        current_drawdown: float,
        period: str = "daily"
    ) -> dict:
        """
        Check drawdown against limit.

        Args:
            current_drawdown: Current drawdown (as decimal, e.g., 0.03 for 3%)
            period: Period ('daily', 'weekly', 'monthly')

        Returns:
            Drawdown limit check result
        """
        limit_key = f"max_drawdown_{period}"
        max_drawdown = self.limits.get("portfolio_limits", {}).get(limit_key, 0.05)

        utilization = abs(current_drawdown) / max_drawdown * 100

        if utilization >= 100:
            status = "BREACH"
        elif utilization >= self.CRITICAL_LEVEL * 100:
            status = "CRITICAL"
        elif utilization >= self.WARNING_LEVEL * 100:
            status = "WARNING"
        else:
            status = "OK"

        return {
            "current_drawdown_pct": round(current_drawdown * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "utilization_pct": round(utilization, 1),
            "period": period,
            "status": status,
        }

    def check_concentration(
        self,
        positions: dict[str, dict],
        portfolio_value: float
    ) -> dict:
        """
        Check concentration limits.

        Args:
            positions: Dict of positions
            portfolio_value: Total portfolio value

        Returns:
            Concentration check result
        """
        if portfolio_value <= 0:
            return {"status": "OK", "concentrations": {}}

        max_single = self.limits.get("concentration_limits", {}).get("max_single_instrument", 0.40)
        max_correlated = self.limits.get("concentration_limits", {}).get("max_correlated_exposure", 0.60)

        # Calculate position weights
        position_values = {}
        for ticker, pos in positions.items():
            contract_type = ticker[:2]
            multiplier = 1000 if contract_type in ["CL", "CO"] else 42000
            value = pos["quantity"] * pos["price"] * multiplier
            position_values[ticker] = value

        # Check single instrument concentration
        concentrations = {}
        breaches = []

        for ticker, value in position_values.items():
            concentration = abs(value) / portfolio_value
            concentrations[ticker] = round(concentration * 100, 1)

            if concentration > max_single:
                breaches.append({
                    "ticker": ticker,
                    "concentration_pct": round(concentration * 100, 1),
                    "limit_pct": round(max_single * 100, 1),
                    "type": "single_instrument",
                })

        # Check correlated exposure (crude oil group)
        crude_exposure = sum(
            abs(v) for t, v in position_values.items()
            if t[:2] in ["CL", "CO"]
        )
        crude_concentration = crude_exposure / portfolio_value

        if crude_concentration > max_correlated:
            breaches.append({
                "group": "crude_oil",
                "concentration_pct": round(crude_concentration * 100, 1),
                "limit_pct": round(max_correlated * 100, 1),
                "type": "correlated_exposure",
            })

        status = "BREACH" if breaches else "OK"

        return {
            "status": status,
            "concentrations": concentrations,
            "breaches": breaches,
            "crude_exposure_pct": round(crude_concentration * 100, 1),
            "max_single_pct": round(max_single * 100, 1),
            "max_correlated_pct": round(max_correlated * 100, 1),
        }

    def check_exposure_limits(
        self,
        gross_exposure: float,
        net_exposure: float
    ) -> dict:
        """
        Check gross and net exposure limits.

        Args:
            gross_exposure: Total absolute position value
            net_exposure: Net position value (long - short)

        Returns:
            Exposure limit check result
        """
        max_gross = self.limits.get("portfolio_limits", {}).get("max_gross_exposure", 20000000)
        max_net = self.limits.get("portfolio_limits", {}).get("max_net_exposure", 15000000)

        gross_util = gross_exposure / max_gross * 100
        net_util = abs(net_exposure) / max_net * 100

        max_util = max(gross_util, net_util)

        if max_util >= 100:
            status = "BREACH"
        elif max_util >= self.CRITICAL_LEVEL * 100:
            status = "CRITICAL"
        elif max_util >= self.WARNING_LEVEL * 100:
            status = "WARNING"
        else:
            status = "OK"

        return {
            "gross_exposure": round(gross_exposure, 2),
            "max_gross": max_gross,
            "gross_utilization_pct": round(gross_util, 1),
            "net_exposure": round(net_exposure, 2),
            "max_net": max_net,
            "net_utilization_pct": round(net_util, 1),
            "status": status,
        }

    def run_pre_trade_checks(
        self,
        positions: dict[str, dict],
        proposed_trade: dict,
        current_var: float,
        current_drawdown: float,
        portfolio_value: float
    ) -> dict:
        """
        Run all pre-trade risk checks.

        Args:
            positions: Current positions
            proposed_trade: Proposed trade details
            current_var: Current portfolio VaR
            current_drawdown: Current drawdown
            portfolio_value: Total portfolio value

        Returns:
            Comprehensive pre-trade check result
        """
        checks = {}
        all_approved = True

        # Position limit check
        ticker = proposed_trade["ticker"]
        current_qty = positions.get(ticker, {}).get("quantity", 0)
        proposed_qty = proposed_trade.get("quantity", 0)
        price = proposed_trade.get("price", 0)

        checks["position"] = self.check_position_limit(ticker, current_qty, proposed_qty, price)
        if not checks["position"]["approved"]:
            all_approved = False

        # VaR impact (simplified)
        var_impact = abs(proposed_qty * price * 1000) * 0.02 * 1.65  # Rough VaR estimate
        checks["var"] = self.check_var_limit(current_var, var_impact)
        if not checks["var"]["approved"]:
            all_approved = False

        # Drawdown check
        checks["drawdown"] = self.check_drawdown_limit(current_drawdown)

        # Update positions for concentration check
        test_positions = positions.copy()
        test_positions[ticker] = {
            "quantity": current_qty + proposed_qty,
            "price": price,
        }
        checks["concentration"] = self.check_concentration(test_positions, portfolio_value)
        if checks["concentration"]["status"] == "BREACH":
            all_approved = False

        # Overall result
        return {
            "approved": all_approved,
            "checks": checks,
            "timestamp": datetime.now().isoformat(),
        }
