"""
Position Sizing Algorithms
==========================
Position sizing methods for risk-adjusted trade sizing.

Features:
- Kelly Criterion for optimal position sizing
- Volatility Targeting for consistent risk exposure
- Risk Parity for balanced risk allocation
- Fixed Fractional for simple percentage-based sizing
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing method enumeration."""
    FIXED = "FIXED"
    FIXED_FRACTIONAL = "FIXED_FRACTIONAL"
    KELLY = "KELLY"
    HALF_KELLY = "HALF_KELLY"
    VOLATILITY_TARGET = "VOLATILITY_TARGET"
    RISK_PARITY = "RISK_PARITY"
    ATR_BASED = "ATR_BASED"
    VAR_BASED = "VAR_BASED"


@dataclass
class SizingConfig:
    """Configuration for position sizing."""
    method: SizingMethod = SizingMethod.VOLATILITY_TARGET
    
    # Account parameters
    account_value: float = 1_000_000
    max_position_pct: float = 0.25  # Max 25% in single position
    max_position_contracts: int = 50  # Max contracts per position
    
    # Risk parameters
    risk_per_trade_pct: float = 0.02  # 2% risk per trade
    target_volatility: float = 0.15  # 15% annual volatility target
    
    # Kelly parameters
    kelly_fraction: float = 0.25  # Use fraction of Kelly
    
    # Contract specifications
    contract_multiplier: float = 1000  # Barrels per contract
    tick_size: float = 0.01
    min_contracts: int = 1


@dataclass
class SizingResult:
    """Result from position sizing calculation."""
    method: SizingMethod
    contracts: int
    notional_value: float
    risk_amount: float
    position_pct: float
    rationale: str
    adjustments: List[str] = field(default_factory=list)


class PositionSizer(ABC):
    """Abstract base class for position sizing algorithms."""
    
    def __init__(self, config: SizingConfig):
        self.config = config
    
    @abstractmethod
    def calculate_size(
        self,
        price: float,
        volatility: float,
        stop_loss_pct: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
        **kwargs,
    ) -> SizingResult:
        """
        Calculate position size.
        
        Args:
            price: Current price
            volatility: Price volatility (annualized)
            stop_loss_pct: Stop loss as percentage of price
            win_rate: Historical win rate (for Kelly)
            avg_win_loss_ratio: Average win/loss ratio (for Kelly)
            
        Returns:
            SizingResult with recommended position size
        """
        pass
    
    def _apply_limits(self, contracts: int, price: float) -> Tuple[int, List[str]]:
        """Apply position limits."""
        adjustments = []
        original = contracts
        
        # Apply max contracts limit
        if contracts > self.config.max_position_contracts:
            contracts = self.config.max_position_contracts
            adjustments.append(f"Reduced from {original} to {contracts} (max contracts limit)")
        
        # Apply max position % limit
        notional = contracts * price * self.config.contract_multiplier
        max_notional = self.config.account_value * self.config.max_position_pct
        
        if notional > max_notional:
            new_contracts = int(max_notional / (price * self.config.contract_multiplier))
            if new_contracts < contracts:
                adjustments.append(f"Reduced from {contracts} to {new_contracts} (max position % limit)")
                contracts = new_contracts
        
        # Ensure minimum
        contracts = max(contracts, self.config.min_contracts)
        
        return contracts, adjustments


class FixedSizer(PositionSizer):
    """Fixed contract size."""
    
    def __init__(self, config: SizingConfig, fixed_contracts: int = 5):
        super().__init__(config)
        self.fixed_contracts = fixed_contracts
    
    def calculate_size(self, price: float, **kwargs) -> SizingResult:
        contracts, adjustments = self._apply_limits(self.fixed_contracts, price)
        
        notional = contracts * price * self.config.contract_multiplier
        position_pct = notional / self.config.account_value * 100
        
        return SizingResult(
            method=SizingMethod.FIXED,
            contracts=contracts,
            notional_value=notional,
            risk_amount=0,  # Unknown without stop loss
            position_pct=position_pct,
            rationale=f"Fixed size of {self.fixed_contracts} contracts",
            adjustments=adjustments,
        )


class FixedFractional(PositionSizer):
    """
    Fixed fractional position sizing.
    
    Risks a fixed percentage of account on each trade.
    Position size = (Account * Risk%) / (Price * Stop%)
    """
    
    def calculate_size(
        self,
        price: float,
        volatility: float = 0,
        stop_loss_pct: Optional[float] = None,
        **kwargs,
    ) -> SizingResult:
        if stop_loss_pct is None or stop_loss_pct <= 0:
            stop_loss_pct = 0.02  # Default 2% stop
        
        # Risk amount in dollars
        risk_amount = self.config.account_value * self.config.risk_per_trade_pct
        
        # Risk per contract (stop distance in dollars)
        stop_distance = price * stop_loss_pct
        risk_per_contract = stop_distance * self.config.contract_multiplier
        
        # Calculate contracts
        raw_contracts = risk_amount / risk_per_contract
        contracts = max(1, int(raw_contracts))
        
        contracts, adjustments = self._apply_limits(contracts, price)
        
        notional = contracts * price * self.config.contract_multiplier
        position_pct = notional / self.config.account_value * 100
        actual_risk = contracts * risk_per_contract
        
        return SizingResult(
            method=SizingMethod.FIXED_FRACTIONAL,
            contracts=contracts,
            notional_value=notional,
            risk_amount=actual_risk,
            position_pct=position_pct,
            rationale=f"Risk {self.config.risk_per_trade_pct*100:.1f}% = ${risk_amount:,.0f}, "
                     f"Stop {stop_loss_pct*100:.1f}% = ${stop_distance:.2f}/contract",
            adjustments=adjustments,
        )


class KellyCriterion(PositionSizer):
    """
    Kelly Criterion position sizing.
    
    f* = (bp - q) / b
    
    Where:
    - f* = fraction of bankroll to bet
    - b = win/loss ratio (avg win / avg loss)
    - p = win rate
    - q = 1 - p (loss rate)
    """
    
    def calculate_size(
        self,
        price: float,
        volatility: float = 0,
        stop_loss_pct: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
        **kwargs,
    ) -> SizingResult:
        # Default parameters if not provided
        if win_rate is None:
            win_rate = 0.55  # 55% win rate
        if avg_win_loss_ratio is None:
            avg_win_loss_ratio = 1.5  # Win 1.5x what you lose
        
        # Validate inputs
        win_rate = max(0.01, min(0.99, win_rate))
        avg_win_loss_ratio = max(0.1, avg_win_loss_ratio)
        
        # Calculate Kelly fraction
        b = avg_win_loss_ratio
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Cap Kelly at reasonable level
        kelly_fraction = max(0, min(1.0, kelly_fraction))
        
        # Apply Kelly fraction (use partial Kelly for safety)
        effective_fraction = kelly_fraction * self.config.kelly_fraction
        
        # Calculate position size
        position_value = self.config.account_value * effective_fraction
        contract_value = price * self.config.contract_multiplier
        
        raw_contracts = position_value / contract_value
        contracts = max(1, int(raw_contracts))
        
        contracts, adjustments = self._apply_limits(contracts, price)
        
        notional = contracts * price * self.config.contract_multiplier
        position_pct = notional / self.config.account_value * 100
        
        return SizingResult(
            method=SizingMethod.KELLY,
            contracts=contracts,
            notional_value=notional,
            risk_amount=notional * (1 - win_rate) * stop_loss_pct if stop_loss_pct else 0,
            position_pct=position_pct,
            rationale=f"Kelly f* = {kelly_fraction*100:.1f}%, Using {self.config.kelly_fraction*100:.0f}% = {effective_fraction*100:.2f}%, "
                     f"Win rate: {win_rate*100:.0f}%, W/L ratio: {avg_win_loss_ratio:.2f}",
            adjustments=adjustments,
        )


class VolatilityTargeting(PositionSizer):
    """
    Volatility-targeted position sizing.
    
    Adjusts position size to target a specific portfolio volatility.
    Position = Target Vol / Asset Vol
    """
    
    def calculate_size(
        self,
        price: float,
        volatility: float,
        stop_loss_pct: Optional[float] = None,
        **kwargs,
    ) -> SizingResult:
        if volatility <= 0:
            volatility = 0.25  # Default 25% annualized volatility
        
        target_vol = self.config.target_volatility
        
        # Calculate vol-adjusted position weight
        position_weight = target_vol / volatility
        
        # Cap at max position size
        position_weight = min(position_weight, self.config.max_position_pct)
        
        # Calculate contracts
        position_value = self.config.account_value * position_weight
        contract_value = price * self.config.contract_multiplier
        
        raw_contracts = position_value / contract_value
        contracts = max(1, int(raw_contracts))
        
        contracts, adjustments = self._apply_limits(contracts, price)
        
        notional = contracts * price * self.config.contract_multiplier
        position_pct = notional / self.config.account_value * 100
        
        # Expected portfolio volatility
        expected_vol = (notional / self.config.account_value) * volatility
        
        return SizingResult(
            method=SizingMethod.VOLATILITY_TARGET,
            contracts=contracts,
            notional_value=notional,
            risk_amount=notional * volatility / np.sqrt(252),  # Daily VaR approximation
            position_pct=position_pct,
            rationale=f"Target vol: {target_vol*100:.0f}%, Asset vol: {volatility*100:.0f}%, "
                     f"Position weight: {position_weight*100:.1f}%, Expected portfolio vol: {expected_vol*100:.1f}%",
            adjustments=adjustments,
        )


class RiskParity(PositionSizer):
    """
    Risk parity position sizing.
    
    Allocates equal risk contribution across assets.
    Useful for multi-asset portfolios.
    """
    
    def calculate_size(
        self,
        price: float,
        volatility: float,
        asset_volatilities: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> SizingResult:
        if volatility <= 0:
            volatility = 0.25
        
        # If single asset, use volatility targeting
        if asset_volatilities is None or len(asset_volatilities) <= 1:
            return VolatilityTargeting(self.config).calculate_size(
                price=price,
                volatility=volatility,
            )
        
        # Calculate risk parity weights
        n_assets = len(asset_volatilities)
        vols = np.array(list(asset_volatilities.values()))
        
        # Inverse volatility weighting
        inv_vols = 1 / vols
        weights = inv_vols / inv_vols.sum()
        
        # Get this asset's weight (assume it's the first in the dict)
        asset_weight = weights[0]
        
        # Adjust for target volatility
        portfolio_vol = np.sqrt(np.sum((weights * vols) ** 2))  # Simplified (no correlations)
        scale_factor = self.config.target_volatility / portfolio_vol
        
        adjusted_weight = asset_weight * scale_factor
        adjusted_weight = min(adjusted_weight, self.config.max_position_pct)
        
        # Calculate contracts
        position_value = self.config.account_value * adjusted_weight
        contract_value = price * self.config.contract_multiplier
        
        raw_contracts = position_value / contract_value
        contracts = max(1, int(raw_contracts))
        
        contracts, adjustments = self._apply_limits(contracts, price)
        
        notional = contracts * price * self.config.contract_multiplier
        position_pct = notional / self.config.account_value * 100
        
        return SizingResult(
            method=SizingMethod.RISK_PARITY,
            contracts=contracts,
            notional_value=notional,
            risk_amount=notional * volatility / np.sqrt(252),
            position_pct=position_pct,
            rationale=f"Risk parity weight: {asset_weight*100:.1f}% (of {n_assets} assets), "
                     f"Adjusted for vol target: {adjusted_weight*100:.1f}%",
            adjustments=adjustments,
        )


class ATRBasedSizing(PositionSizer):
    """
    ATR-based position sizing.
    
    Uses Average True Range to determine position size based on
    typical price movement.
    """
    
    def __init__(self, config: SizingConfig, atr_multiplier: float = 2.0):
        super().__init__(config)
        self.atr_multiplier = atr_multiplier
    
    def calculate_size(
        self,
        price: float,
        volatility: float = 0,
        atr: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        **kwargs,
    ) -> SizingResult:
        if atr is None:
            # Estimate ATR from volatility (rough approximation)
            atr = price * volatility / np.sqrt(252)
        
        # Risk amount in dollars
        risk_amount = self.config.account_value * self.config.risk_per_trade_pct
        
        # Stop distance based on ATR
        stop_distance = atr * self.atr_multiplier
        risk_per_contract = stop_distance * self.config.contract_multiplier
        
        # Calculate contracts
        raw_contracts = risk_amount / risk_per_contract
        contracts = max(1, int(raw_contracts))
        
        contracts, adjustments = self._apply_limits(contracts, price)
        
        notional = contracts * price * self.config.contract_multiplier
        position_pct = notional / self.config.account_value * 100
        actual_risk = contracts * risk_per_contract
        
        return SizingResult(
            method=SizingMethod.ATR_BASED,
            contracts=contracts,
            notional_value=notional,
            risk_amount=actual_risk,
            position_pct=position_pct,
            rationale=f"ATR: ${atr:.2f}, Stop: {self.atr_multiplier}x ATR = ${stop_distance:.2f}, "
                     f"Risk: ${risk_amount:,.0f}",
            adjustments=adjustments,
        )


class VaRBasedSizing(PositionSizer):
    """
    VaR-based position sizing.
    
    Sizes positions to not exceed a maximum VaR contribution.
    """
    
    def __init__(self, config: SizingConfig, max_var_pct: float = 0.02, confidence: float = 0.95):
        super().__init__(config)
        self.max_var_pct = max_var_pct
        self.confidence = confidence
        # Z-score for confidence level
        self.z_score = {0.95: 1.645, 0.99: 2.326}.get(confidence, 1.645)
    
    def calculate_size(
        self,
        price: float,
        volatility: float,
        stop_loss_pct: Optional[float] = None,
        **kwargs,
    ) -> SizingResult:
        if volatility <= 0:
            volatility = 0.25
        
        # Max VaR in dollars
        max_var = self.config.account_value * self.max_var_pct
        
        # Daily volatility
        daily_vol = volatility / np.sqrt(252)
        
        # VaR per dollar notional
        var_per_dollar = daily_vol * self.z_score
        
        # VaR per contract
        contract_value = price * self.config.contract_multiplier
        var_per_contract = contract_value * var_per_dollar
        
        # Calculate contracts
        raw_contracts = max_var / var_per_contract
        contracts = max(1, int(raw_contracts))
        
        contracts, adjustments = self._apply_limits(contracts, price)
        
        notional = contracts * price * self.config.contract_multiplier
        position_pct = notional / self.config.account_value * 100
        actual_var = contracts * var_per_contract
        
        return SizingResult(
            method=SizingMethod.VAR_BASED,
            contracts=contracts,
            notional_value=notional,
            risk_amount=actual_var,
            position_pct=position_pct,
            rationale=f"Max VaR ({self.confidence*100:.0f}%): ${max_var:,.0f}, "
                     f"Daily vol: {daily_vol*100:.2f}%, VaR/contract: ${var_per_contract:,.0f}",
            adjustments=adjustments,
        )


def get_position_sizer(config: SizingConfig) -> PositionSizer:
    """
    Factory function to get appropriate position sizer.
    
    Args:
        config: Sizing configuration
        
    Returns:
        Position sizer instance
    """
    sizers = {
        SizingMethod.FIXED: FixedSizer,
        SizingMethod.FIXED_FRACTIONAL: FixedFractional,
        SizingMethod.KELLY: KellyCriterion,
        SizingMethod.HALF_KELLY: lambda c: KellyCriterion(SizingConfig(
            **{**vars(c), "kelly_fraction": 0.5}
        )),
        SizingMethod.VOLATILITY_TARGET: VolatilityTargeting,
        SizingMethod.RISK_PARITY: RiskParity,
        SizingMethod.ATR_BASED: ATRBasedSizing,
        SizingMethod.VAR_BASED: VaRBasedSizing,
    }
    
    sizer_class = sizers.get(config.method)
    if sizer_class:
        return sizer_class(config)
    
    # Default to volatility targeting
    return VolatilityTargeting(config)


def calculate_optimal_size(
    price: float,
    volatility: float,
    account_value: float = 1_000_000,
    method: SizingMethod = SizingMethod.VOLATILITY_TARGET,
    **kwargs,
) -> SizingResult:
    """
    Convenience function to calculate optimal position size.
    
    Args:
        price: Current price
        volatility: Annualized volatility
        account_value: Account value
        method: Sizing method to use
        **kwargs: Additional parameters for sizer
        
    Returns:
        SizingResult
    """
    config = SizingConfig(
        method=method,
        account_value=account_value,
        **{k: v for k, v in kwargs.items() if hasattr(SizingConfig, k)},
    )
    
    sizer = get_position_sizer(config)
    return sizer.calculate_size(price=price, volatility=volatility, **kwargs)
