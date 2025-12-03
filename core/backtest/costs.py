"""
Transaction Cost Models
========================
Realistic transaction cost modeling for backtesting.

Includes:
- Commission structures (per-trade, per-contract)
- Slippage models (fixed, proportional, volatility-based)
- Market impact models (linear, square-root)
- Bid-ask spread costs
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class TransactionCosts:
    """Container for transaction cost breakdown."""
    commission: float = 0.0
    slippage: float = 0.0
    spread: float = 0.0
    market_impact: float = 0.0
    
    @property
    def total(self) -> float:
        """Total transaction cost."""
        return self.commission + self.slippage + self.spread + self.market_impact
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "commission": round(self.commission, 4),
            "slippage": round(self.slippage, 4),
            "spread": round(self.spread, 4),
            "market_impact": round(self.market_impact, 4),
            "total": round(self.total, 4),
        }


@dataclass
class CostModelConfig:
    """Configuration for cost models."""
    
    # Commission settings
    commission_per_contract: float = 2.50  # $ per contract
    commission_per_trade: float = 0.0  # Fixed $ per trade
    commission_pct: float = 0.0  # Percentage of notional
    
    # Slippage settings (in ticks)
    slippage_ticks: float = 1.0  # Fixed slippage in ticks
    slippage_pct: float = 0.0  # Proportional slippage
    
    # Spread settings
    half_spread_ticks: float = 1.0  # Half bid-ask spread in ticks
    
    # Market impact settings
    market_impact_coefficient: float = 0.1  # Impact coefficient
    
    # Contract specifications
    tick_size: float = 0.01  # Price increment
    contract_multiplier: float = 1000  # Barrels per contract
    
    @classmethod
    def oil_futures_default(cls) -> "CostModelConfig":
        """Default configuration for oil futures."""
        return cls(
            commission_per_contract=2.50,
            slippage_ticks=1.0,
            half_spread_ticks=1.0,
            tick_size=0.01,
            contract_multiplier=1000,
        )
    
    @classmethod
    def low_cost(cls) -> "CostModelConfig":
        """Low-cost configuration for optimistic estimates."""
        return cls(
            commission_per_contract=1.50,
            slippage_ticks=0.5,
            half_spread_ticks=0.5,
            tick_size=0.01,
            contract_multiplier=1000,
        )
    
    @classmethod
    def high_cost(cls) -> "CostModelConfig":
        """High-cost configuration for conservative estimates."""
        return cls(
            commission_per_contract=5.00,
            slippage_ticks=2.0,
            half_spread_ticks=2.0,
            market_impact_coefficient=0.2,
            tick_size=0.01,
            contract_multiplier=1000,
        )


class CostModel(ABC):
    """Abstract base class for cost models."""
    
    @abstractmethod
    def calculate_costs(
        self,
        price: float,
        quantity: int,
        side: OrderSide,
        volatility: Optional[float] = None,
        avg_daily_volume: Optional[float] = None
    ) -> TransactionCosts:
        """
        Calculate transaction costs for an order.
        
        Args:
            price: Execution price
            quantity: Number of contracts (absolute value)
            side: Buy or sell
            volatility: Current volatility (for volatility-based slippage)
            avg_daily_volume: ADV for market impact calculation
            
        Returns:
            TransactionCosts breakdown
        """
        pass


class SimpleCostModel(CostModel):
    """
    Simple cost model with fixed costs.
    
    Suitable for basic backtesting with reasonable defaults.
    """
    
    def __init__(self, config: Optional[CostModelConfig] = None):
        """Initialize with configuration."""
        self.config = config or CostModelConfig.oil_futures_default()
    
    def calculate_costs(
        self,
        price: float,
        quantity: int,
        side: OrderSide,
        volatility: Optional[float] = None,
        avg_daily_volume: Optional[float] = None
    ) -> TransactionCosts:
        """Calculate costs using simple fixed models."""
        quantity = abs(quantity)
        
        # Commission
        commission = (
            self.config.commission_per_contract * quantity +
            self.config.commission_per_trade +
            price * quantity * self.config.contract_multiplier * self.config.commission_pct
        )
        
        # Slippage (in price terms)
        slippage_price = self.config.slippage_ticks * self.config.tick_size
        slippage = slippage_price * quantity * self.config.contract_multiplier
        
        # Spread cost (half-spread, as we pay half when entering and half when exiting)
        spread_price = self.config.half_spread_ticks * self.config.tick_size
        spread = spread_price * quantity * self.config.contract_multiplier
        
        return TransactionCosts(
            commission=commission,
            slippage=slippage,
            spread=spread,
            market_impact=0.0,
        )


class VolatilityAdjustedCostModel(CostModel):
    """
    Cost model that adjusts slippage based on volatility.
    
    Higher volatility leads to higher expected slippage.
    """
    
    def __init__(
        self, 
        config: Optional[CostModelConfig] = None,
        base_volatility: float = 0.02  # 2% daily vol as baseline
    ):
        """Initialize with configuration."""
        self.config = config or CostModelConfig.oil_futures_default()
        self.base_volatility = base_volatility
    
    def calculate_costs(
        self,
        price: float,
        quantity: int,
        side: OrderSide,
        volatility: Optional[float] = None,
        avg_daily_volume: Optional[float] = None
    ) -> TransactionCosts:
        """Calculate costs with volatility-adjusted slippage."""
        quantity = abs(quantity)
        current_vol = volatility or self.base_volatility
        
        # Commission (fixed)
        commission = (
            self.config.commission_per_contract * quantity +
            self.config.commission_per_trade
        )
        
        # Volatility-adjusted slippage
        vol_multiplier = current_vol / self.base_volatility
        adjusted_slippage_ticks = self.config.slippage_ticks * vol_multiplier
        slippage_price = adjusted_slippage_ticks * self.config.tick_size
        slippage = slippage_price * quantity * self.config.contract_multiplier
        
        # Spread (also volatility-adjusted)
        adjusted_spread_ticks = self.config.half_spread_ticks * vol_multiplier
        spread_price = adjusted_spread_ticks * self.config.tick_size
        spread = spread_price * quantity * self.config.contract_multiplier
        
        return TransactionCosts(
            commission=commission,
            slippage=slippage,
            spread=spread,
            market_impact=0.0,
        )


class MarketImpactCostModel(CostModel):
    """
    Cost model with market impact modeling.
    
    Uses square-root market impact model:
    Impact = coefficient * volatility * sqrt(quantity / ADV)
    
    This is the Almgren-Chriss model commonly used in practice.
    """
    
    def __init__(
        self, 
        config: Optional[CostModelConfig] = None,
        default_adv: float = 50000  # Default average daily volume
    ):
        """Initialize with configuration."""
        self.config = config or CostModelConfig.oil_futures_default()
        self.default_adv = default_adv
    
    def calculate_costs(
        self,
        price: float,
        quantity: int,
        side: OrderSide,
        volatility: Optional[float] = None,
        avg_daily_volume: Optional[float] = None
    ) -> TransactionCosts:
        """Calculate costs including market impact."""
        quantity = abs(quantity)
        vol = volatility or 0.02
        adv = avg_daily_volume or self.default_adv
        
        # Commission
        commission = (
            self.config.commission_per_contract * quantity +
            self.config.commission_per_trade
        )
        
        # Fixed slippage
        slippage_price = self.config.slippage_ticks * self.config.tick_size
        slippage = slippage_price * quantity * self.config.contract_multiplier
        
        # Spread
        spread_price = self.config.half_spread_ticks * self.config.tick_size
        spread = spread_price * quantity * self.config.contract_multiplier
        
        # Market impact (square-root model)
        participation_rate = quantity / adv if adv > 0 else 0
        impact_pct = self.config.market_impact_coefficient * vol * np.sqrt(participation_rate)
        impact_price = price * impact_pct
        market_impact = impact_price * quantity * self.config.contract_multiplier
        
        return TransactionCosts(
            commission=commission,
            slippage=slippage,
            spread=spread,
            market_impact=market_impact,
        )


class TieredCommissionModel(CostModel):
    """
    Cost model with tiered commission structure.
    
    Commission rate decreases with volume.
    """
    
    def __init__(
        self,
        tiers: Optional[List[tuple]] = None,
        config: Optional[CostModelConfig] = None
    ):
        """
        Initialize with commission tiers.
        
        Args:
            tiers: List of (volume_threshold, commission_rate) tuples
                   e.g., [(100, 2.50), (500, 2.00), (1000, 1.50), (float('inf'), 1.00)]
        """
        self.config = config or CostModelConfig.oil_futures_default()
        self.tiers = tiers or [
            (100, 2.50),      # Up to 100 contracts: $2.50
            (500, 2.00),      # 101-500 contracts: $2.00
            (1000, 1.50),     # 501-1000 contracts: $1.50
            (float('inf'), 1.00)  # 1000+ contracts: $1.00
        ]
        self._daily_volume = 0
        self._current_date = None
    
    def reset_daily_volume(self):
        """Reset daily volume counter."""
        self._daily_volume = 0
    
    def _get_commission_rate(self, cumulative_volume: int) -> float:
        """Get commission rate based on cumulative volume."""
        for threshold, rate in self.tiers:
            if cumulative_volume <= threshold:
                return rate
        return self.tiers[-1][1]
    
    def calculate_costs(
        self,
        price: float,
        quantity: int,
        side: OrderSide,
        volatility: Optional[float] = None,
        avg_daily_volume: Optional[float] = None
    ) -> TransactionCosts:
        """Calculate costs with tiered commissions."""
        quantity = abs(quantity)
        
        # Update daily volume
        self._daily_volume += quantity
        
        # Get tiered commission rate
        commission_rate = self._get_commission_rate(self._daily_volume)
        commission = commission_rate * quantity
        
        # Slippage
        slippage_price = self.config.slippage_ticks * self.config.tick_size
        slippage = slippage_price * quantity * self.config.contract_multiplier
        
        # Spread
        spread_price = self.config.half_spread_ticks * self.config.tick_size
        spread = spread_price * quantity * self.config.contract_multiplier
        
        return TransactionCosts(
            commission=commission,
            slippage=slippage,
            spread=spread,
            market_impact=0.0,
        )


def estimate_total_costs(
    trades_df: pd.DataFrame,
    cost_model: CostModel,
    price_col: str = "price",
    quantity_col: str = "quantity",
    side_col: str = "side"
) -> pd.DataFrame:
    """
    Estimate total costs for a series of trades.
    
    Args:
        trades_df: DataFrame with trade data
        cost_model: Cost model to use
        price_col: Column name for prices
        quantity_col: Column name for quantities
        side_col: Column name for sides
        
    Returns:
        DataFrame with cost breakdown for each trade
    """
    costs_list = []
    
    for _, row in trades_df.iterrows():
        side = OrderSide.BUY if row[side_col].upper() in ["BUY", "LONG"] else OrderSide.SELL
        
        costs = cost_model.calculate_costs(
            price=row[price_col],
            quantity=row[quantity_col],
            side=side
        )
        
        costs_list.append(costs.to_dict())
    
    costs_df = pd.DataFrame(costs_list)
    
    # Add total row
    totals = costs_df.sum()
    
    return costs_df, totals.to_dict()


def create_cost_comparison(
    price: float = 75.0,
    quantity: int = 10,
    side: OrderSide = OrderSide.BUY
) -> pd.DataFrame:
    """
    Compare costs across different cost models.
    
    Args:
        price: Trade price
        quantity: Number of contracts
        side: Trade side
        
    Returns:
        DataFrame comparing costs across models
    """
    models = {
        "Simple (Default)": SimpleCostModel(CostModelConfig.oil_futures_default()),
        "Simple (Low)": SimpleCostModel(CostModelConfig.low_cost()),
        "Simple (High)": SimpleCostModel(CostModelConfig.high_cost()),
        "Volatility-Adjusted": VolatilityAdjustedCostModel(),
        "Market Impact": MarketImpactCostModel(),
    }
    
    results = []
    for name, model in models.items():
        costs = model.calculate_costs(price, quantity, side, volatility=0.02)
        row = costs.to_dict()
        row["model"] = name
        results.append(row)
    
    df = pd.DataFrame(results)
    df = df.set_index("model")
    
    return df
