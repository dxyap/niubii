"""
Execution Algorithms
====================
Algorithms for optimal order execution.

Features:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- Implementation Shortfall
- Slice scheduling and execution tracking
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd

from .oms import Order, OrderSide

logger = logging.getLogger(__name__)


class AlgorithmType(Enum):
    """Execution algorithm types."""
    TWAP = "TWAP"
    VWAP = "VWAP"
    POV = "POV"  # Percentage of Volume
    IS = "IS"    # Implementation Shortfall
    ICEBERG = "ICEBERG"


@dataclass
class AlgorithmConfig:
    """Configuration for execution algorithms."""
    algo_type: AlgorithmType = AlgorithmType.TWAP

    # Time parameters
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_minutes: int = 60

    # Slicing parameters
    num_slices: int = 12
    min_slice_size: int = 1
    randomize_timing: bool = True
    randomize_size: bool = True
    variance_pct: float = 0.20  # +/- 20% randomization

    # Execution parameters
    use_limit_orders: bool = True
    limit_offset_ticks: float = 1.0  # Ticks from mid
    max_market_order_pct: float = 0.30  # Max % to execute as market

    # VWAP specific
    volume_profile: list[float] | None = None  # Hourly volume weights

    # POV specific
    participation_rate: float = 0.10  # 10% of volume

    # IS specific
    urgency: float = 0.5  # 0 = passive, 1 = aggressive


@dataclass
class ExecutionSlice:
    """Represents a single execution slice."""
    slice_id: str
    parent_order_id: str
    sequence: int
    scheduled_time: datetime
    quantity: int

    # Execution state
    executed_quantity: int = 0
    executed_price: float = 0.0
    status: str = "PENDING"  # PENDING, WORKING, FILLED, CANCELLED
    child_order_id: str | None = None

    # Metrics
    arrival_price: float | None = None
    execution_time: datetime | None = None
    slippage: float = 0.0

    @property
    def is_complete(self) -> bool:
        return self.status in ["FILLED", "CANCELLED"]

    @property
    def is_pending(self) -> bool:
        return self.status == "PENDING"

    def to_dict(self) -> dict:
        return {
            "slice_id": self.slice_id,
            "parent_order_id": self.parent_order_id,
            "sequence": self.sequence,
            "scheduled_time": self.scheduled_time.isoformat(),
            "quantity": self.quantity,
            "executed_quantity": self.executed_quantity,
            "executed_price": self.executed_price,
            "status": self.status,
            "slippage": self.slippage,
        }


@dataclass
class AlgorithmProgress:
    """Progress tracking for algorithm execution."""
    algo_id: str
    algo_type: AlgorithmType
    total_quantity: int
    executed_quantity: int
    remaining_quantity: int
    num_slices: int
    completed_slices: int
    pending_slices: int

    # Execution quality
    avg_execution_price: float
    arrival_price: float
    vwap: float
    slippage_bps: float

    # Timing
    start_time: datetime
    expected_end_time: datetime
    elapsed_minutes: float
    remaining_minutes: float

    @property
    def pct_complete(self) -> float:
        return (self.executed_quantity / self.total_quantity * 100) if self.total_quantity > 0 else 0


class ExecutionAlgorithm(ABC):
    """Abstract base class for execution algorithms."""

    def __init__(self, config: AlgorithmConfig):
        self.config = config
        self.algo_id: str = f"ALGO-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.slices: list[ExecutionSlice] = []
        self.arrival_price: float | None = None
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self._is_active = False

    @abstractmethod
    def generate_schedule(
        self,
        order: Order,
        current_price: float,
        market_data: pd.DataFrame | None = None,
    ) -> list[ExecutionSlice]:
        """
        Generate execution schedule.

        Args:
            order: Parent order to execute
            current_price: Current market price
            market_data: Optional historical/volume data

        Returns:
            List of execution slices
        """
        pass

    def get_next_slice(self) -> ExecutionSlice | None:
        """Get next pending slice to execute."""
        now = datetime.now()
        for slice_ in self.slices:
            if slice_.is_pending and slice_.scheduled_time <= now:
                return slice_
        return None

    def get_pending_slices(self) -> list[ExecutionSlice]:
        """Get all pending slices."""
        return [s for s in self.slices if s.is_pending]

    def mark_slice_executed(
        self,
        slice_id: str,
        executed_quantity: int,
        executed_price: float,
    ):
        """Mark a slice as executed."""
        for slice_ in self.slices:
            if slice_.slice_id == slice_id:
                slice_.executed_quantity = executed_quantity
                slice_.executed_price = executed_price
                slice_.status = "FILLED"
                slice_.execution_time = datetime.now()

                if self.arrival_price:
                    slice_.slippage = (executed_price - self.arrival_price) / self.arrival_price * 10000
                break

    def get_progress(self) -> AlgorithmProgress:
        """Get algorithm progress."""
        executed_qty = sum(s.executed_quantity for s in self.slices)
        total_qty = sum(s.quantity for s in self.slices)
        remaining = total_qty - executed_qty

        completed = len([s for s in self.slices if s.status == "FILLED"])
        pending = len([s for s in self.slices if s.is_pending])

        # Calculate execution metrics
        if executed_qty > 0:
            total_value = sum(s.executed_quantity * s.executed_price for s in self.slices if s.status == "FILLED")
            avg_price = total_value / executed_qty
        else:
            avg_price = 0

        # Calculate VWAP
        filled_slices = [s for s in self.slices if s.status == "FILLED"]
        if filled_slices:
            vwap = sum(s.executed_quantity * s.executed_price for s in filled_slices) / sum(s.executed_quantity for s in filled_slices)
        else:
            vwap = 0

        # Calculate slippage
        if self.arrival_price and avg_price > 0:
            slippage_bps = (avg_price - self.arrival_price) / self.arrival_price * 10000
        else:
            slippage_bps = 0

        # Timing
        now = datetime.now()
        elapsed = (now - self.start_time).total_seconds() / 60 if self.start_time else 0
        remaining_time = (self.end_time - now).total_seconds() / 60 if self.end_time else 0

        return AlgorithmProgress(
            algo_id=self.algo_id,
            algo_type=self.config.algo_type,
            total_quantity=total_qty,
            executed_quantity=executed_qty,
            remaining_quantity=remaining,
            num_slices=len(self.slices),
            completed_slices=completed,
            pending_slices=pending,
            avg_execution_price=avg_price,
            arrival_price=self.arrival_price or 0,
            vwap=vwap,
            slippage_bps=slippage_bps,
            start_time=self.start_time or datetime.now(),
            expected_end_time=self.end_time or datetime.now(),
            elapsed_minutes=elapsed,
            remaining_minutes=max(0, remaining_time),
        )

    def cancel(self):
        """Cancel algorithm execution."""
        self._is_active = False
        for slice_ in self.slices:
            if slice_.is_pending:
                slice_.status = "CANCELLED"
        logger.info(f"Algorithm {self.algo_id} cancelled")


class TWAPAlgorithm(ExecutionAlgorithm):
    """
    Time-Weighted Average Price algorithm.

    Executes order in equal slices over a specified time period.
    Aims to achieve the time-weighted average price.
    """

    def generate_schedule(
        self,
        order: Order,
        current_price: float,
        market_data: pd.DataFrame | None = None,
    ) -> list[ExecutionSlice]:
        """Generate TWAP schedule."""
        self.arrival_price = current_price
        self.start_time = self.config.start_time or datetime.now()
        self.end_time = self.config.end_time or (self.start_time + timedelta(minutes=self.config.duration_minutes))

        num_slices = self.config.num_slices
        total_quantity = order.quantity

        # Calculate time between slices
        total_seconds = (self.end_time - self.start_time).total_seconds()
        slice_interval = total_seconds / num_slices

        # Calculate base slice size
        base_size = total_quantity / num_slices

        self.slices = []
        remaining_qty = total_quantity

        for i in range(num_slices):
            # Calculate slice time
            slice_time = self.start_time + timedelta(seconds=slice_interval * i)

            # Add randomization to timing if configured
            if self.config.randomize_timing and i > 0:
                jitter = np.random.uniform(-slice_interval * 0.2, slice_interval * 0.2)
                slice_time += timedelta(seconds=jitter)

            # Calculate slice quantity
            if i == num_slices - 1:
                # Last slice gets remaining
                slice_qty = remaining_qty
            else:
                slice_qty = int(base_size)

                # Add randomization to size if configured
                if self.config.randomize_size:
                    variance = self.config.variance_pct
                    multiplier = 1 + np.random.uniform(-variance, variance)
                    slice_qty = int(base_size * multiplier)

                slice_qty = max(self.config.min_slice_size, slice_qty)
                slice_qty = min(slice_qty, remaining_qty)

            remaining_qty -= slice_qty

            if slice_qty > 0:
                slice_ = ExecutionSlice(
                    slice_id=f"{self.algo_id}-{i+1:03d}",
                    parent_order_id=order.order_id,
                    sequence=i + 1,
                    scheduled_time=slice_time,
                    quantity=slice_qty,
                    arrival_price=current_price,
                )
                self.slices.append(slice_)

        self._is_active = True
        logger.info(f"TWAP generated {len(self.slices)} slices for {total_quantity} contracts over {self.config.duration_minutes} minutes")

        return self.slices


class VWAPAlgorithm(ExecutionAlgorithm):
    """
    Volume-Weighted Average Price algorithm.

    Executes order following historical volume profile.
    Aims to achieve volume-weighted average price.
    """

    # Default intraday volume profile (hourly buckets, US trading hours 9-16)
    DEFAULT_VOLUME_PROFILE = [
        0.15,  # 09:00-10:00 - High opening volume
        0.12,  # 10:00-11:00
        0.09,  # 11:00-12:00 - Lunch dip
        0.08,  # 12:00-13:00
        0.10,  # 13:00-14:00
        0.12,  # 14:00-15:00
        0.18,  # 15:00-16:00 - High closing volume
        0.16,  # Extended/overnight
    ]

    def generate_schedule(
        self,
        order: Order,
        current_price: float,
        market_data: pd.DataFrame | None = None,
    ) -> list[ExecutionSlice]:
        """Generate VWAP schedule."""
        self.arrival_price = current_price
        self.start_time = self.config.start_time or datetime.now()
        self.end_time = self.config.end_time or (self.start_time + timedelta(minutes=self.config.duration_minutes))

        total_quantity = order.quantity

        # Get volume profile
        volume_profile = self.config.volume_profile or self.DEFAULT_VOLUME_PROFILE

        # Normalize profile
        profile = np.array(volume_profile[:self.config.num_slices])
        profile = profile / profile.sum()

        # Calculate time per slice
        total_seconds = (self.end_time - self.start_time).total_seconds()
        num_slices = len(profile)
        slice_interval = total_seconds / num_slices

        self.slices = []
        remaining_qty = total_quantity

        for i, weight in enumerate(profile):
            slice_time = self.start_time + timedelta(seconds=slice_interval * i)

            # Add randomization
            if self.config.randomize_timing and i > 0:
                jitter = np.random.uniform(-slice_interval * 0.15, slice_interval * 0.15)
                slice_time += timedelta(seconds=jitter)

            # Calculate volume-weighted quantity
            if i == len(profile) - 1:
                slice_qty = remaining_qty
            else:
                slice_qty = int(total_quantity * weight)

                if self.config.randomize_size:
                    variance = self.config.variance_pct * 0.5  # Less variance for VWAP
                    multiplier = 1 + np.random.uniform(-variance, variance)
                    slice_qty = int(slice_qty * multiplier)

                slice_qty = max(self.config.min_slice_size, slice_qty)
                slice_qty = min(slice_qty, remaining_qty)

            remaining_qty -= slice_qty

            if slice_qty > 0:
                slice_ = ExecutionSlice(
                    slice_id=f"{self.algo_id}-{i+1:03d}",
                    parent_order_id=order.order_id,
                    sequence=i + 1,
                    scheduled_time=slice_time,
                    quantity=slice_qty,
                    arrival_price=current_price,
                )
                self.slices.append(slice_)

        self._is_active = True
        logger.info(f"VWAP generated {len(self.slices)} slices following volume profile")

        return self.slices


class POVAlgorithm(ExecutionAlgorithm):
    """
    Percentage of Volume algorithm.

    Executes as a percentage of market volume.
    Adapts to real-time volume.
    """

    def generate_schedule(
        self,
        order: Order,
        current_price: float,
        market_data: pd.DataFrame | None = None,
    ) -> list[ExecutionSlice]:
        """Generate POV schedule (initial schedule, adapts in real-time)."""
        self.arrival_price = current_price
        self.start_time = self.config.start_time or datetime.now()
        self.end_time = self.config.end_time or (self.start_time + timedelta(minutes=self.config.duration_minutes))

        total_quantity = order.quantity
        participation = self.config.participation_rate

        # Estimate expected volume based on historical average
        # In real implementation, would use actual volume forecasts
        estimated_hourly_volume = 5000  # contracts
        total_minutes = self.config.duration_minutes
        estimated_volume = estimated_hourly_volume * (total_minutes / 60)

        # Calculate expected slices based on participation
        target_per_slice = int(estimated_volume * participation / self.config.num_slices)
        target_per_slice = max(self.config.min_slice_size, target_per_slice)

        # Generate initial schedule (will be adjusted in real-time)
        total_seconds = (self.end_time - self.start_time).total_seconds()
        slice_interval = total_seconds / self.config.num_slices

        self.slices = []
        remaining_qty = total_quantity

        for i in range(self.config.num_slices):
            slice_time = self.start_time + timedelta(seconds=slice_interval * i)

            if self.config.randomize_timing:
                jitter = np.random.uniform(-slice_interval * 0.2, slice_interval * 0.2)
                slice_time += timedelta(seconds=jitter)

            if i == self.config.num_slices - 1:
                slice_qty = remaining_qty
            else:
                slice_qty = min(target_per_slice, remaining_qty)

            remaining_qty -= slice_qty

            if slice_qty > 0:
                slice_ = ExecutionSlice(
                    slice_id=f"{self.algo_id}-{i+1:03d}",
                    parent_order_id=order.order_id,
                    sequence=i + 1,
                    scheduled_time=slice_time,
                    quantity=slice_qty,
                    arrival_price=current_price,
                )
                self.slices.append(slice_)

        self._is_active = True
        logger.info(f"POV generated {len(self.slices)} slices at {participation*100:.0f}% participation")

        return self.slices

    def adapt_to_volume(self, current_volume: int, elapsed_minutes: float) -> int:
        """
        Adapt execution to real-time volume.

        Args:
            current_volume: Volume traded since start
            elapsed_minutes: Time elapsed

        Returns:
            Recommended next slice size
        """
        target_executed = int(current_volume * self.config.participation_rate)
        actual_executed = sum(s.executed_quantity for s in self.slices)

        # Calculate how much we should catch up or slow down
        difference = target_executed - actual_executed

        # Adjust next slice
        remaining_slices = len([s for s in self.slices if s.is_pending])
        if remaining_slices > 0:
            adjustment = difference / remaining_slices
            return max(self.config.min_slice_size, int(adjustment))

        return 0


class ImplementationShortfall(ExecutionAlgorithm):
    """
    Implementation Shortfall algorithm.

    Balances market impact vs. timing risk.
    More aggressive execution when price is favorable.
    """

    def generate_schedule(
        self,
        order: Order,
        current_price: float,
        market_data: pd.DataFrame | None = None,
    ) -> list[ExecutionSlice]:
        """Generate IS schedule."""
        self.arrival_price = current_price
        self.start_time = self.config.start_time or datetime.now()
        self.end_time = self.config.end_time or (self.start_time + timedelta(minutes=self.config.duration_minutes))

        total_quantity = order.quantity
        urgency = self.config.urgency

        # Higher urgency = more front-loaded execution
        # Generate exponential decay schedule
        num_slices = self.config.num_slices
        decay_factor = 2 * urgency  # 0 = flat, 1 = steep decay

        weights = np.exp(-decay_factor * np.arange(num_slices) / num_slices)
        weights = weights / weights.sum()

        # Calculate timing
        total_seconds = (self.end_time - self.start_time).total_seconds()
        slice_interval = total_seconds / num_slices

        self.slices = []
        remaining_qty = total_quantity

        for i, weight in enumerate(weights):
            slice_time = self.start_time + timedelta(seconds=slice_interval * i)

            if i == num_slices - 1:
                slice_qty = remaining_qty
            else:
                slice_qty = int(total_quantity * weight)
                slice_qty = max(self.config.min_slice_size, slice_qty)
                slice_qty = min(slice_qty, remaining_qty)

            remaining_qty -= slice_qty

            if slice_qty > 0:
                slice_ = ExecutionSlice(
                    slice_id=f"{self.algo_id}-{i+1:03d}",
                    parent_order_id=order.order_id,
                    sequence=i + 1,
                    scheduled_time=slice_time,
                    quantity=slice_qty,
                    arrival_price=current_price,
                )
                self.slices.append(slice_)

        self._is_active = True
        logger.info(f"IS generated {len(self.slices)} slices with urgency {urgency:.1f}")

        return self.slices

    def adjust_for_price_move(self, current_price: float, order_side: OrderSide) -> float:
        """
        Adjust urgency based on price movement.

        Returns adjusted urgency factor.
        """
        if not self.arrival_price:
            return self.config.urgency

        price_change = (current_price - self.arrival_price) / self.arrival_price

        if order_side == OrderSide.BUY:
            # If price went up, increase urgency (we're missing out)
            if price_change > 0:
                return min(1.0, self.config.urgency + price_change * 2)
            else:
                # Price went down, can be more patient
                return max(0, self.config.urgency + price_change * 2)
        else:
            # Sell order - opposite logic
            if price_change < 0:
                return min(1.0, self.config.urgency - price_change * 2)
            else:
                return max(0, self.config.urgency - price_change * 2)


def get_execution_algorithm(config: AlgorithmConfig) -> ExecutionAlgorithm:
    """
    Factory function to get appropriate execution algorithm.

    Args:
        config: Algorithm configuration

    Returns:
        Execution algorithm instance
    """
    algorithms = {
        AlgorithmType.TWAP: TWAPAlgorithm,
        AlgorithmType.VWAP: VWAPAlgorithm,
        AlgorithmType.POV: POVAlgorithm,
        AlgorithmType.IS: ImplementationShortfall,
    }

    algo_class = algorithms.get(config.algo_type)
    if algo_class:
        return algo_class(config)

    return TWAPAlgorithm(config)
