"""
Execution Algorithms Module (TWAP/VWAP)
=======================================

Features:
1. Time-Weighted Average Price (TWAP) execution
2. Volume-Weighted Average Price (VWAP) execution
3. Adaptive execution based on market conditions
4. Slippage estimation and control
5. Participation rate management

Author: Anonymous
Date: November 2025
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ExecutionAlgo(Enum):
    """Available execution algorithms"""
    TWAP = "twap"
    VWAP = "vwap"
    ADAPTIVE = "adaptive"
    IMMEDIATE = "immediate"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderSlice:
    """A single slice of a parent order"""
    slice_id: int
    parent_order_id: str
    ticker: str
    side: OrderSide
    quantity: int
    target_time: datetime
    executed: bool = False
    executed_time: Optional[datetime] = None
    executed_price: Optional[float] = None
    executed_quantity: int = 0
    slippage: float = 0.0


@dataclass
class ParentOrder:
    """Parent order to be executed over time"""
    order_id: str
    ticker: str
    side: OrderSide
    total_quantity: int
    algo: ExecutionAlgo
    start_time: datetime
    end_time: datetime
    urgency: float = 1.0  # 1.0 = normal, >1 = more aggressive
    max_participation: float = 0.10  # Max % of volume
    slices: List[OrderSlice] = field(default_factory=list)
    status: str = "pending"  # pending, active, completed, cancelled
    filled_quantity: int = 0
    avg_price: float = 0.0
    vwap_benchmark: float = 0.0
    
    @property
    def fill_rate(self) -> float:
        return self.filled_quantity / self.total_quantity if self.total_quantity > 0 else 0
    
    @property
    def slippage_bps(self) -> float:
        if self.avg_price == 0 or self.vwap_benchmark == 0:
            return 0.0
        return (self.avg_price / self.vwap_benchmark - 1) * 10000


@dataclass
class VolumeProfile:
    """Intraday volume profile for VWAP calculation"""
    intervals: List[datetime]  # Time intervals
    volumes: List[float]  # Historical volume at each interval
    cum_volume_pct: List[float]  # Cumulative volume percentage
    
    @classmethod
    def from_historical(cls, 
                       historical_volumes: pd.DataFrame,
                       n_intervals: int = 78) -> 'VolumeProfile':
        """
        Create volume profile from historical intraday data.
        
        Parameters:
        -----------
        historical_volumes : pd.DataFrame
            Historical intraday volume data
        n_intervals : int
            Number of intervals per day (78 = 5-min intervals for 6.5hr day)
        """
        # Aggregate by time-of-day
        if len(historical_volumes) == 0:
            # Return uniform profile if no data
            intervals = list(range(n_intervals))
            volumes = [1.0 / n_intervals] * n_intervals
            cum_pct = np.cumsum(volumes).tolist()
            return cls(intervals=intervals, volumes=volumes, cum_volume_pct=cum_pct)
        
        # Group by time interval and calculate average
        avg_volumes = historical_volumes.groupby(
            historical_volumes.index.time
        ).mean()
        
        volumes = avg_volumes.values.flatten().tolist()
        total = sum(volumes)
        volumes = [v / total for v in volumes]  # Normalize
        
        intervals = list(range(len(volumes)))
        cum_pct = np.cumsum(volumes).tolist()
        
        return cls(intervals=intervals, volumes=volumes, cum_volume_pct=cum_pct)


class ExecutionAlgorithm(ABC):
    """Base class for execution algorithms"""
    
    @abstractmethod
    def generate_schedule(self,
                         order: ParentOrder,
                         volume_profile: Optional[VolumeProfile] = None
                         ) -> List[OrderSlice]:
        """Generate execution schedule"""
        pass
    
    @abstractmethod
    def get_next_slice(self,
                      order: ParentOrder,
                      current_time: datetime,
                      market_data: Dict) -> Optional[OrderSlice]:
        """Get the next slice to execute"""
        pass


class TWAPExecutor(ExecutionAlgorithm):
    """
    Time-Weighted Average Price Execution
    
    Divides order equally over time intervals.
    Simple but effective for reducing market impact.
    """
    
    def __init__(self, n_intervals: int = 10):
        self.n_intervals = n_intervals
    
    def generate_schedule(self,
                         order: ParentOrder,
                         volume_profile: Optional[VolumeProfile] = None
                         ) -> List[OrderSlice]:
        """
        Generate TWAP execution schedule.
        Divides order into equal time slices.
        """
        total_duration = (order.end_time - order.start_time).total_seconds()
        interval_seconds = total_duration / self.n_intervals
        
        quantity_per_slice = order.total_quantity // self.n_intervals
        remainder = order.total_quantity % self.n_intervals
        
        slices = []
        
        for i in range(self.n_intervals):
            target_time = order.start_time + timedelta(seconds=i * interval_seconds)
            
            # Add remainder to last slice
            qty = quantity_per_slice
            if i == self.n_intervals - 1:
                qty += remainder
            
            slice_order = OrderSlice(
                slice_id=i,
                parent_order_id=order.order_id,
                ticker=order.ticker,
                side=order.side,
                quantity=qty,
                target_time=target_time
            )
            slices.append(slice_order)
        
        order.slices = slices
        return slices
    
    def get_next_slice(self,
                      order: ParentOrder,
                      current_time: datetime,
                      market_data: Dict) -> Optional[OrderSlice]:
        """Get the next unexecuted slice that is due"""
        for slice_order in order.slices:
            if not slice_order.executed and slice_order.target_time <= current_time:
                return slice_order
        return None


class VWAPExecutor(ExecutionAlgorithm):
    """
    Volume-Weighted Average Price Execution
    
    Executes proportionally to historical volume profile.
    Aims to achieve execution price close to VWAP.
    """
    
    def __init__(self,
                 max_participation: float = 0.10,
                 n_intervals: int = 78):
        self.max_participation = max_participation
        self.n_intervals = n_intervals
    
    def generate_schedule(self,
                         order: ParentOrder,
                         volume_profile: Optional[VolumeProfile] = None
                         ) -> List[OrderSlice]:
        """
        Generate VWAP execution schedule.
        Allocates shares proportionally to expected volume.
        """
        if volume_profile is None:
            # Use uniform profile if none provided
            volume_weights = [1.0 / self.n_intervals] * self.n_intervals
        else:
            volume_weights = volume_profile.volumes[:self.n_intervals]
            # Pad or truncate to match intervals
            if len(volume_weights) < self.n_intervals:
                remaining_weight = 1.0 - sum(volume_weights)
                additional = self.n_intervals - len(volume_weights)
                volume_weights.extend([remaining_weight / additional] * additional)
        
        total_duration = (order.end_time - order.start_time).total_seconds()
        interval_seconds = total_duration / self.n_intervals
        
        slices = []
        allocated = 0
        
        for i, weight in enumerate(volume_weights):
            target_time = order.start_time + timedelta(seconds=i * interval_seconds)
            
            # Allocate based on volume weight
            qty = int(order.total_quantity * weight)
            
            # Ensure we don't over-allocate
            if allocated + qty > order.total_quantity:
                qty = order.total_quantity - allocated
            
            if qty > 0:
                slice_order = OrderSlice(
                    slice_id=i,
                    parent_order_id=order.order_id,
                    ticker=order.ticker,
                    side=order.side,
                    quantity=qty,
                    target_time=target_time
                )
                slices.append(slice_order)
                allocated += qty
        
        # Handle any remainder
        if allocated < order.total_quantity and slices:
            slices[-1].quantity += order.total_quantity - allocated
        
        order.slices = slices
        return slices
    
    def get_next_slice(self,
                      order: ParentOrder,
                      current_time: datetime,
                      market_data: Dict) -> Optional[OrderSlice]:
        """
        Get the next slice, with participation rate check.
        """
        for slice_order in order.slices:
            if not slice_order.executed and slice_order.target_time <= current_time:
                # Check participation rate
                current_volume = market_data.get('current_volume', float('inf'))
                if slice_order.quantity <= current_volume * self.max_participation:
                    return slice_order
                else:
                    # Reduce quantity to stay within participation limit
                    max_qty = int(current_volume * self.max_participation)
                    if max_qty > 0:
                        slice_order.quantity = min(slice_order.quantity, max_qty)
                        return slice_order
        return None


class AdaptiveExecutor(ExecutionAlgorithm):
    """
    Adaptive Execution Algorithm
    
    Combines TWAP and VWAP with real-time adjustments based on:
    - Market volatility
    - Spread conditions
    - Volume availability
    - Price momentum
    """
    
    def __init__(self,
                 base_algo: str = "vwap",
                 urgency: float = 1.0,
                 max_participation: float = 0.10):
        self.urgency = urgency
        self.max_participation = max_participation
        
        if base_algo == "twap":
            self.base_executor = TWAPExecutor()
        else:
            self.base_executor = VWAPExecutor(max_participation=max_participation)
    
    def generate_schedule(self,
                         order: ParentOrder,
                         volume_profile: Optional[VolumeProfile] = None
                         ) -> List[OrderSlice]:
        """Generate base schedule, to be adapted in real-time"""
        return self.base_executor.generate_schedule(order, volume_profile)
    
    def get_next_slice(self,
                      order: ParentOrder,
                      current_time: datetime,
                      market_data: Dict) -> Optional[OrderSlice]:
        """
        Get next slice with adaptive adjustments.
        """
        base_slice = self.base_executor.get_next_slice(order, current_time, market_data)
        
        if base_slice is None:
            return None
        
        # Adapt based on market conditions
        adjusted_slice = self._adapt_slice(base_slice, market_data)
        
        return adjusted_slice
    
    def _adapt_slice(self,
                    slice_order: OrderSlice,
                    market_data: Dict) -> OrderSlice:
        """
        Adapt slice based on current market conditions.
        """
        # Get market conditions
        volatility = market_data.get('volatility', 0.02)
        spread_bps = market_data.get('spread_bps', 5)
        volume = market_data.get('current_volume', 100000)
        momentum = market_data.get('momentum', 0)  # Price momentum
        
        qty_adjustment = 1.0
        
        # High volatility -> smaller slices
        if volatility > 0.03:
            qty_adjustment *= 0.7
        elif volatility < 0.01:
            qty_adjustment *= 1.2
        
        # Wide spread -> smaller slices
        if spread_bps > 10:
            qty_adjustment *= 0.8
        
        # Low volume -> smaller slices
        if volume < 50000:
            qty_adjustment *= 0.6
        
        # Momentum consideration
        # If buying and price is rising, might want to speed up
        if slice_order.side == OrderSide.BUY and momentum > 0.01:
            qty_adjustment *= 1.1 * self.urgency
        elif slice_order.side == OrderSide.SELL and momentum < -0.01:
            qty_adjustment *= 1.1 * self.urgency
        
        # Apply adjustment
        adjusted_qty = int(slice_order.quantity * qty_adjustment)
        adjusted_qty = max(1, adjusted_qty)  # Minimum 1 share
        
        slice_order.quantity = adjusted_qty
        
        return slice_order


class ExecutionEngine:
    """
    Main execution engine that manages order execution.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.active_orders: Dict[str, ParentOrder] = {}
        self.completed_orders: List[ParentOrder] = []
        
        # Initialize executors
        self.executors = {
            ExecutionAlgo.TWAP: TWAPExecutor(
                n_intervals=self.config.get('twap_intervals', 10)
            ),
            ExecutionAlgo.VWAP: VWAPExecutor(
                max_participation=self.config.get('max_participation', 0.10)
            ),
            ExecutionAlgo.ADAPTIVE: AdaptiveExecutor(
                urgency=self.config.get('urgency', 1.0),
                max_participation=self.config.get('max_participation', 0.10)
            )
        }
        
        self.slippage_model = SlippageModel(
            base_impact_bps=self.config.get('base_impact_bps', 5),
            volatility_factor=self.config.get('volatility_factor', 2.0)
        )
    
    def submit_order(self,
                    ticker: str,
                    side: OrderSide,
                    quantity: int,
                    algo: ExecutionAlgo,
                    duration_minutes: int = 60,
                    urgency: float = 1.0,
                    volume_profile: Optional[VolumeProfile] = None
                    ) -> ParentOrder:
        """
        Submit a parent order for algorithmic execution.
        """
        order_id = f"{ticker}_{side.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        order = ParentOrder(
            order_id=order_id,
            ticker=ticker,
            side=side,
            total_quantity=quantity,
            algo=algo,
            start_time=start_time,
            end_time=end_time,
            urgency=urgency
        )
        
        # Generate execution schedule
        executor = self.executors.get(algo, self.executors[ExecutionAlgo.TWAP])
        executor.generate_schedule(order, volume_profile)
        
        order.status = "active"
        self.active_orders[order_id] = order
        
        logger.info(f"Submitted {algo.value} order: {order_id}, "
                   f"{side.value} {quantity} {ticker}")
        
        return order
    
    def execute_due_slices(self,
                          current_time: datetime,
                          market_data: Dict[str, Dict]
                          ) -> List[Tuple[str, OrderSlice]]:
        """
        Execute all due slices across active orders.
        
        Parameters:
        -----------
        current_time : datetime
            Current time
        market_data : Dict[str, Dict]
            Market data per ticker {ticker: {price, volume, spread, etc.}}
        
        Returns:
        --------
        List of (order_id, executed_slice) tuples
        """
        executed = []
        
        for order_id, order in list(self.active_orders.items()):
            if order.status != "active":
                continue
            
            ticker_data = market_data.get(order.ticker, {})
            executor = self.executors.get(order.algo, self.executors[ExecutionAlgo.TWAP])
            
            # Get next due slice
            slice_order = executor.get_next_slice(order, current_time, ticker_data)
            
            if slice_order is not None:
                # Execute the slice
                fill_price, actual_qty = self._execute_slice(
                    slice_order, ticker_data
                )
                
                # Update slice
                slice_order.executed = True
                slice_order.executed_time = current_time
                slice_order.executed_price = fill_price
                slice_order.executed_quantity = actual_qty
                
                # Update parent order
                order.filled_quantity += actual_qty
                
                # Update average price
                total_value = order.avg_price * (order.filled_quantity - actual_qty)
                total_value += fill_price * actual_qty
                order.avg_price = total_value / order.filled_quantity
                
                executed.append((order_id, slice_order))
                
                logger.debug(f"Executed slice {slice_order.slice_id} of {order_id}: "
                           f"{actual_qty} @ ${fill_price:.2f}")
            
            # Check if order is complete
            if order.filled_quantity >= order.total_quantity:
                order.status = "completed"
                self.completed_orders.append(order)
                del self.active_orders[order_id]
                
                logger.info(f"Completed order {order_id}: "
                          f"Avg price ${order.avg_price:.2f}, "
                          f"Slippage {order.slippage_bps:.1f} bps")
        
        return executed
    
    def _execute_slice(self,
                      slice_order: OrderSlice,
                      market_data: Dict) -> Tuple[float, int]:
        """
        Execute a single slice and return fill price and quantity.
        Applies slippage model.
        """
        mid_price = market_data.get('price', 100.0)
        volatility = market_data.get('volatility', 0.02)
        volume = market_data.get('volume', 100000)
        spread_bps = market_data.get('spread_bps', 5)
        
        # Calculate slippage
        slippage = self.slippage_model.estimate_slippage(
            quantity=slice_order.quantity,
            side=slice_order.side,
            mid_price=mid_price,
            volatility=volatility,
            volume=volume,
            spread_bps=spread_bps
        )
        
        # Apply slippage to get fill price
        if slice_order.side == OrderSide.BUY:
            fill_price = mid_price * (1 + slippage)
        else:
            fill_price = mid_price * (1 - slippage)
        
        slice_order.slippage = slippage
        
        return fill_price, slice_order.quantity
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        if order_id in self.active_orders:
            order = self.active_orders.pop(order_id)
            order.status = "cancelled"
            self.completed_orders.append(order)
            logger.info(f"Cancelled order {order_id}")
            return True
        return False
    
    def get_execution_summary(self) -> Dict:
        """Get summary of execution performance"""
        if not self.completed_orders:
            return {'n_orders': 0}
        
        slippages = [o.slippage_bps for o in self.completed_orders]
        fill_rates = [o.fill_rate for o in self.completed_orders]
        
        return {
            'n_orders': len(self.completed_orders),
            'avg_slippage_bps': np.mean(slippages),
            'max_slippage_bps': max(slippages),
            'min_slippage_bps': min(slippages),
            'avg_fill_rate': np.mean(fill_rates),
            'fully_filled': sum(1 for fr in fill_rates if fr >= 1.0)
        }


class SlippageModel:
    """
    Models expected slippage based on order and market characteristics.
    """
    
    def __init__(self,
                 base_impact_bps: float = 5,
                 volatility_factor: float = 2.0,
                 size_factor: float = 0.5):
        self.base_impact_bps = base_impact_bps
        self.volatility_factor = volatility_factor
        self.size_factor = size_factor
    
    def estimate_slippage(self,
                         quantity: int,
                         side: OrderSide,
                         mid_price: float,
                         volatility: float,
                         volume: float,
                         spread_bps: float) -> float:
        """
        Estimate expected slippage as a decimal (not bps).
        
        Components:
        1. Half spread (always pay half the spread)
        2. Market impact (based on size relative to volume)
        3. Volatility risk
        """
        # Half spread component
        half_spread = (spread_bps / 10000) / 2
        
        # Market impact (square root model)
        size_ratio = quantity * mid_price / (volume * mid_price)
        market_impact = self.base_impact_bps / 10000 * np.sqrt(size_ratio) * self.size_factor
        
        # Volatility component
        vol_impact = volatility * self.volatility_factor / 100
        
        # Total slippage
        total_slippage = half_spread + market_impact + vol_impact
        
        return total_slippage


if __name__ == "__main__":
    # Test execution engine
    print("Execution Engine Test")
    print("=" * 50)
    
    # Create engine
    engine = ExecutionEngine({
        'twap_intervals': 10,
        'max_participation': 0.10
    })
    
    # Submit TWAP order
    order = engine.submit_order(
        ticker="AAPL",
        side=OrderSide.BUY,
        quantity=1000,
        algo=ExecutionAlgo.TWAP,
        duration_minutes=30
    )
    
    print(f"Submitted order: {order.order_id}")
    print(f"Number of slices: {len(order.slices)}")
    print(f"Quantity per slice: {order.slices[0].quantity}")
    
    # Simulate execution
    market_data = {
        'AAPL': {
            'price': 150.0,
            'volume': 500000,
            'volatility': 0.02,
            'spread_bps': 3
        }
    }
    
    current_time = order.start_time
    
    for i in range(15):
        current_time += timedelta(minutes=2)
        executed = engine.execute_due_slices(current_time, market_data)
        
        if executed:
            print(f"Time {i*2}min: Executed {len(executed)} slices")
    
    # Summary
    summary = engine.get_execution_summary()
    print("\nExecution Summary:")
    print(f"  Orders: {summary['n_orders']}")
    print(f"  Avg Slippage: {summary['avg_slippage_bps']:.1f} bps")
    print(f"  Avg Fill Rate: {summary['avg_fill_rate']:.1%}")
