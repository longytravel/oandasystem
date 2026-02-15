"""
Base class for strategies that can be fast-optimized.

To make a strategy work with the ultra-fast optimizer:
1. Inherit from FastStrategy
2. Implement get_parameter_space() - return parameter grid
3. Implement precompute() - pre-compute ALL possible signals (done ONCE)
4. Implement filter_signals() - filter pre-computed signals by params (done per trial)

The key insight: separate what CAN be pre-computed from what MUST be filtered per trial.

For full-featured optimization (35+ params with trade management):
5. Implement get_parameter_groups() - organize params into optimization stages
6. Implement get_management_arrays() - return arrays for trailing/BE/partial closes
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, NamedTuple, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


class FastSignal(NamedTuple):
    """Pre-computed signal with all filterable attributes."""
    bar: int              # Bar index for entry
    direction: int        # 1=buy, -1=sell
    price: float          # Entry price
    hour: int             # Hour of day (for time filters)
    day: int              # Day of week (for day filters)
    # Strategy-specific attributes stored here:
    attributes: Dict[str, Any]


@dataclass
class ParameterDef:
    """Definition for a single parameter."""
    name: str
    values: List[Any]
    default: Any
    param_type: str = 'categorical'  # categorical, int, float

    def get_tight_range(self, locked_value: Any, n_values: int = 5) -> List[Any]:
        """Get tight range around a locked value for final optimization."""
        if locked_value not in self.values:
            return [locked_value]

        idx = self.values.index(locked_value)
        n = len(self.values)

        half = n_values // 2
        start = max(0, idx - half)
        end = min(n, idx + half + 1)

        if start == 0:
            end = min(n, n_values)
        if end == n:
            start = max(0, n - n_values)

        return self.values[start:end]


@dataclass
class ParameterGroup:
    """A group of related parameters for staged optimization."""
    name: str
    description: str
    parameters: Dict[str, ParameterDef] = field(default_factory=dict)

    def add_param(self, name: str, values: List[Any], default: Any, param_type: str = 'categorical'):
        """Add a parameter to this group."""
        self.parameters[name] = ParameterDef(name, values, default, param_type)

    def get_defaults(self) -> Dict[str, Any]:
        """Get default values for all params in group."""
        return {name: p.default for name, p in self.parameters.items()}

    def get_space_size(self) -> int:
        """Get total combinations in this group."""
        size = 1
        for p in self.parameters.values():
            size *= len(p.values)
        return size

    def get_param_space(self) -> Dict[str, List[Any]]:
        """Get parameter space for this group."""
        return {name: p.values for name, p in self.parameters.items()}


class FastStrategy(ABC):
    """
    Base class for fast-optimizable strategies.

    The optimization flow:
    1. precompute(df) is called ONCE per dataset
    2. filter_signals(params) is called per trial (must be FAST)
    3. compute_sl_tp(signal, params) returns SL/TP prices

    Performance optimization:
    Strategies can implement _build_signal_arrays() to enable vectorized
    filtering (numpy boolean masks instead of Python for-loops). This gives
    10-50x speedup on the per-trial filter+SL/TP path.
    """

    name: str = "FastStrategy"
    version: str = "1.0"

    def __init__(self):
        self._precomputed_signals: List[FastSignal] = []
        self._df: pd.DataFrame = None
        self._pip_size: float = 0.0001
        # Vectorized signal arrays (populated by _build_signal_arrays)
        self._vec_arrays: Optional[Dict[str, np.ndarray]] = None

    @abstractmethod
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Return parameter grid for optimization.

        Format:
        {
            "param_name": [value1, value2, value3, ...],
        }

        Note: These are DISCRETE values, not ranges.
        The optimizer will test all combinations.
        """
        pass

    @abstractmethod
    def precompute(self, df: pd.DataFrame) -> List[FastSignal]:
        """
        Pre-compute ALL possible signals from data.

        This is called ONCE per dataset. Compute signals for ALL
        parameter combinations - filtering happens later.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            List of FastSignal objects with all possible signals
        """
        pass

    @abstractmethod
    def filter_signals(
        self,
        signals: List[FastSignal],
        params: Dict[str, Any]
    ) -> List[FastSignal]:
        """
        Filter pre-computed signals based on parameters.

        This is called PER TRIAL - must be very fast.
        Only do simple comparisons here, no heavy computation.

        Args:
            signals: Pre-computed signals from precompute()
            params: Trial parameters

        Returns:
            Filtered list of signals
        """
        pass

    @abstractmethod
    def compute_sl_tp(
        self,
        signal: FastSignal,
        params: Dict[str, Any],
        pip_size: float
    ) -> tuple:
        """
        Compute stop loss and take profit for a signal.

        Args:
            signal: The signal
            params: Trial parameters
            pip_size: Pip size for the instrument

        Returns:
            (sl_price, tp_price)
        """
        pass

    def set_pip_size(self, pair: str):
        """Set pip size based on pair."""
        self._pip_size = 0.01 if 'JPY' in pair else 0.0001

    def precompute_for_dataset(self, df: pd.DataFrame) -> int:
        """
        Pre-compute signals and store internally.

        Returns number of signals found.
        """
        self._df = df
        self._precomputed_signals = self.precompute(df)
        # Build vectorized arrays if strategy supports it
        if hasattr(self, '_build_signal_arrays'):
            self._vec_arrays = self._build_signal_arrays(self._precomputed_signals)
        else:
            self._vec_arrays = None
        return len(self._precomputed_signals)

    def get_filtered_arrays(
        self,
        params: Dict[str, Any],
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> tuple:
        """
        Get numpy arrays ready for backtest engine.

        Returns:
            (entry_bars, entry_prices, directions, sl_prices, tp_prices)
            All as numpy arrays.
        """
        # Fast vectorized path
        if self._vec_arrays is not None and hasattr(self, '_filter_vectorized') and hasattr(self, '_compute_sl_tp_vectorized'):
            mask = self._filter_vectorized(params)
            if not np.any(mask):
                return (
                    np.array([], dtype=np.int64),
                    np.array([], dtype=np.float64),
                    np.array([], dtype=np.int64),
                    np.array([], dtype=np.float64),
                    np.array([], dtype=np.float64),
                )
            va = self._vec_arrays
            entry_bars = va['bars'][mask]
            entry_prices = va['prices'][mask]
            directions = va['directions'][mask]
            sl_prices, tp_prices = self._compute_sl_tp_vectorized(mask, params, self._pip_size)
            return entry_bars, entry_prices, directions, sl_prices, tp_prices

        # Fallback: original Python-loop path
        signals = self.filter_signals(self._precomputed_signals, params)

        if not signals:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )

        entry_bars = np.array([s.bar for s in signals], dtype=np.int64)
        entry_prices = np.array([s.price for s in signals], dtype=np.float64)
        directions = np.array([s.direction for s in signals], dtype=np.int64)

        sl_prices = np.zeros(len(signals), dtype=np.float64)
        tp_prices = np.zeros(len(signals), dtype=np.float64)

        for i, signal in enumerate(signals):
            sl, tp = self.compute_sl_tp(signal, params, self._pip_size)
            sl_prices[i] = sl
            tp_prices[i] = tp

        return entry_bars, entry_prices, directions, sl_prices, tp_prices

    def get_parameter_groups(self) -> Optional[Dict[str, ParameterGroup]]:
        """
        Return parameters organized into groups for staged optimization.

        Override this in strategies that support 35+ parameters.

        Groups should be ordered by optimization priority:
        1. signal - Signal generation parameters (RSI period, swing strength, etc.)
        2. filters - Entry filters (slope, trend, time)
        3. risk - Stop loss and take profit settings
        4. management - Trailing, breakeven, partial closes
        5. time - Trading hours and day filters

        Returns:
            Dict of group_name -> ParameterGroup, or None if not supported
        """
        return None

    def get_management_arrays(
        self,
        signals: List[FastSignal],
        params: Dict[str, Any],
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Get trade management arrays for full-featured backtest.

        Override this in strategies that support trailing/BE/partial closes.

        Args:
            signals: Filtered signals
            params: Parameter dictionary

        Returns:
            Dictionary with keys:
            - use_trailing, trail_start_pips, trail_step_pips
            - use_breakeven, be_trigger_pips, be_offset_pips
            - use_partial, partial_pct, partial_target_pips
            - max_bars
            Or None if strategy doesn't support management features.
        """
        return None

    def get_all_arrays(
        self,
        params: Dict[str, Any],
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        days: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Get all arrays needed for full-featured backtest.

        Returns:
            (signal_arrays, management_arrays)
            signal_arrays has: entry_bars, entry_prices, directions, sl_prices, tp_prices
            management_arrays has: all trade management arrays (or empty dict if not supported)
        """
        empty_int = np.array([], dtype=np.int64)
        empty_float = np.array([], dtype=np.float64)

        # Fast vectorized path
        if (self._vec_arrays is not None
            and hasattr(self, '_filter_vectorized')
            and hasattr(self, '_compute_sl_tp_vectorized')
            and hasattr(self, '_get_management_arrays_vectorized')):

            mask = self._filter_vectorized(params)
            if not np.any(mask):
                return {
                    'entry_bars': empty_int,
                    'entry_prices': empty_float,
                    'directions': empty_int,
                    'sl_prices': empty_float,
                    'tp_prices': empty_float,
                }, {}

            va = self._vec_arrays
            sl_prices, tp_prices = self._compute_sl_tp_vectorized(mask, params, self._pip_size)

            signal_arrays = {
                'entry_bars': va['bars'][mask],
                'entry_prices': va['prices'][mask],
                'directions': va['directions'][mask],
                'sl_prices': sl_prices,
                'tp_prices': tp_prices,
            }

            mgmt_arrays = self._get_management_arrays_vectorized(mask, params, sl_prices)
            if mgmt_arrays is None:
                mgmt_arrays = {}

            return signal_arrays, mgmt_arrays

        # Fallback: original Python-loop path
        signals = self.filter_signals(self._precomputed_signals, params)

        if not signals:
            return {
                'entry_bars': empty_int,
                'entry_prices': empty_float,
                'directions': empty_int,
                'sl_prices': empty_float,
                'tp_prices': empty_float,
            }, {}

        n = len(signals)
        entry_bars = np.array([s.bar for s in signals], dtype=np.int64)
        entry_prices = np.array([s.price for s in signals], dtype=np.float64)
        directions = np.array([s.direction for s in signals], dtype=np.int64)

        sl_prices = np.zeros(n, dtype=np.float64)
        tp_prices = np.zeros(n, dtype=np.float64)

        for i, signal in enumerate(signals):
            sl, tp = self.compute_sl_tp(signal, params, self._pip_size)
            sl_prices[i] = sl
            tp_prices[i] = tp

        signal_arrays = {
            'entry_bars': entry_bars,
            'entry_prices': entry_prices,
            'directions': directions,
            'sl_prices': sl_prices,
            'tp_prices': tp_prices,
        }

        # Get management arrays if supported
        mgmt_arrays = self.get_management_arrays(signals, params)
        if mgmt_arrays is None:
            mgmt_arrays = {}

        return signal_arrays, mgmt_arrays

    def get_signal_attributes(self, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Get strategy-specific signal attribute arrays for ML feature enrichment.

        Returns arrays from _vec_arrays that represent strategy-specific signal
        qualities (e.g. rsi_diff, bars_between, price_slope), excluding core
        arrays (bars, prices, directions, hours, days) which are already used
        by the market-level features.

        The arrays are filtered by the strategy's current parameter set.

        Returns:
            Dict mapping attribute name -> np.ndarray of values per signal.
            Empty dict if vectorized arrays are not available.
        """
        if self._vec_arrays is None:
            return {}

        # Core arrays that are NOT strategy-specific attributes
        core_keys = {'bars', 'prices', 'directions', 'hours', 'days'}

        # Return all non-core arrays as strategy attributes
        attrs = {}
        for key, arr in self._vec_arrays.items():
            if key not in core_keys:
                attrs[key] = arr

        return attrs

    def supports_full_management(self) -> bool:
        """Check if strategy supports full trade management features."""
        return self.get_parameter_groups() is not None

    def supports_grid(self) -> bool:
        """Check if strategy supports multiple concurrent positions (grid trading).

        Override and return True in grid strategies. When True, the optimizer
        and pipeline stages will use grid_backtest_numba instead of the
        single-position engines.
        """
        return False

    def get_grid_arrays(
        self,
        params: Dict[str, Any],
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Get signal arrays with group_ids for grid backtesting.

        Returns:
            (entry_bars, entry_prices, directions, sl_prices, tp_prices, group_ids)
            where group_ids links signals from the same grid setup.
            Returns None if strategy doesn't support grid.
        """
        return None
