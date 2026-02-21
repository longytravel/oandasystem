"""
Unified Optimizer - Single class for all optimization modes.

Implements MT5-style ranking:
- R² equity curve smoothness
- OnTester score: Profit × R² × PF × √Trades / (DD + 5)
- Combined Rank = Back_Rank + Forward_Rank (lower is better)

Modes:
- quick: Basic SL/TP, ~700 trials/sec (signal param tuning)
- full: All trade management, ~400 trials/sec (full param tuning)
- staged: Group-by-group optimization with parameter locking

Usage:
    from strategies.rsi_full_v3 import RSIDivergenceFullFastV3
    from optimization.unified_optimizer import UnifiedOptimizer

    strategy = RSIDivergenceFullFastV3()
    optimizer = UnifiedOptimizer(strategy)

    # Staged mode with combined ranking
    results = optimizer.run(df_back, df_forward, mode='staged', trials_per_stage=5000)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, NamedTuple, Tuple
from pathlib import Path
from datetime import datetime
import json
import itertools
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import gc
warnings.filterwarnings('ignore')

from loguru import logger

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from optimization.fast_strategy import FastStrategy, ParameterGroup
from optimization.numba_backtest import (
    basic_backtest_numba, full_backtest_numba, grid_backtest_numba,
    Metrics, get_quote_conversion_rate,
    calculate_quality_score, quality_score_from_metrics,
)


_TF_BARS_PER_YEAR = {
    'M1': 252 * 1440, 'M5': 252 * 288, 'M15': 252 * 96,
    'M30': 252 * 48, 'H1': 252 * 22, 'H4': 252 * 6, 'D': 252,
}


def _bars_per_year(timeframe: str) -> float:
    """Return annualised bar count for a given timeframe string."""
    return float(_TF_BARS_PER_YEAR.get(timeframe, 252 * 22))


class UnifiedOptimizer:
    """
    Unified optimizer with quality-score-based combined back+forward ranking.

    Quality Score = Sortino × R² × min(PF,5) × √min(Trades,200) × (1 + min(Return%,200)/100) / (Ulcer + MaxDD%/2 + 5)

    This universal metric rewards smooth equity curves (R²), risk-adjusted profit (Sortino),
    absolute profit (return multiplier), efficiency (PF), and penalizes drawdown (Ulcer + MaxDD).

    Ranking Method:
    1. Run back optimization, maximize quality_score
    2. Forward test ALL valid results
    3. Filter: forward quality_score >= back × min_forward_ratio (reject overfit)
    4. Rank by back quality_score (1 = best)
    5. Rank by forward quality_score (1 = best)
    6. Combined Rank = back_rank + forward_rank × forward_rank_weight
    7. Sort by combined rank (ascending - lower is better)

    A result that's consistent in BOTH periods beats one that's amazing
    in one but terrible in the other.
    """

    def __init__(
        self,
        strategy: FastStrategy,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 1.0,
        spread_pips: float = 1.5,
        slippage_pips: float = 0.0,
        min_trades: int = 20,
        # V2 parameters
        min_forward_ratio: float = 0.0,     # Min forward/back OnTester ratio (0=disabled)
        forward_rank_weight: float = 1.0,   # Weight for forward rank in combined ranking
        # Performance parameters
        n_jobs: int = 1,                    # Parallel workers (-1 = all cores)
        # V3 parameters
        pair: str = 'GBP_USD',              # Currency pair for pip value calculation
        account_currency: str = 'USD',      # Account currency for conversion
        timeframe: str = 'H1',              # Timeframe for bars_per_year calculation
        # Hard pre-filters
        max_dd_hard_limit: float = 30.0,    # MaxDD > this → instant reject
        min_r2_hard: float = 0.5,           # R² < this → instant reject
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.min_trades = min_trades

        # V2 parameters
        self.min_forward_ratio = min_forward_ratio
        self.forward_rank_weight = forward_rank_weight

        # Performance
        self.n_jobs = n_jobs

        # Hard pre-filters
        self.max_dd_hard_limit = max_dd_hard_limit
        self.min_r2_hard = min_r2_hard

        # V3: Store pair and account currency for pip value calculation
        self.pair = pair
        self.account_currency = account_currency

        # Compute bars_per_year from timeframe
        self.bars_per_year = _bars_per_year(timeframe)

        # Get pip_size from strategy (must call strategy.set_pip_size(pair) first)
        # This handles JPY pairs (0.01) vs other pairs (0.0001) automatically
        if hasattr(strategy, '_pip_size') and strategy._pip_size > 0:
            self.pip_size = strategy._pip_size
        else:
            self.pip_size = 0.0001  # Default for EUR/USD, GBP/USD, etc.

        # V3 FIX: Get quote conversion rate for cross-currency pairs
        self.quote_conversion_rate = get_quote_conversion_rate(pair, account_currency)

        # Data storage
        self.df_back: pd.DataFrame = None
        self.df_forward: pd.DataFrame = None
        self.back_arrays: Dict[str, np.ndarray] = {}
        self.fwd_arrays: Dict[str, np.ndarray] = {}

        # Signals storage
        self.back_signals = None
        self.fwd_signals = None

        # Results
        self.results: List[Dict] = []
        self.all_back_results: List[Dict] = []
        self.stage_results: Dict[str, Dict] = {}
        self.locked_params: Dict[str, Any] = {}
        self.param_importance: Dict[str, float] = {}
        self.optuna_study: Optional['optuna.Study'] = None

    def _prepare_data(self, df_back: pd.DataFrame, df_forward: pd.DataFrame):
        """Prepare price arrays from dataframes."""
        self.df_back = df_back
        self.df_forward = df_forward

        # Back arrays
        self.back_arrays = {
            'highs': df_back['high'].values.astype(np.float64),
            'lows': df_back['low'].values.astype(np.float64),
            'closes': df_back['close'].values.astype(np.float64),
            'days': df_back.index.dayofweek.values.astype(np.int64),
        }

        # Forward arrays
        self.fwd_arrays = {
            'highs': df_forward['high'].values.astype(np.float64),
            'lows': df_forward['low'].values.astype(np.float64),
            'closes': df_forward['close'].values.astype(np.float64),
            'days': df_forward.index.dayofweek.values.astype(np.int64),
        }

    def _precompute_signals(self):
        """Pre-compute signals for both datasets."""
        logger.info(f"Pre-computing signals for {self.strategy.name}...")

        start = time.time()
        n_back = self.strategy.precompute_for_dataset(self.df_back)
        self.back_signals = self.strategy._precomputed_signals.copy()
        # Also save vectorized arrays so we can swap them correctly later
        self.back_vec_arrays = getattr(self.strategy, '_vec_arrays', None)
        back_time = time.time() - start

        start = time.time()
        n_fwd = self.strategy.precompute_for_dataset(self.df_forward)
        self.fwd_signals = self.strategy._precomputed_signals.copy()
        self.fwd_vec_arrays = getattr(self.strategy, '_vec_arrays', None)
        fwd_time = time.time() - start

        logger.info(f"Pre-computed: {n_back} back, {n_fwd} forward signals "
                   f"({back_time:.1f}s + {fwd_time:.1f}s)")

    def _run_basic_trial(self, params: Dict, arrays: Dict) -> Metrics:
        """Run basic SL/TP only backtest."""
        entry_bars, entry_prices, directions, sl_prices, tp_prices = \
            self.strategy.get_filtered_arrays(
                params,
                arrays['highs'],
                arrays['lows'],
                arrays['closes']
            )

        if len(entry_bars) < 3:
            return Metrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        result = basic_backtest_numba(
            entry_bars, entry_prices, directions,
            sl_prices, tp_prices,
            arrays['highs'],
            arrays['lows'],
            arrays['closes'],
            self.initial_capital,
            self.risk_per_trade,
            self.pip_size,
            self.quote_conversion_rate,  # V3 FIX: Pass quote conversion rate
            self.bars_per_year,
            self.spread_pips,
            slippage_pips=self.slippage_pips,
        )

        return Metrics(*result)

    def _run_full_trial(self, params: Dict, arrays: Dict) -> Metrics:
        """Run full-featured backtest with trade management."""
        signal_arrays, mgmt_arrays = self.strategy.get_all_arrays(
            params,
            arrays['highs'],
            arrays['lows'],
            arrays['closes'],
            arrays['days'],
        )

        if len(signal_arrays['entry_bars']) < 3:
            return Metrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # Get management arrays (with defaults if not provided)
        n = len(signal_arrays['entry_bars'])

        use_trailing = mgmt_arrays.get('use_trailing', np.zeros(n, dtype=np.bool_))
        trail_start = mgmt_arrays.get('trail_start_pips', np.zeros(n, dtype=np.float64))
        trail_step = mgmt_arrays.get('trail_step_pips', np.zeros(n, dtype=np.float64))
        use_be = mgmt_arrays.get('use_breakeven', np.zeros(n, dtype=np.bool_))
        be_trigger = mgmt_arrays.get('be_trigger_pips', np.zeros(n, dtype=np.float64))
        be_offset = mgmt_arrays.get('be_offset_pips', np.zeros(n, dtype=np.float64))
        use_partial = mgmt_arrays.get('use_partial', np.zeros(n, dtype=np.bool_))
        partial_pct = mgmt_arrays.get('partial_pct', np.zeros(n, dtype=np.float64))
        partial_target = mgmt_arrays.get('partial_target_pips', np.zeros(n, dtype=np.float64))
        max_bars = mgmt_arrays.get('max_bars', np.zeros(n, dtype=np.int64))

        # V5: New exit management arrays (defaults for backward compat with V3/V4)
        trail_mode = mgmt_arrays.get('trail_mode', np.zeros(n, dtype=np.int64))
        chandelier_atr_mult = mgmt_arrays.get('chandelier_atr_mult', np.full(n, 3.0, dtype=np.float64))
        atr_pips_arr = mgmt_arrays.get('atr_pips', np.full(n, 35.0, dtype=np.float64))
        stale_exit_bars = mgmt_arrays.get('stale_exit_bars', np.zeros(n, dtype=np.int64))

        # V6: ML exit arrays
        n_bars = len(arrays['highs'])
        use_ml = mgmt_arrays.get('use_ml_exit', np.zeros(n, dtype=np.bool_))
        ml_min_hold_arr = mgmt_arrays.get('ml_min_hold', np.zeros(n, dtype=np.int64))
        ml_threshold_arr = mgmt_arrays.get('ml_threshold', np.ones(n, dtype=np.float64))

        # Compute ML scores if strategy supports it and ML exit is enabled
        if hasattr(self.strategy, 'get_ml_score_arrays') and np.any(use_ml):
            ml_long, ml_short = self.strategy.get_ml_score_arrays(
                params, arrays['highs'], arrays['lows'], arrays['closes']
            )
        else:
            ml_long = np.zeros(n_bars, dtype=np.float64)
            ml_short = np.zeros(n_bars, dtype=np.float64)

        # V2: Get quality multiplier if available (empty array if not)
        quality_mult = mgmt_arrays.get('quality_mult', np.empty(0, dtype=np.float64))

        result = full_backtest_numba(
            signal_arrays['entry_bars'],
            signal_arrays['entry_prices'],
            signal_arrays['directions'],
            signal_arrays['sl_prices'],
            signal_arrays['tp_prices'],
            use_trailing, trail_start, trail_step,
            use_be, be_trigger, be_offset,
            use_partial, partial_pct, partial_target,
            max_bars,
            trail_mode, chandelier_atr_mult, atr_pips_arr, stale_exit_bars,
            ml_long, ml_short, use_ml, ml_min_hold_arr, ml_threshold_arr,
            arrays['highs'],
            arrays['lows'],
            arrays['closes'],
            arrays['days'],
            self.initial_capital,
            self.risk_per_trade,
            self.pip_size,
            params.get('max_daily_trades', 0),
            params.get('max_daily_loss_pct', 0.0),
            quality_mult,  # V2: Quality-based position sizing
            self.quote_conversion_rate,  # V3 FIX: Quote currency conversion
            self.bars_per_year,
            0,  # ml_exit_cooldown_bars (positional required for numba)
            self.spread_pips,
            slippage_pips=self.slippage_pips,
        )

        return Metrics(*result)

    def _swap_to_back(self):
        """Swap strategy signals and vec_arrays to back-test data."""
        self.strategy._precomputed_signals = self.back_signals
        self.strategy._vec_arrays = getattr(self, 'back_vec_arrays', None)

    def _swap_to_forward(self):
        """Swap strategy signals and vec_arrays to forward-test data."""
        self.strategy._precomputed_signals = self.fwd_signals
        self.strategy._vec_arrays = getattr(self, 'fwd_vec_arrays', None)

    def _select_trial_runner(self, use_full: bool):
        """Select the appropriate trial runner based on strategy capabilities."""
        if self.strategy.supports_grid():
            return self._run_grid_trial
        return self._run_full_trial if use_full else self._run_basic_trial

    def _run_grid_trial(self, params: Dict, arrays: Dict) -> Metrics:
        """Run grid backtest for strategies supporting multiple concurrent positions."""
        grid_result = self.strategy.get_grid_arrays(
            params, arrays['highs'], arrays['lows'], arrays['closes']
        )

        if grid_result is None:
            return Metrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        entry_bars, entry_prices, directions, sl_prices, tp_prices, group_ids = grid_result

        if len(entry_bars) < 3:
            return Metrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        max_concurrent = params.get('grid_orders', 0) + 1
        if max_concurrent < 2:
            max_concurrent = 11  # Default: 1 initial + 10 grid

        result = grid_backtest_numba(
            entry_bars, entry_prices, directions,
            sl_prices, tp_prices, group_ids,
            arrays['highs'], arrays['lows'], arrays['closes'],
            self.initial_capital,
            self.risk_per_trade,
            self.pip_size,
            max_concurrent,
            self.quote_conversion_rate,
            self.bars_per_year,
            self.spread_pips,
            slippage_pips=self.slippage_pips,
        )

        return Metrics(*result)

    def _generate_param_grid(self, param_space: Dict, n_trials: int) -> List[Dict]:
        """Generate parameter combinations."""
        keys = list(param_space.keys())
        all_combos = list(itertools.product(*[param_space[k] for k in keys]))
        total = len(all_combos)

        logger.info(f"Parameter space: {len(keys)} params, {total:,} total combinations")

        if total > n_trials:
            np.random.seed(42)
            indices = np.random.choice(total, n_trials, replace=False)
            selected = [all_combos[i] for i in indices]
            logger.info(f"Randomly sampling {n_trials:,} from {total:,}")
        else:
            selected = all_combos
            logger.info(f"Testing all {total:,} combinations")

        return [{k: v for k, v in zip(keys, combo)} for combo in selected]

    def _run_grid_optimization(
        self,
        n_trials: int,
        use_full: bool,
        min_sharpe: float,
        param_space: Dict = None,
        base_params: Dict = None,
    ) -> List[Dict]:
        """Run grid-based optimization."""
        if param_space is None:
            param_space = self.strategy.get_parameter_space()
        if base_params is None:
            base_params = {}

        param_list = self._generate_param_grid(param_space, n_trials)
        self._swap_to_back()

        start = time.time()
        back_results = []

        run_trial = self._select_trial_runner(use_full)

        for i, params in enumerate(param_list):
            # Merge with base params
            full_params = {**base_params, **params}

            metrics = run_trial(full_params, self.back_arrays)

            # Validity: positive quality_score (requires Sharpe>0, R²>0, PF>0)
            # plus minimum Sharpe threshold
            qs = quality_score_from_metrics(metrics)
            if (metrics.trades >= self.min_trades and
                qs > 0 and
                metrics.sharpe >= min_sharpe):
                back_results.append({
                    'trial_id': i,
                    'params': full_params,
                    'back': metrics,
                })

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                logger.info(f"  {i+1:,}/{len(param_list):,} ({rate:.0f}/sec)")

        back_time = time.time() - start
        logger.info(f"Back complete: {len(back_results)} valid in {back_time:.1f}s "
                   f"({len(param_list)/back_time:.0f}/sec)")

        return back_results

    def _run_optuna_optimization(
        self,
        n_trials: int,
        use_full: bool,
        min_sharpe: float,
        param_space: Dict = None,
        base_params: Dict = None,
    ) -> Tuple[List[Dict], 'optuna.Study']:
        """Run Optuna TPE optimization.

        Uses batched execution to avoid a known heap corruption bug in
        Optuna's in-memory storage that corrupts trial state after ~500-5000
        trials on Windows/Python 3.11.  Each batch creates a fresh study with
        an independent TPE sampler.  With categorical parameters and
        independent (non-multivariate) TPE, each parameter's distribution is
        modeled separately, so per-batch TPE converges quickly and the total
        search coverage equals a single monolithic run.
        """
        if param_space is None:
            param_space = self.strategy.get_parameter_space()
        if base_params is None:
            base_params = {}

        self._swap_to_back()
        run_trial = self._select_trial_runner(use_full)

        def objective(trial: 'optuna.Trial') -> float:
            params = dict(base_params)
            for key, values in param_space.items():
                params[key] = trial.suggest_categorical(key, values)

            # Prune invalid parameter combinations early
            if params.get('use_trailing', False):
                if params.get('trail_step_pips', 10) > params.get('trail_start_pips', 50):
                    return -200  # trail_step > trail_start is invalid
            if params.get('use_break_even', False):
                be_offset = params.get('be_offset_pips', 5)
                be_trigger = params.get('be_atr_mult', 0.5) * params.get('atr_pips', 35.0)
                if be_offset > be_trigger:
                    return -200  # BE offset > trigger is invalid

            metrics = run_trial(params, self.back_arrays)

            trial.set_user_attr('params', params)
            trial.set_user_attr('metrics', metrics)

            if metrics.trades < self.min_trades:
                return -100 + metrics.trades

            # Hard pre-filters: reject garbage early so optimizer focuses on viable systems
            if metrics.max_dd > self.max_dd_hard_limit:
                return -300
            if metrics.r_squared < self.min_r2_hard:
                return -200
            if metrics.total_return < 2.0:
                return -150  # Reject near-zero-return regimes (e.g. BE offset cancels spread+slippage)
            if metrics.max_dd < 0.5:
                return -125  # MaxDD < 0.5% over full backtest is always an artifact

            # Optimize for quality_score: Sortino × R² × min(PF,5) × √min(Trades,200) / (Ulcer + MaxDD/2 + 5)
            return quality_score_from_metrics(metrics)

        # --- Batched TPE execution ---
        # Fresh study per batch avoids heap corruption while preserving
        # full Bayesian optimization quality.  Each batch varies the seed
        # so different regions of the search space are explored.
        BATCH_SIZE = 500
        all_completed = []  # FrozenTrial objects from all batches
        n_batches = (n_trials + BATCH_SIZE - 1) // BATCH_SIZE

        start = time.time()

        for b in range(n_batches):
            batch_n = min(BATCH_SIZE, n_trials - b * BATCH_SIZE)
            sampler = optuna.samplers.TPESampler(seed=42 + b)
            study = optuna.create_study(direction='maximize', sampler=sampler)

            try:
                study.optimize(objective, n_trials=batch_n,
                               show_progress_bar=False, n_jobs=1)
            except Exception as e:
                logger.warning(f"Batch {b+1}/{n_batches} crashed after "
                               f"{len(study.trials)} trials: {e}")

            # Collect completed trials from this batch
            for t in study.trials:
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
                    all_completed.append(t)

            del study
            gc.collect()

            # Progress logging every 1000 trials
            total_done = len(all_completed)
            elapsed = time.time() - start
            if total_done > 0 and (total_done % 1000 < BATCH_SIZE or b == n_batches - 1):
                rate = total_done / elapsed
                best_val = max(t.value for t in all_completed)
                logger.info(f"  {total_done:,}/{n_trials:,} ({rate:.0f}/sec) best={best_val:.2f}")

        back_time = time.time() - start

        # Build a merged study for param importance analysis
        final_sampler = optuna.samplers.TPESampler(seed=42)
        final_study = optuna.create_study(direction='maximize', sampler=final_sampler)
        for t in all_completed:
            final_study.add_trial(t)

        # Extract valid results: positive quality_score + min Sharpe
        back_results = []
        for i, trial in enumerate(all_completed):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                metrics = trial.user_attrs.get('metrics')
                if metrics:
                    qs = quality_score_from_metrics(metrics)
                    if (metrics.trades >= self.min_trades and
                        qs > 0 and
                        metrics.sharpe >= min_sharpe):
                        back_results.append({
                            'trial_id': i,
                            'params': trial.user_attrs['params'],
                            'back': metrics,
                        })

        logger.info(f"Back complete: {len(back_results)} valid in {back_time:.1f}s "
                   f"({n_trials/back_time:.0f}/sec)")

        return back_results, final_study

    def _run_staged_optimization(
        self,
        trials_per_stage: int,
        final_trials: int,
        use_full: bool,
        min_sharpe: float,
    ) -> List[Dict]:
        """Run staged group-by-group optimization."""
        groups = self.strategy.get_parameter_groups()
        if not groups:
            raise ValueError(f"Strategy {self.strategy.name} doesn't support staged optimization")

        group_order = list(groups.keys())
        self.locked_params = {}
        self.stage_results = {}

        # Get all defaults
        all_defaults = {}
        for group in groups.values():
            all_defaults.update(group.get_defaults())

        logger.info("\n" + "="*70)
        logger.info("STAGED OPTIMIZATION PLAN")
        logger.info("="*70)

        total_params = 0
        for i, name in enumerate(group_order):
            group = groups[name]
            n_params = len(group.parameters)
            total_params += n_params
            logger.info(f"  Stage {i+1}: {name:<20} ({n_params} params, {group.get_space_size():,} combos)")

        logger.info(f"  Final:  All {total_params} params with tight ranges")
        logger.info(f"\nTrials: {trials_per_stage:,}/stage x {len(group_order)} + {final_trials:,} final")
        logger.info("="*70 + "\n")

        # === STAGE OPTIMIZATION ===
        for group_name in group_order:
            group = groups[group_name]
            param_space = group.get_param_space()

            # Base params: locked + defaults (excluding this group)
            base_params = dict(all_defaults)
            base_params.update(self.locked_params)

            logger.info(f"\n{'='*60}")
            logger.info(f"STAGE: {group_name.upper()} - {len(param_space)} params")
            logger.info(f"{'='*60}")

            # Run optimization for this group
            self._swap_to_back()
            back_results, study = self._run_optuna_optimization(
                trials_per_stage,
                use_full,
                min_sharpe,
                param_space=param_space,
                base_params=base_params,
            )

            # Get best result by quality_score
            if back_results:
                back_results.sort(key=lambda x: quality_score_from_metrics(x['back']), reverse=True)
                best = back_results[0]
                best_qs = quality_score_from_metrics(best['back'])

                # Lock params from this group
                for param_name in param_space:
                    self.locked_params[param_name] = best['params'][param_name]
                    logger.info(f"  Locked: {param_name} = {self.locked_params[param_name]}")

                self.stage_results[group_name] = {
                    'best_quality_score': best_qs,
                    'best_sharpe': best['back'].sharpe,
                    'best_r2': best['back'].r_squared,
                    'n_valid': len(back_results),
                    'n_trials': trials_per_stage,
                    'best_params': best['params'],
                }
            else:
                logger.warning(f"No valid results for stage {group_name}")
                # Use defaults
                for param_name in param_space:
                    self.locked_params[param_name] = group.get_defaults()[param_name]

        # === BUILD TIGHT RANGES ===
        logger.info("\n" + "="*60)
        logger.info("BUILDING TIGHT RANGES FOR FINAL OPTIMIZATION")
        logger.info("="*60)

        tight_ranges = {}
        for group_name, group in groups.items():
            for param_name, param_def in group.parameters.items():
                locked_val = self.locked_params.get(param_name, param_def.default)
                tight = param_def.get_tight_range(locked_val, 5)
                tight_ranges[param_name] = tight
                logger.info(f"  {param_name}: {locked_val} -> {tight}")

        tight_space = 1
        for values in tight_ranges.values():
            tight_space *= len(values)
        logger.info(f"\nTight search space: {tight_space:,} combinations")

        # === FINAL OPTIMIZATION ===
        logger.info(f"\n{'='*60}")
        logger.info("FINAL OPTIMIZATION")
        logger.info(f"{'='*60}")

        back_results, study = self._run_optuna_optimization(
            final_trials,
            use_full,
            min_sharpe,
            param_space=tight_ranges,
            base_params={},
        )

        self.optuna_study = study
        return back_results

    def _forward_test_all(self, back_results: List[Dict], use_full: bool) -> List[Dict]:
        """
        Run forward testing on ALL valid back results.

        This is required for combined ranking - we need forward scores for all results.
        """
        logger.info(f"\nForward testing ALL {len(back_results)} valid results...")

        self._swap_to_forward()
        run_trial = self._select_trial_runner(use_full)

        start = time.time()

        # Parallel forward testing using thread pool (Numba releases GIL)
        n_workers = self.n_jobs if self.n_jobs > 0 else os.cpu_count()
        if n_workers > 1 and len(back_results) > 10:
            # Parallel execution
            def eval_forward(idx_result):
                idx, result = idx_result
                return idx, run_trial(result['params'], self.fwd_arrays)

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(eval_forward, (i, r)): i
                          for i, r in enumerate(back_results)}
                for future in as_completed(futures):
                    idx, metrics = future.result()
                    back_results[idx]['forward'] = metrics
        else:
            # Sequential (single-threaded or small batch)
            for result in back_results:
                metrics = run_trial(result['params'], self.fwd_arrays)
                result['forward'] = metrics

        fwd_time = time.time() - start
        logger.info(f"Forward complete: {len(back_results)} trials in {fwd_time:.1f}s")

        return back_results

    def _apply_combined_ranking(self, results: List[Dict]) -> List[Dict]:
        """
        Apply quality-score-based combined ranking.

        Uses quality_score (Sharpe × R² × PF × √Trades / (DD+5)) for all ranking,
        ensuring smooth equity + profit + low drawdown are rewarded universally.

        Combined Rank = Back_Rank + Forward_Rank × forward_rank_weight

        A result consistent in BOTH periods beats one amazing in one but poor in the other.
        Lower combined rank = better.
        """
        # Filter to results with forward trades
        valid = [r for r in results if r.get('forward') and r['forward'].trades >= 5]

        if not valid:
            return []

        # Apply forward performance threshold using quality_score
        if self.min_forward_ratio > 0:
            filtered = []
            rejected_count = 0
            for r in valid:
                back_qs = quality_score_from_metrics(r['back'])
                fwd_qs = quality_score_from_metrics(r['forward'])

                # Require forward quality_score to be at least X% of back
                if back_qs > 0 and fwd_qs >= back_qs * self.min_forward_ratio:
                    filtered.append(r)
                else:
                    rejected_count += 1

            if filtered:
                logger.info(f"Forward threshold filter: kept {len(filtered)}, rejected {rejected_count} "
                           f"(quality_score ratio < {self.min_forward_ratio:.1%})")
                valid = filtered
            else:
                logger.warning(f"Forward threshold filter rejected ALL results - disabling filter")

        # Rank by back quality_score (1 = best)
        valid.sort(key=lambda x: quality_score_from_metrics(x['back']), reverse=True)
        for i, r in enumerate(valid):
            r['back_rank'] = i + 1

        # Rank by forward quality_score (1 = best)
        valid.sort(key=lambda x: quality_score_from_metrics(x['forward']), reverse=True)
        for i, r in enumerate(valid):
            r['forward_rank'] = i + 1

        # Weighted combined rank
        for r in valid:
            r['combined_rank'] = r['back_rank'] + r['forward_rank'] * self.forward_rank_weight

        # Sort by combined rank (ascending - lower is better)
        valid.sort(key=lambda x: x['combined_rank'])

        return valid

    def run(
        self,
        df_back: pd.DataFrame,
        df_forward: pd.DataFrame,
        mode: str = 'quick',
        n_trials: int = 10000,
        trials_per_stage: int = 5000,
        final_trials: int = 10000,
        top_n: int = 200,  # Now only used for display, not filtering
        min_back_sharpe: float = 1.0,
        use_optuna: bool = True,
    ) -> List[Dict]:
        """
        Run optimization with MT5-style combined ranking.

        Args:
            df_back: Back-testing data
            df_forward: Forward validation data
            mode: 'quick' (basic SL/TP), 'full' (all features), 'staged' (group-by-group),
                  'fullspace' (full params without staging - may find more diverse solutions)
            n_trials: Trials for quick/full/fullspace mode
            trials_per_stage: Trials per stage for staged mode
            final_trials: Final stage trials for staged mode
            top_n: Top N results to display (all are forward tested)
            min_back_sharpe: Minimum back Sharpe to qualify
            use_optuna: Use Optuna TPE (True) or grid sampling (False)

        Returns:
            List of results sorted by COMBINED RANK (lower = better)
        """
        # Prepare data
        self._prepare_data(df_back, df_forward)
        self._precompute_signals()

        use_full = mode in ('full', 'staged', 'fullspace')

        # === BACK OPTIMIZATION ===
        logger.info(f"\n{'='*70}")
        logger.info(f"BACK OPTIMIZATION - Mode: {mode.upper()}")
        logger.info(f"{'='*70}")

        if mode == 'staged':
            back_results = self._run_staged_optimization(
                trials_per_stage, final_trials, use_full, min_back_sharpe
            )
        elif mode == 'fullspace':
            # Full parameter space optimization - no staging
            # This explores the entire space without greedy locking, potentially
            # finding parameter interactions that staged optimization misses
            logger.info("FULLSPACE MODE: Searching entire parameter space without staging")
            logger.info("This may find more diverse solutions but takes longer to converge")

            param_space = self.strategy.get_parameter_space()
            total_combos = 1
            for values in param_space.values():
                total_combos *= len(values)
            logger.info(f"Total parameter space: {len(param_space)} params, {total_combos:,} combinations")

            if use_optuna and OPTUNA_AVAILABLE:
                back_results, self.optuna_study = self._run_optuna_optimization(
                    n_trials, use_full, min_back_sharpe, param_space=param_space
                )
            else:
                back_results = self._run_grid_optimization(
                    n_trials, use_full, min_back_sharpe, param_space=param_space
                )
        elif use_optuna and OPTUNA_AVAILABLE:
            back_results, self.optuna_study = self._run_optuna_optimization(
                n_trials, use_full, min_back_sharpe
            )
        else:
            back_results = self._run_grid_optimization(
                n_trials, use_full, min_back_sharpe
            )

        self.all_back_results = back_results

        if not back_results:
            logger.warning("No valid back results!")
            return []

        # === FORWARD TEST ALL RESULTS ===
        # Must test ALL for combined ranking to work properly
        back_results = self._forward_test_all(back_results, use_full)

        # === APPLY COMBINED RANKING ===
        self.results = self._apply_combined_ranking(back_results)

        # Calculate param importance then release study
        if self.optuna_study:
            self._calculate_param_importance()
            self.optuna_study = None  # Release Optuna study memory

        # Release all_back_results - only top N in self.results are needed downstream
        n_back = len(self.all_back_results)
        self.all_back_results = []
        logger.info(f"Released {n_back} back results from memory (kept top {len(self.results)})")

        return self.results

    def _calculate_param_importance(self):
        """Calculate parameter importance from Optuna study."""
        if not self.optuna_study:
            return

        try:
            importance = optuna.importance.get_param_importances(self.optuna_study)
            self.param_importance = dict(importance)
        except Exception as e:
            logger.warning(f"Could not calculate param importance: {e}")

    def print_results(self, top_n: int = 30):
        """Print comparison table with combined ranking."""
        print("\n" + "=" * 130)
        print(f"COMBINED RANKING RESULTS - {self.strategy.name}")
        if self.forward_rank_weight != 1.0:
            print(f"Forward Rank Weight: {self.forward_rank_weight}x")
        if self.min_forward_ratio > 0:
            print(f"Min Forward Ratio: {self.min_forward_ratio:.1%}")
        print("=" * 130)
        print(f"{'Comb':<6} {'Back#':<7} {'Fwd#':<7} {'BackQS':<10} {'FwdQS':<10} "
              f"{'BackSh':<9} {'FwdSh':<9} {'BackR²':<8} {'FwdR²':<8} {'BackTr':<8} {'FwdTr':<8} {'BackDD%':<9} {'FwdDD%':<9}")
        print("-" * 140)

        for r in self.results[:top_n]:
            back = r['back']
            fwd = r['forward']

            print(f"{r['combined_rank']:<6} "
                  f"{r['back_rank']:<7} "
                  f"{r['forward_rank']:<7} "
                  f"{quality_score_from_metrics(back):<10.3f} "
                  f"{quality_score_from_metrics(fwd):<10.3f} "
                  f"{back.sharpe:<9.3f} "
                  f"{fwd.sharpe:<9.3f} "
                  f"{back.r_squared:<8.3f} "
                  f"{fwd.r_squared:<8.3f} "
                  f"{back.trades:<8} "
                  f"{fwd.trades:<8} "
                  f"{back.max_dd:<9.1f} "
                  f"{fwd.max_dd:<9.1f}")

        print("=" * 130)

        if self.results:
            print(f"\nBest combined result: Trial #{self.results[0]['trial_id']} "
                  f"(Combined Rank: {self.results[0]['combined_rank']})")

    def print_stage_summary(self):
        """Print summary of staged optimization."""
        if not self.stage_results:
            print("No stage results (use mode='staged')")
            return

        print("\n" + "="*80)
        print("STAGE SUMMARY")
        print("="*80)
        print(f"{'Stage':<20} {'Best QS':<12} {'Best Sharpe':<15} {'Best R²':<10} {'Valid/Trials':<20}")
        print("-"*80)

        for name, result in self.stage_results.items():
            qs = result.get('best_quality_score', result.get('best_sharpe', 0))
            print(f"{name:<20} {qs:<12.3f} {result.get('best_sharpe', 0):<15.3f} "
                  f"{result['best_r2']:<10.3f} "
                  f"{result['n_valid']}/{result['n_trials']}")

        print("="*80)

    def print_param_importance(self, top_n: int = 20):
        """Print parameter importance."""
        if not self.param_importance:
            print("\nParameter importance not available (run with use_optuna=True)")
            return

        print("\n" + "=" * 60)
        print("PARAMETER IMPORTANCE")
        print("=" * 60)

        sorted_params = sorted(
            self.param_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for param, importance in sorted_params[:top_n]:
            bar = "#" * int(importance * 50)
            print(f"{param:<30} {importance:>6.1%} {bar}")

        print("=" * 60)

    def analyze_parameter_stability(
        self,
        params: Dict,
        use_full: bool = True,
        test_forward: bool = True,
    ) -> Dict:
        """
        Analyze parameter stability by testing neighboring values.

        A robust parameter set should have neighbors that also perform well.
        If performance degrades dramatically with small changes, it's likely overfit.

        FIX: Boolean filter parameters are now EXEMPTED from stability testing.
        These parameters (like require_pullback, use_rsi_extreme_filter) are binary
        decisions that legitimately can change trade counts dramatically - they're
        not "fragile", they're just different operating modes.

        Returns:
            Dict with stability scores per parameter and overall stability rating
        """
        if not self.back_signals:
            raise ValueError("Must run optimization first to precompute signals")

        logger.info("\n" + "="*70)
        logger.info("PARAMETER STABILITY ANALYSIS")
        logger.info("="*70)

        self._swap_to_back()
        run_trial = self._select_trial_runner(use_full)

        # Get baseline performance using quality_score (smooth equity + profit + low DD)
        base_back = run_trial(params, self.back_arrays)
        base_score = quality_score_from_metrics(base_back)

        base_fwd = None
        base_fwd_score = 0
        if test_forward and self.fwd_signals:
            self._swap_to_forward()
            base_fwd = run_trial(params, self.fwd_arrays)
            base_fwd_score = quality_score_from_metrics(base_fwd)
            self._swap_to_back()

        logger.info(f"Baseline: Back QS={base_score:.3f}, Forward QS={base_fwd_score:.3f}")

        # Get parameter space for neighbor testing
        groups = self.strategy.get_parameter_groups()
        all_spaces = {}
        for group in groups.values():
            all_spaces.update(group.get_param_space())

        stability_results = {}
        skipped_booleans = []

        for param_name, values in all_spaces.items():
            if param_name not in params:
                continue

            current_val = params[param_name]
            if current_val not in values:
                continue

            # Skip boolean parameters - they represent different operating modes,
            # not points on a continuous optimization surface. Testing neighbors
            # (True->False or False->True) isn't meaningful for stability analysis.
            if len(values) == 2 and set(values) == {True, False}:
                skipped_booleans.append(param_name)
                stability_results[param_name] = {
                    'current_value': current_val,
                    'base_score': base_score,
                    'neighbor_scores': [],
                    'avg_neighbor_score': 0,
                    'stability_ratio': 1.0,  # Exempt = treated as stable
                    'min_stability': 1.0,
                    'skipped': True,
                    'skip_reason': 'boolean_filter',
                }
                continue

            # Skip categorical/mode parameters (e.g., sl_mode=['fixed','atr','swing'],
            # tp_mode=['rr','atr','fixed']). Switching between modes is comparing
            # different strategies, not testing parameter stability.
            if all(isinstance(v, str) for v in values):
                skipped_booleans.append(param_name)
                stability_results[param_name] = {
                    'current_value': current_val,
                    'base_score': base_score,
                    'neighbor_scores': [],
                    'avg_neighbor_score': 0,
                    'stability_ratio': 1.0,  # Exempt = treated as stable
                    'min_stability': 1.0,
                    'skipped': True,
                    'skip_reason': 'categorical_mode',
                }
                continue

            # Use ±10% perturbation instead of grid neighbors.
            # Grid neighbors can be massive jumps (e.g., RSI 14 -> 7 = 50% change)
            # which tests a different strategy, not parameter stability.
            # Industry standard: ±5-15% perturbation (Alvarez Quant Trading).
            perturbation_pct = 0.10
            neighbor_scores = []

            if isinstance(current_val, (int, np.integer)):
                step = max(1, round(abs(current_val) * perturbation_pct))
                neighbor_vals = [current_val - step, current_val + step]
            elif isinstance(current_val, (float, np.floating)):
                step = abs(current_val) * perturbation_pct
                neighbor_vals = [current_val - step, current_val + step]
            else:
                neighbor_vals = []

            for neighbor_val in neighbor_vals:
                # Clamp to parameter range bounds
                min_val = min(values)
                max_val = max(values)
                clamped = type(current_val)(max(min_val, min(max_val, neighbor_val)))
                if clamped == current_val:
                    continue

                test_params = dict(params)
                test_params[param_name] = clamped

                metrics = run_trial(test_params, self.back_arrays)
                neighbor_scores.append(quality_score_from_metrics(metrics))

            if neighbor_scores and base_score > 0:
                # Calculate stability as ratio of neighbor avg to base
                avg_neighbor = np.mean(neighbor_scores)
                stability = avg_neighbor / base_score
                min_neighbor = min(neighbor_scores)
                min_stability = min_neighbor / base_score
            else:
                stability = 0
                min_stability = 0

            stability_results[param_name] = {
                'current_value': current_val,
                'base_score': base_score,
                'neighbor_scores': neighbor_scores,
                'avg_neighbor_score': np.mean(neighbor_scores) if neighbor_scores else 0,
                'stability_ratio': stability,  # 1.0 = neighbors perform same, <0.5 = unstable
                'min_stability': min_stability,
                'skipped': False,
            }

        # Calculate overall stability (excluding skipped boolean params)
        tested_results = [r for r in stability_results.values() if not r.get('skipped', False)]
        stabilities = [r['stability_ratio'] for r in tested_results if r['stability_ratio'] > 0]
        min_stabilities = [r['min_stability'] for r in tested_results if r['min_stability'] > 0]

        overall = {
            'mean_stability': np.mean(stabilities) if stabilities else 0,
            'min_stability': np.min(min_stabilities) if min_stabilities else 0,
            'n_stable_params': sum(1 for s in stabilities if s > 0.7),
            'n_unstable_params': sum(1 for s in stabilities if s < 0.3),
            'n_total_params': len(stabilities),
            'n_skipped_booleans': len(skipped_booleans),
        }

        # Rate overall robustness
        if overall['mean_stability'] > 0.8 and overall['min_stability'] > 0.5:
            overall['rating'] = 'ROBUST'
        elif overall['mean_stability'] > 0.6:
            overall['rating'] = 'MODERATE'
        elif overall['mean_stability'] > 0.4:
            overall['rating'] = 'FRAGILE'
        else:
            overall['rating'] = 'OVERFIT'

        # Print results
        logger.info(f"\n{'Parameter':<30} {'Current':<12} {'Avg Neighbor':<15} {'Stability':<12}")
        logger.info("-"*70)

        for param, result in sorted(stability_results.items(), key=lambda x: x[1]['stability_ratio']):
            if result.get('skipped'):
                logger.info(f"{param:<30} {str(result['current_value']):<12} "
                           f"{'(skipped)':<15} {'EXEMPT':<10} [boolean]")
            else:
                stability_pct = result['stability_ratio'] * 100
                marker = "!!!" if stability_pct < 50 else ("!" if stability_pct < 70 else "")
                logger.info(f"{param:<30} {str(result['current_value']):<12} "
                           f"{result['avg_neighbor_score']:<15.1f} {stability_pct:<10.0f}% {marker}")

        logger.info(f"\n{'='*70}")
        logger.info(f"OVERALL STABILITY: {overall['rating']}")
        logger.info(f"  Mean stability:   {overall['mean_stability']:.1%}")
        logger.info(f"  Min stability:    {overall['min_stability']:.1%}")
        logger.info(f"  Stable params:    {overall['n_stable_params']}/{overall['n_total_params']}")
        logger.info(f"  Unstable params:  {overall['n_unstable_params']}/{overall['n_total_params']}")
        if skipped_booleans:
            logger.info(f"  Skipped booleans: {len(skipped_booleans)} ({', '.join(skipped_booleans)})")
        logger.info(f"{'='*70}")

        return {
            'params': stability_results,
            'overall': overall,
            'baseline_back': base_score,
            'baseline_forward': base_fwd_score,
        }

    def run_robust_optimization(
        self,
        df_back: pd.DataFrame,
        df_forward: pd.DataFrame,
        n_trials: int = 5000,
        n_candidates: int = 50,
        min_stability: float = 0.6,
        min_back_sharpe: float = 1.0,
    ) -> List[Dict]:
        """
        Robustness-focused optimization.

        Instead of finding THE best parameters, finds parameters that:
        1. Perform well on backtest
        2. Perform well on forward test
        3. Have STABLE neighboring parameters (robustness)

        Args:
            df_back: Back-testing data
            df_forward: Forward validation data
            n_trials: Optimization trials
            n_candidates: Top N candidates to stability-test
            min_stability: Minimum mean stability ratio to accept
            min_back_sharpe: Minimum back Sharpe

        Returns:
            Results sorted by combined rank, with stability scores
        """
        logger.info("\n" + "="*70)
        logger.info("ROBUST OPTIMIZATION MODE")
        logger.info(f"Testing top {n_candidates} candidates for parameter stability")
        logger.info("="*70)

        # Run standard optimization first
        results = self.run(
            df_back, df_forward,
            mode='staged',
            trials_per_stage=n_trials // 6,  # Divide among stages + final
            final_trials=n_trials // 3,
            min_back_sharpe=min_back_sharpe,
        )

        if not results:
            logger.warning("No valid results from optimization")
            return []

        # Test stability of top candidates
        logger.info(f"\n{'='*70}")
        logger.info(f"STABILITY TESTING TOP {min(n_candidates, len(results))} CANDIDATES")
        logger.info(f"{'='*70}")

        stable_results = []

        for i, r in enumerate(results[:n_candidates]):
            logger.info(f"\n--- Candidate {i+1} (Combined Rank: {r['combined_rank']}) ---")

            stability = self.analyze_parameter_stability(
                r['params'],
                use_full=True,
                test_forward=True,
            )

            r['stability'] = stability

            if stability['overall']['mean_stability'] >= min_stability:
                stable_results.append(r)
                logger.info(f"  -> ACCEPTED (stability: {stability['overall']['mean_stability']:.1%})")
            else:
                logger.info(f"  -> REJECTED (stability: {stability['overall']['mean_stability']:.1%} < {min_stability:.1%})")

        # Sort by combined rank among stable results
        stable_results.sort(key=lambda x: x['combined_rank'])

        logger.info(f"\n{'='*70}")
        logger.info(f"ROBUST OPTIMIZATION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"  Candidates tested: {min(n_candidates, len(results))}")
        logger.info(f"  Passed stability:  {len(stable_results)}")
        logger.info(f"  Pass rate:         {len(stable_results)/min(n_candidates, len(results)):.1%}")

        if stable_results:
            best = stable_results[0]
            logger.info(f"\nBest Robust Result:")
            logger.info(f"  Combined Rank:    {best['combined_rank']}")
            logger.info(f"  Back Sharpe:      {best['back'].sharpe:.3f}")
            logger.info(f"  Forward Sharpe:   {best['forward'].sharpe:.3f}")
            logger.info(f"  Stability:        {best['stability']['overall']['mean_stability']:.1%}")
            logger.info(f"  Rating:           {best['stability']['overall']['rating']}")

        return stable_results

    def save_results(self, output_dir: Path) -> Path:
        """Save results to JSON with combined ranking info."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"{self.strategy.name}_{timestamp}.json"

        data = {
            'timestamp': timestamp,
            'strategy': self.strategy.name,
            'ranking_method': f'combined (back_rank + forward_rank * {self.forward_rank_weight})',
            'score_formula': 'Sortino × R² × min(PF,5) × √min(Trades,200) × ReturnMult / (Ulcer + MaxDD/2 + 5)',
            'min_forward_ratio': self.min_forward_ratio,
            'forward_rank_weight': self.forward_rank_weight,
            'n_results': len(self.results),
            'n_back_tested': len(self.all_back_results) if self.all_back_results else 'released',
            'locked_params': self.locked_params,
            'stage_results': {
                name: {k: v for k, v in result.items() if k != 'best_params'}
                for name, result in self.stage_results.items()
            },
            'param_importance': self.param_importance,
            'results': []
        }

        for r in self.results[:100]:
            result_data = {
                'trial_id': r['trial_id'],
                'combined_rank': r['combined_rank'],
                'back_rank': r['back_rank'],
                'forward_rank': r['forward_rank'],
                'params': r['params'],
                'back': {
                    'trades': r['back'].trades,
                    'quality_score': quality_score_from_metrics(r['back']),
                    'ontester_score': r['back'].ontester_score,
                    'r_squared': r['back'].r_squared,
                    'sharpe': r['back'].sharpe,
                    'sortino': r['back'].sortino,
                    'ulcer': r['back'].ulcer,
                    'win_rate': r['back'].win_rate,
                    'profit_factor': r['back'].profit_factor,
                    'max_dd': r['back'].max_dd,
                    'total_return': r['back'].total_return,
                },
            }
            if 'forward' in r:
                result_data['forward'] = {
                    'trades': r['forward'].trades,
                    'quality_score': quality_score_from_metrics(r['forward']),
                    'ontester_score': r['forward'].ontester_score,
                    'r_squared': r['forward'].r_squared,
                    'sharpe': r['forward'].sharpe,
                    'sortino': r['forward'].sortino,
                    'ulcer': r['forward'].ulcer,
                    'win_rate': r['forward'].win_rate,
                    'profit_factor': r['forward'].profit_factor,
                    'max_dd': r['forward'].max_dd,
                    'total_return': r['forward'].total_return,
                }
            data['results'].append(result_data)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved to {filepath}")
        return filepath
