"""
Stage 3: Walk-Forward Validation

Rolling window validation to test parameter robustness across time:
- 12 month training window
- 3 month test window
- Roll forward by 3 months
- Test that parameters work across ALL windows
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, List, Tuple, Optional

from loguru import logger

from optimization.unified_optimizer import UnifiedOptimizer
from optimization.fast_strategy import FastStrategy
from optimization.numba_backtest import Metrics, get_quote_conversion_rate
from pipeline.config import PipelineConfig
from pipeline.state import PipelineState
from pipeline.stages.s2_optimization import get_strategy

# ML Exit integration (optional - graceful degradation if unavailable)
try:
    from pipeline.ml_exit.dataset_builder import build_exit_dataset
    from pipeline.ml_exit.train import train_ml_exit_models, get_ml_backend
    from pipeline.ml_exit.inference import predict_exit_scores
    from pipeline.ml_exit.policy import apply_exit_policy, generate_ml_score_arrays
    from pipeline.ml_exit.features import precompute_market_features, compute_decision_features, get_feature_names
    ML_EXIT_AVAILABLE = True
except ImportError:
    ML_EXIT_AVAILABLE = False

# Meta-labeling signal filter (optional)
try:
    from pipeline.ml_exit.signal_features import (
        compute_signal_features, train_signal_classifier, predict_signal_filter,
        calibrate_threshold,
    )
    SIGNAL_FILTER_AVAILABLE = True
except ImportError:
    SIGNAL_FILTER_AVAILABLE = False


class WalkForwardStage:
    """Stage 3: Walk-forward validation with rolling windows."""

    name = "walkforward"

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(
        self,
        state: PipelineState,
        df: pd.DataFrame,
        candidates: List[Dict[str, Any]],
        back_end_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        """
        Execute walk-forward validation.

        For each candidate, test performance across multiple rolling windows.

        Args:
            state: Pipeline state
            df: Full dataset
            candidates: Candidates from optimization stage
            back_end_date: End date of optimization back-test period.
                Windows whose test period overlaps with this are flagged
                as in-sample and excluded from the pass/fail decision.

        Returns:
            Dict with:
            - candidates: Updated candidates with walk-forward results
            - windows: Window definitions
            - summary: Overall walk-forward statistics
        """
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 3: WALK-FORWARD VALIDATION")
        logger.info("=" * 70)

        # Generate window definitions
        windows = self._generate_windows(df)

        # Mark windows as in-sample or out-of-sample relative to optimization back-test
        if back_end_date is not None:
            n_oos = 0
            for w in windows:
                # A window is out-of-sample if its test period starts after back-test ends
                w['out_of_sample'] = w['test_start'] >= back_end_date
                if w['out_of_sample']:
                    n_oos += 1
            logger.info(f"  Back-test ends: {back_end_date.date()}")
            logger.info(f"  Out-of-sample windows: {n_oos}/{len(windows)}")
            if n_oos == 0:
                logger.warning("  WARNING: All WF windows overlap with optimization training data!")
                logger.warning("  Consider using more historical data or shorter training windows.")
        else:
            for w in windows:
                w['out_of_sample'] = True  # Default: treat all as OOS if no split info

        if len(windows) < self.config.walkforward.min_windows:
            logger.error(f"FAILED: Only {len(windows)} windows available, need {self.config.walkforward.min_windows}")
            logger.error("Walk-forward validation requires sufficient data for multiple windows.")
            logger.error("Options: (1) Add more historical data, (2) Reduce train_months, or (3) Lower min_windows requirement")

            # FIX: Fail the stage instead of proceeding with insufficient evidence
            return {
                'candidates': [],  # No candidates pass
                'all_candidates': candidates,
                'windows': windows,
                'summary': {
                    'n_windows': len(windows),
                    'n_candidates_tested': 0,
                    'n_candidates_passed': 0,
                    'pass_rate': 0,
                    'failed_reason': f'insufficient_windows ({len(windows)} < {self.config.walkforward.min_windows})',
                },
            }

        logger.info(f"\nWindow Configuration:")
        logger.info(f"  Train period: {self.config.walkforward.train_months} months")
        logger.info(f"  Test period:  {self.config.walkforward.test_months} months")
        logger.info(f"  Roll step:    {self.config.walkforward.roll_step_months} months")
        logger.info(f"  Total windows: {len(windows)}")
        logger.info(f"  Min pass rate: {self.config.walkforward.min_window_pass_rate:.0%}")

        # Print window schedule
        logger.info("\nWindow Schedule:")
        for i, w in enumerate(windows):
            oos_tag = " [OOS]" if w.get('out_of_sample', True) else " [IS]"
            logger.info(f"  W{i+1}: Train {w['train_start'].date()} to {w['train_end'].date()}, "
                       f"Test {w['test_start'].date()} to {w['test_end'].date()}{oos_tag}")

        # Get strategy for testing
        strategy = get_strategy(self.config.strategy_name)
        strategy.set_pip_size(self.config.pair)

        # OPT-4: Precompute signals ONCE on full dataset, then filter by window range.
        # This eliminates redundant precompute calls (was: N_candidates * N_windows times).
        logger.info("\nPre-computing signals on full dataset (once)...")
        strategy.precompute_for_dataset(df)
        full_signals = strategy._precomputed_signals
        full_vec_arrays = getattr(strategy, '_vec_arrays', None)
        logger.info(f"  Pre-computed {len(full_signals)} signals")

        # Build bar-index lookup for window slicing
        # Map each window's test period to bar index range
        window_bar_ranges = []
        for w in windows:
            test_mask = (df.index >= w['test_start']) & (df.index < w['test_end'])
            test_indices = np.where(test_mask)[0]
            if len(test_indices) > 0:
                window_bar_ranges.append((int(test_indices[0]), int(test_indices[-1]) + 1, int(test_mask.sum())))
            else:
                window_bar_ranges.append((0, 0, 0))

        # Prepare full data arrays once
        full_highs = df['high'].values.astype(np.float64)
        full_lows = df['low'].values.astype(np.float64)
        full_closes = df['close'].values.astype(np.float64)
        full_days = df.index.dayofweek.values.astype(np.int64)
        pip_size = 0.01 if 'JPY' in self.config.pair else 0.0001
        quote_rate = get_quote_conversion_rate(self.config.pair, 'USD')

        # Test each candidate across all windows
        validated_candidates = []
        window_results_all = {}

        for i, candidate in enumerate(candidates):
            logger.info(f"\n--- Candidate {candidate['rank']} (Combined Rank: {candidate['combined_rank']}) ---")

            window_results = self._test_candidate_windows_fast(
                candidate, windows, window_bar_ranges, df, strategy,
                full_signals, full_vec_arrays,
                full_highs, full_lows, full_closes, full_days,
                pip_size, quote_rate,
            )

            # Attach OOS flag to each window result
            for j, wr in enumerate(window_results):
                if j < len(windows):
                    wr['out_of_sample'] = windows[j].get('out_of_sample', True)

            # Calculate walk-forward statistics
            wf_stats = self._calculate_wf_stats(window_results)

            # Update candidate with walk-forward results
            candidate['walkforward'] = {
                'window_results': window_results,
                'stats': wf_stats,
            }

            window_results_all[candidate['rank']] = window_results

            # Check if candidate passes
            passed = self._candidate_passes(wf_stats)
            candidate['walkforward']['passed'] = passed

            if passed:
                validated_candidates.append(candidate)
                logger.info(f"  PASSED: {wf_stats['pass_rate']:.0%} all windows, "
                           f"OOS {wf_stats.get('oos_pass_rate', 0):.0%} ({wf_stats.get('oos_n_passed', 0)}/{wf_stats.get('oos_n_windows', 0)}), "
                           f"mean Sharpe {wf_stats['mean_sharpe']:.2f}")
            else:
                logger.info(f"  FAILED: {wf_stats['pass_rate']:.0%} all windows, "
                           f"OOS {wf_stats.get('oos_pass_rate', 0):.0%} ({wf_stats.get('oos_n_passed', 0)}/{wf_stats.get('oos_n_windows', 0)}), "
                           f"mean Sharpe {wf_stats['mean_sharpe']:.2f}")

        # Update state with validated candidates
        state.candidates = validated_candidates

        # Build summary
        summary = {
            'n_windows': len(windows),
            'n_candidates_tested': len(candidates),
            'n_candidates_passed': len(validated_candidates),
            'pass_rate': len(validated_candidates) / len(candidates) if candidates else 0,
        }

        if validated_candidates:
            summary['best_wf_pass_rate'] = max(
                c['walkforward']['stats']['pass_rate'] for c in validated_candidates
            )
            summary['best_mean_sharpe'] = max(
                c['walkforward']['stats']['mean_sharpe'] for c in validated_candidates
            )

        # Save stage output
        output_data = {
            'summary': summary,
            'windows': [
                {
                    'train_start': str(w['train_start']),
                    'train_end': str(w['train_end']),
                    'test_start': str(w['test_start']),
                    'test_end': str(w['test_end']),
                    'out_of_sample': w.get('out_of_sample', True),
                }
                for w in windows
            ],
            'candidates': [
                {
                    'rank': c['rank'],
                    'combined_rank': c['combined_rank'],
                    'walkforward': c['walkforward'],
                }
                for c in candidates
            ],
        }
        state.save_stage_output('walkforward', output_data)

        logger.info(f"\nWalk-Forward Complete:")
        logger.info(f"  Candidates tested: {len(candidates)}")
        logger.info(f"  Candidates passed: {len(validated_candidates)}")
        logger.info(f"  Pass rate: {summary['pass_rate']:.0%}")

        return {
            'candidates': validated_candidates,
            'all_candidates': candidates,  # Include failed ones too
            'windows': windows,
            'summary': summary,
        }

    def _generate_windows(self, df: pd.DataFrame) -> List[Dict[str, datetime]]:
        """
        Generate rolling window definitions.

        Windows are generated from the data, not from fixed dates,
        to ensure we always have valid data for each window.
        """
        windows = []

        data_start = df.index[0].to_pydatetime()
        data_end = df.index[-1].to_pydatetime()

        train_months = self.config.walkforward.train_months
        test_months = self.config.walkforward.test_months
        roll_step = self.config.walkforward.roll_step_months

        # Start first window after enough data for training
        # Anchor start 6 months in to have some buffer
        anchor_months = 6
        window_start = data_start + relativedelta(months=anchor_months)

        while True:
            train_start = window_start
            train_end = train_start + relativedelta(months=train_months)
            test_start = train_end
            test_end = test_start + relativedelta(months=test_months)

            # Check if we have enough data for this window
            # Allow 7-day tolerance for month-boundary rounding
            if test_end > data_end + timedelta(days=7):
                break
            # Clamp test_end to actual data end
            if test_end > data_end:
                test_end = data_end

            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
            })

            # Roll forward
            window_start = window_start + relativedelta(months=roll_step)

        return windows

    def _test_candidate_windows_fast(
        self,
        candidate: Dict[str, Any],
        windows: List[Dict],
        window_bar_ranges: List[Tuple[int, int, int]],
        df: pd.DataFrame,
        strategy: FastStrategy,
        full_signals: list,
        full_vec_arrays: Optional[Dict[str, np.ndarray]],
        full_highs: np.ndarray,
        full_lows: np.ndarray,
        full_closes: np.ndarray,
        full_days: np.ndarray,
        pip_size: float,
        quote_rate: float,
    ) -> List[Dict[str, Any]]:
        """
        Test a candidate across all windows using precomputed signals.

        OPT-4: Instead of re-precomputing signals per window, we use the
        full-dataset signals and filter by bar index range per window.
        """
        from optimization.numba_backtest import full_backtest_numba

        params = candidate['params']
        results = []

        # Get filtered signal arrays once for this candidate (on full dataset)
        # The strategy's vectorized path uses the already-stored _precomputed_signals
        # and _vec_arrays from the precompute_for_dataset() call in run().
        signal_arrays, mgmt_arrays = strategy.get_all_arrays(
            params, full_highs, full_lows, full_closes, full_days
        )

        all_entry_bars = signal_arrays['entry_bars']
        all_entry_prices = signal_arrays['entry_prices']
        all_directions = signal_arrays['directions']
        all_sl_prices = signal_arrays['sl_prices']
        all_tp_prices = signal_arrays['tp_prices']

        n_all = len(all_entry_bars)

        # Pre-extract management arrays for the full signal set
        all_use_trailing = mgmt_arrays.get('use_trailing', np.zeros(n_all, dtype=np.bool_))
        all_trail_start = mgmt_arrays.get('trail_start_pips', np.zeros(n_all, dtype=np.float64))
        all_trail_step = mgmt_arrays.get('trail_step_pips', np.zeros(n_all, dtype=np.float64))
        all_use_be = mgmt_arrays.get('use_breakeven', np.zeros(n_all, dtype=np.bool_))
        all_be_trigger = mgmt_arrays.get('be_trigger_pips', np.zeros(n_all, dtype=np.float64))
        all_be_offset = mgmt_arrays.get('be_offset_pips', np.zeros(n_all, dtype=np.float64))
        all_use_partial = mgmt_arrays.get('use_partial', np.zeros(n_all, dtype=np.bool_))
        all_partial_pct = mgmt_arrays.get('partial_pct', np.zeros(n_all, dtype=np.float64))
        all_partial_target = mgmt_arrays.get('partial_target_pips', np.zeros(n_all, dtype=np.float64))
        all_max_bars = mgmt_arrays.get('max_bars', np.zeros(n_all, dtype=np.int64))
        all_trail_mode = mgmt_arrays.get('trail_mode', np.zeros(n_all, dtype=np.int64))
        all_chandelier_atr_mult = mgmt_arrays.get('chandelier_atr_mult', np.full(n_all, 3.0, dtype=np.float64))
        all_atr_pips_arr = mgmt_arrays.get('atr_pips', np.full(n_all, 35.0, dtype=np.float64))
        all_stale_exit_bars = mgmt_arrays.get('stale_exit_bars', np.zeros(n_all, dtype=np.int64))

        # V6: ML exit arrays (from strategy, if any)
        all_use_ml = mgmt_arrays.get('use_ml_exit', np.zeros(n_all, dtype=np.bool_))
        all_ml_min_hold = mgmt_arrays.get('ml_min_hold', np.zeros(n_all, dtype=np.int64))
        all_ml_threshold = mgmt_arrays.get('ml_threshold', np.ones(n_all, dtype=np.float64))

        # Compute ML scores on full dataset if strategy supports it and ML is enabled
        n_full_bars = len(full_highs)
        if hasattr(strategy, 'get_ml_score_arrays') and np.any(all_use_ml):
            full_ml_long, full_ml_short = strategy.get_ml_score_arrays(
                params, full_highs, full_lows, full_closes
            )
        else:
            full_ml_long = np.zeros(n_full_bars, dtype=np.float64)
            full_ml_short = np.zeros(n_full_bars, dtype=np.float64)

        # ML Exit integration: check if pipeline-level ML exit is enabled
        ml_exit_enabled = (
            self.config.ml_exit.enabled
            and self.config.ml_exit.ml_mode == 'exit'
            and ML_EXIT_AVAILABLE
            and get_ml_backend() is not None
        )
        if ml_exit_enabled:
            logger.info(f"  ML Exit enabled (backend: {get_ml_backend()})")

        # Meta-labeling signal filter: check if entry_filter mode is active
        entry_filter_enabled = (
            self.config.ml_exit.enabled
            and self.config.ml_exit.ml_mode == 'entry_filter'
            and SIGNAL_FILTER_AVAILABLE
        )
        if entry_filter_enabled:
            logger.info(f"  Signal filter enabled (meta-labeling, threshold={self.config.ml_exit.signal_filter_threshold})")

        # Get strategy-specific signal attributes for ML feature enrichment
        # These are aligned with the full filtered signal set (same mask as all_entry_bars)
        full_strategy_attrs = None
        if entry_filter_enabled and hasattr(strategy, 'get_signal_attributes'):
            full_strategy_attrs = strategy.get_signal_attributes(params)
            if full_strategy_attrs:
                # The attrs from get_signal_attributes() are for ALL precomputed signals.
                # We need to apply the same filter mask the strategy uses.
                # get_all_arrays uses _filter_vectorized which produces a mask over _vec_arrays.
                # The signal_arrays are already filtered, so we need the same mask.
                if hasattr(strategy, '_filter_vectorized') and strategy._vec_arrays is not None:
                    filter_mask = strategy._filter_vectorized(params)
                    filtered_attrs = {}
                    for key, arr in full_strategy_attrs.items():
                        if len(arr) == len(filter_mask):
                            filtered_attrs[key] = arr[filter_mask]
                        else:
                            filtered_attrs[key] = arr
                    full_strategy_attrs = filtered_attrs
                n_strat_features = len(full_strategy_attrs)
                logger.info(f"  Strategy attributes: {n_strat_features} features ({', '.join(sorted(full_strategy_attrs.keys()))})")

        # Adaptive SL: ML predicts which signals have low adverse excursion, tightens their SL
        adaptive_sl_mode = self.config.ml_exit.ml_mode if self.config.ml_exit.enabled else None
        adaptive_sl_enabled = (
            self.config.ml_exit.enabled
            and adaptive_sl_mode in ('adaptive_sl', 'adaptive_sl_reg')
            and SIGNAL_FILTER_AVAILABLE
        )
        use_regression = (adaptive_sl_mode == 'adaptive_sl_reg')
        if adaptive_sl_enabled:
            mode_str = "regression (predict MAE_r)" if use_regression else "classification (easy/hard)"
            logger.info(f"  Adaptive SL enabled ({mode_str})")

        for i, window in enumerate(windows):
            bar_start, bar_end, n_bars = window_bar_ranges[i]

            if n_bars < 50:
                logger.warning(f"  Window {i+1}: Insufficient test data ({n_bars} bars)")
                results.append({
                    'window': i + 1,
                    'status': 'insufficient_data',
                    'train_candles': 0,
                    'test_candles': n_bars,
                })
                continue

            # Find signals whose entry_bar falls within this window's test range
            sig_mask = (all_entry_bars >= bar_start) & (all_entry_bars < bar_end)
            n_window_sigs = int(np.sum(sig_mask))

            if n_window_sigs < 1:
                results.append({
                    'window': i + 1,
                    'status': 'completed',
                    'train_candles': 0,
                    'test_candles': n_bars,
                    'trades': 0,
                    'ontester': 0.0,
                    'sharpe': 0.0,
                    'return': 0.0,
                    'max_dd': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'passed': False,
                })
                continue

            # Slice signal arrays for this window
            w_entry_bars = all_entry_bars[sig_mask] - bar_start  # Re-index to window start
            w_directions = all_directions[sig_mask]
            w_sl_prices = all_sl_prices[sig_mask]
            w_tp_prices = all_tp_prices[sig_mask]

            # Apply spread to entry prices
            w_raw_prices = all_entry_prices[sig_mask]
            w_entry_prices = np.where(
                w_directions == 1,
                w_raw_prices + (self.config.spread_pips + self.config.slippage_pips) * pip_size,
                w_raw_prices - (self.config.spread_pips + self.config.slippage_pips) * pip_size,
            )

            # Slice management arrays
            w_use_trailing = all_use_trailing[sig_mask]
            w_trail_start = all_trail_start[sig_mask]
            w_trail_step = all_trail_step[sig_mask]
            w_use_be = all_use_be[sig_mask]
            w_be_trigger = all_be_trigger[sig_mask]
            w_be_offset = all_be_offset[sig_mask]
            w_use_partial = all_use_partial[sig_mask]
            w_partial_pct = all_partial_pct[sig_mask]
            w_partial_target = all_partial_target[sig_mask]
            w_max_bars = all_max_bars[sig_mask]
            w_trail_mode = all_trail_mode[sig_mask]
            w_chandelier_atr_mult = all_chandelier_atr_mult[sig_mask]
            w_atr_pips_arr = all_atr_pips_arr[sig_mask]
            w_stale_exit_bars = all_stale_exit_bars[sig_mask]

            # V6: Slice ML arrays (from strategy)
            w_use_ml = all_use_ml[sig_mask]
            w_ml_min_hold = all_ml_min_hold[sig_mask]
            w_ml_threshold = all_ml_threshold[sig_mask]

            # Slice market data for this window
            w_highs = full_highs[bar_start:bar_end]
            w_lows = full_lows[bar_start:bar_end]
            w_closes = full_closes[bar_start:bar_end]
            w_days = full_days[bar_start:bar_end]

            # V6: Default ML score arrays (from strategy or zeros)
            w_ml_long = full_ml_long[bar_start:bar_end]
            w_ml_short = full_ml_short[bar_start:bar_end]

            filter_result = None  # initialized for scope

            # --- Entry Filter (meta-labeling): train on signals before window, filter test signals ---
            if entry_filter_enabled:
                filter_result = self._train_and_apply_signal_filter(
                    all_entry_bars, all_directions, all_sl_prices, all_tp_prices,
                    all_entry_prices, sig_mask,
                    full_highs, full_lows, full_closes,
                    df, bar_start, bar_end, pip_size,
                    params, strategy, full_days,
                    mgmt_arrays={
                        'use_trailing': all_use_trailing,
                        'trail_start_pips': all_trail_start,
                        'trail_step_pips': all_trail_step,
                        'use_breakeven': all_use_be,
                        'be_trigger_pips': all_be_trigger,
                        'be_offset_pips': all_be_offset,
                        'use_partial': all_use_partial,
                        'partial_pct': all_partial_pct,
                        'partial_target_pips': all_partial_target,
                        'max_bars': all_max_bars,
                        'trail_mode': all_trail_mode,
                        'chandelier_atr_mult': all_chandelier_atr_mult,
                        'atr_pips': all_atr_pips_arr,
                        'stale_exit_bars': all_stale_exit_bars,
                        'use_ml_exit': all_use_ml,
                        'ml_min_hold': all_ml_min_hold,
                        'ml_threshold': all_ml_threshold,
                    },
                    window_idx=i,
                    strategy_attrs=full_strategy_attrs,
                )
                if filter_result is not None:
                    keep_mask = filter_result['keep_mask']
                    # Re-slice all window arrays with the filter mask
                    w_entry_bars = w_entry_bars[keep_mask]
                    w_directions = w_directions[keep_mask]
                    w_sl_prices = w_sl_prices[keep_mask]
                    w_tp_prices = w_tp_prices[keep_mask]
                    w_entry_prices = w_entry_prices[keep_mask]
                    w_use_trailing = w_use_trailing[keep_mask]
                    w_trail_start = w_trail_start[keep_mask]
                    w_trail_step = w_trail_step[keep_mask]
                    w_use_be = w_use_be[keep_mask]
                    w_be_trigger = w_be_trigger[keep_mask]
                    w_be_offset = w_be_offset[keep_mask]
                    w_use_partial = w_use_partial[keep_mask]
                    w_partial_pct = w_partial_pct[keep_mask]
                    w_partial_target = w_partial_target[keep_mask]
                    w_max_bars = w_max_bars[keep_mask]
                    w_trail_mode = w_trail_mode[keep_mask]
                    w_chandelier_atr_mult = w_chandelier_atr_mult[keep_mask]
                    w_atr_pips_arr = w_atr_pips_arr[keep_mask]
                    w_stale_exit_bars = w_stale_exit_bars[keep_mask]
                    w_use_ml = w_use_ml[keep_mask]
                    w_ml_min_hold = w_ml_min_hold[keep_mask]
                    w_ml_threshold = w_ml_threshold[keep_mask]
                    n_window_sigs = int(keep_mask.sum())

                    if n_window_sigs < 1:
                        results.append({
                            'window': i + 1,
                            'status': 'completed',
                            'train_candles': 0,
                            'test_candles': n_bars,
                            'trades': 0,
                            'ontester': 0.0,
                            'sharpe': 0.0,
                            'return': 0.0,
                            'max_dd': 0.0,
                            'win_rate': 0.0,
                            'profit_factor': 0.0,
                            'passed': False,
                            'signal_filter': filter_result.get('metrics', {}),
                        })
                        continue

            # --- Adaptive SL: predict MAE and tighten SL for easy trades ---
            adaptive_sl_result = None
            if adaptive_sl_enabled:
                adaptive_sl_result = self._train_and_apply_adaptive_sl(
                    all_entry_bars, all_directions, all_sl_prices, all_tp_prices,
                    all_entry_prices, sig_mask,
                    full_highs, full_lows, full_closes,
                    df, bar_start, bar_end, pip_size,
                    params, strategy, full_days,
                    mgmt_arrays={
                        'use_trailing': all_use_trailing,
                        'trail_start_pips': all_trail_start,
                        'trail_step_pips': all_trail_step,
                        'use_breakeven': all_use_be,
                        'be_trigger_pips': all_be_trigger,
                        'be_offset_pips': all_be_offset,
                        'use_partial': all_use_partial,
                        'partial_pct': all_partial_pct,
                        'partial_target_pips': all_partial_target,
                        'max_bars': all_max_bars,
                        'trail_mode': all_trail_mode,
                        'chandelier_atr_mult': all_chandelier_atr_mult,
                        'atr_pips': all_atr_pips_arr,
                        'stale_exit_bars': all_stale_exit_bars,
                        'use_ml_exit': all_use_ml,
                        'ml_min_hold': all_ml_min_hold,
                        'ml_threshold': all_ml_threshold,
                    },
                    window_idx=i,
                    use_regression=use_regression,
                )
                if adaptive_sl_result is not None:
                    sl_scale = adaptive_sl_result['sl_scale']
                    # Tighten SL for predicted-easy signals (keep TP unchanged!)
                    for j in range(len(w_sl_prices)):
                        if sl_scale[j] < 1.0:
                            sl_dist = abs(w_entry_prices[j] - w_sl_prices[j])
                            new_sl_dist = sl_dist * sl_scale[j]
                            if w_directions[j] == 1:  # long
                                w_sl_prices[j] = w_entry_prices[j] - new_sl_dist
                            else:  # short
                                w_sl_prices[j] = w_entry_prices[j] + new_sl_dist

            # --- ML Exit: train on data before window, predict on test window ---
            ml_window_info = {}
            if ml_exit_enabled:
                ml_result = self._train_and_predict_ml_exit(
                    params, df, strategy, window, bar_start, bar_end,
                    full_highs, full_lows, full_closes, full_days, pip_size,
                )
                if ml_result is not None:
                    w_ml_long, w_ml_short = ml_result['score_arrays']
                    # Override: enable ML exit for all signals in this window
                    w_use_ml = np.ones(n_window_sigs, dtype=np.bool_)
                    w_ml_min_hold = np.full(n_window_sigs, self.config.ml_exit.ml_min_hold_bars, dtype=np.int64)
                    w_ml_threshold = np.full(n_window_sigs, self.config.ml_exit.ml_exit_threshold, dtype=np.float64)
                    ml_window_info = ml_result.get('metrics', {})

            quality_mult = np.empty(0, dtype=np.float64)

            # V6.2: Pass cooldown when ML exit is active
            cooldown = self.config.ml_exit.ml_exit_cooldown_bars if ml_exit_enabled and ml_result is not None else 0

            result = full_backtest_numba(
                w_entry_bars,
                w_entry_prices,
                w_directions,
                w_sl_prices,
                w_tp_prices,
                w_use_trailing, w_trail_start, w_trail_step,
                w_use_be, w_be_trigger, w_be_offset,
                w_use_partial, w_partial_pct, w_partial_target,
                w_max_bars,
                w_trail_mode, w_chandelier_atr_mult, w_atr_pips_arr, w_stale_exit_bars,
                w_ml_long, w_ml_short, w_use_ml, w_ml_min_hold, w_ml_threshold,
                w_highs, w_lows, w_closes, w_days,
                self.config.initial_capital,
                self.config.risk_per_trade,
                pip_size,
                params.get('max_daily_trades', 0),
                params.get('max_daily_loss_pct', 0.0),
                quality_mult,
                quote_rate,
                5544.0,  # bars_per_year
                cooldown,
            )

            metrics = Metrics(*result)

            # Fix 5: Diagnostic — check if ML changed trade count vs Pass 1
            if ml_window_info and ml_window_info.get('status') == 'success':
                pass1_n_trades = ml_window_info.get('n_trades', 0)
                pass3_n_trades = metrics.trades
                if pass3_n_trades != pass1_n_trades:
                    logger.warning(
                        f"  W{i+1} ML changed trade count: Pass1={pass1_n_trades}, Pass3={pass3_n_trades}"
                    )
                ml_window_info['pass3_n_trades'] = pass3_n_trades
                ml_window_info['trade_count_changed'] = (pass3_n_trades != pass1_n_trades)

            min_trades = self.config.walkforward.min_trades_per_window
            window_passed = (
                metrics.trades >= min_trades and
                metrics.ontester_score > 0
            )

            window_result = {
                'window': i + 1,
                'status': 'completed',
                'train_candles': 0,
                'test_candles': n_bars,
                'trades': metrics.trades,
                'ontester': metrics.ontester_score,
                'sharpe': metrics.sharpe,
                'return': metrics.total_return,
                'max_dd': metrics.max_dd,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'passed': window_passed,
            }
            if ml_window_info:
                window_result['ml_exit'] = ml_window_info
            if entry_filter_enabled and filter_result is not None:
                window_result['signal_filter'] = filter_result.get('metrics', {})
            if adaptive_sl_enabled and adaptive_sl_result is not None:
                window_result['adaptive_sl'] = adaptive_sl_result.get('metrics', {})
            results.append(window_result)

        return results

    def _train_ml_for_window(
        self,
        df_train: pd.DataFrame,
        candidate: Dict[str, Any],
        strategy: FastStrategy,
    ):
        """Train ML exit models on training data for a walk-forward window.

        Args:
            df_train: DataFrame slice covering the training period (all data before test window).
            candidate: Candidate dict with 'params' key.
            strategy: FastStrategy instance. Note: this method will call
                strategy.precompute_for_dataset(df_train) internally via build_exit_dataset,
                which mutates the strategy's cached signals. The caller is responsible for
                restoring the full-dataset precomputation afterwards.

        Returns:
            MLExitModels or None on failure.
        """
        try:
            dataset = build_exit_dataset(
                df_train,
                candidate['params'],
                strategy,
                self.config,
                pair=self.config.pair,
                min_trade_bars=3,
            )
            if dataset is None or len(dataset) < 50:
                n_rows = len(dataset) if dataset is not None else 0
                logger.warning(
                    f"  ML training: insufficient dataset rows ({n_rows}), "
                    f"need >= 50. Skipping ML for this window."
                )
                return None

            models = train_ml_exit_models(
                dataset,
                n_optuna_trials=self.config.ml_exit.n_optuna_trials,
                cv_folds=self.config.ml_exit.cv_folds,
                early_stopping_rounds=self.config.ml_exit.early_stopping_rounds,
            )
            if models is None:
                logger.warning("  ML training: skipped (low sample count)")
                return None
            logger.info(
                f"  ML training complete: backend={models.backend}, "
                f"n_samples={models.training_metrics.get('n_samples', 0)}, "
                f"hold_r2={models.training_metrics.get('hold_value', {}).get('val_r2', 'N/A')}, "
                f"risk_auc={models.training_metrics.get('adverse_risk', {}).get('val_auc', 'N/A')}"
            )
            return models
        except Exception as e:
            logger.warning(f"  ML training failed: {e}")
            return None

    def _generate_ml_arrays_for_window(
        self,
        models,
        w_highs: np.ndarray,
        w_lows: np.ndarray,
        w_closes: np.ndarray,
        w_opens,
        w_days: np.ndarray,
        candidate: Dict[str, Any],
        strategy: FastStrategy,
        signal_arrays: Dict[str, np.ndarray],
        mgmt_arrays: Dict[str, np.ndarray],
        pip_size: float,
        quote_rate: float,
        n_bars: int,
        w_hours: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generate ML score arrays for a walk-forward test window.

        Two-pass approach:
        1. Run backtest without ML to get trade entry/exit bar positions.
        2. For each active trade at each bar, compute 14 decision features.
        3. Predict exit scores, apply policy, and map to per-bar arrays.

        Args:
            models: Trained MLExitModels from _train_ml_for_window.
            w_highs, w_lows, w_closes: Window market data arrays, shape (n_bars,).
            w_opens: Window open prices or None.
            w_days: Window day-of-week array, shape (n_bars,).
            candidate: Candidate dict with 'params'.
            strategy: FastStrategy instance.
            signal_arrays: Pre-sliced signal arrays for this window. Keys:
                entry_bars (re-indexed to window start), entry_prices (spread-adjusted),
                directions, sl_prices, tp_prices.
            mgmt_arrays: Pre-sliced management arrays for this window.
            pip_size: Pip size for the currency pair.
            quote_rate: Quote conversion rate.
            n_bars: Number of bars in the window.

        Returns:
            Tuple of (ml_long_scores, ml_short_scores, metrics_dict).
            On failure returns (zeros, zeros, error_metrics).
        """
        from optimization.numba_backtest import full_backtest_with_telemetry

        zeros = np.zeros(n_bars, dtype=np.float64)
        try:
            params = candidate['params']
            n_sigs = len(signal_arrays['entry_bars'])

            # === Pass 1: Run backtest WITHOUT ML to get trade telemetry ===
            use_trailing = mgmt_arrays.get('use_trailing', np.zeros(n_sigs, dtype=np.bool_))
            trail_start = mgmt_arrays.get('trail_start_pips', np.zeros(n_sigs, dtype=np.float64))
            trail_step = mgmt_arrays.get('trail_step_pips', np.zeros(n_sigs, dtype=np.float64))
            use_be = mgmt_arrays.get('use_breakeven', np.zeros(n_sigs, dtype=np.bool_))
            be_trigger = mgmt_arrays.get('be_trigger_pips', np.zeros(n_sigs, dtype=np.float64))
            be_offset = mgmt_arrays.get('be_offset_pips', np.zeros(n_sigs, dtype=np.float64))
            use_partial = mgmt_arrays.get('use_partial', np.zeros(n_sigs, dtype=np.bool_))
            partial_pct = mgmt_arrays.get('partial_pct', np.zeros(n_sigs, dtype=np.float64))
            partial_target = mgmt_arrays.get('partial_target_pips', np.zeros(n_sigs, dtype=np.float64))
            max_bars_arr = mgmt_arrays.get('max_bars', np.zeros(n_sigs, dtype=np.int64))
            trail_mode = mgmt_arrays.get('trail_mode', np.zeros(n_sigs, dtype=np.int64))
            chandelier_atr_mult = mgmt_arrays.get('chandelier_atr_mult', np.full(n_sigs, 3.0, dtype=np.float64))
            atr_pips_arr = mgmt_arrays.get('atr_pips', np.full(n_sigs, 35.0, dtype=np.float64))
            stale_exit_bars = mgmt_arrays.get('stale_exit_bars', np.zeros(n_sigs, dtype=np.int64))

            # ML disabled for pass 1
            no_ml = np.zeros(n_sigs, dtype=np.bool_)
            no_ml_hold = np.zeros(n_sigs, dtype=np.int64)
            no_ml_thresh = np.ones(n_sigs, dtype=np.float64)
            ml_long_zeros = np.zeros(n_bars, dtype=np.float64)
            ml_short_zeros = np.zeros(n_bars, dtype=np.float64)
            quality_mult = np.empty(0, dtype=np.float64)

            result = full_backtest_with_telemetry(
                signal_arrays['entry_bars'],
                signal_arrays['entry_prices'],
                signal_arrays['directions'],
                signal_arrays['sl_prices'],
                signal_arrays['tp_prices'],
                use_trailing, trail_start, trail_step,
                use_be, be_trigger, be_offset,
                use_partial, partial_pct, partial_target,
                max_bars_arr,
                trail_mode, chandelier_atr_mult, atr_pips_arr, stale_exit_bars,
                ml_long_zeros, ml_short_zeros, no_ml, no_ml_hold, no_ml_thresh,
                w_highs, w_lows, w_closes, w_days,
                self.config.initial_capital,
                self.config.risk_per_trade,
                pip_size,
                params.get('max_daily_trades', 0),
                params.get('max_daily_loss_pct', 0.0),
                quality_mult,
                quote_rate,
            )

            # Unpack telemetry:
            # (pnls, equity_curve, exit_reasons, bars_held, entry_bar_indices,
            #  exit_bar_indices, mfe_r, mae_r, signal_indices,
            #  n_trades, win_rate, pf, sharpe, max_dd, total_return, r_squared, ontester)
            entry_bar_indices = result[4]
            exit_bar_indices = result[5]
            signal_indices_arr = result[8]
            n_trades = result[9]

            if n_trades < 3:
                logger.warning(f"  ML inference: too few trades ({n_trades}) for feature generation")
                return zeros.copy(), zeros.copy(), {'status': 'too_few_trades', 'n_trades': int(n_trades)}

            # === Pass 2: Build features for each (trade, bar) pair ===
            market_data = precompute_market_features(w_highs, w_lows, w_closes, w_opens)
            feature_names = get_feature_names()

            # Use real hour data to match training features (Fix 2: was zeros, causing train/test skew)
            hours = w_hours if w_hours is not None else np.zeros(n_bars, dtype=np.int64)
            day_of_week = w_days

            sig_entry_prices = signal_arrays['entry_prices']
            sig_sl_prices = signal_arrays['sl_prices']
            sig_tp_prices = signal_arrays['tp_prices']
            sig_directions = signal_arrays['directions']

            rows = []
            bar_indices_list = []

            for trade_i in range(n_trades):
                entry_bar = int(entry_bar_indices[trade_i])
                exit_bar = int(exit_bar_indices[trade_i])
                trade_duration = exit_bar - entry_bar

                if trade_duration < 3:
                    continue

                # Map trade to its original signal index (signals can be skipped)
                sig_idx = int(signal_indices_arr[trade_i])
                if sig_idx >= len(sig_entry_prices):
                    continue

                direction = int(sig_directions[sig_idx])
                entry_price = float(sig_entry_prices[sig_idx])
                sl_price = float(sig_sl_prices[sig_idx])
                tp_price = float(sig_tp_prices[sig_idx])
                initial_sl_dist = abs(entry_price - sl_price)

                if initial_sl_dist < 1e-10:
                    continue

                running_mfe_r = 0.0
                running_mae_r = 0.0

                for bar in range(entry_bar, exit_bar + 1):
                    if bar >= n_bars:
                        break

                    # Update running MFE/MAE
                    if direction == 1:
                        fav_excursion = (w_highs[bar] - entry_price) / initial_sl_dist
                        adv_excursion = (entry_price - w_lows[bar]) / initial_sl_dist
                    else:
                        fav_excursion = (entry_price - w_lows[bar]) / initial_sl_dist
                        adv_excursion = (w_highs[bar] - entry_price) / initial_sl_dist

                    if fav_excursion > running_mfe_r:
                        running_mfe_r = fav_excursion
                    if adv_excursion > running_mae_r:
                        running_mae_r = adv_excursion

                    trade_state = {
                        'direction': direction,
                        'entry_bar': entry_bar,
                        'entry_price': entry_price,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'initial_sl_dist': initial_sl_dist,
                        'mfe_r': running_mfe_r,
                        'mae_r': running_mae_r,
                        'hour': int(hours[bar]) if bar < len(hours) else 0,
                        'day_of_week': int(day_of_week[bar]) if bar < len(day_of_week) else 0,
                        'max_hold_bars': int(max_bars_arr[sig_idx]) if sig_idx < len(max_bars_arr) else 0,
                    }

                    features = compute_decision_features(trade_state, market_data, bar)
                    row = {}
                    for j, fname in enumerate(feature_names):
                        row[fname] = features[j]
                    rows.append(row)
                    bar_indices_list.append(bar)

            if not rows:
                logger.warning("  ML inference: no feature rows generated")
                return zeros.copy(), zeros.copy(), {'status': 'no_features'}

            features_df = pd.DataFrame(rows)
            bar_indices_arr = np.array(bar_indices_list, dtype=np.int64)

            # === Pass 3: Predict and apply policy ===
            predictions = predict_exit_scores(models, features_df)

            policy_scores = apply_exit_policy(
                predictions,
                hold_value_threshold=self.config.ml_exit.min_hold_value,
                adverse_risk_threshold=self.config.ml_exit.max_adverse_risk,
                min_confidence=self.config.ml_exit.min_confidence,
                policy_mode=self.config.ml_exit.policy_mode,
            )

            ml_long, ml_short = generate_ml_score_arrays(
                policy_scores, bar_indices_arr, n_bars,
            )

            n_exit_signals = int(np.sum(policy_scores > 0))
            n_hold_below = int((predictions['hold_value_pred'] < self.config.ml_exit.min_hold_value).sum())
            n_risk_above = int((predictions['adverse_risk_pred'] > self.config.ml_exit.max_adverse_risk).sum())
            metrics = {
                'status': 'success',
                'n_trades': int(n_trades),
                'n_feature_rows': len(rows),
                'n_exit_signals': n_exit_signals,
                'mean_confidence': float(predictions['confidence'].mean()),
                'mean_hold_value': float(predictions['hold_value_pred'].mean()),
                'mean_adverse_risk': float(predictions['adverse_risk_pred'].mean()),
                'hold_value_range': [
                    float(predictions['hold_value_pred'].min()),
                    float(predictions['hold_value_pred'].max()),
                ],
                'adverse_risk_range': [
                    float(predictions['adverse_risk_pred'].min()),
                    float(predictions['adverse_risk_pred'].max()),
                ],
            }
            logger.info(
                f"  ML inference: {n_trades} trades, {len(rows)} feature rows, "
                f"{n_exit_signals} exit signals "
                f"(hold<{self.config.ml_exit.min_hold_value}: {n_hold_below}, "
                f"risk>{self.config.ml_exit.max_adverse_risk}: {n_risk_above})"
            )

            return ml_long, ml_short, metrics

        except Exception as e:
            logger.warning(f"  ML inference failed: {e}")
            return zeros.copy(), zeros.copy(), {'status': 'error', 'error': str(e)}

    def _train_and_predict_ml_exit(
        self,
        params: Dict[str, Any],
        df: pd.DataFrame,
        strategy: FastStrategy,
        window: Dict,
        bar_start: int,
        bar_end: int,
        full_highs: np.ndarray,
        full_lows: np.ndarray,
        full_closes: np.ndarray,
        full_days: np.ndarray,
        pip_size: float,
    ) -> Optional[Dict[str, Any]]:
        """Orchestrate ML exit training and inference for a single walk-forward window.

        1. Slice training data (all data before this window's test start).
        2. Train ML models on training data via _train_ml_for_window.
        3. Restore full-dataset precomputed signals (training mutated them).
        4. Run two-pass inference via _generate_ml_arrays_for_window to generate
           per-bar ML score arrays for the numba backtest engine.

        Args:
            params: Candidate parameter dict.
            df: Full DataFrame (entire dataset).
            strategy: FastStrategy instance (precomputed on full dataset).
            window: Window dict with train_start, train_end, test_start, test_end.
            bar_start: Bar index of window test start in the full dataset.
            bar_end: Bar index of window test end in the full dataset.
            full_highs, full_lows, full_closes, full_days: Full market data arrays.
            pip_size: Pip size for the currency pair.

        Returns:
            Dict with 'score_arrays' (ml_long, ml_short) and 'metrics' dict,
            or None on failure.
        """
        try:
            # Step 1: Get training data -- all data BEFORE this window's test start
            df_train_ml = df[df.index < window['test_start']]
            if len(df_train_ml) < 200:
                logger.warning(
                    f"  ML: insufficient training data ({len(df_train_ml)} bars), need >= 200"
                )
                return None

            candidate = {'params': params}

            # Step 2: Train ML models
            # NOTE: build_exit_dataset calls strategy.precompute_for_dataset(df_train_ml)
            # which mutates the strategy's cached signals. We must restore afterwards.
            models = self._train_ml_for_window(df_train_ml, candidate, strategy)

            # Step 3: Restore precomputed signals on full dataset
            strategy.precompute_for_dataset(df)

            if models is None:
                return None

            # Step 4: Prepare test window arrays for two-pass inference
            w_highs = full_highs[bar_start:bar_end]
            w_lows = full_lows[bar_start:bar_end]
            w_closes = full_closes[bar_start:bar_end]
            w_days = full_days[bar_start:bar_end]
            w_opens = (
                df['open'].values[bar_start:bar_end].astype(np.float64)
                if 'open' in df.columns else None
            )
            w_hours = df.index[bar_start:bar_end].hour.values.astype(np.int64) if hasattr(df.index, 'hour') else np.zeros(bar_end - bar_start, dtype=np.int64)
            n_bars = bar_end - bar_start

            # Re-fetch signal arrays from the restored full-dataset precomputation
            signal_arrays_full, mgmt_arrays_full = strategy.get_all_arrays(
                params, full_highs, full_lows, full_closes, full_days
            )

            all_entry_bars = signal_arrays_full['entry_bars']
            sig_mask = (all_entry_bars >= bar_start) & (all_entry_bars < bar_end)
            n_window_sigs = int(np.sum(sig_mask))

            if n_window_sigs < 1:
                logger.warning("  ML: no signals in test window")
                return None

            # Build window-local signal arrays (re-indexed to window start, spread applied)
            w_signal_arrays = {
                'entry_bars': all_entry_bars[sig_mask] - bar_start,
                'entry_prices': np.where(
                    signal_arrays_full['directions'][sig_mask] == 1,
                    signal_arrays_full['entry_prices'][sig_mask] + (self.config.spread_pips + self.config.slippage_pips) * pip_size,
                    signal_arrays_full['entry_prices'][sig_mask] - (self.config.spread_pips + self.config.slippage_pips) * pip_size,
                ),
                'directions': signal_arrays_full['directions'][sig_mask],
                'sl_prices': signal_arrays_full['sl_prices'][sig_mask],
                'tp_prices': signal_arrays_full['tp_prices'][sig_mask],
            }

            # Slice management arrays for this window
            w_mgmt_arrays = {}
            mgmt_defaults = {
                'use_trailing': np.zeros(n_window_sigs, dtype=np.bool_),
                'trail_start_pips': np.zeros(n_window_sigs, dtype=np.float64),
                'trail_step_pips': np.zeros(n_window_sigs, dtype=np.float64),
                'use_breakeven': np.zeros(n_window_sigs, dtype=np.bool_),
                'be_trigger_pips': np.zeros(n_window_sigs, dtype=np.float64),
                'be_offset_pips': np.zeros(n_window_sigs, dtype=np.float64),
                'use_partial': np.zeros(n_window_sigs, dtype=np.bool_),
                'partial_pct': np.zeros(n_window_sigs, dtype=np.float64),
                'partial_target_pips': np.zeros(n_window_sigs, dtype=np.float64),
                'max_bars': np.zeros(n_window_sigs, dtype=np.int64),
                'trail_mode': np.zeros(n_window_sigs, dtype=np.int64),
                'chandelier_atr_mult': np.full(n_window_sigs, 3.0, dtype=np.float64),
                'atr_pips': np.full(n_window_sigs, 35.0, dtype=np.float64),
                'stale_exit_bars': np.zeros(n_window_sigs, dtype=np.int64),
            }
            for key, default_val in mgmt_defaults.items():
                full_arr = mgmt_arrays_full.get(key, default_val)
                if len(full_arr) == len(all_entry_bars):
                    w_mgmt_arrays[key] = full_arr[sig_mask]
                else:
                    w_mgmt_arrays[key] = default_val

            quote_rate = get_quote_conversion_rate(self.config.pair, 'USD')

            # Step 5: Generate ML score arrays via two-pass approach
            ml_long, ml_short, metrics = self._generate_ml_arrays_for_window(
                models, w_highs, w_lows, w_closes, w_opens, w_days,
                candidate, strategy,
                w_signal_arrays, w_mgmt_arrays,
                pip_size, quote_rate, n_bars,
                w_hours=w_hours,
            )

            # Enrich metrics with training info
            metrics['backend'] = models.backend
            metrics['training_metrics'] = models.training_metrics
            metrics['top_features'] = dict(sorted(
                models.feature_importances.items(),
                key=lambda x: x[1], reverse=True,
            )[:5])

            return {
                'score_arrays': (ml_long, ml_short),
                'metrics': metrics,
            }

        except Exception as e:
            logger.warning(f"  ML exit train+predict failed: {e}")
            return None

    def _train_and_apply_signal_filter(
        self,
        all_entry_bars: np.ndarray,
        all_directions: np.ndarray,
        all_sl_prices: np.ndarray,
        all_tp_prices: np.ndarray,
        all_entry_prices: np.ndarray,
        test_sig_mask: np.ndarray,
        full_highs: np.ndarray,
        full_lows: np.ndarray,
        full_closes: np.ndarray,
        df: pd.DataFrame,
        bar_start: int,
        bar_end: int,
        pip_size: float,
        params: Dict[str, Any],
        strategy: FastStrategy,
        full_days: np.ndarray,
        mgmt_arrays: Dict[str, np.ndarray],
        window_idx: int,
        strategy_attrs: Optional[Dict[str, np.ndarray]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Train a signal classifier on signals before the test window and filter test signals.

        Meta-labeling approach: one prediction per signal (not per bar), binary classification.
        Includes OOS skill check, strategy-specific features, and threshold calibration.

        Returns:
            Dict with 'keep_mask' (bool array for test signals) and 'metrics', or None.
        """
        from optimization.numba_backtest import full_backtest_with_telemetry

        try:
            # === Step 1: Get training signals (signals BEFORE this test window) ===
            train_sig_mask = all_entry_bars < bar_start
            n_train_sigs = int(np.sum(train_sig_mask))

            if n_train_sigs < 10:
                logger.warning(f"  W{window_idx+1} Signal filter: too few training signals ({n_train_sigs}), need >= 10")
                return None

            # === Step 2: Run backtest on training signals to get trade outcomes ===
            # Find the range of bars that training signals span
            train_entry_bars = all_entry_bars[train_sig_mask]
            train_bar_start = max(0, int(train_entry_bars.min()) - 50)  # buffer for lookback
            train_bar_end = bar_start  # up to test window start

            # Re-index training signals to local range
            t_entry_bars = train_entry_bars - train_bar_start
            t_directions = all_directions[train_sig_mask]
            t_sl_prices = all_sl_prices[train_sig_mask]
            t_tp_prices = all_tp_prices[train_sig_mask]

            # Apply spread
            t_raw_prices = all_entry_prices[train_sig_mask]
            t_entry_prices = np.where(
                t_directions == 1,
                t_raw_prices + (self.config.spread_pips + self.config.slippage_pips) * pip_size,
                t_raw_prices - (self.config.spread_pips + self.config.slippage_pips) * pip_size,
            )

            # Slice management arrays for training signals
            t_use_trailing = mgmt_arrays['use_trailing'][train_sig_mask]
            t_trail_start = mgmt_arrays['trail_start_pips'][train_sig_mask]
            t_trail_step = mgmt_arrays['trail_step_pips'][train_sig_mask]
            t_use_be = mgmt_arrays['use_breakeven'][train_sig_mask]
            t_be_trigger = mgmt_arrays['be_trigger_pips'][train_sig_mask]
            t_be_offset = mgmt_arrays['be_offset_pips'][train_sig_mask]
            t_use_partial = mgmt_arrays['use_partial'][train_sig_mask]
            t_partial_pct = mgmt_arrays['partial_pct'][train_sig_mask]
            t_partial_target = mgmt_arrays['partial_target_pips'][train_sig_mask]
            t_max_bars = mgmt_arrays['max_bars'][train_sig_mask]
            t_trail_mode = mgmt_arrays['trail_mode'][train_sig_mask]
            t_chandelier_atr_mult = mgmt_arrays['chandelier_atr_mult'][train_sig_mask]
            t_atr_pips = mgmt_arrays['atr_pips'][train_sig_mask]
            t_stale_exit_bars = mgmt_arrays['stale_exit_bars'][train_sig_mask]

            # Slice market data for training range
            t_highs = full_highs[train_bar_start:train_bar_end]
            t_lows = full_lows[train_bar_start:train_bar_end]
            t_closes = full_closes[train_bar_start:train_bar_end]
            t_days = full_days[train_bar_start:train_bar_end]
            n_train_bars = train_bar_end - train_bar_start

            # No ML for training backtest
            no_ml = np.zeros(n_train_sigs, dtype=np.bool_)
            no_ml_hold = np.zeros(n_train_sigs, dtype=np.int64)
            no_ml_thresh = np.ones(n_train_sigs, dtype=np.float64)
            ml_zeros = np.zeros(n_train_bars, dtype=np.float64)
            quality_mult = np.empty(0, dtype=np.float64)
            quote_rate = get_quote_conversion_rate(self.config.pair, 'USD')

            result = full_backtest_with_telemetry(
                t_entry_bars, t_entry_prices, t_directions,
                t_sl_prices, t_tp_prices,
                t_use_trailing, t_trail_start, t_trail_step,
                t_use_be, t_be_trigger, t_be_offset,
                t_use_partial, t_partial_pct, t_partial_target,
                t_max_bars,
                t_trail_mode, t_chandelier_atr_mult, t_atr_pips, t_stale_exit_bars,
                ml_zeros, ml_zeros, no_ml, no_ml_hold, no_ml_thresh,
                t_highs, t_lows, t_closes, t_days,
                self.config.initial_capital,
                self.config.risk_per_trade,
                pip_size,
                params.get('max_daily_trades', 0),
                params.get('max_daily_loss_pct', 0.0),
                quality_mult,
                quote_rate,
            )

            # Unpack telemetry
            pnls = result[0]
            signal_indices = result[8]
            n_trades = int(result[9])

            if n_trades < 10:
                logger.warning(f"  W{window_idx+1} Signal filter: too few training trades ({n_trades}), need >= 10")
                return None

            # === Step 3: Map trade outcomes to signals ===
            # Binary labels: 1 = profitable, 0 = losing trade
            trade_labels = np.zeros(n_train_sigs, dtype=np.float64)
            trade_pnls = np.zeros(n_train_sigs, dtype=np.float64)
            trade_has_outcome = np.zeros(n_train_sigs, dtype=bool)

            for trade_i in range(n_trades):
                sig_idx = int(signal_indices[trade_i])
                if sig_idx < n_train_sigs:
                    trade_labels[sig_idx] = 1.0 if pnls[trade_i] > 0 else 0.0
                    trade_pnls[sig_idx] = pnls[trade_i]
                    trade_has_outcome[sig_idx] = True

            # Only use signals that actually became trades
            labeled_mask = trade_has_outcome
            n_labeled = int(labeled_mask.sum())

            if n_labeled < 10:
                logger.warning(f"  W{window_idx+1} Signal filter: too few labeled signals ({n_labeled})")
                return None

            # === Step 4: Compute features for labeled training signals ===
            # Use original bar indices (not re-indexed) for feature computation
            train_signal_bars = all_entry_bars[train_sig_mask][labeled_mask]
            train_signal_dirs = all_directions[train_sig_mask][labeled_mask]
            labels = trade_labels[labeled_mask]
            labeled_pnls = trade_pnls[labeled_mask]

            full_opens = df['open'].values.astype(np.float64) if 'open' in df.columns else full_closes.copy()
            full_hours = df.index.hour.values.astype(np.int64) if hasattr(df.index, 'hour') else np.zeros(len(full_closes), dtype=np.int64)
            full_day_of_week = df.index.dayofweek.values.astype(np.int64)

            # Slice strategy-specific attributes for training signals
            train_strat_attrs = None
            if strategy_attrs:
                train_strat_attrs = {}
                for key, arr in strategy_attrs.items():
                    # strategy_attrs arrays are aligned with all signals (full dataset)
                    # Apply train_sig_mask then labeled_mask to get labeled training subset
                    train_strat_attrs[key] = arr[train_sig_mask][labeled_mask]

            train_features = compute_signal_features(
                train_signal_bars, train_signal_dirs,
                full_closes, full_highs, full_lows, full_opens,
                full_hours, full_day_of_week,
                strategy_attrs=train_strat_attrs,
            )

            if len(train_features) < 10:
                logger.warning(f"  W{window_idx+1} Signal filter: too few feature rows ({len(train_features)})")
                return None

            # === Step 5: Train classifier (with OOS skill check) ===
            train_result = train_signal_classifier(train_features, labels)
            if train_result is None:
                return None
            clf, clf_metrics = train_result

            # Compute training profit factor for context
            train_wins = labeled_pnls[labeled_pnls > 0].sum()
            train_losses = abs(labeled_pnls[labeled_pnls < 0].sum())
            train_pf = train_wins / train_losses if train_losses > 0 else 999.0

            # === Step 5b: Calibrate threshold on training data ===
            calibrated_thresh = calibrate_threshold(clf, train_features, labels, labeled_pnls)
            logger.info(f"  W{window_idx+1} Signal filter: calibrated threshold={calibrated_thresh:.2f} "
                        f"(config default={self.config.ml_exit.signal_filter_threshold})")

            # === Step 6: Compute features for TEST window signals ===
            test_signal_bars = all_entry_bars[test_sig_mask]
            test_signal_dirs = all_directions[test_sig_mask]
            n_test_sigs = int(test_sig_mask.sum())

            # Slice strategy attributes for test signals
            test_strat_attrs = None
            if strategy_attrs:
                test_strat_attrs = {}
                for key, arr in strategy_attrs.items():
                    test_strat_attrs[key] = arr[test_sig_mask]

            test_features = compute_signal_features(
                test_signal_bars, test_signal_dirs,
                full_closes, full_highs, full_lows, full_opens,
                full_hours, full_day_of_week,
                strategy_attrs=test_strat_attrs,
            )

            # === Step 7: Predict and filter (using calibrated threshold) ===
            keep_mask = predict_signal_filter(clf, test_features, threshold=calibrated_thresh)

            n_kept = int(keep_mask.sum())
            n_profitable = int(labels.sum())
            win_rate = n_profitable / n_labeled if n_labeled > 0 else 0

            metrics = {
                'n_train_signals': n_labeled,
                'n_train_profitable': n_profitable,
                'train_win_rate': round(win_rate, 3),
                'train_pf': round(train_pf, 3),
                'val_auc': clf_metrics.get('val_auc', -1.0),
                'train_auc': clf_metrics.get('train_auc', -1.0),
                'skill_detected': clf_metrics.get('skill_detected', False),
                'backend': clf_metrics.get('backend', ''),
                'calibrated_threshold': round(calibrated_thresh, 3),
                'n_test_signals': n_test_sigs,
                'n_kept': n_kept,
                'filter_rate': round(1 - n_kept / n_test_sigs, 3) if n_test_sigs > 0 else 0,
                'n_strategy_features': len(train_strat_attrs) if train_strat_attrs else 0,
            }

            # Feature importance from classifier
            try:
                if hasattr(clf, 'feature_importances_'):
                    importances = clf.feature_importances_
                    feature_names = list(train_features.columns)
                    top_features = sorted(
                        zip(feature_names, importances),
                        key=lambda x: x[1], reverse=True
                    )[:5]
                    metrics['top_features'] = {name: round(float(imp), 4) for name, imp in top_features}
            except Exception:
                pass

            logger.info(
                f"  W{window_idx+1} Signal filter: train={n_labeled} signals "
                f"(WR={win_rate:.0%}, PF={train_pf:.2f}), "
                f"val_AUC={clf_metrics.get('val_auc', -1):.3f}, "
                f"threshold={calibrated_thresh:.2f}, "
                f"test={n_test_sigs} -> kept {n_kept} ({n_kept/n_test_sigs:.0%})"
            )

            return {'keep_mask': keep_mask, 'metrics': metrics}

        except Exception as e:
            logger.warning(f"  W{window_idx+1} Signal filter failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _train_and_apply_adaptive_sl(
        self,
        all_entry_bars: np.ndarray,
        all_directions: np.ndarray,
        all_sl_prices: np.ndarray,
        all_tp_prices: np.ndarray,
        all_entry_prices: np.ndarray,
        test_sig_mask: np.ndarray,
        full_highs: np.ndarray,
        full_lows: np.ndarray,
        full_closes: np.ndarray,
        df: pd.DataFrame,
        bar_start: int,
        bar_end: int,
        pip_size: float,
        params: Dict[str, Any],
        strategy: FastStrategy,
        full_days: np.ndarray,
        mgmt_arrays: Dict[str, np.ndarray],
        window_idx: int,
        use_regression: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Train ML to predict MAE (adverse excursion), tighten SL for easy trades.

        Instead of filtering signals (keep/reject), this approach tightens the
        stop-loss for signals predicted to have LOW adverse excursion. This:
        - Keeps ALL signals (no removing good signals)
        - Reduces risk on easy trades (tighter SL = less to lose)
        - Preserves TP (winners still capture full move)
        - Directly competes with fixed SL ("dumb SL")

        Returns:
            Dict with 'sl_scale' (float array, one per test signal) and 'metrics'.
        """
        from optimization.numba_backtest import full_backtest_with_telemetry
        from pipeline.ml_exit.signal_features import compute_signal_features, train_signal_classifier

        try:
            # === Step 1: Get training signals (before this test window) ===
            train_sig_mask = all_entry_bars < bar_start
            n_train_sigs = int(np.sum(train_sig_mask))

            if n_train_sigs < 20:
                logger.warning(f"  W{window_idx+1} Adaptive SL: too few training signals ({n_train_sigs})")
                return None

            # === Step 2: Run telemetry backtest on training signals ===
            train_entry_bars = all_entry_bars[train_sig_mask]
            train_bar_start = max(0, int(train_entry_bars.min()) - 50)
            train_bar_end = bar_start

            t_entry_bars = train_entry_bars - train_bar_start
            t_directions = all_directions[train_sig_mask]
            t_sl_prices = all_sl_prices[train_sig_mask]
            t_tp_prices = all_tp_prices[train_sig_mask]

            t_raw_prices = all_entry_prices[train_sig_mask]
            t_entry_prices = np.where(
                t_directions == 1,
                t_raw_prices + (self.config.spread_pips + self.config.slippage_pips) * pip_size,
                t_raw_prices - (self.config.spread_pips + self.config.slippage_pips) * pip_size,
            )

            # Slice management arrays
            t_use_trailing = mgmt_arrays['use_trailing'][train_sig_mask]
            t_trail_start = mgmt_arrays['trail_start_pips'][train_sig_mask]
            t_trail_step = mgmt_arrays['trail_step_pips'][train_sig_mask]
            t_use_be = mgmt_arrays['use_breakeven'][train_sig_mask]
            t_be_trigger = mgmt_arrays['be_trigger_pips'][train_sig_mask]
            t_be_offset = mgmt_arrays['be_offset_pips'][train_sig_mask]
            t_use_partial = mgmt_arrays['use_partial'][train_sig_mask]
            t_partial_pct = mgmt_arrays['partial_pct'][train_sig_mask]
            t_partial_target = mgmt_arrays['partial_target_pips'][train_sig_mask]
            t_max_bars = mgmt_arrays['max_bars'][train_sig_mask]
            t_trail_mode = mgmt_arrays['trail_mode'][train_sig_mask]
            t_chandelier_atr_mult = mgmt_arrays['chandelier_atr_mult'][train_sig_mask]
            t_atr_pips = mgmt_arrays['atr_pips'][train_sig_mask]
            t_stale_exit_bars = mgmt_arrays['stale_exit_bars'][train_sig_mask]

            t_highs = full_highs[train_bar_start:train_bar_end]
            t_lows = full_lows[train_bar_start:train_bar_end]
            t_closes = full_closes[train_bar_start:train_bar_end]
            t_days = full_days[train_bar_start:train_bar_end]
            n_train_bars = train_bar_end - train_bar_start

            no_ml = np.zeros(n_train_sigs, dtype=np.bool_)
            no_ml_hold = np.zeros(n_train_sigs, dtype=np.int64)
            no_ml_thresh = np.ones(n_train_sigs, dtype=np.float64)
            ml_zeros = np.zeros(n_train_bars, dtype=np.float64)
            quality_mult = np.empty(0, dtype=np.float64)
            quote_rate = get_quote_conversion_rate(self.config.pair, 'USD')

            result = full_backtest_with_telemetry(
                t_entry_bars, t_entry_prices, t_directions,
                t_sl_prices, t_tp_prices,
                t_use_trailing, t_trail_start, t_trail_step,
                t_use_be, t_be_trigger, t_be_offset,
                t_use_partial, t_partial_pct, t_partial_target,
                t_max_bars,
                t_trail_mode, t_chandelier_atr_mult, t_atr_pips, t_stale_exit_bars,
                ml_zeros, ml_zeros, no_ml, no_ml_hold, no_ml_thresh,
                t_highs, t_lows, t_closes, t_days,
                self.config.initial_capital,
                self.config.risk_per_trade,
                pip_size,
                params.get('max_daily_trades', 0),
                params.get('max_daily_loss_pct', 0.0),
                quality_mult,
                quote_rate,
            )

            # Unpack telemetry (index 6=mfe_r, 7=mae_r, 8=signal_indices, 9=n_trades)
            mae_r = result[7]
            signal_indices = result[8]
            n_trades = int(result[9])

            if n_trades < 20:
                logger.warning(f"  W{window_idx+1} Adaptive SL: too few training trades ({n_trades})")
                return None

            # === Step 3: Create labels ===
            MAE_THRESHOLD = 0.5
            trade_mae_values = np.zeros(n_train_sigs, dtype=np.float64)
            trade_labels = np.zeros(n_train_sigs, dtype=np.float64)
            trade_has_outcome = np.zeros(n_train_sigs, dtype=bool)

            n_easy = 0
            for trade_i in range(n_trades):
                sig_idx = int(signal_indices[trade_i])
                if sig_idx < n_train_sigs:
                    trade_mae_values[sig_idx] = mae_r[trade_i]
                    is_easy = 1.0 if mae_r[trade_i] < MAE_THRESHOLD else 0.0
                    trade_labels[sig_idx] = is_easy
                    trade_has_outcome[sig_idx] = True
                    if is_easy > 0:
                        n_easy += 1

            labeled_mask = trade_has_outcome
            n_labeled = int(labeled_mask.sum())

            if n_labeled < 50:
                logger.warning(f"  W{window_idx+1} Adaptive SL: too few labeled signals ({n_labeled})")
                return None

            # === Step 3b: Data-confidence scaling ===
            MIN_TRAIN_FOR_TIGHTENING = 100
            data_confidence = min(1.0, max(0.0, (n_labeled - MIN_TRAIN_FOR_TIGHTENING) / 200))

            if data_confidence <= 0:
                logger.info(
                    f"  W{window_idx+1} Adaptive SL: insufficient data confidence "
                    f"({n_labeled} labeled < {MIN_TRAIN_FOR_TIGHTENING} min) -> skipping"
                )
                return None

            # === Step 4: Compute features ===
            train_signal_bars = all_entry_bars[train_sig_mask][labeled_mask]
            train_signal_dirs = all_directions[train_sig_mask][labeled_mask]
            labels = trade_labels[labeled_mask]
            mae_labels = trade_mae_values[labeled_mask]

            full_opens = df['open'].values.astype(np.float64) if 'open' in df.columns else full_closes.copy()
            full_hours = df.index.hour.values.astype(np.int64) if hasattr(df.index, 'hour') else np.zeros(len(full_closes), dtype=np.int64)
            full_day_of_week = df.index.dayofweek.values.astype(np.int64)

            train_features = compute_signal_features(
                train_signal_bars, train_signal_dirs,
                full_closes, full_highs, full_lows, full_opens,
                full_hours, full_day_of_week,
            )

            if len(train_features) < 20:
                logger.warning(f"  W{window_idx+1} Adaptive SL: too few feature rows ({len(train_features)})")
                return None

            # === Step 5+6+7: Train model and compute SL scaling ===
            test_signal_bars = all_entry_bars[test_sig_mask]
            test_signal_dirs = all_directions[test_sig_mask]
            n_test_sigs = int(test_sig_mask.sum())

            test_features = compute_signal_features(
                test_signal_bars, test_signal_dirs,
                full_closes, full_highs, full_lows, full_opens,
                full_hours, full_day_of_week,
            )

            easy_rate = n_easy / n_labeled if n_labeled > 0 else 0

            if use_regression:
                # --- REGRESSION MODE: Predict MAE_r directly, set SL proportionally ---
                from pipeline.ml_exit.signal_features import train_mae_regressor

                reg = train_mae_regressor(train_features, mae_labels)
                if reg is None:
                    return None

                try:
                    predicted_mae = reg.predict(test_features.values)
                    sl_scale = np.ones(n_test_sigs, dtype=np.float64)

                    # SL = max(MIN_SL_PCT, predicted_MAE * BUFFER) * original_SL
                    # Buffer gives safety margin above predicted MAE
                    MIN_SL_PCT = 0.50  # Never tighten below 50% of original SL
                    BUFFER_MULT = 1.5  # SL = 1.5x predicted MAE
                    n_tightened = 0

                    for j in range(n_test_sigs):
                        # predicted_mae is MAE_r (fraction of original SL)
                        target_sl_pct = max(MIN_SL_PCT, predicted_mae[j] * BUFFER_MULT)
                        # Apply data confidence: blend target with 1.0 (no change)
                        sl_scale[j] = 1.0 - data_confidence * (1.0 - target_sl_pct)
                        if sl_scale[j] < 0.99:
                            n_tightened += 1

                except Exception as e:
                    logger.warning(f"  W{window_idx+1} Adaptive SL regression prediction failed: {e}")
                    return None

                mean_sl_scale = float(sl_scale[sl_scale < 0.99].mean()) if n_tightened > 0 else 1.0
                mean_pred_mae = float(predicted_mae.mean())
                metrics = {
                    'mode': 'regression',
                    'n_train_signals': n_labeled,
                    'n_train_easy': n_easy,
                    'easy_rate': round(easy_rate, 3),
                    'data_confidence': round(data_confidence, 3),
                    'mean_mae_train': round(float(mae_labels.mean()), 3),
                    'mean_mae_predicted': round(mean_pred_mae, 3),
                    'min_sl_pct': MIN_SL_PCT,
                    'buffer_mult': BUFFER_MULT,
                    'n_test_signals': n_test_sigs,
                    'n_tightened': n_tightened,
                    'tighten_rate': round(n_tightened / n_test_sigs, 3) if n_test_sigs > 0 else 0,
                    'mean_sl_scale_tightened': round(mean_sl_scale, 3),
                }

                # Feature importance
                try:
                    if hasattr(reg, 'feature_importances_'):
                        from pipeline.ml_exit.signal_features import SIGNAL_FEATURE_NAMES
                        importances = reg.feature_importances_
                        top_features = sorted(
                            zip(SIGNAL_FEATURE_NAMES, importances),
                            key=lambda x: x[1], reverse=True
                        )[:5]
                        metrics['top_features'] = {name: round(float(imp), 4) for name, imp in top_features}
                except Exception:
                    pass

                logger.info(
                    f"  W{window_idx+1} Adaptive SL [REG]: train={n_labeled} (conf={data_confidence:.2f}), "
                    f"mae_pred={mean_pred_mae:.3f}, tightened {n_tightened}/{n_test_sigs} (mean_scale={mean_sl_scale:.3f})"
                )

            else:
                # --- CLASSIFICATION MODE: easy/hard + gradient tightening ---
                clf_result = train_signal_classifier(train_features, labels)
                if clf_result is None:
                    return None
                clf = clf_result[0]  # Unpack (clf, metrics) tuple

                MAX_TIGHT = 0.45
                SL_TIGHT_FACTOR = 1.0 - MAX_TIGHT * data_confidence
                CONFIDENCE_CUTOFF = 0.50

                try:
                    probs = clf.predict_proba(test_features.values)[:, 1]  # P(easy)
                    sl_scale = np.ones(n_test_sigs, dtype=np.float64)

                    n_tightened = 0
                    for j in range(n_test_sigs):
                        if probs[j] >= CONFIDENCE_CUTOFF:
                            prob_scale = (probs[j] - CONFIDENCE_CUTOFF) / (1.0 - CONFIDENCE_CUTOFF)
                            sl_scale[j] = 1.0 - (1.0 - SL_TIGHT_FACTOR) * prob_scale
                            n_tightened += 1

                except Exception as e:
                    logger.warning(f"  W{window_idx+1} Adaptive SL prediction failed: {e}")
                    return None

                mean_sl_scale = float(sl_scale[sl_scale < 1.0].mean()) if n_tightened > 0 else 1.0
                metrics = {
                    'mode': 'classification',
                    'n_train_signals': n_labeled,
                    'n_train_easy': n_easy,
                    'easy_rate': round(easy_rate, 3),
                    'data_confidence': round(data_confidence, 3),
                    'mae_threshold': MAE_THRESHOLD,
                    'sl_tight_factor': round(SL_TIGHT_FACTOR, 3),
                    'confidence_cutoff': CONFIDENCE_CUTOFF,
                    'n_test_signals': n_test_sigs,
                    'n_tightened': n_tightened,
                    'tighten_rate': round(n_tightened / n_test_sigs, 3) if n_test_sigs > 0 else 0,
                    'mean_prob_easy': round(float(probs.mean()), 3),
                    'mean_sl_scale_tightened': round(mean_sl_scale, 3),
                }

                # Feature importance
                try:
                    if hasattr(clf, 'feature_importances_'):
                        from pipeline.ml_exit.signal_features import SIGNAL_FEATURE_NAMES
                        importances = clf.feature_importances_
                        top_features = sorted(
                            zip(SIGNAL_FEATURE_NAMES, importances),
                            key=lambda x: x[1], reverse=True
                        )[:5]
                        metrics['top_features'] = {name: round(float(imp), 4) for name, imp in top_features}
                except Exception:
                    pass

                logger.info(
                    f"  W{window_idx+1} Adaptive SL [CLF]: train={n_labeled} (conf={data_confidence:.2f}), "
                    f"tightened {n_tightened}/{n_test_sigs} (SL*{SL_TIGHT_FACTOR:.2f}, mean_scale={mean_sl_scale:.3f})"
                )

            return {'sl_scale': sl_scale, 'metrics': metrics}

        except Exception as e:
            logger.warning(f"  W{window_idx+1} Adaptive SL failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _test_candidate_windows(
        self,
        candidate: Dict[str, Any],
        windows: List[Dict],
        df: pd.DataFrame,
        strategy: FastStrategy,
    ) -> List[Dict[str, Any]]:
        """Test a candidate's parameters across all windows (legacy fallback)."""
        results = []

        for i, window in enumerate(windows):
            # Slice data for this window
            df_train = df[
                (df.index >= window['train_start']) &
                (df.index < window['train_end'])
            ]
            df_test = df[
                (df.index >= window['test_start']) &
                (df.index < window['test_end'])
            ]

            if len(df_train) < 100 or len(df_test) < 50:
                logger.warning(f"  Window {i+1}: Insufficient data (train={len(df_train)}, test={len(df_test)})")
                results.append({
                    'window': i + 1,
                    'status': 'insufficient_data',
                    'train_candles': len(df_train),
                    'test_candles': len(df_test),
                })
                continue

            # Test parameters on this window
            metrics = self._backtest_params(
                candidate['params'],
                df_test,
                strategy,
            )

            min_trades = self.config.walkforward.min_trades_per_window
            window_passed = (
                metrics.trades >= min_trades and
                metrics.ontester_score > 0
            )

            results.append({
                'window': i + 1,
                'status': 'completed',
                'train_candles': len(df_train),
                'test_candles': len(df_test),
                'trades': metrics.trades,
                'ontester': metrics.ontester_score,
                'sharpe': metrics.sharpe,
                'return': metrics.total_return,
                'max_dd': metrics.max_dd,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'passed': window_passed,
            })

        return results

    def _backtest_params(
        self,
        params: Dict[str, Any],
        df: pd.DataFrame,
        strategy: FastStrategy,
    ) -> Metrics:
        """Run backtest with given parameters on data."""
        from optimization.numba_backtest import full_backtest_numba

        # Prepare arrays
        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)
        closes = df['close'].values.astype(np.float64)
        days = df.index.dayofweek.values.astype(np.int64)

        # Precompute signals
        strategy.precompute_for_dataset(df)

        # Get filtered signals and arrays
        signal_arrays, mgmt_arrays = strategy.get_all_arrays(
            params, highs, lows, closes, days
        )

        if len(signal_arrays['entry_bars']) < 3:
            return Metrics(0, 0, 0, 0, 0, 0, 0, 0)

        # Apply spread
        pip_size = 0.01 if 'JPY' in self.config.pair else 0.0001
        entry_prices = np.where(
            signal_arrays['directions'] == 1,
            signal_arrays['entry_prices'] + (self.config.spread_pips + self.config.slippage_pips) * pip_size,
            signal_arrays['entry_prices'] - (self.config.spread_pips + self.config.slippage_pips) * pip_size
        )

        # Get management arrays
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
        trail_mode = mgmt_arrays.get('trail_mode', np.zeros(n, dtype=np.int64))
        chandelier_atr_mult = mgmt_arrays.get('chandelier_atr_mult', np.full(n, 3.0, dtype=np.float64))
        atr_pips_arr = mgmt_arrays.get('atr_pips', np.full(n, 35.0, dtype=np.float64))
        stale_exit_bars = mgmt_arrays.get('stale_exit_bars', np.zeros(n, dtype=np.int64))
        quality_mult = np.empty(0, dtype=np.float64)

        # V6: ML exit arrays
        n_bars = len(highs)
        use_ml = mgmt_arrays.get('use_ml_exit', np.zeros(n, dtype=np.bool_))
        ml_min_hold_arr = mgmt_arrays.get('ml_min_hold', np.zeros(n, dtype=np.int64))
        ml_threshold_arr = mgmt_arrays.get('ml_threshold', np.ones(n, dtype=np.float64))

        if hasattr(strategy, 'get_ml_score_arrays') and np.any(use_ml):
            ml_long, ml_short = strategy.get_ml_score_arrays(params, highs, lows, closes)
        else:
            ml_long = np.zeros(n_bars, dtype=np.float64)
            ml_short = np.zeros(n_bars, dtype=np.float64)

        result = full_backtest_numba(
            signal_arrays['entry_bars'],
            entry_prices,
            signal_arrays['directions'],
            signal_arrays['sl_prices'],
            signal_arrays['tp_prices'],
            use_trailing, trail_start, trail_step,
            use_be, be_trigger, be_offset,
            use_partial, partial_pct, partial_target,
            max_bars,
            trail_mode, chandelier_atr_mult, atr_pips_arr, stale_exit_bars,
            ml_long, ml_short, use_ml, ml_min_hold_arr, ml_threshold_arr,
            highs, lows, closes, days,
            self.config.initial_capital,
            self.config.risk_per_trade,
            pip_size,
            params.get('max_daily_trades', 0),
            params.get('max_daily_loss_pct', 0.0),
            quality_mult,
            get_quote_conversion_rate(self.config.pair, 'USD'),
        )

        return Metrics(*result)

    def _calculate_wf_stats(self, window_results: List[Dict]) -> Dict[str, Any]:
        """Calculate walk-forward statistics from window results."""
        completed = [r for r in window_results if r.get('status') == 'completed']

        if not completed:
            return {
                'n_windows': 0,
                'n_passed': 0,
                'pass_rate': 0.0,
                'mean_sharpe': 0.0,
                'mean_ontester': 0.0,
                'mean_return': 0.0,
                'max_dd': 0.0,
                'consistency': 0.0,
                'oos_n_windows': 0,
                'oos_n_passed': 0,
                'oos_pass_rate': 0.0,
            }

        passed = [r for r in completed if r.get('passed', False)]

        sharpes = [r['sharpe'] for r in completed]
        ontesters = [r['ontester'] for r in completed]
        returns = [r['return'] for r in completed]
        max_dds = [r['max_dd'] for r in completed]

        # Out-of-sample window stats (only windows after optimization back-test period)
        oos_completed = [r for r in completed if r.get('out_of_sample', True)]
        oos_passed = [r for r in oos_completed if r.get('passed', False)]

        stats = {
            'n_windows': len(completed),
            'n_passed': len(passed),
            'pass_rate': len(passed) / len(completed) if completed else 0,
            'mean_sharpe': np.mean(sharpes),
            'std_sharpe': np.std(sharpes),
            'min_sharpe': min(sharpes),
            'max_sharpe': max(sharpes),
            'mean_ontester': np.mean(ontesters),
            'mean_return': np.mean(returns),
            'total_return': sum(returns),
            'max_dd': max(max_dds),
            'mean_dd': np.mean(max_dds),
            'consistency': np.mean(sharpes) / (np.std(sharpes) + 0.01),  # Sharpe ratio of sharpes
            # Out-of-sample specific metrics
            'oos_n_windows': len(oos_completed),
            'oos_n_passed': len(oos_passed),
            'oos_pass_rate': len(oos_passed) / len(oos_completed) if oos_completed else 0.0,
        }

        if oos_completed:
            oos_sharpes = [r['sharpe'] for r in oos_completed]
            stats['oos_mean_sharpe'] = np.mean(oos_sharpes)

        return stats

    def _candidate_passes(self, wf_stats: Dict[str, Any]) -> bool:
        """Check if candidate passes walk-forward criteria.

        Uses out-of-sample pass rate when OOS windows are available,
        since in-sample windows overlap with optimization training data
        and don't provide independent validation.
        """
        # Use OOS pass rate if we have OOS windows, otherwise fall back to all windows
        oos_n = wf_stats.get('oos_n_windows', 0)
        if oos_n > 0:
            pass_rate = wf_stats['oos_pass_rate']
            mean_sharpe = wf_stats.get('oos_mean_sharpe', wf_stats['mean_sharpe'])
        else:
            # No OOS windows available - use all windows (backward compat)
            pass_rate = wf_stats['pass_rate']
            mean_sharpe = wf_stats['mean_sharpe']

        # Pass rate threshold
        if pass_rate < self.config.walkforward.min_window_pass_rate:
            return False

        # Mean Sharpe threshold
        if mean_sharpe < self.config.walkforward.min_mean_sharpe:
            return False

        return True
