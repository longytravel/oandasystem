"""
Stage 5: Monte Carlo Simulation

Randomizes trade order to estimate return distribution:
- Shuffle trade PnLs N times (default 500)
- Calculate equity curve for each shuffle
- Compute statistics: mean, std, percentiles
- 5th percentile = worst case at 95% confidence

Enhanced: Also collects detailed trade data for report visualization:
- Per-trade records (entry/exit bar, direction, PnL, pips)
- Equity curve with bar indices
- Back/forward split equity curves
- Raw MC return/DD arrays for interactive histograms
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from numba import njit, prange

from loguru import logger

from optimization.numba_backtest import full_backtest_with_trades, full_backtest_with_telemetry, get_quote_conversion_rate
from pipeline.config import PipelineConfig
from pipeline.state import PipelineState
from pipeline.stages.s2_optimization import get_strategy


@njit(cache=True)
def simulate_equity_curve(pnls: np.ndarray, initial_capital: float) -> Tuple[float, float]:
    """
    Simulate equity curve from PnL array.

    Returns: (final_return_pct, max_drawdown_pct)
    """
    n = len(pnls)
    if n == 0:
        return 0.0, 0.0

    equity = initial_capital
    peak = initial_capital
    max_dd = 0.0

    for i in range(n):
        equity += pnls[i]
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    final_return = (equity - initial_capital) / initial_capital * 100
    return final_return, max_dd * 100


@njit(cache=True)
def _shuffle_and_simulate(pnls: np.ndarray, initial_capital: float, seed: int) -> Tuple[float, float]:
    """Single Monte Carlo iteration - shuffle and simulate."""
    np.random.seed(seed)
    n_trades = len(pnls)
    shuffled = pnls.copy()

    # Fisher-Yates shuffle
    for j in range(n_trades - 1, 0, -1):
        k = np.random.randint(0, j + 1)
        shuffled[j], shuffled[k] = shuffled[k], shuffled[j]

    return simulate_equity_curve(shuffled, initial_capital)


@njit(parallel=True, cache=True)
def run_monte_carlo_numba(
    pnls: np.ndarray,
    initial_capital: float,
    n_iterations: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Monte Carlo simulation in parallel with Numba.

    Fixed: Each parallel iteration gets unique seed for thread safety.

    Args:
        pnls: Array of trade PnLs
        initial_capital: Starting capital
        n_iterations: Number of iterations
        seed: Base random seed (each iteration uses seed + i)

    Returns:
        (returns_array, max_dd_array) - arrays of final returns and max DDs
    """
    returns = np.zeros(n_iterations, dtype=np.float64)
    max_dds = np.zeros(n_iterations, dtype=np.float64)

    for i in prange(n_iterations):
        # Each iteration gets unique seed = base_seed + iteration
        iter_seed = seed + i
        ret, dd = _shuffle_and_simulate(pnls, initial_capital, iter_seed)
        returns[i] = ret
        max_dds[i] = dd

    return returns, max_dds


@njit(cache=True)
def _bootstrap_metrics(pnls: np.ndarray, initial_capital: float, seed: int) -> Tuple[float, float, float]:
    """Single bootstrap iteration - sample with replacement and compute metrics.

    Returns: (sharpe_estimate, win_rate, profit_factor)
    """
    np.random.seed(seed)
    n = len(pnls)
    resampled = np.zeros(n, dtype=np.float64)

    # Sample with replacement
    for i in range(n):
        idx = np.random.randint(0, n)
        resampled[i] = pnls[idx]

    # Compute metrics on bootstrap sample
    mean_pnl = 0.0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    for i in range(n):
        mean_pnl += resampled[i]
        if resampled[i] > 0:
            wins += 1
            gross_profit += resampled[i]
        else:
            gross_loss -= resampled[i]
    mean_pnl /= n

    var = 0.0
    for i in range(n):
        var += (resampled[i] - mean_pnl) ** 2
    std = np.sqrt(var / (n - 1)) if n > 1 else 1.0

    sharpe = np.sqrt(n) * mean_pnl / std if std > 0 else 0.0
    win_rate = wins / n if n > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

    return sharpe, win_rate, pf


@njit(parallel=True, cache=True)
def run_bootstrap_numba(
    pnls: np.ndarray,
    initial_capital: float,
    n_iterations: int,
    seed: int = 12345
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run bootstrap resampling in parallel with Numba.

    Samples trades WITH REPLACEMENT to estimate confidence intervals
    on key metrics (Sharpe, win rate, profit factor).

    Returns:
        (sharpe_array, win_rate_array, pf_array)
    """
    sharpes = np.zeros(n_iterations, dtype=np.float64)
    win_rates = np.zeros(n_iterations, dtype=np.float64)
    pfs = np.zeros(n_iterations, dtype=np.float64)

    for i in prange(n_iterations):
        iter_seed = seed + i
        s, w, p = _bootstrap_metrics(pnls, initial_capital, iter_seed)
        sharpes[i] = s
        win_rates[i] = w
        pfs[i] = p

    return sharpes, win_rates, pfs


@njit(cache=True)
def _permutation_sharpe(pnls: np.ndarray, seed: int) -> float:
    """Single permutation iteration - randomly flip trade signs and compute Sharpe.

    This tests the null hypothesis that trade direction has no predictive value.
    By randomly flipping the sign of each PnL, we simulate random entry timing
    while preserving the magnitude distribution.
    """
    np.random.seed(seed)
    n = len(pnls)
    permuted = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if np.random.random() < 0.5:
            permuted[i] = pnls[i]
        else:
            permuted[i] = -pnls[i]

    mean_pnl = 0.0
    for i in range(n):
        mean_pnl += permuted[i]
    mean_pnl /= n

    var = 0.0
    for i in range(n):
        var += (permuted[i] - mean_pnl) ** 2
    std = np.sqrt(var / (n - 1)) if n > 1 else 1.0

    return np.sqrt(n) * mean_pnl / std if std > 0 else 0.0


@njit(parallel=True, cache=True)
def run_permutation_test(pnls: np.ndarray, n_iterations: int, seed: int = 54321) -> np.ndarray:
    """Run permutation significance test in parallel.

    Returns array of Sharpe ratios from randomized sign-flipped trade sequences.
    Compare original Sharpe to this distribution to get p-value.
    """
    sharpes = np.zeros(n_iterations, dtype=np.float64)
    for i in prange(n_iterations):
        sharpes[i] = _permutation_sharpe(pnls, seed + i)
    return sharpes


class MonteCarloStage:
    """Stage 5: Monte Carlo simulation for risk estimation."""

    name = "montecarlo"

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(
        self,
        state: PipelineState,
        df_back: pd.DataFrame,
        df_forward: pd.DataFrame,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Execute Monte Carlo simulation for each candidate.

        Extracts trade PnLs and shuffles to estimate return distribution.

        Args:
            state: Pipeline state
            df_back: Back-test data (not used, kept for consistency)
            df_forward: Forward-test data
            candidates: Candidates from stability stage

        Returns:
            Dict with:
            - candidates: Updated candidates with MC results
            - summary: Overall MC statistics
        """
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 5: MONTE CARLO SIMULATION")
        logger.info("=" * 70)

        if not candidates:
            logger.warning("No candidates to analyze!")
            # Fix Finding 12: persist stage output on early return
            empty_result = {'n_analyzed': 0}
            state.save_stage_output('montecarlo', {'summary': empty_result, 'candidates': []})
            return {
                'candidates': [],
                'summary': empty_result,
            }

        logger.info(f"Iterations: {self.config.montecarlo.iterations}")
        logger.info(f"Confidence level: {self.config.montecarlo.confidence_level:.0%}")
        logger.info(f"Min trades required: {self.config.montecarlo.min_trades_for_mc}")

        # Get strategy
        strategy = get_strategy(self.config.strategy_name)
        strategy.set_pip_size(self.config.pair)
        pip_size = 0.01 if 'JPY' in self.config.pair else 0.0001

        # Combine back and forward data for MC analysis
        df_full = pd.concat([df_back, df_forward])

        # Analyze each candidate
        for i, candidate in enumerate(candidates):
            logger.info(f"\n--- Candidate {candidate['rank']} ---")

            # Get trade PnLs
            pnls, equity = self._get_trade_pnls(
                candidate['params'],
                df_full,
                strategy,
                pip_size,
            )

            if len(pnls) < self.config.montecarlo.min_trades_for_mc:
                logger.warning(f"  Insufficient trades for MC: {len(pnls)} < {self.config.montecarlo.min_trades_for_mc}")
                candidate['montecarlo'] = {
                    'status': 'insufficient_trades',
                    'n_trades': len(pnls),
                }
                continue

            # Run Monte Carlo
            mc_results, mc_raw_returns, mc_raw_dds = self._run_monte_carlo(pnls, len(pnls))

            candidate['montecarlo'] = {
                'status': 'completed',
                'n_trades': len(pnls),
                'n_iterations': self.config.montecarlo.iterations,
                'results': mc_results,
                'raw_returns': mc_raw_returns.tolist(),
                'raw_max_dds': mc_raw_dds.tolist(),
            }

            # Log key metrics
            logger.info(f"  Trades: {len(pnls)}")
            logger.info(f"  Original return: {mc_results['original_return']:.1f}%")
            logger.info(f"  MC Mean return:  {mc_results['mean_return']:.1f}%")
            logger.info(f"  5th percentile:  {mc_results['pct_5_return']:.1f}%")
            logger.info(f"  95th %ile DD:    {mc_results['pct_95_dd']:.1f}%")
            if 'bootstrap' in mc_results:
                bs = mc_results['bootstrap']
                logger.info(f"  Bootstrap Sharpe CI: [{bs['sharpe_ci_lower']:.2f}, {bs['sharpe_ci_upper']:.2f}]")
                logger.info(f"  Bootstrap WinRate CI: [{bs['win_rate_ci_lower']:.1%}, {bs['win_rate_ci_upper']:.1%}]")
            if 'permutation' in mc_results:
                perm = mc_results['permutation']
                sig = "YES" if perm['significant_at_05'] else "NO"
                logger.info(f"  Permutation p-value: {perm['p_value']:.4f} (significant: {sig})")

        # Collect detailed trade data for the best candidate (for report)
        # Find the candidate with best combined_rank
        best_candidate = min(candidates, key=lambda c: c.get('combined_rank', 999))
        logger.info(f"\nCollecting trade details for best candidate (rank {best_candidate['rank']})...")
        trade_details = self._collect_trade_details(
            best_candidate['params'],
            df_back, df_forward, strategy, pip_size,
        )
        best_candidate['trade_details'] = trade_details

        # Update state
        state.candidates = candidates

        # Build summary
        completed = [c for c in candidates if c.get('montecarlo', {}).get('status') == 'completed']

        summary = {
            'n_analyzed': len(candidates),
            'n_completed': len(completed),
            'iterations': self.config.montecarlo.iterations,
        }

        if completed:
            summary['avg_5th_pct_return'] = np.mean([
                c['montecarlo']['results']['pct_5_return'] for c in completed
            ])
            summary['best_5th_pct_return'] = max([
                c['montecarlo']['results']['pct_5_return'] for c in completed
            ])

        # Save stage output
        output_data = {
            'summary': summary,
            'candidates': [
                {
                    'rank': c['rank'],
                    'combined_rank': c['combined_rank'],
                    'montecarlo': c.get('montecarlo', {}),
                }
                for c in candidates
            ],
        }
        state.save_stage_output('montecarlo', output_data)

        logger.info(f"\nMonte Carlo Complete:")
        logger.info(f"  Candidates analyzed: {len(candidates)}")
        logger.info(f"  Successfully completed: {len(completed)}")

        return {
            'candidates': candidates,
            'summary': summary,
        }

    def _get_trade_pnls(
        self,
        params: Dict[str, Any],
        df: pd.DataFrame,
        strategy,
        pip_size: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract trade PnLs from backtest."""
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
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        # Apply slippage to entry prices (spread is deducted in engine via PnL)
        entry_prices = np.where(
            signal_arrays['directions'] == 1,
            signal_arrays['entry_prices'] + self.config.slippage_pips * pip_size,
            signal_arrays['entry_prices'] - self.config.slippage_pips * pip_size
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

        result = full_backtest_with_trades(
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
            spread_pips=self.config.spread_pips,
        )

        pnls, equity = result[0], result[1]
        return pnls, equity

    def _run_monte_carlo(
        self,
        pnls: np.ndarray,
        n_trades: int,
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """Run Monte Carlo simulation on PnL array.

        Returns: (results_dict, raw_returns_array, raw_max_dds_array)
        """
        # Run parallel MC simulation
        returns, max_dds = run_monte_carlo_numba(
            pnls,
            self.config.initial_capital,
            self.config.montecarlo.iterations,
            seed=42,
        )

        # Calculate original metrics
        original_return, original_dd = simulate_equity_curve(pnls, self.config.initial_capital)

        # Calculate percentiles
        confidence = self.config.montecarlo.confidence_level
        low_pct = (1 - confidence) * 100  # 5th percentile for 95% confidence
        high_pct = confidence * 100       # 95th percentile

        results = {
            # Original (actual order)
            'original_return': original_return,
            'original_max_dd': original_dd,

            # Return distribution
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'pct_5_return': np.percentile(returns, low_pct),
            'pct_25_return': np.percentile(returns, 25),
            'pct_50_return': np.percentile(returns, 50),
            'pct_75_return': np.percentile(returns, 75),
            'pct_95_return': np.percentile(returns, high_pct),

            # Drawdown distribution
            'mean_dd': np.mean(max_dds),
            'std_dd': np.std(max_dds),
            'max_dd': np.max(max_dds),
            'pct_5_dd': np.percentile(max_dds, low_pct),
            'pct_50_dd': np.percentile(max_dds, 50),
            'pct_95_dd': np.percentile(max_dds, high_pct),

            # VaR metrics
            'var_95': -np.percentile(returns, low_pct) if np.percentile(returns, low_pct) < 0 else 0,
            'expected_shortfall': np.mean(returns[returns <= np.percentile(returns, low_pct)]) if len(returns[returns <= np.percentile(returns, low_pct)]) > 0 else 0,

            # Confidence metrics
            'prob_positive': np.mean(returns > 0) * 100,
            'prob_above_5pct': np.mean(returns > 5) * 100,
            'prob_above_10pct': np.mean(returns > 10) * 100,
        }

        # Bootstrap resampling for confidence intervals on key metrics
        bootstrap_iters = self.config.montecarlo.bootstrap_iterations
        if bootstrap_iters > 0 and len(pnls) >= self.config.montecarlo.min_trades_for_mc:
            sharpes, win_rates, pfs = run_bootstrap_numba(
                pnls,
                self.config.initial_capital,
                bootstrap_iters,
                seed=12345,
            )

            results['bootstrap'] = {
                'sharpe_mean': float(np.mean(sharpes)),
                'sharpe_std': float(np.std(sharpes)),
                'sharpe_ci_lower': float(np.percentile(sharpes, low_pct)),
                'sharpe_ci_upper': float(np.percentile(sharpes, high_pct)),
                'win_rate_mean': float(np.mean(win_rates)),
                'win_rate_ci_lower': float(np.percentile(win_rates, low_pct)),
                'win_rate_ci_upper': float(np.percentile(win_rates, high_pct)),
                'pf_mean': float(np.mean(pfs)),
                'pf_ci_lower': float(np.percentile(pfs, low_pct)),
                'pf_ci_upper': float(np.percentile(pfs, high_pct)),
            }

        # Permutation significance test - does the strategy beat random entry?
        if len(pnls) >= self.config.montecarlo.min_trades_for_mc:
            # Compute original Sharpe for comparison
            mean_pnl = np.mean(pnls)
            std_pnl = np.std(pnls, ddof=1) if len(pnls) > 1 else 1.0
            original_sharpe = np.sqrt(len(pnls)) * mean_pnl / std_pnl if std_pnl > 0 else 0.0

            perm_sharpes = run_permutation_test(
                pnls,
                n_iterations=self.config.montecarlo.iterations,
                seed=54321,
            )

            # p-value: fraction of permuted Sharpes >= original Sharpe
            p_value = float(np.mean(perm_sharpes >= original_sharpe))

            results['permutation'] = {
                'original_sharpe': float(original_sharpe),
                'p_value': p_value,
                'significant_at_05': p_value < 0.05,
                'significant_at_01': p_value < 0.01,
                'perm_mean_sharpe': float(np.mean(perm_sharpes)),
                'perm_95th': float(np.percentile(perm_sharpes, 95)),
            }

        return results, returns, max_dds

    def _run_telemetry_backtest(
        self,
        params: Dict[str, Any],
        df: pd.DataFrame,
        strategy,
        pip_size: float,
    ) -> Tuple:
        """Run full_backtest_with_telemetry on a dataframe.

        Returns the raw telemetry result tuple, or None if insufficient signals.
        """
        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)
        closes = df['close'].values.astype(np.float64)
        days = df.index.dayofweek.values.astype(np.int64)

        strategy.precompute_for_dataset(df)
        signal_arrays, mgmt_arrays = strategy.get_all_arrays(
            params, highs, lows, closes, days
        )

        if len(signal_arrays['entry_bars']) < 3:
            return None

        directions = signal_arrays['directions']
        # Apply slippage to entry prices (spread is deducted in engine via PnL)
        entry_prices = np.where(
            directions == 1,
            signal_arrays['entry_prices'] + self.config.slippage_pips * pip_size,
            signal_arrays['entry_prices'] - self.config.slippage_pips * pip_size
        )

        n = len(signal_arrays['entry_bars'])
        n_bars = len(highs)

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

        use_ml = mgmt_arrays.get('use_ml_exit', np.zeros(n, dtype=np.bool_))
        ml_min_hold_arr = mgmt_arrays.get('ml_min_hold', np.zeros(n, dtype=np.int64))
        ml_threshold_arr = mgmt_arrays.get('ml_threshold', np.ones(n, dtype=np.float64))

        if hasattr(strategy, 'get_ml_score_arrays') and np.any(use_ml):
            ml_long, ml_short = strategy.get_ml_score_arrays(params, highs, lows, closes)
        else:
            ml_long = np.zeros(n_bars, dtype=np.float64)
            ml_short = np.zeros(n_bars, dtype=np.float64)

        result = full_backtest_with_telemetry(
            signal_arrays['entry_bars'],
            entry_prices,
            directions,
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
            spread_pips=self.config.spread_pips,
        )

        return result, signal_arrays, entry_prices

    def _collect_trade_details(
        self,
        params: Dict[str, Any],
        df_back: pd.DataFrame,
        df_forward: pd.DataFrame,
        strategy,
        pip_size: float,
    ) -> Dict[str, Any]:
        """Collect detailed trade data for report visualization.

        Uses full_backtest_with_telemetry to get per-trade exit reasons,
        bars held, MFE/MAE in R-multiples, and entry/exit bar indices.

        Returns dict with:
        - trades: list of per-trade dicts (enriched with telemetry)
        - equity_curve: list of {bar_idx, equity, timestamp}
        - back_equity / forward_equity: split equity curves
        - summary stats: gross_profit, gross_loss, etc.
        """
        EXIT_REASON_NAMES = {0: 'sl', 1: 'tp', 2: 'trailing', 3: 'time',
                             4: 'stale', 5: 'ml', 6: 'force_close'}

        result = {}

        for period_name, df in [('back', df_back), ('forward', df_forward)]:
            telemetry_result = self._run_telemetry_backtest(params, df, strategy, pip_size)

            if telemetry_result is None:
                result[f'{period_name}_trades'] = []
                result[f'{period_name}_equity'] = []
                continue

            bt_result, signal_arrays, entry_prices = telemetry_result
            (pnls, equity_curve, exit_reasons, bars_held, entry_bar_indices,
             exit_bar_indices, mfe_r_arr, mae_r_arr, signal_indices_arr,
             n_trades, win_rate, pf, sharpe, max_dd, total_ret, r_sq, ontester) = bt_result

            if n_trades == 0:
                result[f'{period_name}_trades'] = []
                result[f'{period_name}_equity'] = []
                continue

            timestamps = df.index
            directions = signal_arrays['directions']
            sl_prices = signal_arrays['sl_prices']
            tp_prices = signal_arrays['tp_prices']

            trades = []
            equity_pts = []
            running_equity = self.config.initial_capital

            for i in range(n_trades):
                running_equity += pnls[i]

                trade = {
                    'pnl': float(pnls[i]),
                    'equity_after': float(running_equity),
                    'exit_reason': EXIT_REASON_NAMES.get(int(exit_reasons[i]), 'unknown'),
                    'bars_held': int(bars_held[i]),
                    'mfe_r': float(mfe_r_arr[i]),
                    'mae_r': float(mae_r_arr[i]),
                }

                # Map entry/exit bars to timestamps
                e_bar = int(entry_bar_indices[i])
                x_bar = int(exit_bar_indices[i])

                if e_bar < len(timestamps):
                    trade['entry_time'] = str(timestamps[e_bar])
                    trade['entry_bar'] = e_bar
                if x_bar < len(timestamps):
                    trade['exit_time'] = str(timestamps[x_bar])
                    trade['exit_bar'] = x_bar

                # Map to signal data using signal_indices (trade_i may != signal_i)
                sig_idx = int(signal_indices_arr[i])
                if sig_idx < len(directions):
                    trade['direction'] = 'long' if int(directions[sig_idx]) == 1 else 'short'
                    trade['entry_price'] = float(entry_prices[sig_idx])
                    trade['sl_price'] = float(sl_prices[sig_idx])
                    trade['tp_price'] = float(tp_prices[sig_idx])

                    # Calculate pips (use sig_idx for correct entry price)
                    if pip_size == 0.01:
                        trade['pips'] = float(pnls[i]) / (float(entry_prices[sig_idx]) * 0.01)
                    else:
                        trade['pips'] = float(pnls[i]) / 10.0

                trades.append(trade)
                equity_pts.append({
                    'trade_num': i,
                    'equity': float(running_equity),
                    'timestamp': trade.get('entry_time', ''),
                })

            result[f'{period_name}_trades'] = trades
            result[f'{period_name}_equity'] = equity_pts

        # Compute summary stats from all trades
        all_trades = result.get('back_trades', []) + result.get('forward_trades', [])
        all_pnls = [t['pnl'] for t in all_trades]

        if all_pnls:
            wins = [p for p in all_pnls if p > 0]
            losses = [p for p in all_pnls if p <= 0]

            # Consecutive wins/losses
            max_consec_wins = 0
            max_consec_losses = 0
            cur_wins = 0
            cur_losses = 0
            for p in all_pnls:
                if p > 0:
                    cur_wins += 1
                    cur_losses = 0
                    max_consec_wins = max(max_consec_wins, cur_wins)
                else:
                    cur_losses += 1
                    cur_wins = 0
                    max_consec_losses = max(max_consec_losses, cur_losses)

            # Exit reason breakdown
            exit_reason_counts = {}
            for t in all_trades:
                reason = t.get('exit_reason', 'unknown')
                exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1

            # MFE/MAE stats
            all_mfe = [t['mfe_r'] for t in all_trades if 'mfe_r' in t]
            all_mae = [t['mae_r'] for t in all_trades if 'mae_r' in t]
            all_bars = [t['bars_held'] for t in all_trades if 'bars_held' in t]

            result['summary'] = {
                'total_trades': len(all_pnls),
                'total_net_profit': sum(all_pnls),
                'gross_profit': sum(wins) if wins else 0,
                'gross_loss': sum(losses) if losses else 0,
                'profit_factor': sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0,
                'win_rate': len(wins) / len(all_pnls),
                'avg_win': np.mean(wins) if wins else 0,
                'avg_loss': np.mean(losses) if losses else 0,
                'largest_win': max(wins) if wins else 0,
                'largest_loss': min(losses) if losses else 0,
                'max_consecutive_wins': max_consec_wins,
                'max_consecutive_losses': max_consec_losses,
                'expected_payoff': np.mean(all_pnls),
                'long_trades': len([t for t in all_trades if t.get('direction') == 'long']),
                'short_trades': len([t for t in all_trades if t.get('direction') == 'short']),
                'long_wins': len([t for t in all_trades if t.get('direction') == 'long' and t['pnl'] > 0]),
                'short_wins': len([t for t in all_trades if t.get('direction') == 'short' and t['pnl'] > 0]),
                'exit_reason_counts': exit_reason_counts,
                'avg_bars_held': float(np.mean(all_bars)) if all_bars else 0,
                'avg_mfe_r': float(np.mean(all_mfe)) if all_mfe else 0,
                'avg_mae_r': float(np.mean(all_mae)) if all_mae else 0,
            }
        else:
            result['summary'] = {'total_trades': 0}

        return result
