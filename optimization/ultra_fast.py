"""
Ultra-Fast MT5-Style Generic Optimizer with Optuna Support.

Works with ANY strategy that implements FastStrategy interface.
Supports both grid sampling and intelligent Optuna TPE sampling.

Speed techniques:
1. Pre-compute ALL signals once (not per trial)
2. Numba JIT for backtest core
3. Vectorized numpy operations
4. Fast signal filtering per trial

Usage:
    from strategies.rsi_fast import get_strategy

    strategy = get_strategy('rsi_full')
    optimizer = UltraFastOptimizer(strategy)

    # Grid sampling (fast, exhaustive)
    results = optimizer.run(df_back, df_forward, n_trials=10000, use_optuna=False)

    # Optuna sampling (intelligent, finds optima faster)
    results = optimizer.run(df_back, df_forward, n_trials=10000, use_optuna=True)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
from pathlib import Path
from datetime import datetime
import json
import itertools
import time
from numba import njit
import warnings
warnings.filterwarnings('ignore')

from loguru import logger

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from optimization.fast_strategy import FastStrategy


class Metrics(NamedTuple):
    """Lightweight metrics container."""
    trades: int
    win_rate: float
    profit_factor: float
    sharpe: float
    max_dd: float
    total_return: float


@njit(cache=True)
def fast_backtest_numba(
    entry_bars: np.ndarray,
    entry_prices: np.ndarray,
    directions: np.ndarray,
    sl_prices: np.ndarray,
    tp_prices: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    initial_capital: float,
    risk_pct: float,
) -> Tuple[int, float, float, float, float, float]:
    """
    Numba-compiled backtest engine.

    Returns: (trades, win_rate, profit_factor, sharpe, max_dd, total_return)
    """
    n_signals = len(entry_bars)
    if n_signals == 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0)

    pnls = np.zeros(n_signals, dtype=np.float64)
    n_trades = 0
    equity = initial_capital
    peak_equity = initial_capital
    max_dd = 0.0

    sig_idx = 0
    in_pos = False
    pos_dir = 0
    pos_entry = 0.0
    pos_sl = 0.0
    pos_tp = 0.0
    pos_size = 0.0

    n_bars = len(highs)

    for bar in range(n_bars):
        # Check exit
        if in_pos:
            exited = False
            exit_price = 0.0

            if pos_dir == 1:  # Long
                if lows[bar] <= pos_sl:
                    exit_price = pos_sl
                    exited = True
                elif highs[bar] >= pos_tp:
                    exit_price = pos_tp
                    exited = True
            else:  # Short
                if highs[bar] >= pos_sl:
                    exit_price = pos_sl
                    exited = True
                elif lows[bar] <= pos_tp:
                    exit_price = pos_tp
                    exited = True

            if exited:
                if pos_dir == 1:
                    pnl = (exit_price - pos_entry) * pos_size * 10000
                else:
                    pnl = (pos_entry - exit_price) * pos_size * 10000

                pnls[n_trades] = pnl
                n_trades += 1
                equity += pnl

                # Track drawdown
                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity
                if dd > max_dd:
                    max_dd = dd

                in_pos = False

        # Check entry
        if not in_pos and sig_idx < n_signals:
            if entry_bars[sig_idx] == bar:
                sl_dist = abs(entry_prices[sig_idx] - sl_prices[sig_idx])
                risk_amt = equity * risk_pct / 100.0
                pos_size = risk_amt / (sl_dist * 10000) if sl_dist > 0 else 0.01

                pos_entry = entry_prices[sig_idx]
                pos_sl = sl_prices[sig_idx]
                pos_tp = tp_prices[sig_idx]
                pos_dir = directions[sig_idx]
                in_pos = True
                sig_idx += 1

    # Force close remaining position
    if in_pos and n_bars > 0:
        exit_price = closes[n_bars - 1]
        if pos_dir == 1:
            pnl = (exit_price - pos_entry) * pos_size * 10000
        else:
            pnl = (pos_entry - exit_price) * pos_size * 10000
        pnls[n_trades] = pnl
        n_trades += 1
        equity += pnl

    if n_trades == 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Calculate metrics
    pnls = pnls[:n_trades]

    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    for i in range(n_trades):
        if pnls[i] > 0:
            wins += 1
            gross_profit += pnls[i]
        else:
            gross_loss -= pnls[i]

    win_rate = wins / n_trades
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Sharpe ratio
    mean_pnl = 0.0
    for i in range(n_trades):
        mean_pnl += pnls[i]
    mean_pnl /= n_trades

    var = 0.0
    for i in range(n_trades):
        var += (pnls[i] - mean_pnl) ** 2
    std = np.sqrt(var / n_trades) if n_trades > 1 else 1.0

    sharpe = np.sqrt(252) * mean_pnl / std if std > 0 else 0.0

    total_ret = (equity - initial_capital) / initial_capital * 100

    return (n_trades, win_rate, pf, sharpe, max_dd * 100, total_ret)


class UltraFastOptimizer:
    """
    Ultra-fast optimizer that works with any FastStrategy.

    Supports:
    - Grid sampling (exhaustive coverage)
    - Optuna TPE sampling (intelligent search)
    - Parameter importance analysis
    """

    def __init__(
        self,
        strategy: FastStrategy,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 1.0,
        spread_pips: float = 1.5,
        pip_size: float = 0.0001,
        min_trades: int = 20,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.spread_pips = spread_pips
        self.pip_size = pip_size
        self.min_trades = min_trades

        # Will be set during run()
        self.df_back: pd.DataFrame = None
        self.df_forward: pd.DataFrame = None
        self.back_highs: np.ndarray = None
        self.back_lows: np.ndarray = None
        self.back_closes: np.ndarray = None
        self.fwd_highs: np.ndarray = None
        self.fwd_lows: np.ndarray = None
        self.fwd_closes: np.ndarray = None

        # Results
        self.results: List[Dict] = []
        self.all_back_results: List[Dict] = []  # Store ALL back results
        self.optuna_study: Optional['optuna.Study'] = None
        self.param_importance: Dict[str, float] = {}

    def _run_trial(
        self,
        params: Dict,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> Metrics:
        """Run a single trial with given parameters."""
        # Get filtered signal arrays from strategy
        entry_bars, entry_prices, directions, sl_prices, tp_prices = \
            self.strategy.get_filtered_arrays(params, highs, lows, closes)

        if len(entry_bars) < 3:
            return Metrics(0, 0, 0, 0, 0, 0)

        # Apply spread to entry prices
        entry_prices = np.where(
            directions == 1,
            entry_prices + self.spread_pips * self.pip_size,
            entry_prices - self.spread_pips * self.pip_size
        )

        # Run numba backtest
        trades, wr, pf, sharpe, dd, ret = fast_backtest_numba(
            entry_bars, entry_prices, directions,
            sl_prices, tp_prices,
            highs.astype(np.float64),
            lows.astype(np.float64),
            closes.astype(np.float64),
            self.initial_capital,
            self.risk_per_trade,
        )

        return Metrics(trades, wr, pf, sharpe, dd, ret)

    def generate_param_grid(self, n_trials: int) -> List[Dict]:
        """Generate parameter combinations using grid sampling."""
        param_space = self.strategy.get_parameter_space()
        keys = list(param_space.keys())

        all_combos = list(itertools.product(*[param_space[k] for k in keys]))
        total_combos = len(all_combos)

        logger.info(f"Parameter space: {len(keys)} params, {total_combos:,} total combinations")

        if total_combos > n_trials:
            np.random.seed(42)
            indices = np.random.choice(total_combos, n_trials, replace=False)
            selected = [all_combos[i] for i in indices]
            logger.info(f"Randomly sampling {n_trials:,} from {total_combos:,}")
        else:
            selected = all_combos
            logger.info(f"Testing all {total_combos:,} combinations")

        return [{k: v for k, v in zip(keys, combo)} for combo in selected]

    def _create_optuna_objective(self, back_signals):
        """Create Optuna objective function."""
        param_space = self.strategy.get_parameter_space()

        def objective(trial: 'optuna.Trial') -> float:
            # Sample parameters using Optuna
            params = {}
            for key, values in param_space.items():
                if isinstance(values[0], bool):
                    params[key] = trial.suggest_categorical(key, values)
                elif isinstance(values[0], int):
                    params[key] = trial.suggest_categorical(key, values)
                elif isinstance(values[0], float):
                    params[key] = trial.suggest_categorical(key, values)
                else:
                    params[key] = trial.suggest_categorical(key, values)

            # Run backtest
            self.strategy._precomputed_signals = back_signals
            metrics = self._run_trial(
                params, self.back_highs, self.back_lows, self.back_closes
            )

            # Store result
            trial.set_user_attr('params', params)
            trial.set_user_attr('metrics', metrics)

            # Penalize low trade count
            if metrics.trades < self.min_trades:
                return -100

            # Optimize for Sharpe ratio
            return metrics.sharpe

        return objective

    def run(
        self,
        df_back: pd.DataFrame,
        df_forward: pd.DataFrame,
        n_trials: int = 10000,
        top_n: int = 100,
        min_back_sharpe: float = 1.0,
        use_optuna: bool = False,
        forward_all: bool = False,  # Test ALL back results on forward
    ) -> List[Dict]:
        """
        Run full optimization: back + forward testing.

        Args:
            df_back: Back-testing data
            df_forward: Forward-testing data
            n_trials: Number of trials to run
            top_n: Top N results to forward test (ignored if forward_all=True)
            min_back_sharpe: Minimum back Sharpe to qualify for forward
            use_optuna: Use Optuna TPE sampling instead of grid
            forward_all: Test ALL valid back results on forward

        Returns:
            List of results sorted by forward Sharpe
        """
        self.df_back = df_back
        self.df_forward = df_forward

        # Store price arrays
        self.back_highs = df_back['high'].values
        self.back_lows = df_back['low'].values
        self.back_closes = df_back['close'].values
        self.fwd_highs = df_forward['high'].values
        self.fwd_lows = df_forward['low'].values
        self.fwd_closes = df_forward['close'].values

        # Pre-compute signals for both datasets
        logger.info(f"Pre-computing signals for {self.strategy.name}...")

        start = time.time()
        n_back = self.strategy.precompute_for_dataset(df_back)
        back_signals = self.strategy._precomputed_signals.copy()
        precompute_back = time.time() - start

        start = time.time()
        n_fwd = self.strategy.precompute_for_dataset(df_forward)
        fwd_signals = self.strategy._precomputed_signals.copy()
        precompute_fwd = time.time() - start

        logger.info(f"Pre-computed: {n_back} back, {n_fwd} forward signals "
                   f"({precompute_back:.1f}s + {precompute_fwd:.1f}s)")

        # === PHASE 1: BACK OPTIMIZATION ===
        if use_optuna and OPTUNA_AVAILABLE:
            back_results = self._run_optuna_optimization(
                n_trials, back_signals, min_back_sharpe
            )
        else:
            back_results = self._run_grid_optimization(
                n_trials, back_signals, min_back_sharpe
            )

        self.all_back_results = back_results

        # Sort by back Sharpe
        back_results.sort(key=lambda x: x['back'].sharpe, reverse=True)

        # Determine what to forward test
        if forward_all:
            to_forward = back_results
            logger.info(f"Forward testing ALL {len(to_forward)} valid back results...")
        else:
            to_forward = back_results[:top_n]
            logger.info(f"Forward testing top {len(to_forward)} by back Sharpe...")

        # === PHASE 2: FORWARD TESTING ===
        start = time.time()
        self.strategy._precomputed_signals = fwd_signals

        for result in to_forward:
            metrics = self._run_trial(
                result['params'], self.fwd_highs, self.fwd_lows, self.fwd_closes
            )
            result['forward'] = metrics

        fwd_time = time.time() - start
        logger.info(f"Forward complete: {len(to_forward)} trials in {fwd_time:.1f}s")

        # Filter and sort by forward Sharpe
        final_results = [r for r in to_forward if r['forward'].trades >= 5]
        final_results.sort(key=lambda x: x['forward'].sharpe, reverse=True)

        self.results = final_results

        # Calculate parameter importance if using Optuna
        if use_optuna and OPTUNA_AVAILABLE and self.optuna_study:
            self._calculate_param_importance()

        return final_results

    def _run_grid_optimization(
        self,
        n_trials: int,
        back_signals,
        min_back_sharpe: float
    ) -> List[Dict]:
        """Run optimization using grid sampling."""
        logger.info("Phase 1: BACK optimization (grid sampling)...")

        param_list = self.generate_param_grid(n_trials)
        self.strategy._precomputed_signals = back_signals

        start = time.time()
        back_results = []

        for i, params in enumerate(param_list):
            metrics = self._run_trial(
                params, self.back_highs, self.back_lows, self.back_closes
            )
            if metrics.trades >= self.min_trades and metrics.sharpe >= min_back_sharpe:
                back_results.append({
                    'trial_id': i,
                    'params': params,
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
        back_signals,
        min_back_sharpe: float
    ) -> List[Dict]:
        """Run optimization using Optuna TPE sampler."""
        logger.info("Phase 1: BACK optimization (Optuna TPE)...")

        # Create study
        sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
        self.optuna_study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
        )

        # Create objective
        objective = self._create_optuna_objective(back_signals)

        # Run optimization with progress callback
        start = time.time()

        def callback(study, trial):
            if (trial.number + 1) % 1000 == 0:
                elapsed = time.time() - start
                rate = (trial.number + 1) / elapsed
                best = study.best_value if study.best_trial else 0
                logger.info(f"  {trial.number+1:,}/{n_trials:,} ({rate:.0f}/sec) best={best:.2f}")

        self.optuna_study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[callback],
            show_progress_bar=False,
        )

        back_time = time.time() - start

        # Extract results
        back_results = []
        for i, trial in enumerate(self.optuna_study.trials):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                metrics = trial.user_attrs.get('metrics')
                if metrics and metrics.trades >= self.min_trades and metrics.sharpe >= min_back_sharpe:
                    back_results.append({
                        'trial_id': i,
                        'params': trial.user_attrs['params'],
                        'back': metrics,
                    })

        logger.info(f"Back complete: {len(back_results)} valid in {back_time:.1f}s "
                   f"({n_trials/back_time:.0f}/sec)")

        return back_results

    def _calculate_param_importance(self):
        """Calculate parameter importance using Optuna."""
        if not self.optuna_study:
            return

        try:
            importance = optuna.importance.get_param_importances(self.optuna_study)
            self.param_importance = dict(importance)
        except Exception as e:
            logger.warning(f"Could not calculate param importance: {e}")

    def print_results(self, top_n: int = 30):
        """Print comparison table of back vs forward results."""
        print("\n" + "=" * 110)
        print(f"BACK vs FORWARD RESULTS - {self.strategy.name}")
        print("=" * 110)
        print(f"{'#':<6} {'Back Sharpe':<13} {'Fwd Sharpe':<13} {'Decay':<10} "
              f"{'Back WR':<10} {'Fwd WR':<10} {'Back Tr':<10} {'Fwd Tr':<8}")
        print("-" * 110)

        for i, r in enumerate(self.results[:top_n]):
            back = r['back']
            fwd = r['forward']
            decay = (back.sharpe - fwd.sharpe) / back.sharpe * 100 if back.sharpe > 0 else 0

            print(f"{r['trial_id']:<6} "
                  f"{back.sharpe:<13.2f} "
                  f"{fwd.sharpe:<13.2f} "
                  f"{decay:>+8.1f}% "
                  f"{back.win_rate*100:<10.1f} "
                  f"{fwd.win_rate*100:<10.1f} "
                  f"{back.trades:<10} "
                  f"{fwd.trades:<8}")

        print("=" * 110)

        if self.results:
            print(f"\nBest forward result: Trial #{self.results[0]['trial_id']}")
            print(f"Parameters: {self.results[0]['params']}")

    def print_param_importance(self, top_n: int = 20):
        """Print parameter importance from Optuna."""
        if not self.param_importance:
            print("\nParameter importance not available (run with use_optuna=True)")
            return

        print("\n" + "=" * 60)
        print("PARAMETER IMPORTANCE (contribution to Sharpe)")
        print("=" * 60)

        sorted_params = sorted(
            self.param_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for param, importance in sorted_params[:top_n]:
            bar = "█" * int(importance * 50)
            print(f"{param:<30} {importance:>6.1%} {bar}")

        print("=" * 60)

    def save_results(self, output_dir: Path) -> Path:
        """Save results to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"{self.strategy.name}_{timestamp}.json"

        data = {
            'timestamp': timestamp,
            'strategy': self.strategy.name,
            'n_results': len(self.results),
            'n_back_tested': len(self.all_back_results),
            'param_importance': self.param_importance,
            'results': []
        }

        for r in self.results[:100]:  # Save top 100
            data['results'].append({
                'trial_id': r['trial_id'],
                'params': r['params'],
                'back': {
                    'trades': r['back'].trades,
                    'sharpe': r['back'].sharpe,
                    'win_rate': r['back'].win_rate,
                    'profit_factor': r['back'].profit_factor,
                    'max_dd': r['back'].max_dd,
                },
                'forward': {
                    'trades': r['forward'].trades,
                    'sharpe': r['forward'].sharpe,
                    'win_rate': r['forward'].win_rate,
                    'profit_factor': r['forward'].profit_factor,
                    'max_dd': r['forward'].max_dd,
                },
            })

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved to {filepath}")
        return filepath
