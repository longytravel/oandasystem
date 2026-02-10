"""
Staged Parameter Optimizer with Optuna.

Optimizes parameters in logical groups, locks winners, then does
a final tight-range optimization across all parameters.

Works with ANY EA that defines parameter groups.

Flow:
1. Optimize each group sequentially (others at defaults/locked)
2. Lock best values from each group
3. Final optimization with tight ranges around all locked values
4. Forward validation on final candidates
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, NamedTuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import time
from copy import deepcopy

from loguru import logger

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("Optuna required: pip install optuna")

from optimization.fast_strategy import FastStrategy


class Metrics(NamedTuple):
    """Lightweight metrics container."""
    trades: int
    win_rate: float
    profit_factor: float
    sharpe: float
    max_dd: float
    total_return: float


@dataclass
class ParameterDef:
    """Definition for a single parameter."""
    name: str
    values: List[Any]  # Full range of values
    default: Any  # Default value
    param_type: str = 'categorical'  # categorical, int, float

    def get_tight_range(self, locked_value: Any, n_values: int = 5) -> List[Any]:
        """Get tight range around a locked value."""
        if locked_value not in self.values:
            return [locked_value]

        idx = self.values.index(locked_value)
        n = len(self.values)

        # Get neighboring values
        half = n_values // 2
        start = max(0, idx - half)
        end = min(n, idx + half + 1)

        # Adjust if at edges
        if start == 0:
            end = min(n, n_values)
        if end == n:
            start = max(0, n - n_values)

        return self.values[start:end]


@dataclass
class ParameterGroup:
    """A group of related parameters."""
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


class StagedOptimizer:
    """
    Multi-stage optimizer that works with any EA.

    Usage:
        optimizer = StagedOptimizer(backtest_func)

        # Define parameter groups
        optimizer.add_group('signal', 'Entry signal parameters')
        optimizer.add_param('signal', 'rsi_period', [5,7,9,11,14,18,21], default=14)
        optimizer.add_param('signal', 'swing_strength', [3,5,7,9], default=5)

        optimizer.add_group('risk', 'Risk management')
        optimizer.add_param('risk', 'stop_loss', [20,30,40,50], default=30)

        # Run staged optimization
        results = optimizer.optimize(
            df_back, df_forward,
            trials_per_stage=5000,
            final_trials=10000
        )
    """

    def __init__(
        self,
        backtest_func: Callable[[Dict[str, Any], pd.DataFrame], Metrics],
        min_trades: int = 20,
        min_sharpe: float = 1.0,
    ):
        """
        Args:
            backtest_func: Function that takes (params_dict, dataframe) and returns Metrics
            min_trades: Minimum trades to consider valid
            min_sharpe: Minimum Sharpe for back-test qualification
        """
        self.backtest_func = backtest_func
        self.min_trades = min_trades
        self.min_sharpe = min_sharpe

        self.groups: Dict[str, ParameterGroup] = {}
        self.group_order: List[str] = []
        self.locked_params: Dict[str, Any] = {}

        # Results storage
        self.stage_results: Dict[str, Dict] = {}
        self.final_results: List[Dict] = []

    def add_group(self, name: str, description: str = ""):
        """Add a parameter group."""
        self.groups[name] = ParameterGroup(name, description)
        self.group_order.append(name)
        logger.info(f"Added group: {name} - {description}")

    def add_param(
        self,
        group: str,
        name: str,
        values: List[Any],
        default: Any,
        param_type: str = 'categorical'
    ):
        """Add a parameter to a group."""
        if group not in self.groups:
            raise ValueError(f"Group '{group}' not found. Add it first with add_group()")
        self.groups[group].add_param(name, values, default, param_type)

    def get_all_defaults(self) -> Dict[str, Any]:
        """Get defaults for all parameters across all groups."""
        defaults = {}
        for group in self.groups.values():
            defaults.update(group.get_defaults())
        return defaults

    def get_current_params(self, group_to_optimize: str = None) -> Dict[str, Any]:
        """
        Get current parameter values.
        - Locked params use locked values
        - Group being optimized: will be sampled by Optuna
        - Other groups: use defaults
        """
        params = self.get_all_defaults()
        params.update(self.locked_params)  # Override with locked
        return params

    def _create_objective(
        self,
        df: pd.DataFrame,
        group_name: str,
        tight_ranges: Dict[str, List[Any]] = None
    ) -> Callable:
        """Create Optuna objective for a specific group or final optimization."""

        if tight_ranges:
            # Final optimization - all params with tight ranges
            param_space = tight_ranges
        else:
            # Stage optimization - just this group's params
            param_space = {
                name: p.values
                for name, p in self.groups[group_name].parameters.items()
            }

        def objective(trial: optuna.Trial) -> float:
            # Start with locked/default params
            params = self.get_current_params()

            # Sample the params we're optimizing
            for name, values in param_space.items():
                params[name] = trial.suggest_categorical(name, values)

            # Run backtest
            try:
                metrics = self.backtest_func(params, df)
            except Exception as e:
                logger.warning(f"Backtest failed: {e}")
                return -1000

            # Store for later
            trial.set_user_attr('params', params.copy())
            trial.set_user_attr('metrics', metrics)

            # Penalize low trade count
            if metrics.trades < self.min_trades:
                return -100 + metrics.trades  # Still prefer more trades

            return metrics.sharpe

        return objective

    def _run_stage(
        self,
        df: pd.DataFrame,
        group_name: str,
        n_trials: int,
        tight_ranges: Dict[str, List[Any]] = None
    ) -> Dict:
        """Run optimization for one stage/group."""

        if tight_ranges:
            stage_name = "FINAL"
            space_desc = f"{len(tight_ranges)} params with tight ranges"
        else:
            stage_name = group_name.upper()
            group = self.groups[group_name]
            space_desc = f"{len(group.parameters)} params, {group.get_space_size():,} combos"

        logger.info(f"\n{'='*60}")
        logger.info(f"STAGE: {stage_name} - {space_desc}")
        logger.info(f"{'='*60}")

        # Create study
        sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
        study = optuna.create_study(direction='maximize', sampler=sampler)

        # Create objective
        objective = self._create_objective(df, group_name, tight_ranges)

        # Progress tracking
        start = time.time()

        def callback(study, trial):
            if (trial.number + 1) % 500 == 0:
                elapsed = time.time() - start
                rate = (trial.number + 1) / elapsed
                best = study.best_value if study.best_trial else 0
                logger.info(f"  {trial.number+1:,}/{n_trials:,} ({rate:.0f}/sec) best_sharpe={best:.2f}")

        # Run optimization
        study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)

        elapsed = time.time() - start

        # Get best result
        best_trial = study.best_trial
        best_params = best_trial.user_attrs['params']
        best_metrics = best_trial.user_attrs['metrics']

        logger.info(f"\nBest {stage_name}: Sharpe={best_metrics.sharpe:.2f}, "
                   f"Trades={best_metrics.trades}, WR={best_metrics.win_rate*100:.1f}%")

        # Get top N for analysis
        valid_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.user_attrs.get('metrics', Metrics(0,0,0,0,0,0)).trades >= self.min_trades
        ]
        valid_trials.sort(key=lambda t: t.value, reverse=True)

        return {
            'stage': stage_name,
            'best_params': best_params,
            'best_metrics': best_metrics,
            'best_sharpe': best_metrics.sharpe,
            'n_trials': n_trials,
            'n_valid': len(valid_trials),
            'time_sec': elapsed,
            'study': study,
            'top_trials': valid_trials[:100],
        }

    def optimize(
        self,
        df_back: pd.DataFrame,
        df_forward: pd.DataFrame,
        trials_per_stage: int = 5000,
        final_trials: int = 10000,
        tight_range_size: int = 5,
        forward_top_n: int = 200,
    ) -> List[Dict]:
        """
        Run full staged optimization.

        Args:
            df_back: Back-testing data
            df_forward: Forward validation data
            trials_per_stage: Optuna trials per group stage
            final_trials: Trials for final tight-range optimization
            tight_range_size: Number of values around locked param for final
            forward_top_n: Top N from final to forward-test

        Returns:
            List of results sorted by forward Sharpe
        """
        total_start = time.time()

        # Print optimization plan
        logger.info("\n" + "="*70)
        logger.info("STAGED OPTIMIZATION PLAN")
        logger.info("="*70)

        total_params = 0
        for i, group_name in enumerate(self.group_order):
            group = self.groups[group_name]
            n_params = len(group.parameters)
            total_params += n_params
            logger.info(f"  Stage {i+1}: {group_name:<20} ({n_params} params, {group.get_space_size():,} combos)")

        logger.info(f"  Final:  All {total_params} params with tight ranges")
        logger.info(f"\nTrials: {trials_per_stage:,}/stage × {len(self.group_order)} stages + {final_trials:,} final")
        logger.info("="*70 + "\n")

        # === STAGE OPTIMIZATION ===
        for group_name in self.group_order:
            result = self._run_stage(df_back, group_name, trials_per_stage)
            self.stage_results[group_name] = result

            # Lock the best params from this group
            group = self.groups[group_name]
            for param_name in group.parameters:
                self.locked_params[param_name] = result['best_params'][param_name]
                logger.info(f"  Locked: {param_name} = {self.locked_params[param_name]}")

        # === BUILD TIGHT RANGES ===
        logger.info("\n" + "="*60)
        logger.info("BUILDING TIGHT RANGES FOR FINAL OPTIMIZATION")
        logger.info("="*60)

        tight_ranges = {}
        for group_name, group in self.groups.items():
            for param_name, param_def in group.parameters.items():
                locked_val = self.locked_params[param_name]
                tight = param_def.get_tight_range(locked_val, tight_range_size)
                tight_ranges[param_name] = tight
                logger.info(f"  {param_name}: {locked_val} → {tight}")

        # Calculate tight space size
        tight_space = 1
        for values in tight_ranges.values():
            tight_space *= len(values)
        logger.info(f"\nTight search space: {tight_space:,} combinations")

        # === FINAL OPTIMIZATION ===
        final_result = self._run_stage(
            df_back,
            self.group_order[0],  # Dummy, not used when tight_ranges provided
            final_trials,
            tight_ranges=tight_ranges
        )

        # === FORWARD VALIDATION ===
        logger.info("\n" + "="*60)
        logger.info(f"FORWARD VALIDATION - Top {forward_top_n} results")
        logger.info("="*60)

        fwd_start = time.time()

        # Get top trials from final stage
        top_trials = final_result['top_trials'][:forward_top_n]

        forward_results = []
        for trial in top_trials:
            params = trial.user_attrs['params']
            back_metrics = trial.user_attrs['metrics']

            # Run forward test
            try:
                fwd_metrics = self.backtest_func(params, df_forward)
            except Exception as e:
                logger.warning(f"Forward test failed: {e}")
                continue

            if fwd_metrics.trades >= 5:
                forward_results.append({
                    'trial_id': trial.number,
                    'params': params,
                    'back': back_metrics,
                    'forward': fwd_metrics,
                })

        fwd_time = time.time() - fwd_start
        logger.info(f"Forward complete: {len(forward_results)} valid in {fwd_time:.1f}s")

        # Sort by forward Sharpe
        forward_results.sort(key=lambda x: x['forward'].sharpe, reverse=True)
        self.final_results = forward_results

        # === SUMMARY ===
        total_time = time.time() - total_start

        logger.info("\n" + "="*70)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Stages completed: {len(self.stage_results)}")
        logger.info(f"Final valid results: {len(forward_results)}")

        if forward_results:
            best = forward_results[0]
            logger.info(f"\nBEST RESULT:")
            logger.info(f"  Back:    Sharpe={best['back'].sharpe:.2f}, Trades={best['back'].trades}, WR={best['back'].win_rate*100:.1f}%")
            logger.info(f"  Forward: Sharpe={best['forward'].sharpe:.2f}, Trades={best['forward'].trades}, WR={best['forward'].win_rate*100:.1f}%")
            logger.info(f"  Params:  {best['params']}")

        return forward_results

    def print_stage_summary(self):
        """Print summary of all stages."""
        print("\n" + "="*80)
        print("STAGE SUMMARY")
        print("="*80)
        print(f"{'Stage':<20} {'Best Sharpe':<15} {'Valid/Trials':<15} {'Time':<10}")
        print("-"*80)

        for name, result in self.stage_results.items():
            print(f"{name:<20} {result['best_sharpe']:<15.2f} "
                  f"{result['n_valid']}/{result['n_trials']:<15} {result['time_sec']:.1f}s")

        print("="*80)

    def print_results(self, top_n: int = 30):
        """Print forward validation results."""
        if not self.final_results:
            print("No results yet. Run optimize() first.")
            return

        print("\n" + "="*110)
        print("BACK vs FORWARD RESULTS")
        print("="*110)
        print(f"{'#':<6} {'Back Sharpe':<13} {'Fwd Sharpe':<13} {'Decay':<10} "
              f"{'Back WR':<10} {'Fwd WR':<10} {'Back Tr':<10} {'Fwd Tr':<8}")
        print("-"*110)

        for r in self.final_results[:top_n]:
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

        print("="*110)

    def save_results(self, output_dir: Path) -> Path:
        """Save all results to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"staged_optimization_{timestamp}.json"

        data = {
            'timestamp': timestamp,
            'stages': {},
            'locked_params': self.locked_params,
            'n_final_results': len(self.final_results),
            'results': []
        }

        # Stage summaries
        for name, result in self.stage_results.items():
            data['stages'][name] = {
                'best_sharpe': result['best_sharpe'],
                'n_valid': result['n_valid'],
                'n_trials': result['n_trials'],
                'time_sec': result['time_sec'],
            }

        # Top results
        for r in self.final_results[:100]:
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


def create_optimizer_from_groups(
    groups_config: Dict[str, Dict[str, Dict]],
    backtest_func: Callable,
    **kwargs
) -> StagedOptimizer:
    """
    Create optimizer from a config dict.

    Example config:
    {
        'signal': {
            'rsi_period': {'values': [5,7,9,14], 'default': 14},
            'swing_strength': {'values': [3,5,7], 'default': 5},
        },
        'risk': {
            'stop_loss': {'values': [20,30,40], 'default': 30},
        }
    }
    """
    optimizer = StagedOptimizer(backtest_func, **kwargs)

    for group_name, params in groups_config.items():
        optimizer.add_group(group_name)
        for param_name, config in params.items():
            optimizer.add_param(
                group_name,
                param_name,
                config['values'],
                config['default'],
                config.get('type', 'categorical')
            )

    return optimizer
