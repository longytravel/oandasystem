"""
Stage 2: Initial Optimization

Runs staged Optuna optimization with MT5-style combined ranking.
Produces top N candidates for further validation.
"""
import gc
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from loguru import logger

from optimization.unified_optimizer import UnifiedOptimizer
from optimization.fast_strategy import FastStrategy
from optimization.numba_backtest import quality_score_from_metrics
from pipeline.config import PipelineConfig
from pipeline.state import PipelineState


def get_strategy(strategy_name: str) -> FastStrategy:
    """Get strategy instance by name."""
    # Import available strategies
    from strategies.rsi_full_v3 import RSIDivergenceFullFastV3
    from strategies.rsi_full_v4 import RSIDivergenceFullFastV4
    from strategies.rsi_full_v5 import RSIDivergenceFullFastV5
    from strategies.ema_cross_ml import EMACrossMLStrategy
    from strategies.fair_price_ma import FairPriceMAStrategy
    from strategies.donchian_breakout import DonchianBreakoutStrategy
    from strategies.bollinger_squeeze import BollingerSqueezeStrategy
    from strategies.london_breakout import LondonBreakoutStrategy
    from strategies.stochastic_adx import StochasticADXStrategy

    strategies = {
        # V3 - RSI Stability-Hardened (multi-RSI consensus, adaptive swings)
        'RSI_Divergence_v3': RSIDivergenceFullFastV3,
        'RSIDivergenceFullFastV3': RSIDivergenceFullFastV3,
        'rsi_v3': RSIDivergenceFullFastV3,
        # V4 - RSI Trade Management Optimization (chain BE->trail, lower TP)
        'RSI_Divergence_v4': RSIDivergenceFullFastV4,
        'RSIDivergenceFullFastV4': RSIDivergenceFullFastV4,
        'rsi_v4': RSIDivergenceFullFastV4,
        # V5 - Chandelier Exit + Stale Exit (ATR-adaptive trail, wider BE, 1R partial)
        'RSI_Divergence_v5': RSIDivergenceFullFastV5,
        'RSIDivergenceFullFastV5': RSIDivergenceFullFastV5,
        'rsi_v5': RSIDivergenceFullFastV5,
        # V6 - EMA Cross + ML Exit (decoupled entry/exit, high trade count)
        'EMA_Cross_ML': EMACrossMLStrategy,
        'EMACrossMLStrategy': EMACrossMLStrategy,
        'ema_cross_ml': EMACrossMLStrategy,
        # Fair Price MA (converted from fairPriceMP v4.0 EA, grid strategy)
        'Fair_Price_MA': FairPriceMAStrategy,
        'FairPriceMA': FairPriceMAStrategy,
        'FairPriceMAStrategy': FairPriceMAStrategy,
        'fair_price_ma': FairPriceMAStrategy,
        # Donchian Channel Breakout (Turtle Trading)
        'Donchian_Breakout': DonchianBreakoutStrategy,
        'donchian_breakout': DonchianBreakoutStrategy,
        'donchian': DonchianBreakoutStrategy,
        # Bollinger Band Squeeze Breakout
        'Bollinger_Squeeze': BollingerSqueezeStrategy,
        'bollinger_squeeze': BollingerSqueezeStrategy,
        'bollinger': BollingerSqueezeStrategy,
        # London Session Breakout
        'London_Breakout': LondonBreakoutStrategy,
        'london_breakout': LondonBreakoutStrategy,
        'london': LondonBreakoutStrategy,
        # Stochastic + ADX
        'Stochastic_ADX': StochasticADXStrategy,
        'stochastic_adx': StochasticADXStrategy,
        'stoch_adx': StochasticADXStrategy,
    }

    if strategy_name not in strategies:
        available = list(strategies.keys())
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available}")

    return strategies[strategy_name]()


class OptimizationStage:
    """Stage 2: Initial optimization with combined ranking."""

    name = "optimization"

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(
        self,
        state: PipelineState,
        df_back: pd.DataFrame,
        df_forward: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Execute optimization stage.

        Args:
            state: Pipeline state
            df_back: Back-test data
            df_forward: Forward-test data

        Returns:
            Dict with keys:
            - candidates: Top N candidates with metrics
            - stage_results: Per-stage optimization results
        """
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 2: INITIAL OPTIMIZATION")
        logger.info("=" * 70)

        # Get strategy
        strategy = get_strategy(self.config.strategy_name)

        # Set pip size for pair
        pip_size = 0.01 if 'JPY' in self.config.pair else 0.0001
        strategy.set_pip_size(self.config.pair)

        logger.info(f"Strategy: {strategy.name}")
        logger.info(f"Pair: {self.config.pair} (pip_size={pip_size})")
        logger.info(f"Mode: {self.config.optimization.mode}")
        logger.info(f"Trials/stage: {self.config.optimization.trials_per_stage}")
        logger.info(f"Final trials: {self.config.optimization.final_trials}")

        # Create optimizer
        optimizer = UnifiedOptimizer(
            strategy=strategy,
            initial_capital=self.config.initial_capital,
            risk_per_trade=self.config.risk_per_trade,
            spread_pips=self.config.spread_pips,
            slippage_pips=self.config.slippage_pips,
            min_trades=self.config.optimization.min_trades,
            min_forward_ratio=self.config.optimization.min_forward_ratio,
            forward_rank_weight=self.config.optimization.forward_rank_weight,
            n_jobs=self.config.optimization.n_jobs,  # Parallel workers
            pair=self.config.pair,  # For pip value calculation
            account_currency='USD',
            timeframe=self.config.timeframe,
            max_dd_hard_limit=self.config.optimization.max_dd_hard_limit,
            min_r2_hard=self.config.optimization.min_r2_hard,
        )

        # Run optimization
        results = optimizer.run(
            df_back=df_back,
            df_forward=df_forward,
            mode=self.config.optimization.mode,
            trials_per_stage=self.config.optimization.trials_per_stage,
            final_trials=self.config.optimization.final_trials,
            min_back_sharpe=self.config.optimization.min_back_sharpe,
            use_optuna=True,
        )

        if not results:
            logger.error("Optimization produced no valid results!")
            return {
                'candidates': [],
                'stage_results': {},
                'summary': {'n_candidates': 0, 'best_sharpe': 0},
            }

        # Print results summary
        optimizer.print_results(top_n=20)
        optimizer.print_stage_summary()

        # Extract top N candidates with diversity
        top_n = self.config.optimization.top_n_candidates
        max_same = getattr(self.config.optimization, 'max_same_signature', 1)
        diverse_results = self._select_diverse_candidates(results, top_n, max_same, strategy)
        candidates = self._extract_candidates(diverse_results, strategy)

        # Store in state
        state.candidates = candidates

        # Build summary
        best = candidates[0] if candidates else {}
        summary = {
            'n_candidates': len(candidates),
            'n_total_results': len(results),
            'best_quality_score': best.get('back_quality_score', 0),
            'best_sharpe': best.get('back_sharpe', 0),
            'best_forward_sharpe': best.get('forward_sharpe', 0),
            'best_combined_rank': best.get('combined_rank', 999),
        }

        # Save stage output
        output_data = {
            'summary': summary,
            'candidates': candidates,
            'stage_results': {
                name: {k: v for k, v in res.items() if k != 'best_params'}
                for name, res in optimizer.stage_results.items()
            },
            'locked_params': optimizer.locked_params,
            'param_importance': optimizer.param_importance,
        }
        state.save_stage_output('optimization', output_data)

        logger.info(f"\nOptimization complete: {len(candidates)} candidates")
        if candidates:
            logger.info(f"Best combined rank: {candidates[0]['combined_rank']}")
            logger.info(f"Best back QS: {candidates[0].get('back_quality_score', 0):.3f}")
            logger.info(f"Best back Sharpe: {candidates[0]['back_sharpe']:.3f}")
            logger.info(f"Best forward Sharpe: {candidates[0]['forward_sharpe']:.3f}")

        # Extract what we need before releasing the optimizer
        stage_results = optimizer.stage_results
        param_importance = optimizer.param_importance

        # Release optimizer's heavy data (DataFrames, arrays, signals)
        optimizer.df_back = None
        optimizer.df_forward = None
        optimizer.back_arrays = {}
        optimizer.fwd_arrays = {}
        optimizer.back_signals = None
        optimizer.fwd_signals = None
        optimizer.strategy = None
        gc.collect()

        return {
            'candidates': candidates,
            'stage_results': stage_results,
            'param_importance': param_importance,
            'summary': summary,
        }

    def _get_signature_params(self, strategy: FastStrategy) -> List[str]:
        """
        Get key parameters that define strategy "signature" for diversity.

        Works for ANY EA by detecting signal-generating params from parameter groups.
        Falls back to all params if groups aren't defined.
        """
        groups = strategy.get_parameter_groups()

        if not groups:
            # No groups defined - can't determine key params, use all
            return []

        # Key params are from 'signal' and 'filters' groups (or first 2 groups)
        # These are the params that fundamentally change what trades are generated
        # NOT 'risk', 'management', 'time' which just change how trades are managed
        key_params = []

        priority_groups = ['signal', 'filters']  # These define the strategy identity

        for group_name in priority_groups:
            if group_name in groups:
                key_params.extend(groups[group_name].parameters.keys())

        # If no standard groups found, use first 2 groups
        if not key_params:
            group_names = list(groups.keys())[:2]
            for name in group_names:
                key_params.extend(groups[name].parameters.keys())

        # Also include sl_mode and tp_mode if they exist (risk structure matters)
        if 'risk' in groups:
            for param in ['sl_mode', 'tp_mode']:
                if param in groups['risk'].parameters and param not in key_params:
                    key_params.append(param)

        return key_params

    def _select_diverse_candidates(
        self,
        results: List[Dict],
        top_n: int,
        max_same_signature: int = 1,
        strategy: FastStrategy = None,
    ) -> List[Dict]:
        """
        Select top N candidates ensuring diversity in key parameters.

        Strategy:
        1. First pass: Select 1 candidate per unique signature (maximum diversity)
        2. Second pass: If slots remain, allow up to max_same_signature per signature
        3. Final pass: Fill any remaining slots with next best by rank

        This ensures we test DIFFERENT strategies before testing variations of the same one.

        Args:
            results: All results sorted by combined rank
            top_n: Number of candidates to select
            max_same_signature: Max candidates per signature after diversity pass (default 1)
            strategy: Strategy instance (to get key params dynamically)

        Returns:
            List of diverse candidates
        """
        if len(results) <= top_n:
            return results

        # Get key parameters dynamically from strategy (works for any EA)
        if strategy:
            key_params = self._get_signature_params(strategy)
        else:
            key_params = []

        if not key_params:
            # Fallback: just take top N if we can't determine key params
            logger.warning("Could not determine key params for diversity - taking top N by rank")
            return results[:top_n]

        logger.info(f"\nSelecting {top_n} diverse candidates from {len(results)} results...")
        logger.info(f"  Signature params: {', '.join(key_params)}")

        def get_signature(params: Dict) -> tuple:
            """Get a signature tuple of key parameter values."""
            return tuple(params.get(k, None) for k in key_params)

        # Track all unique signatures and their best candidate
        signature_best = {}  # sig -> list of results with this sig
        for r in results:
            sig = get_signature(r['params'])
            if sig not in signature_best:
                signature_best[sig] = []
            signature_best[sig].append(r)

        logger.info(f"  Found {len(signature_best)} unique signatures in {len(results)} results")

        selected = []
        signature_counts = {}

        # PASS 1: Take the BEST candidate from each unique signature (max diversity)
        # Sort signatures by their best candidate's rank
        sorted_sigs = sorted(signature_best.keys(),
                            key=lambda s: signature_best[s][0]['combined_rank'])

        for sig in sorted_sigs:
            if len(selected) >= top_n:
                break
            best_for_sig = signature_best[sig][0]  # Already sorted by rank
            selected.append(best_for_sig)
            signature_counts[sig] = 1

        logger.info(f"  Pass 1 (diversity): Selected {len(selected)} candidates (1 per signature)")

        # PASS 2: If we have slots and max_same_signature > 1, add more from each signature
        if len(selected) < top_n and max_same_signature > 1:
            for sig in sorted_sigs:
                if len(selected) >= top_n:
                    break
                # Add more candidates from this signature (up to max)
                candidates_for_sig = signature_best[sig]
                current_count = signature_counts.get(sig, 0)

                for r in candidates_for_sig[current_count:]:  # Skip already selected
                    if len(selected) >= top_n:
                        break
                    if signature_counts.get(sig, 0) >= max_same_signature:
                        break
                    selected.append(r)
                    signature_counts[sig] = signature_counts.get(sig, 0) + 1

            logger.info(f"  Pass 2 (fill): Selected {len(selected)} total")

        # PASS 3: If still need more, take next best by rank regardless of signature
        if len(selected) < top_n:
            selected_ids = {id(r) for r in selected}
            for r in results:
                if len(selected) >= top_n:
                    break
                if id(r) not in selected_ids:
                    selected.append(r)
                    sig = get_signature(r['params'])
                    signature_counts[sig] = signature_counts.get(sig, 0) + 1

        # Log final diversity stats
        unique_in_selected = len(set(get_signature(r['params']) for r in selected))
        logger.info(f"  Final: {len(selected)} candidates, {unique_in_selected} unique signatures")

        if unique_in_selected < min(5, top_n // 2) and len(signature_best) > unique_in_selected:
            logger.warning(f"  Low diversity despite {len(signature_best)} available signatures")
            logger.warning(f"  Consider using mode='fullspace' for better exploration")

        # Show signature distribution (top 5)
        sig_summary = sorted(signature_counts.items(), key=lambda x: -x[1])[:5]
        for sig, count in sig_summary:
            sig_str = ", ".join(f"{k}={v}" for k, v in zip(key_params, sig) if v is not None)
            logger.info(f"    {count}x: {sig_str[:60]}{'...' if len(sig_str) > 60 else ''}")

        return selected

    def _extract_candidates(
        self,
        results: List[Dict],
        strategy: FastStrategy
    ) -> List[Dict[str, Any]]:
        """Extract candidate info for pipeline."""
        candidates = []

        for i, r in enumerate(results):
            back = r['back']
            fwd = r['forward']
            back_qs = quality_score_from_metrics(back)
            fwd_qs = quality_score_from_metrics(fwd)

            candidate = {
                'rank': i + 1,
                'trial_id': r['trial_id'],
                'combined_rank': r['combined_rank'],
                'back_rank': r['back_rank'],
                'forward_rank': r['forward_rank'],
                'params': r['params'],

                # Back metrics
                'back_trades': back.trades,
                'back_quality_score': back_qs,
                'back_ontester': back.ontester_score,
                'back_r_squared': back.r_squared,
                'back_sharpe': back.sharpe,
                'back_sortino': back.sortino,
                'back_ulcer': back.ulcer,
                'back_win_rate': back.win_rate,
                'back_profit_factor': back.profit_factor,
                'back_max_dd': back.max_dd,
                'back_return': back.total_return,

                # Forward metrics
                'forward_trades': fwd.trades,
                'forward_quality_score': fwd_qs,
                'forward_ontester': fwd.ontester_score,
                'forward_r_squared': fwd.r_squared,
                'forward_sharpe': fwd.sharpe,
                'forward_sortino': fwd.sortino,
                'forward_ulcer': fwd.ulcer,
                'forward_win_rate': fwd.win_rate,
                'forward_profit_factor': fwd.profit_factor,
                'forward_max_dd': fwd.max_dd,
                'forward_return': fwd.total_return,

                # Derived - quality_score ratio (universal, not inflated by compounding)
                'forward_back_ratio': fwd_qs / back_qs if back_qs > 0 else 0,
            }

            candidates.append(candidate)

        return candidates
