"""
Stage 4: Parameter Stability Analysis

Tests neighboring parameter values to ensure robustness:
- For each parameter, test ±1 step
- Calculate stability ratio (neighbor_avg / baseline)
- Flag fragile parameters
- Rate overall robustness
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from loguru import logger

from optimization.unified_optimizer import UnifiedOptimizer
from pipeline.config import PipelineConfig
from pipeline.state import PipelineState
from pipeline.stages.s2_optimization import get_strategy


class StabilityStage:
    """Stage 4: Parameter stability analysis."""

    name = "stability"

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
        Execute stability analysis for each candidate.

        Tests each parameter's neighbors to detect fragile parameters.

        Args:
            state: Pipeline state
            df_back: Back-test data
            df_forward: Forward-test data
            candidates: Candidates from walk-forward stage

        Returns:
            Dict with:
            - candidates: Updated candidates with stability results
            - summary: Overall stability statistics
        """
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 4: PARAMETER STABILITY ANALYSIS")
        logger.info("=" * 70)

        if not candidates:
            logger.warning("No candidates to analyze!")
            # Fix Finding 12: persist stage output on early return
            empty_result = {'n_analyzed': 0}
            state.save_stage_output('stability', {'summary': empty_result, 'candidates': []})
            return {
                'candidates': [],
                'summary': empty_result,
            }

        # Get strategy
        strategy = get_strategy(self.config.strategy_name)
        strategy.set_pip_size(self.config.pair)
        pip_size = 0.01 if 'JPY' in self.config.pair else 0.0001

        # Create optimizer for stability testing
        optimizer = UnifiedOptimizer(
            strategy=strategy,
            initial_capital=self.config.initial_capital,
            risk_per_trade=self.config.risk_per_trade,
            spread_pips=self.config.spread_pips + self.config.slippage_pips,
            min_trades=self.config.optimization.min_trades,
        )

        # Prepare data
        optimizer._prepare_data(df_back, df_forward)
        optimizer._precompute_signals()

        # Analyze each candidate
        # Stability is ADVISORY, not a gate. All candidates pass through to MC/Confidence.
        n_passed = 0

        for i, candidate in enumerate(candidates):
            logger.info(f"\n--- Candidate {candidate['rank']} ---")

            stability = optimizer.analyze_parameter_stability(
                params=candidate['params'],
                use_full=True,
                test_forward=self.config.stability.test_forward,
            )

            # Add stability to candidate
            candidate['stability'] = {
                'overall': stability['overall'],
                'params': stability['params'],
                'baseline_back': stability['baseline_back'],
                'baseline_forward': stability['baseline_forward'],
            }

            # Check stability threshold (advisory only - does NOT filter candidates)
            mean_stability = stability['overall']['mean_stability']
            min_stability = stability['overall']['min_stability']
            rating = stability['overall']['rating']

            passed = (
                mean_stability >= self.config.stability.min_stability and
                min_stability >= self.config.stability.min_single_stability
            )

            candidate['stability']['passed'] = passed

            if passed:
                n_passed += 1
                logger.info(f"  PASSED: mean={mean_stability:.1%}, min={min_stability:.1%}, rating={rating}")
            else:
                logger.info(f"  ADVISORY WARNING: mean={mean_stability:.1%}, min={min_stability:.1%}, rating={rating}")

            # Log fragile parameters
            fragile_params = self._get_fragile_params(stability['params'])
            if fragile_params:
                logger.info(f"  Fragile params: {', '.join(fragile_params)}")

        # All candidates pass through - stability is advisory, not a gate
        state.candidates = candidates

        # Build summary
        summary = {
            'n_analyzed': len(candidates),
            'n_passed': n_passed,
            'pass_rate': n_passed / len(candidates) if candidates else 0,
            'advisory_only': True,
        }

        if candidates:
            summary['best_mean_stability'] = max(
                c['stability']['overall']['mean_stability'] for c in candidates
            )
            summary['avg_mean_stability'] = np.mean([
                c['stability']['overall']['mean_stability'] for c in candidates
            ])

        # Save stage output
        output_data = {
            'summary': summary,
            'candidates': [
                {
                    'rank': c['rank'],
                    'combined_rank': c['combined_rank'],
                    'stability': c['stability'],
                }
                for c in candidates
            ],
        }
        state.save_stage_output('stability', output_data)

        logger.info(f"\nStability Analysis Complete (ADVISORY):")
        logger.info(f"  Candidates analyzed: {len(candidates)}")
        logger.info(f"  Stability passed:    {n_passed}")
        logger.info(f"  Pass rate:          {summary['pass_rate']:.0%}")
        logger.info(f"  All {len(candidates)} candidates forwarded to MC/Confidence stages")

        return {
            'candidates': candidates,  # All candidates pass through
            'summary': summary,
        }

    def _get_fragile_params(
        self,
        param_stability: Dict[str, Dict],
        threshold: float = 0.5
    ) -> List[str]:
        """Get list of fragile parameters below threshold."""
        fragile = []
        for param, result in param_stability.items():
            if result.get('stability_ratio', 1.0) < threshold:
                fragile.append(param)
        return fragile
