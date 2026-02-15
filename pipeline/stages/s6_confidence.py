"""
Stage 6: Confidence Scoring

Combines all validation evidence into a single 0-100 score:
- Backtest Quality (15%) - absolute backtest performance (PF, Sharpe, R², trades, DD)
- Forward/Back Ratio (15%) - penalizes suspicious extreme ratios
- Walk-Forward Consistency (25%)
- Parameter Stability (15%)
- Monte Carlo Risk (15%)
- Quality Score (15%) - universal metric: smooth equity + profit + low drawdown

Score thresholds:
- RED (<40): DO NOT TRADE
- YELLOW (40-70): Re-optimize or proceed with caution
- GREEN (70-85): Paper trade first
- GREEN (85-100): Ready for live (small size)
"""
import numpy as np
from typing import Dict, Any, List, Optional

from loguru import logger

from pipeline.config import PipelineConfig
from pipeline.state import PipelineState


class ConfidenceStage:
    """Stage 6: Confidence scoring system."""

    name = "confidence"

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(
        self,
        state: PipelineState,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate confidence scores for each candidate.

        Args:
            state: Pipeline state
            candidates: Candidates from Monte Carlo stage

        Returns:
            Dict with:
            - candidates: Updated candidates with confidence scores
            - best_candidate: Top scoring candidate
            - summary: Overall scoring statistics
        """
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 6: CONFIDENCE SCORING")
        logger.info("=" * 70)

        if not candidates:
            logger.warning("No candidates to score!")
            # Fix Finding 12: persist stage output on early return
            empty_result = {'n_scored': 0}
            state.save_stage_output('confidence', {'summary': empty_result, 'candidates': [], 'best_candidate': None})
            return {
                'candidates': [],
                'best_candidate': None,
                'summary': empty_result,
            }

        # Score each candidate
        for i, candidate in enumerate(candidates):
            score_breakdown = self._calculate_score(candidate)
            candidate['confidence'] = score_breakdown

            logger.info(f"\n--- Candidate {candidate['rank']} ---")
            logger.info(f"  BT Quality:    {score_breakdown['backtest_quality_score']:.1f}/100 "
                       f"(w={self.config.confidence.backtest_quality_weight:.0%})")
            logger.info(f"  Forward/Back:  {score_breakdown['forward_back_score']:.1f}/100 "
                       f"(w={self.config.confidence.forward_back_weight:.0%})")
            logger.info(f"  Walk-Forward:  {score_breakdown['walkforward_score']:.1f}/100 "
                       f"(w={self.config.confidence.walkforward_weight:.0%})")
            logger.info(f"  Stability:     {score_breakdown['stability_score']:.1f}/100 "
                       f"(w={self.config.confidence.stability_weight:.0%})")
            logger.info(f"  Monte Carlo:   {score_breakdown['montecarlo_score']:.1f}/100 "
                       f"(w={self.config.confidence.montecarlo_weight:.0%})")
            logger.info(f"  Sharpe:        {score_breakdown['sharpe_score']:.1f}/100 "
                       f"(w={self.config.confidence.sharpe_weight:.0%})")
            logger.info(f"  -----------------------------------------")
            logger.info(f"  TOTAL SCORE:   {score_breakdown['total_score']:.1f}/100")
            logger.info(f"  RATING:        {score_breakdown['rating']}")

        # Sort by confidence score
        candidates.sort(key=lambda x: x['confidence']['total_score'], reverse=True)

        # Get best candidate
        best = candidates[0] if candidates else None
        if best:
            state.best_candidate = best
            state.final_score = best['confidence']['total_score']
            state.final_rating = best['confidence']['rating']

        # Update state candidates
        state.candidates = candidates

        # Build summary
        summary = {
            'n_scored': len(candidates),
            'best_score': best['confidence']['total_score'] if best else 0,
            'best_rating': best['confidence']['rating'] if best else 'RED',
            'avg_score': np.mean([c['confidence']['total_score'] for c in candidates]) if candidates else 0,
        }

        # Count by rating
        ratings = [c['confidence']['rating'] for c in candidates]
        summary['n_green'] = sum(1 for r in ratings if r == 'GREEN')
        summary['n_yellow'] = sum(1 for r in ratings if r == 'YELLOW')
        summary['n_red'] = sum(1 for r in ratings if r == 'RED')

        # Save stage output
        output_data = {
            'summary': summary,
            'best_candidate': {
                'rank': best['rank'],
                'params': best['params'],
                'confidence': best['confidence'],
            } if best else None,
            'candidates': [
                {
                    'rank': c['rank'],
                    'combined_rank': c['combined_rank'],
                    'confidence': c['confidence'],
                }
                for c in candidates
            ],
        }
        state.save_stage_output('confidence', output_data)

        logger.info(f"\nConfidence Scoring Complete:")
        logger.info(f"  Candidates scored: {len(candidates)}")
        logger.info(f"  GREEN:  {summary['n_green']}")
        logger.info(f"  YELLOW: {summary['n_yellow']}")
        logger.info(f"  RED:    {summary['n_red']}")
        if best:
            logger.info(f"\n  Best candidate: Rank {best['rank']}")
            logger.info(f"  Best score: {best['confidence']['total_score']:.1f}/100")
            logger.info(f"  Rating: {best['confidence']['rating']}")

        return {
            'candidates': candidates,
            'best_candidate': best,
            'summary': summary,
        }

    def _backtest_quality_score(self, candidate: Dict[str, Any]) -> float:
        """
        Evaluate absolute backtest quality (0-100).

        Uses 5 equally-weighted sub-components (20% each):
        - Profit Factor: 1.0->0, 2.0->50, 3.0+->100
        - Sortino: 0->0, 1.5->50, 3.0+->100 (Sortino replaces Sharpe — better for asymmetric)
        - R² (equity smoothness): 0->0, 0.5->50, 0.9+->100
        - Trade count (statistical significance): <30->0, 100+->100
        - Blended DD (lower=better): uses Ulcer + MaxDD/2, penalizes chronic + peak drawdown
        """
        back_stats = candidate.get('back_stats', {})

        # Profit Factor: 1.0->0, 2.0->50, 3.0+->100
        profit_factor = back_stats.get('profit_factor', candidate.get('back_profit_factor', 1.0))
        pf_sub = max(0, min(100, (profit_factor - 1.0) * 50))

        # Sortino: 0->0, 3.0+->100 (replaces Sharpe for asymmetric strategy fairness)
        back_sortino = back_stats.get('sortino', candidate.get('back_sortino', 0))
        sortino_sub = max(0, min(100, back_sortino / 3.0 * 100))

        # R² (equity smoothness): 0->0, 0.9+->100
        r_squared = back_stats.get('r_squared', candidate.get('back_r_squared', 0))
        r2_sub = max(0, min(100, r_squared / 0.9 * 100))

        # Trade count: <30->0, 100+->100
        trades = back_stats.get('trades', candidate.get('back_trades', 0))
        trade_sub = max(0, min(100, (trades - 30) / 70 * 100))

        # Blended DD (lower=better): Ulcer + MaxDD/2, scale 0->100, 30+->0
        max_dd = back_stats.get('max_dd', candidate.get('back_max_dd', 40))
        ulcer = back_stats.get('ulcer', candidate.get('back_ulcer', 0))
        blended_dd = ulcer + max_dd / 2.0
        dd_sub = max(0, min(100, (30 - blended_dd) / 30 * 100))

        return (pf_sub + sortino_sub + r2_sub + trade_sub + dd_sub) / 5

    def _calculate_score(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence score for a candidate."""
        cfg = self.config.confidence

        # 1. Backtest Quality (15%) - absolute backtest performance
        bt_quality_score = self._backtest_quality_score(candidate)

        # 2. Forward/Back Ratio (15%) - penalizes suspicious extreme ratios
        fb_ratio = candidate.get('forward_back_ratio', 0)
        if fb_ratio <= 0:
            fb_score = 0.0
        elif fb_ratio <= 1.0:
            fb_score = fb_ratio * 100        # linear 0->100
        elif fb_ratio <= 2.0:
            fb_score = 100.0                  # 1.0-2.0 is fine, stays 100
        else:
            # Above 2.0: suspicious over-performance, diminishing returns
            fb_score = max(0, 100 - (fb_ratio - 2.0) * 20)

        # 3. Walk-Forward Consistency (25%)
        wf = candidate.get('walkforward', {})
        wf_stats = wf.get('stats', {})
        # Prefer out-of-sample pass rate when available (more honest metric)
        oos_n = wf_stats.get('oos_n_windows', 0)
        if oos_n > 0:
            wf_pass_rate = wf_stats.get('oos_pass_rate', 0)
        else:
            wf_pass_rate = wf_stats.get('pass_rate', 0)
        wf_score = wf_pass_rate * 100

        # Sharpe quality factor: larger downside range to differentiate when pass_rate=1.0
        # A strategy where all windows pass but Sharpe is mediocre should score lower
        wf_mean_sharpe = (wf_stats.get('oos_mean_sharpe', wf_stats.get('mean_sharpe', 0))
                          if oos_n > 0 else wf_stats.get('mean_sharpe', 0))
        sharpe_adj = min(10, max(-40, (wf_mean_sharpe - 1.0) * 30))
        wf_score = max(0, min(100, wf_score + sharpe_adj))

        # When only 1 OOS window, blend with all-windows pass rate as supplementary evidence
        if oos_n == 1:
            all_pass_rate = wf_stats.get('pass_rate', 0)
            wf_score = wf_score * 0.7 + (all_pass_rate * 100) * 0.3

        # Bonus for high consistency
        if wf_stats.get('consistency', 0) > 2:
            wf_score = min(wf_score + 10, 100)

        # 4. Parameter Stability (15%)
        stability = candidate.get('stability', {})
        stability_overall = stability.get('overall', {})
        mean_stability = stability_overall.get('mean_stability', 0)
        stability_score = min(100, mean_stability * 100)

        # Penalty for fragile parameters
        n_unstable = stability_overall.get('n_unstable_params', 0)
        if n_unstable > 3:
            stability_score *= 0.8

        # 5. Monte Carlo Risk Assessment (15%)
        # Fix Finding 1: shuffle-based MC returns are degenerate (same sum regardless of order).
        # Use bootstrap Sharpe CI and DD distribution instead of the meaningless pct_5_return.
        mc = candidate.get('montecarlo', {})
        mc_results = mc.get('results', {})
        bootstrap = mc_results.get('bootstrap', {})

        mc_score = 50.0  # Neutral default when MC data unavailable
        if bootstrap:
            # Bootstrap Sharpe lower CI: <0 -> 0, 0 -> 25, 1.0 -> 75, 2.0+ -> 100
            sharpe_ci_lower = bootstrap.get('sharpe_ci_lower', 0)
            mc_score = max(0, min(100, sharpe_ci_lower / 2.0 * 100))

            # Bonus if bootstrap profit factor CI lower bound > 1.0
            pf_ci_lower = bootstrap.get('pf_ci_lower', 0)
            if pf_ci_lower > 1.0:
                mc_score = min(mc_score + 10, 100)
        else:
            # Fallback: use drawdown distribution (which IS valid from shuffle MC)
            pct_95_dd = mc_results.get('pct_95_dd', 50)
            # Scale: 50% DD -> 0, 20% DD -> 60, 5% DD -> 90, 0% DD -> 100
            mc_score = max(0, min(100, (50 - pct_95_dd) * 2))

        # Penalty if permutation test shows strategy is not statistically significant
        perm = mc_results.get('permutation', {})
        if perm:
            if not perm.get('significant_at_05', True):
                # Strategy doesn't beat random entry at 5% significance - heavy penalty
                mc_score *= 0.5

        # 6. Quality Score (15%) - universal metric: smooth equity + profit + low DD
        # Priority: OOS mean quality_score > all-windows > optimization forward
        wf_stats_for_qs = wf.get('stats', {})
        oos_qs = (wf_stats_for_qs.get('oos_mean_quality_score', 0)
                  if wf_stats_for_qs.get('oos_n_windows', 0) > 0 else 0)
        wf_all_qs = wf_stats_for_qs.get('mean_quality_score', 0)
        if oos_qs > 0:
            fwd_quality = oos_qs
        elif wf_all_qs > 0:
            fwd_quality = wf_all_qs
        else:
            fwd_quality = candidate.get('forward_quality_score', 0)
        # Scale: 0 -> 0, 1.5 -> 50, 3.0+ -> 100
        sharpe_score = min(fwd_quality / 3.0, 1.0) * 100

        # Hard backtest quality gate: if full backtest shows terrible equity curve
        # or extreme drawdown, cap score at YELLOW regardless of other components.
        # This catches candidates that optimize well per-window but have random overall equity.
        back_r2 = candidate.get('back_r_squared', 0)
        back_dd = candidate.get('back_max_dd', 0)
        back_qs = candidate.get('back_quality_score', 0)
        bt_hard_cap = None
        if back_r2 < 0.3 and back_r2 > 0:
            bt_hard_cap = 50  # R² < 0.3 = random walk equity, YELLOW at best
            logger.warning(f"  Hard gate: back R²={back_r2:.3f} < 0.3 → capped at {bt_hard_cap}")
        if back_dd > 50:
            bt_hard_cap = min(bt_hard_cap or 50, 40)  # MaxDD > 50% = near-ruin, RED
            logger.warning(f"  Hard gate: back MaxDD={back_dd:.1f}% > 50% → capped at {bt_hard_cap}")
        if back_qs > 0 and back_qs < 0.5:
            bt_hard_cap = min(bt_hard_cap or 50, 50)  # Quality score near zero
            logger.warning(f"  Hard gate: back quality_score={back_qs:.3f} < 0.5 → capped at {bt_hard_cap}")

        # Calculate weighted total (capped at 100)
        total_score = min(100, (
            bt_quality_score * cfg.backtest_quality_weight +
            fb_score * cfg.forward_back_weight +
            wf_score * cfg.walkforward_weight +
            stability_score * cfg.stability_weight +
            mc_score * cfg.montecarlo_weight +
            sharpe_score * cfg.sharpe_weight
        ))

        # Apply hard backtest cap
        if bt_hard_cap is not None:
            total_score = min(total_score, bt_hard_cap)

        # Determine rating
        if total_score < cfg.red_threshold:
            rating = 'RED'
            recommendation = 'DO NOT TRADE'
        elif total_score < cfg.yellow_threshold:
            rating = 'YELLOW'
            recommendation = 'Proceed with caution, consider re-optimization'
        elif total_score < 85:
            rating = 'GREEN'
            recommendation = 'Paper trade first'
        else:
            rating = 'GREEN'
            recommendation = 'Ready for live trading (small size)'

        return {
            'total_score': round(total_score, 1),
            'rating': rating,
            'recommendation': recommendation,

            # Component scores
            'backtest_quality_score': round(bt_quality_score, 1),
            'forward_back_score': round(fb_score, 1),
            'walkforward_score': round(wf_score, 1),
            'stability_score': round(stability_score, 1),
            'montecarlo_score': round(mc_score, 1),
            'sharpe_score': round(sharpe_score, 1),

            # Raw values
            'raw_values': {
                'forward_back_ratio': round(fb_ratio, 3),
                'wf_pass_rate': round(wf_pass_rate, 3),
                'wf_oos_windows': oos_n,
                'wf_mean_sharpe': round(wf_mean_sharpe, 3),
                'wf_all_pass_rate': round(wf_stats.get('pass_rate', 0), 3),
                'mean_stability': round(mean_stability, 3),
                'mc_bootstrap_sharpe_ci_lower': round(bootstrap.get('sharpe_ci_lower', 0), 2) if bootstrap else None,
                'mc_pct_95_dd': round(mc_results.get('pct_95_dd', 0), 1),
                'forward_quality_score': round(fwd_quality, 3),
                'permutation_p_value': round(perm.get('p_value', 1.0), 4) if perm else None,
                'permutation_significant': perm.get('significant_at_05', None) if perm else None,
                'bt_hard_cap': bt_hard_cap,
                'back_r_squared': round(back_r2, 4),
                'back_max_dd': round(back_dd, 1),
                'back_quality_score': round(back_qs, 3),
            },

            # Weights used
            'weights': {
                'backtest_quality': cfg.backtest_quality_weight,
                'forward_back': cfg.forward_back_weight,
                'walkforward': cfg.walkforward_weight,
                'stability': cfg.stability_weight,
                'montecarlo': cfg.montecarlo_weight,
                'sharpe': cfg.sharpe_weight,
            },
        }
