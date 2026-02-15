"""Collects and enriches data from all pipeline stages for the report."""
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

from loguru import logger

from pipeline.config import PipelineConfig
from pipeline.state import PipelineState


def _get_stage_summary(result: Dict[str, Any], state: PipelineState, stage_name: str) -> Dict[str, Any]:
    """Get stage summary, falling back to state.json when result is empty (pipeline resume)."""
    summary = result.get('summary', {})
    if not summary and state:
        stage_state = state.stages.get(stage_name)
        if stage_state:
            summary = stage_state.summary or {}
    return summary


def collect_report_data(
    state: PipelineState,
    config: PipelineConfig,
    candidates: List[Dict[str, Any]],
    data_result: Dict[str, Any],
    optimization_result: Dict[str, Any],
    walkforward_result: Dict[str, Any],
    stability_result: Dict[str, Any],
    montecarlo_result: Dict[str, Any],
    confidence_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build all data needed for the visual report.

    Enriches raw stage outputs with derived metrics (monthly returns,
    drawdown curve, recovery factor, etc.).
    """
    best = state.best_candidate or (candidates[0] if candidates else {})
    trade_details = best.get('trade_details', {})

    # Regenerate trade_details if missing.
    # MC puts trade_details on the best-combined-rank candidate, but the
    # confidence-best candidate (state.best_candidate) may be different.
    # Always regenerate for the actual best candidate's params.
    if not trade_details and best.get('params') and data_result.get('df_back') is not None:
        trade_details = _regenerate_trade_details(best, data_result, config)
        best['trade_details'] = trade_details

    # Build combined equity curve from trade details
    back_equity = trade_details.get('back_equity', [])
    forward_equity = trade_details.get('forward_equity', [])
    trade_summary = trade_details.get('summary', {})

    # Compute drawdown curve from combined equity
    all_equity = back_equity + forward_equity
    dd_curve = _compute_drawdown_curve(all_equity)

    # Compute monthly returns
    all_trades = trade_details.get('back_trades', []) + trade_details.get('forward_trades', [])
    monthly_returns = _compute_monthly_returns(all_trades)

    # Recovery factor
    total_profit = trade_summary.get('total_net_profit', 0)
    max_dd_abs = _compute_max_dd_absolute(all_equity, config.initial_capital)
    recovery_factor = total_profit / max_dd_abs if max_dd_abs > 0 else 0

    # MC raw data
    mc_data = best.get('montecarlo', {})
    raw_returns = mc_data.get('raw_returns', [])
    raw_max_dds = mc_data.get('raw_max_dds', [])

    # ML Exit data
    ml_exit_data = _collect_ml_exit_data(best, config)

    return {
        'meta': {
            'run_id': state.run_id,
            'pair': state.pair,
            'timeframe': state.timeframe,
            'strategy': state.strategy_name,
            'description': getattr(state, 'description', ''),
            'generated_at': datetime.now().isoformat(),
        },
        'decision': {
            'score': state.final_score,
            'rating': state.final_rating,
            'recommendation': best.get('confidence', {}).get('recommendation', 'Unknown'),
        },
        'best_candidate': best,
        'candidates': candidates[:config.report.leaderboard_top_n],
        'data_summary': _get_stage_summary(data_result, state, 'data'),
        'optimization_summary': _get_stage_summary(optimization_result, state, 'optimization'),
        'walkforward_summary': _get_stage_summary(walkforward_result, state, 'walkforward'),
        'walkforward_windows': walkforward_result.get('windows', []),
        'stability_summary': _get_stage_summary(stability_result, state, 'stability'),
        'montecarlo_summary': _get_stage_summary(montecarlo_result, state, 'montecarlo'),
        'confidence_summary': _get_stage_summary(confidence_result, state, 'confidence'),

        # Enriched data for charts
        'trade_details': trade_details,
        'trade_summary': trade_summary,
        'back_equity': back_equity,
        'forward_equity': forward_equity,
        'drawdown_curve': dd_curve,
        'monthly_returns': monthly_returns,
        'recovery_factor': recovery_factor,
        'max_dd_absolute': max_dd_abs,
        'mc_raw_returns': raw_returns,
        'mc_raw_max_dds': raw_max_dds,
        'config': config.to_dict(),
        'ml_exit': ml_exit_data,
    }


def _compute_drawdown_curve(equity_pts: List[Dict]) -> List[Dict]:
    """Compute drawdown percentage at each equity point."""
    if not equity_pts:
        return []

    dd_curve = []
    peak = equity_pts[0]['equity'] if equity_pts else 10000

    for pt in equity_pts:
        eq = pt['equity']
        if eq > peak:
            peak = eq
        dd_pct = (peak - eq) / peak * 100 if peak > 0 else 0
        dd_curve.append({
            'trade_num': pt.get('trade_num', 0),
            'timestamp': pt.get('timestamp', ''),
            'drawdown': dd_pct,
        })

    return dd_curve


def _compute_max_dd_absolute(equity_pts: List[Dict], initial_capital: float) -> float:
    """Compute max drawdown in absolute currency terms."""
    if not equity_pts:
        return 0

    peak = initial_capital
    max_dd = 0

    for pt in equity_pts:
        eq = pt['equity']
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    return max_dd


def _compute_monthly_returns(trades: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Group trades by year-month and compute returns.

    Returns: { "2024": { "Jan": 1.5, "Feb": -0.3, ... }, ... }
    """
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    monthly: Dict[str, Dict[str, float]] = {}

    for trade in trades:
        ts = trade.get('entry_time', '')
        if not ts or len(ts) < 7:
            continue
        try:
            year = ts[:4]
            month_num = int(ts[5:7])
            month_name = month_names[month_num - 1]
        except (ValueError, IndexError):
            continue

        if year not in monthly:
            monthly[year] = {}
        monthly[year][month_name] = monthly[year].get(month_name, 0) + trade['pnl']

    return monthly


def _collect_ml_exit_data(best: Dict, config: PipelineConfig) -> Dict[str, Any]:
    """Collect ML exit model data from the best candidate."""
    ml_data = {
        'enabled': getattr(config, 'ml_exit', None) is not None and config.ml_exit.enabled,
        'feature_importances': {},
        'training_metrics': {},
        'window_ml_metrics': [],
    }

    if not ml_data['enabled']:
        return ml_data

    # Extract from walkforward window results
    wf = best.get('walkforward', {})
    for wr in wf.get('window_results', []):
        ml_metrics = wr.get('ml_exit', {})
        if ml_metrics:
            ml_data['window_ml_metrics'].append({
                'window': wr.get('window', 0),
                **ml_metrics,
            })

    # Feature importances and training metrics from the last trained model
    if ml_data['window_ml_metrics']:
        last = ml_data['window_ml_metrics'][-1]
        ml_data['feature_importances'] = last.get('top_features', {})
        ml_data['training_metrics'] = last.get('training_metrics', {})

    return ml_data


def _regenerate_trade_details(
    best: Dict[str, Any],
    data_result: Dict[str, Any],
    config: PipelineConfig,
) -> Dict[str, Any]:
    """Regenerate trade_details by running backtest on best candidate.

    This is needed when resuming from a later stage (confidence/report)
    because trade_details are too large to serialize in state.json.
    """
    from pipeline.stages.s5_montecarlo import MonteCarloStage

    logger.info("Regenerating trade details for report (not in cache)...")

    mc_stage = MonteCarloStage(config)
    df_back = data_result['df_back']
    df_forward = data_result['df_forward']

    from pipeline.stages.s2_optimization import get_strategy
    strategy = get_strategy(config.strategy_name)

    pair = config.pair
    if 'JPY' in pair:
        pip_size = 0.01
    else:
        pip_size = 0.0001

    trade_details = mc_stage._collect_trade_details(
        best['params'], df_back, df_forward, strategy, pip_size,
    )

    logger.info(f"  Regenerated: {len(trade_details.get('back_trades', []))} back trades, "
                f"{len(trade_details.get('forward_trades', []))} forward trades")

    return trade_details
