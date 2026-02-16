"""Build JSON data for the web leaderboard.

Generates web/data/leaderboard.json from pipeline run data, reusing the
same collect_runs() and collect_compare_data() functions from generate_index.py.

Includes enriched detail data (MC, confidence, WF, stability, trade summary,
drawdown curve, monthly returns) for each run with report_data.json.

Usage:
    python scripts/build_web.py
    python scripts/build_web.py --pipelines-dir results/pipelines
"""
import json
import os
import argparse
from datetime import datetime
from pathlib import Path

from generate_index import collect_runs, collect_compare_data


def _make_serializable(obj):
    """Convert any non-JSON-serializable objects (datetimes, etc.) to strings."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    return obj


def _safe_get(d, *keys, default=None):
    """Safely traverse nested dicts."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key)
        if d is None:
            return default
    return d


def collect_detail_data(pipelines_dir: str, runs: list) -> dict:
    """Load rich detail data from report_data.json for the web detail panel.

    Returns a dict keyed by run_id with comprehensive data for each run:
    - MC results (summary stats, bootstrap CIs, permutation test)
    - Confidence breakdown (all 6 component scores + weights)
    - WF window results (per-window stats)
    - Stability overall (rating, stable/unstable counts)
    - Trade summary (exit reasons, consecutive W/L, MFE/MAE)
    - Drawdown curve
    - Monthly returns
    - Extended metrics (sortino, ulcer, quality_score, forward versions)
    """
    details = {}
    for r in runs:
        if not r.get('has_report_data'):
            continue
        path = os.path.join(pipelines_dir, r['run_id'], 'report_data.json')
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        best = data.get('best_candidate', {}) or {}
        mc = best.get('montecarlo', {}) or {}
        conf = best.get('confidence', {}) or {}
        stab = best.get('stability', {}) or {}
        wf = best.get('walkforward', {}) or {}
        ts = data.get('trade_summary') or data.get('trade_details', {}).get('summary', {}) or {}

        # MC results - summary stats only (no raw arrays)
        mc_results = mc.get('results', {}) or {}
        mc_bootstrap = mc_results.get('bootstrap', {}) or {}
        mc_permutation = mc_results.get('permutation', {}) or {}
        mc_detail = {
            'status': mc.get('status', 'unknown'),
            'n_trades': mc.get('n_trades', 0),
            'n_iterations': mc.get('n_iterations', 0),
            'original_return': mc_results.get('original_return'),
            'original_max_dd': mc_results.get('original_max_dd'),
            'mean_return': mc_results.get('mean_return'),
            'std_return': mc_results.get('std_return'),
            'pct_5_return': mc_results.get('pct_5_return'),
            'pct_25_return': mc_results.get('pct_25_return'),
            'pct_50_return': mc_results.get('pct_50_return'),
            'pct_75_return': mc_results.get('pct_75_return'),
            'pct_95_return': mc_results.get('pct_95_return'),
            'mean_dd': mc_results.get('mean_dd'),
            'pct_5_dd': mc_results.get('pct_5_dd'),
            'pct_50_dd': mc_results.get('pct_50_dd'),
            'pct_95_dd': mc_results.get('pct_95_dd'),
            'max_dd': mc_results.get('max_dd'),
            'var_95': mc_results.get('var_95'),
            'expected_shortfall': mc_results.get('expected_shortfall'),
            'prob_positive': mc_results.get('prob_positive'),
            'prob_above_5pct': mc_results.get('prob_above_5pct'),
            'prob_above_10pct': mc_results.get('prob_above_10pct'),
            'bootstrap_sharpe_mean': _safe_get(mc_bootstrap, 'sharpe', 'mean'),
            'bootstrap_sharpe_ci_lower': _safe_get(mc_bootstrap, 'sharpe', 'ci_lower'),
            'bootstrap_sharpe_ci_upper': _safe_get(mc_bootstrap, 'sharpe', 'ci_upper'),
            'bootstrap_wr_mean': _safe_get(mc_bootstrap, 'win_rate', 'mean'),
            'bootstrap_wr_ci_lower': _safe_get(mc_bootstrap, 'win_rate', 'ci_lower'),
            'bootstrap_wr_ci_upper': _safe_get(mc_bootstrap, 'win_rate', 'ci_upper'),
            'bootstrap_pf_mean': _safe_get(mc_bootstrap, 'profit_factor', 'mean'),
            'bootstrap_pf_ci_lower': _safe_get(mc_bootstrap, 'profit_factor', 'ci_lower'),
            'bootstrap_pf_ci_upper': _safe_get(mc_bootstrap, 'profit_factor', 'ci_upper'),
            'perm_p_value': mc_permutation.get('p_value'),
            'perm_significant_05': mc_permutation.get('significant_at_05'),
            'perm_significant_01': mc_permutation.get('significant_at_01'),
            'perm_original_sharpe': mc_permutation.get('original_sharpe'),
        }

        # Confidence breakdown
        conf_detail = {
            'total_score': conf.get('total_score'),
            'rating': conf.get('rating'),
            'recommendation': conf.get('recommendation'),
            'backtest_quality_score': conf.get('backtest_quality_score'),
            'forward_back_score': conf.get('forward_back_score'),
            'walkforward_score': conf.get('walkforward_score'),
            'stability_score': conf.get('stability_score'),
            'montecarlo_score': conf.get('montecarlo_score'),
            'quality_score': conf.get('quality_score'),
            'weights': conf.get('weights', {}),
            'raw_values': conf.get('raw_values', {}),
        }

        # WF window results
        wf_windows = wf.get('window_results', []) or []
        wf_stats = wf.get('stats', {}) or {}
        wf_detail = {
            'stats': {
                'n_windows': wf_stats.get('n_windows', 0),
                'n_passed': wf_stats.get('n_passed', 0),
                'pass_rate': wf_stats.get('pass_rate', 0),
                'mean_sharpe': wf_stats.get('mean_sharpe', 0),
                'mean_return': wf_stats.get('mean_return', 0),
                'total_return': wf_stats.get('total_return', 0),
                'max_dd': wf_stats.get('max_dd', 0),
                'mean_dd': wf_stats.get('mean_dd', 0),
                'consistency': wf_stats.get('consistency', 0),
                'oos_n_windows': wf_stats.get('oos_n_windows', 0),
                'oos_n_passed': wf_stats.get('oos_n_passed', 0),
                'oos_pass_rate': wf_stats.get('oos_pass_rate', 0),
                'oos_mean_sharpe': wf_stats.get('oos_mean_sharpe', 0),
                'mean_quality_score': wf_stats.get('mean_quality_score', 0),
                'oos_mean_quality_score': wf_stats.get('oos_mean_quality_score', 0),
            },
            'windows': [{
                'window': w.get('window', i + 1),
                'status': w.get('status', ''),
                'trades': w.get('trades', 0),
                'sharpe': w.get('sharpe', 0),
                'sortino': w.get('sortino', 0),
                'r_squared': w.get('r_squared', 0),
                'return': w.get('return', 0),
                'max_dd': w.get('max_dd', 0),
                'win_rate': w.get('win_rate', 0),
                'profit_factor': w.get('profit_factor', 0),
                'quality_score': w.get('quality_score', 0),
                'passed': w.get('passed', False),
                'out_of_sample': w.get('out_of_sample', False),
                'ulcer': w.get('ulcer', 0),
            } for i, w in enumerate(wf_windows)],
        }

        # Stability overall
        stab_overall = stab.get('overall', {}) or {}
        stab_detail = {
            'mean_stability': stab_overall.get('mean_stability', 0),
            'min_stability': stab_overall.get('min_stability', 0),
            'n_stable': stab_overall.get('n_stable_params', 0),
            'n_unstable': stab_overall.get('n_unstable_params', 0),
            'n_total': stab_overall.get('n_total_params', 0),
            'n_skipped': stab_overall.get('n_skipped_booleans', 0),
            'rating': stab_overall.get('rating', 'UNKNOWN'),
        }

        # Trade summary
        trade_detail = {
            'total_trades': ts.get('total_trades', 0),
            'total_net_profit': ts.get('total_net_profit'),
            'gross_profit': ts.get('gross_profit'),
            'gross_loss': ts.get('gross_loss'),
            'profit_factor': ts.get('profit_factor'),
            'win_rate': ts.get('win_rate'),
            'avg_win': ts.get('avg_win'),
            'avg_loss': ts.get('avg_loss'),
            'largest_win': ts.get('largest_win'),
            'largest_loss': ts.get('largest_loss'),
            'max_consecutive_wins': ts.get('max_consecutive_wins'),
            'max_consecutive_losses': ts.get('max_consecutive_losses'),
            'expected_payoff': ts.get('expected_payoff'),
            'long_trades': ts.get('long_trades'),
            'short_trades': ts.get('short_trades'),
            'long_wins': ts.get('long_wins'),
            'short_wins': ts.get('short_wins'),
            'exit_reason_counts': ts.get('exit_reason_counts', {}),
            'avg_bars_held': ts.get('avg_bars_held'),
            'avg_mfe_r': ts.get('avg_mfe_r'),
            'avg_mae_r': ts.get('avg_mae_r'),
        }

        # Drawdown curve
        dd_curve = data.get('drawdown_curve', []) or []

        # Monthly returns
        monthly = data.get('monthly_returns', {}) or {}

        # Extended best candidate metrics
        extended = {
            'back_sortino': best.get('back_sortino', 0),
            'forward_sortino': best.get('forward_sortino', 0),
            'back_ulcer': best.get('back_ulcer', 0),
            'forward_ulcer': best.get('forward_ulcer'),
            'back_quality_score': best.get('back_quality_score', 0),
            'forward_quality_score': best.get('forward_quality_score', 0),
            'forward_win_rate': best.get('forward_win_rate', 0),
            'forward_profit_factor': best.get('forward_profit_factor', 0),
            'forward_max_dd': best.get('forward_max_dd', 0),
            'forward_r_squared': best.get('forward_r_squared', 0),
            'forward_back_ratio': best.get('forward_back_ratio', 0),
            'combined_rank': best.get('combined_rank', 0),
            'params': best.get('params', {}),
        }

        # Config info
        cfg = data.get('config', {}) or {}
        config_summary = {
            'initial_capital': cfg.get('initial_capital', 10000),
            'spread_pips': cfg.get('spread_pips', 1.5),
            'data_years': _safe_get(cfg, 'data', 'years'),
            'train_months': _safe_get(cfg, 'walkforward', 'train_months'),
            'test_months': _safe_get(cfg, 'walkforward', 'test_months'),
            'holdout_months': _safe_get(cfg, 'data', 'holdout_months', default=0),
            'trials_per_stage': _safe_get(cfg, 'optimization', 'trials_per_stage'),
            'final_trials': _safe_get(cfg, 'optimization', 'final_trials'),
        }

        # Decision
        decision = data.get('decision', {}) or {}

        details[r['run_id']] = {
            'montecarlo': mc_detail,
            'confidence': conf_detail,
            'walkforward': wf_detail,
            'stability': stab_detail,
            'trade_summary': trade_detail,
            'drawdown_curve': dd_curve,
            'monthly_returns': monthly,
            'extended': extended,
            'config': config_summary,
            'decision': decision,
            'recovery_factor': data.get('recovery_factor'),
            'data_summary': data.get('data_summary', {}),
        }

    return details


def build_web_data(pipelines_dir: str, output_path: str) -> dict:
    """Build and write the leaderboard JSON file.

    Returns the data dict that was written, or None if no pipelines dir exists.
    """
    if not os.path.exists(pipelines_dir):
        print(f"No pipelines directory found at {pipelines_dir}")
        return None

    runs = collect_runs(pipelines_dir)
    compare = collect_compare_data(pipelines_dir, runs)
    detail = collect_detail_data(pipelines_dir, runs)

    payload = {
        'runs': _make_serializable(runs),
        'compare': _make_serializable(compare),
        'detail': _make_serializable(detail),
        'generated_at': datetime.now().isoformat(timespec='seconds'),
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, separators=(',', ':'))

    file_size = os.path.getsize(output_path)
    print(f"Web data: {len(runs)} runs, {len(compare)} with compare data, {len(detail)} with detail data")
    print(f"  Output: {output_path}")
    print(f"  Size:   {file_size:,} bytes ({file_size / 1024:.1f} KB)")

    return payload


def main():
    parser = argparse.ArgumentParser(description='Build web leaderboard JSON')
    parser.add_argument('--pipelines-dir', default=None,
                        help='Pipelines directory (default: results/pipelines)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output path (default: web/data/leaderboard.json)')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    pipelines_dir = args.pipelines_dir or str(project_root / 'results' / 'pipelines')
    output_path = args.output or str(project_root / 'web' / 'data' / 'leaderboard.json')

    build_web_data(pipelines_dir, output_path)


if __name__ == '__main__':
    main()
