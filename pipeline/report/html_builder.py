"""Assembles the complete HTML report with 7-tab layout.

Charts are serialized as JSON and initialized per-tab on first click
to keep the page responsive.
"""
import json
from typing import Dict, Any, List

from pipeline.report import style
from pipeline.report.chart_generators import (
    confidence_gauge, score_breakdown_bar,
    equity_curve, drawdown_chart, monthly_heatmap, yearly_bars,
    pnl_scatter, exit_reason_donut, hourly_distribution, daily_distribution,
    walkforward_bars, walkforward_consistency,
    mc_return_histogram, mc_dd_histogram,
    stability_bars,
    ml_feature_importance, ml_training_metrics_per_window,
)


def build_report_html(data: Dict[str, Any]) -> str:
    """Build self-contained HTML report from enriched data."""
    rating = data['decision']['rating']
    score = data['decision']['score']
    rating_color = style.RATING_COLORS.get(rating, style.BLUE)

    best = data.get('best_candidate', {})
    confidence = best.get('confidence', {})
    trade_details = data.get('trade_details', {})
    all_trades = trade_details.get('back_trades', []) + trade_details.get('forward_trades', [])
    trade_summary = data.get('trade_summary', {})
    mc_results = best.get('montecarlo', {}).get('results', {})
    wf_results = best.get('walkforward', {}).get('window_results', [])

    # Pre-generate all chart JSON
    initial_capital = data.get('config', {}).get('initial_capital', 10000)
    charts = {
        'gauge': confidence_gauge(score, rating),
        'dash_equity': equity_curve(
            data.get('back_equity', []),
            data.get('forward_equity', []),
            initial_capital,
        ),
        'score_bar': score_breakdown_bar(confidence),
        'equity': equity_curve(
            data.get('back_equity', []),
            data.get('forward_equity', []),
            initial_capital,
        ),
        'drawdown': drawdown_chart(data.get('drawdown_curve', [])),
        'monthly': monthly_heatmap(data.get('monthly_returns', {})),
        'yearly': yearly_bars(data.get('monthly_returns', {})),
        'pnl_scatter': pnl_scatter(all_trades),
        'donut': exit_reason_donut(all_trades),
        'hourly': hourly_distribution(all_trades),
        'daily': daily_distribution(all_trades),
        'wf_bars': walkforward_bars(wf_results),
        'wf_consistency': walkforward_consistency(wf_results),
        'mc_returns': mc_return_histogram(data.get('mc_raw_returns', []), mc_results),
        'mc_dd': mc_dd_histogram(data.get('mc_raw_max_dds', []), mc_results),
        'stability': stability_bars(best),
    }

    # ML Exit charts (only if enabled)
    if data.get('ml_exit', {}).get('enabled'):
        charts['ml_importance'] = ml_feature_importance(data['ml_exit'])
        charts['ml_window_metrics'] = ml_training_metrics_per_window(data['ml_exit'])

    css = style.get_css()

    ml_exit_enabled = data.get('ml_exit', {}).get('enabled', False)
    ml_tab_btn = '<button class="tab-btn" data-tab="ml-exit">ML Exit</button>' if ml_exit_enabled else ''
    ml_tab_content = f"""
<!-- ===== TAB: ML EXIT ===== -->
<div class="tab-content" id="tab-ml-exit">
    {_build_ml_exit_tab(data, charts)}
</div>""" if ml_exit_enabled else ''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{data['meta']['pair']} {data['meta']['timeframe']} - {data['meta'].get('description', '')} - Pipeline Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>{css}</style>
</head>
<body>

<!-- ===== BREADCRUMB ===== -->
<div class="breadcrumb">
    <a href="../index.html">&larr; Back to Leaderboard</a>
</div>

<!-- ===== HEADER ===== -->
<div class="report-header">
    <div class="header-top">
        <div class="header-title">
            <span class="pair-badge">{data['meta']['pair']}</span>
            <span class="tf-badge">{data['meta']['timeframe']}</span>
            <span style="color:{style.TEXT_MUTED};font-size:0.8rem">{data['meta']['strategy']}</span>
            {f'<div style="color:{style.TEXT_MUTED};font-size:0.72rem;margin-top:4px;opacity:0.7">{data["meta"].get("description", "")}</div>' if data['meta'].get('description') else ''}
        </div>
        <div class="score-pill">
            <div>
                <div class="score-number" style="color:{rating_color}">{score:.0f}</div>
                <div class="score-label">/ 100</div>
            </div>
            <span class="rating-badge" style="background:{rating_color};color:white">{rating}</span>
        </div>
    </div>
    <nav class="tab-nav">
        <button class="tab-btn active" data-tab="dashboard">Dashboard</button>
        <button class="tab-btn" data-tab="performance">Performance</button>
        <button class="tab-btn" data-tab="trades">Trades</button>
        <button class="tab-btn" data-tab="walkforward">Walk-Forward</button>
        <button class="tab-btn" data-tab="montecarlo">Monte Carlo</button>
        <button class="tab-btn" data-tab="stability">Stability</button>
        {ml_tab_btn}
        <button class="tab-btn" data-tab="settings">Settings</button>
    </nav>
</div>

<!-- ===== TAB: DASHBOARD ===== -->
<div class="tab-content active" id="tab-dashboard">
    {_build_dashboard_tab(data, charts, rating_color, trade_summary, mc_results, confidence)}
</div>

<!-- ===== TAB: PERFORMANCE ===== -->
<div class="tab-content" id="tab-performance">
    {_build_performance_tab(data, charts, trade_summary)}
</div>

<!-- ===== TAB: TRADES ===== -->
<div class="tab-content" id="tab-trades">
    {_build_trades_tab(data, charts, all_trades, trade_summary)}
</div>

<!-- ===== TAB: WALK-FORWARD ===== -->
<div class="tab-content" id="tab-walkforward">
    {_build_walkforward_tab(data, charts, wf_results)}
</div>

<!-- ===== TAB: MONTE CARLO ===== -->
<div class="tab-content" id="tab-montecarlo">
    {_build_montecarlo_tab(data, charts, mc_results)}
</div>

<!-- ===== TAB: STABILITY ===== -->
<div class="tab-content" id="tab-stability">
    {_build_stability_tab(data, charts, best)}
</div>

{ml_tab_content}

<!-- ===== TAB: SETTINGS ===== -->
<div class="tab-content" id="tab-settings">
    {_build_settings_tab(data, best)}
</div>

<!-- ===== FOOTER ===== -->
<div class="report-footer">
    OANDA Trading System Pipeline &middot; Generated {data['meta']['generated_at'][:19]} &middot;
    {data['meta']['run_id']}
</div>

<!-- ===== SCRIPT ===== -->
<script>
{_build_javascript(charts)}
</script>

</body>
</html>"""


# ─────────────────────────────────────────────────────────────
#  TAB BUILDERS
# ─────────────────────────────────────────────────────────────

def _build_dashboard_tab(data, charts, rating_color, trade_summary, mc_results, confidence):
    rec = data['decision'].get('recommendation', '')
    best = data.get('best_candidate', {})

    net_profit = trade_summary.get('total_net_profit', 0)
    total_trades = trade_summary.get('total_trades', 0)
    win_rate = trade_summary.get('win_rate', 0) * 100
    pf = trade_summary.get('profit_factor', 0)
    max_dd = best.get('back_max_dd', 0)
    fwd_sharpe = best.get('forward_sharpe', 0)
    recovery = data.get('recovery_factor', 0)
    prob_positive = mc_results.get('prob_positive', 0)

    return f"""
    <div style="text-align:center;padding:8px 0 4px;color:{style.TEXT_MUTED};font-size:0.85rem">{rec}</div>

    <div class="chart-panel">
        <div class="chart-panel-header">Equity Curve (Back + Forward)</div>
        <div class="chart-panel-body"><div id="chart-dash-equity" style="height:350px"></div></div>
    </div>

    <div class="chart-grid">
        <div class="chart-panel">
            <div class="chart-panel-header">Confidence Score</div>
            <div class="chart-panel-body"><div id="chart-gauge" style="height:200px"></div></div>
        </div>
        <div class="chart-panel">
            <div class="chart-panel-header">Score Breakdown</div>
            <div class="chart-panel-body"><div id="chart-score-bar" style="height:200px"></div></div>
        </div>
    </div>

    <div class="metrics-row">
        {_metric('Total Net Profit', f'${net_profit:,.0f}', cls='positive' if net_profit > 0 else 'negative')}
        {_metric('Total Trades', f'{total_trades}')}
        {_metric('Win Rate', f'{win_rate:.1f}%')}
        {_metric('Profit Factor', f'{pf:.2f}')}
        {_metric('Max Drawdown', f'{max_dd:.1f}%')}
        {_metric('Forward Sharpe', f'{fwd_sharpe:.2f}')}
        {_metric('Recovery Factor', f'{recovery:.2f}')}
        {_metric('Prob. Positive (MC)', f'{prob_positive:.0f}%')}
    </div>

    {_build_leaderboard(data.get('candidates', []))}
    """


def _build_performance_tab(data, charts, trade_summary):
    best = data.get('best_candidate', {})
    net = trade_summary.get('total_net_profit', 0)
    gross_p = trade_summary.get('gross_profit', 0)
    gross_l = trade_summary.get('gross_loss', 0)
    expected = trade_summary.get('expected_payoff', 0)
    max_dd_abs = data.get('max_dd_absolute', 0)

    return f"""
    <div class="metrics-row">
        {_metric('Total Net Profit', f'${net:,.0f}', cls='positive' if net > 0 else 'negative')}
        {_metric('Gross Profit', f'${gross_p:,.0f}', cls='positive')}
        {_metric('Gross Loss', f'${abs(gross_l):,.0f}', cls='negative')}
        {_metric('Expected Payoff', f'${expected:,.2f}')}
        {_metric('Max DD ($)', f'${max_dd_abs:,.0f}')}
        {_metric('Recovery Factor', f'{data.get("recovery_factor", 0):.2f}')}
        {_metric('Back Sharpe', f'{best.get("back_sharpe", 0):.2f}')}
        {_metric('Forward Sharpe', f'{best.get("forward_sharpe", 0):.2f}')}
    </div>

    <div class="chart-panel">
        <div class="chart-panel-header">Equity Curve (Back + Forward)</div>
        <div class="chart-panel-body"><div id="chart-equity" style="height:360px"></div></div>
    </div>

    <div class="chart-panel">
        <div class="chart-panel-header">Underwater Drawdown</div>
        <div class="chart-panel-body"><div id="chart-drawdown" style="height:200px"></div></div>
    </div>

    <div class="chart-grid">
        <div class="chart-panel">
            <div class="chart-panel-header">Monthly P&L</div>
            <div class="chart-panel-body"><div id="chart-monthly" style="height:240px"></div></div>
        </div>
        <div class="chart-panel">
            <div class="chart-panel-header">Yearly P&L</div>
            <div class="chart-panel-body"><div id="chart-yearly" style="height:200px"></div></div>
        </div>
    </div>
    """


def _build_trades_tab(data, charts, all_trades, trade_summary):
    total = trade_summary.get('total_trades', 0)
    long_t = trade_summary.get('long_trades', 0)
    short_t = trade_summary.get('short_trades', 0)
    long_w = trade_summary.get('long_wins', 0)
    short_w = trade_summary.get('short_wins', 0)
    avg_w = trade_summary.get('avg_win', 0)
    avg_l = trade_summary.get('avg_loss', 0)
    largest_w = trade_summary.get('largest_win', 0)
    largest_l = trade_summary.get('largest_loss', 0)
    consec_w = trade_summary.get('max_consecutive_wins', 0)
    consec_l = trade_summary.get('max_consecutive_losses', 0)

    long_wr = (long_w / long_t * 100) if long_t > 0 else 0
    short_wr = (short_w / short_t * 100) if short_t > 0 else 0

    return f"""
    <div class="metrics-row">
        {_metric('Total Trades', str(total))}
        {_metric('Long', f'{long_t} ({long_wr:.0f}% win)')}
        {_metric('Short', f'{short_t} ({short_wr:.0f}% win)')}
        {_metric('Avg Win', f'${avg_w:,.2f}', cls='positive')}
        {_metric('Avg Loss', f'${avg_l:,.2f}', cls='negative')}
        {_metric('Largest Win', f'${largest_w:,.2f}', cls='positive')}
        {_metric('Largest Loss', f'${largest_l:,.2f}', cls='negative')}
        {_metric('Consec W/L', f'{consec_w} / {consec_l}')}
    </div>

    <div class="chart-panel">
        <div class="chart-panel-header">Trade P&L</div>
        <div class="chart-panel-body"><div id="chart-pnl-scatter" style="height:300px"></div></div>
    </div>

    <div class="chart-grid">
        <div class="chart-panel">
            <div class="chart-panel-header">Direction Breakdown</div>
            <div class="chart-panel-body"><div id="chart-donut" style="height:250px"></div></div>
        </div>
        <div class="chart-panel">
            <div class="chart-panel-header">By Hour</div>
            <div class="chart-panel-body"><div id="chart-hourly" style="height:250px"></div></div>
        </div>
    </div>

    <div class="chart-panel">
        <div class="chart-panel-header">By Day of Week</div>
        <div class="chart-panel-body"><div id="chart-daily" style="height:250px"></div></div>
    </div>

    {_build_trade_table(all_trades)}
    """


def _build_walkforward_tab(data, charts, wf_results):
    wf_summary = data.get('walkforward_summary', {})
    best = data.get('best_candidate', {})
    wf_data = best.get('walkforward', {})
    wf_stats = wf_data.get('stats', {})

    n_windows = wf_summary.get('n_windows', 0)
    pass_rate = wf_stats.get('pass_rate', 0)
    mean_sharpe = wf_stats.get('mean_sharpe', 0)
    consistency = wf_stats.get('consistency', 0)

    return f"""
    <div class="metrics-row">
        {_metric('Windows', str(n_windows))}
        {_metric('Pass Rate', f'{pass_rate:.0%}')}
        {_metric('Mean Sharpe', f'{mean_sharpe:.2f}')}
        {_metric('Consistency', f'{consistency:.2f}')}
    </div>

    <div class="chart-panel">
        <div class="chart-panel-header">Window Results</div>
        <div class="chart-panel-body"><div id="chart-wf-bars" style="height:300px"></div></div>
    </div>

    <div class="chart-panel">
        <div class="chart-panel-header">OnTester per Window</div>
        <div class="chart-panel-body"><div id="chart-wf-consistency" style="height:{max(200, len(wf_results)*40+60)}px"></div></div>
    </div>

    {_build_wf_table(wf_results)}
    """


def _build_montecarlo_tab(data, charts, mc_results):
    mc_summary = data.get('montecarlo_summary', {})
    iterations = mc_summary.get('iterations', 500)

    return f"""
    <div class="metrics-row">
        {_metric('Original Return', f'{mc_results.get("original_return", 0):.1f}%')}
        {_metric('MC Mean Return', f'{mc_results.get("mean_return", 0):.1f}%')}
        {_metric('5th Percentile', f'{mc_results.get("pct_5_return", 0):.1f}%',
                 cls='positive' if mc_results.get('pct_5_return', 0) > 0 else 'negative')}
        {_metric('95th %ile Max DD', f'{mc_results.get("pct_95_dd", 0):.1f}%')}
        {_metric('Prob Positive', f'{mc_results.get("prob_positive", 0):.0f}%')}
        {_metric('VaR 95%', f'{mc_results.get("var_95", 0):.1f}%')}
        {_metric('Prob > 5%', f'{mc_results.get("prob_above_5pct", 0):.0f}%')}
        {_metric('Prob > 10%', f'{mc_results.get("prob_above_10pct", 0):.0f}%')}
    </div>

    <div class="chart-grid">
        <div class="chart-panel">
            <div class="chart-panel-header">Return Distribution ({iterations} iterations)</div>
            <div class="chart-panel-body"><div id="chart-mc-returns" style="height:300px"></div></div>
        </div>
        <div class="chart-panel">
            <div class="chart-panel-header">Max Drawdown Distribution</div>
            <div class="chart-panel-body"><div id="chart-mc-dd" style="height:300px"></div></div>
        </div>
    </div>
    """


def _build_stability_tab(data, charts, best):
    stability = best.get('stability', {})
    overall = stability.get('overall', {})

    rating = overall.get('rating', 'N/A')
    mean_stab = overall.get('mean_stability', 0)
    min_stab = overall.get('min_stability', 0)
    n_stable = overall.get('n_stable_params', 0)
    n_total = overall.get('n_total_params', 0)
    n_fragile = overall.get('n_unstable_params', 0)

    stab_colors = {
        'ROBUST': style.GREEN, 'MODERATE': style.YELLOW,
        'FRAGILE': style.ORANGE, 'OVERFIT': style.RED,
    }
    stab_color = stab_colors.get(rating, style.TEXT_MUTED)

    # Build per-param HTML bars
    per_param = stability.get('per_param', stability.get('params', {}))
    param_bars_html = ''
    for pname in sorted(per_param.keys()):
        pdata = per_param[pname]
        ratio = pdata.get('stability_ratio', 0)
        bar_color = style.GREEN if ratio >= 0.8 else (style.YELLOW if ratio >= 0.5 else style.RED)
        pct = min(ratio * 100, 100)
        param_bars_html += f"""
        <div class="stability-bar-container">
            <div class="stability-bar-label">
                <span style="color:{style.TEXT_SECONDARY}">{pname}</span>
                <span style="color:{bar_color};font-family:'JetBrains Mono',monospace;font-size:0.78rem">{ratio:.1%}</span>
            </div>
            <div class="stability-bar-track">
                <div class="stability-bar-fill" style="width:{pct}%;background:{bar_color}"></div>
            </div>
        </div>"""

    # Fragile param warnings
    fragile_html = ''
    fragile_params = [p for p, d in per_param.items() if d.get('stability_ratio', 1) < 0.5]
    if fragile_params:
        fragile_items = ''.join(
            f'<div class="risk-card"><div class="risk-icon" style="background:{style.RED_DIM}">!</div>'
            f'<div><div class="risk-title">{p}</div>'
            f'<div class="risk-desc">Stability: {per_param[p].get("stability_ratio", 0):.1%} - '
            f'Small changes may break strategy performance</div></div></div>'
            for p in fragile_params
        )
        fragile_html = f"""
        <div class="chart-panel" style="margin-top:16px">
            <div class="chart-panel-header" style="color:{style.RED}">Fragile Parameter Warnings</div>
            <div class="chart-panel-body" style="padding:12px">{fragile_items}</div>
        </div>"""

    return f"""
    <div class="metrics-row">
        {_metric('Rating', rating, cls='positive' if rating == 'ROBUST' else ('negative' if rating in ('FRAGILE', 'OVERFIT') else ''))}
        {_metric('Mean Stability', f'{mean_stab:.1%}')}
        {_metric('Min Stability', f'{min_stab:.1%}')}
        {_metric('Stable / Total', f'{n_stable} / {n_total}')}
    </div>

    <div class="chart-panel">
        <div class="chart-panel-header">Parameter Stability Ratios</div>
        <div class="chart-panel-body"><div id="chart-stability" style="height:{max(200, len(per_param)*28+60)}px"></div></div>
    </div>

    <div class="chart-panel">
        <div class="chart-panel-header">Detailed Stability</div>
        <div class="chart-panel-body" style="padding:16px">{param_bars_html}</div>
    </div>

    {fragile_html}
    """


def _build_ml_exit_tab(data, charts):
    """Build the ML Exit diagnostics tab content."""
    ml_data = data.get('ml_exit', {})
    if not ml_data.get('enabled'):
        return f'<div style="padding:20px;color:{style.TEXT_MUTED}">ML Exit not enabled for this run.</div>'

    training = ml_data.get('training_metrics', {})
    backend = training.get('backend', 'unknown')
    n_windows = len(ml_data.get('window_ml_metrics', []))

    # Hold model metrics
    hold_metrics = training.get('hold_value', {})
    # Risk model metrics
    risk_metrics = training.get('adverse_risk', {})

    # Total exit signals across all windows
    total_exits = sum(m.get('n_exit_signals', 0) for m in ml_data.get('window_ml_metrics', []))
    total_train_rows = sum(m.get('n_train_rows', 0) for m in ml_data.get('window_ml_metrics', []))

    return f"""
    <div class="metrics-row">
        {_metric('ML Backend', backend)}
        {_metric('Windows Trained', str(n_windows))}
        {_metric('Hold Model RMSE', f'{hold_metrics.get("val_rmse", 0):.3f}')}
        {_metric('Hold Model R2', f'{hold_metrics.get("val_r2", 0):.3f}')}
        {_metric('Risk Model AUC', f'{risk_metrics.get("val_auc", 0):.3f}')}
        {_metric('Risk Model LogLoss', f'{risk_metrics.get("val_logloss", 0):.3f}')}
        {_metric('Total Exit Signals', str(total_exits))}
        {_metric('Total Train Rows', str(total_train_rows))}
    </div>

    <div class="chart-grid">
        <div class="chart-panel">
            <div class="chart-panel-header">Feature Importance</div>
            <div class="chart-panel-body"><div id="chart-ml-importance" style="height:350px"></div></div>
        </div>
        <div class="chart-panel">
            <div class="chart-panel-header">Model Performance per Window</div>
            <div class="chart-panel-body"><div id="chart-ml-window-metrics" style="height:300px"></div></div>
        </div>
    </div>

    {_build_ml_window_table(ml_data.get('window_ml_metrics', []))}
    """


def _build_ml_window_table(window_metrics):
    """Table showing ML training details per walk-forward window."""
    if not window_metrics:
        return ''

    rows = []
    for m in window_metrics:
        tm = m.get('training_metrics', {})
        hold = tm.get('hold_value', {})
        risk = tm.get('adverse_risk', {})
        rows.append(f"""<tr>
            <td>W{m.get('window', '?')}</td>
            <td>{m.get('backend', '-')}</td>
            <td>{m.get('n_train_rows', 0)}</td>
            <td>{hold.get('val_rmse', 0):.3f}</td>
            <td>{hold.get('val_r2', 0):.3f}</td>
            <td>{risk.get('val_auc', 0):.3f}</td>
            <td>{risk.get('val_logloss', 0):.3f}</td>
            <td>{m.get('n_exit_signals', 0)}</td>
        </tr>""")

    return f"""
    <div class="chart-panel" style="margin-top:16px">
        <div class="chart-panel-header">ML Training Details per Window</div>
        <div class="chart-panel-body">
            <table class="data-table">
                <thead><tr>
                    <th>Window</th><th>Backend</th><th>Train Rows</th>
                    <th>Hold RMSE</th><th>Hold R2</th>
                    <th>Risk AUC</th><th>Risk LogLoss</th><th>Exit Signals</th>
                </tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
    </div>"""


def _build_settings_tab(data, best):
    params = best.get('params', {})
    config = data.get('config', {})

    # Group parameters by category
    categories = {
        'Signal Detection': ['rsi_period', 'rsi_overbought', 'rsi_oversold', 'min_rsi_diff',
                             'swing_strength', 'min_bars_between', 'max_bars_between', 'require_pullback'],
        'Filters': ['use_slope_filter', 'min_price_slope', 'max_price_slope',
                     'use_rsi_extreme_filter', 'use_trend_filter', 'trend_ma_period', 'max_spread_pips'],
        'Risk Management': ['sl_mode', 'sl_fixed_pips', 'sl_atr_mult', 'sl_swing_buffer',
                            'tp_mode', 'tp_rr_ratio', 'tp_atr_mult', 'tp_fixed_pips'],
        'Trade Management': ['use_trailing', 'trail_start_pips', 'trail_step_pips',
                             'use_break_even', 'be_trigger_pips', 'be_offset_pips',
                             'use_partial_close', 'partial_close_pct'],
        'Time Filters': ['use_time_filter', 'trade_start_hour', 'trade_end_hour',
                         'trade_monday', 'trade_friday', 'friday_close_hour'],
    }

    params_html = ''
    assigned_params = set()
    for cat_name, cat_keys in categories.items():
        items = []
        for key in cat_keys:
            if key in params:
                assigned_params.add(key)
                val = params[key]
                items.append(f"""<div class="param-row">
                    <span class="param-key">{key}</span>
                    <span class="param-val">{val}</span>
                </div>""")
        if items:
            params_html += f"""<div class="params-section">
                <div class="params-section-title">{cat_name}</div>
                <div class="params-grid">{''.join(items)}</div>
            </div>"""

    # Unassigned params
    remaining = {k: v for k, v in params.items() if k not in assigned_params}
    if remaining:
        items = ''.join(
            f'<div class="param-row"><span class="param-key">{k}</span>'
            f'<span class="param-val">{v}</span></div>'
            for k, v in sorted(remaining.items())
        )
        params_html += f"""<div class="params-section">
            <div class="params-section-title">Other</div>
            <div class="params-grid">{items}</div>
        </div>"""

    # Risk recommendations
    mc_results = best.get('montecarlo', {}).get('results', {})
    max_dd_95 = mc_results.get('pct_95_dd', 25)
    target_dd = 20
    suggested_size = min(1.0, target_dd / max_dd_95) if max_dd_95 > 0 else 0.5

    risk_html = f"""
    <div class="risk-card">
        <div class="risk-icon" style="background:{style.BLUE_DIM}">S</div>
        <div>
            <div class="risk-title">Suggested Position Size</div>
            <div class="risk-desc">Start at {suggested_size:.0%} of normal size (MC 95th %ile DD: {max_dd_95:.1f}%)</div>
        </div>
    </div>
    <div class="risk-card">
        <div class="risk-icon" style="background:{style.GREEN_DIM}">+</div>
        <div>
            <div class="risk-title">Scaling Plan</div>
            <div class="risk-desc">0.25x &rarr; 0.5x &rarr; 1.0x over 3 months if performance matches backtest</div>
        </div>
    </div>
    <div class="risk-card">
        <div class="risk-icon" style="background:{style.RED_DIM}">!</div>
        <div>
            <div class="risk-title">Circuit Breaker</div>
            <div class="risk-desc">Stop if DD exceeds {min(max_dd_95 * 1.5, 30):.0f}% or 3 consecutive losing months</div>
        </div>
    </div>
    """

    # Pipeline config
    config_json = json.dumps(config, indent=2, default=str)

    return f"""
    <div class="chart-panel">
        <div class="chart-panel-header">Strategy Parameters</div>
        <div class="chart-panel-body" style="padding:16px">{params_html}</div>
    </div>

    <div class="chart-panel">
        <div class="chart-panel-header">Risk Recommendations</div>
        <div class="chart-panel-body" style="padding:12px">{risk_html}</div>
    </div>

    <div class="chart-panel">
        <div class="chart-panel-header" style="cursor:pointer" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'">
            Pipeline Configuration (click to toggle)
        </div>
        <div class="chart-panel-body" style="display:none;padding:16px">
            <pre style="font-size:0.75rem;color:{style.TEXT_MUTED};overflow-x:auto;font-family:'JetBrains Mono',monospace">{config_json}</pre>
        </div>
    </div>
    """


# ─────────────────────────────────────────────────────────────
#  COMPONENT HELPERS
# ─────────────────────────────────────────────────────────────

def _metric(label: str, value: str, cls: str = '', sub: str = '') -> str:
    """Single metric card."""
    cls_attr = f' class="metric-value {cls}"' if cls else ' class="metric-value"'
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ''
    return f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div{cls_attr}>{value}</div>
        {sub_html}
    </div>"""


def _build_leaderboard(candidates: List[Dict]) -> str:
    """Sortable candidate leaderboard table."""
    if not candidates:
        return ''

    rows = []
    for c in candidates:
        rank = c.get('rank', 0)
        score = c.get('confidence', {}).get('total_score', 0)
        score_cls = 'positive' if score >= 70 else ('negative' if score < 40 else '')

        rows.append(f"""<tr>
            <td>{rank}</td>
            <td class="{score_cls}">{score:.1f}</td>
            <td>{c.get('back_ontester', 0):.0f}</td>
            <td>{c.get('forward_ontester', 0):.0f}</td>
            <td>{c.get('back_trades', 0)}</td>
            <td>{c.get('forward_trades', 0)}</td>
            <td>{c.get('back_sharpe', 0):.2f}</td>
            <td>{c.get('forward_sharpe', 0):.2f}</td>
            <td>{c.get('back_max_dd', 0):.1f}%</td>
            <td>{c.get('forward_back_ratio', 0):.2f}</td>
        </tr>""")

    return f"""
    <div class="chart-panel" style="margin-top:16px">
        <div class="chart-panel-header">Candidate Leaderboard (Top {len(candidates)})</div>
        <div class="chart-panel-body">
            <div class="table-scroll">
                <table class="data-table" id="leaderboard">
                    <thead><tr>
                        <th>Rank</th><th>Score</th><th>Back OT</th><th>Fwd OT</th>
                        <th>Back Tr</th><th>Fwd Tr</th><th>Back Sh</th><th>Fwd Sh</th>
                        <th>Max DD</th><th>F/B Ratio</th>
                    </tr></thead>
                    <tbody>{''.join(rows)}</tbody>
                </table>
            </div>
        </div>
    </div>"""


def _build_trade_table(trades: List[Dict]) -> str:
    """Scrollable trade list table."""
    if not trades:
        return ''

    rows = []
    for i, t in enumerate(trades):
        pnl = t.get('pnl', 0)
        cls = 'positive' if pnl > 0 else 'negative'
        direction = t.get('direction', '-')
        entry_time = t.get('entry_time', '-')
        if entry_time and len(entry_time) > 19:
            entry_time = entry_time[:19]

        rows.append(f"""<tr>
            <td>{i+1}</td>
            <td>{entry_time}</td>
            <td>{direction}</td>
            <td>{t.get('entry_price', '-')}</td>
            <td>{t.get('sl_price', '-')}</td>
            <td>{t.get('tp_price', '-')}</td>
            <td class="{cls}">${pnl:,.2f}</td>
        </tr>""")

    return f"""
    <div class="chart-panel" style="margin-top:16px">
        <div class="chart-panel-header">Trade List ({len(trades)} trades)</div>
        <div class="chart-panel-body">
            <div class="table-scroll">
                <table class="data-table" id="trade-table">
                    <thead><tr>
                        <th>#</th><th>Entry Time</th><th>Dir</th>
                        <th>Entry</th><th>SL</th><th>TP</th><th>P&L</th>
                    </tr></thead>
                    <tbody>{''.join(rows)}</tbody>
                </table>
            </div>
        </div>
    </div>"""


def _build_wf_table(wf_results: List[Dict]) -> str:
    """Walk-forward window results table."""
    if not wf_results:
        return ''

    rows = []
    for wr in wf_results:
        status_cls = 'positive' if wr.get('passed') else 'negative'
        status_text = 'Pass' if wr.get('passed') else 'Fail'
        rows.append(f"""<tr>
            <td>W{wr.get('window', '?')}</td>
            <td>{wr.get('trades', 0)}</td>
            <td>{wr.get('ontester', 0):.1f}</td>
            <td>{wr.get('sharpe', 0):.2f}</td>
            <td>{wr.get('return', 0):.1f}%</td>
            <td>{wr.get('win_rate', 0)*100:.0f}%</td>
            <td>{wr.get('max_dd', 0):.1f}%</td>
            <td class="{status_cls}">{status_text}</td>
        </tr>""")

    return f"""
    <div class="chart-panel" style="margin-top:16px">
        <div class="chart-panel-header">Window Details</div>
        <div class="chart-panel-body">
            <table class="data-table">
                <thead><tr>
                    <th>Window</th><th>Trades</th><th>OnTester</th><th>Sharpe</th>
                    <th>Return</th><th>Win Rate</th><th>Max DD</th><th>Status</th>
                </tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
    </div>"""


# ─────────────────────────────────────────────────────────────
#  JAVASCRIPT
# ─────────────────────────────────────────────────────────────

def _build_javascript(charts: Dict[str, str]) -> str:
    """Build all JavaScript: tab switching, chart init, table sort."""

    # Chart ID mapping: chart_key -> (div_id, tab_id)
    chart_map = {
        'gauge': ('chart-gauge', 'dashboard'),
        'dash_equity': ('chart-dash-equity', 'dashboard'),
        'score_bar': ('chart-score-bar', 'dashboard'),
        'equity': ('chart-equity', 'performance'),
        'drawdown': ('chart-drawdown', 'performance'),
        'monthly': ('chart-monthly', 'performance'),
        'yearly': ('chart-yearly', 'performance'),
        'pnl_scatter': ('chart-pnl-scatter', 'trades'),
        'donut': ('chart-donut', 'trades'),
        'hourly': ('chart-hourly', 'trades'),
        'daily': ('chart-daily', 'trades'),
        'wf_bars': ('chart-wf-bars', 'walkforward'),
        'wf_consistency': ('chart-wf-consistency', 'walkforward'),
        'mc_returns': ('chart-mc-returns', 'montecarlo'),
        'mc_dd': ('chart-mc-dd', 'montecarlo'),
        'stability': ('chart-stability', 'stability'),
        'ml_importance': ('chart-ml-importance', 'ml-exit'),
        'ml_window_metrics': ('chart-ml-window-metrics', 'ml-exit'),
    }

    # Build chart data object
    chart_data_js = '{\n'
    for key, json_str in charts.items():
        div_id, tab = chart_map.get(key, (key, 'dashboard'))
        chart_data_js += f'    "{div_id}": {{ tab: "{tab}", spec: {json_str} }},\n'
    chart_data_js += '}'

    return f"""
(function() {{
    // ===== Chart data =====
    var chartData = {chart_data_js};
    var rendered = {{}};

    function renderChartsForTab(tabId) {{
        Object.keys(chartData).forEach(function(divId) {{
            var entry = chartData[divId];
            if (entry.tab === tabId && !rendered[divId]) {{
                var el = document.getElementById(divId);
                if (el && entry.spec.data) {{
                    Plotly.newPlot(el, entry.spec.data, entry.spec.layout, {{
                        responsive: true,
                        displayModeBar: false
                    }});
                    rendered[divId] = true;
                }}
            }}
        }});
    }}

    // ===== Tab switching =====
    var tabs = document.querySelectorAll('.tab-btn');
    var panels = document.querySelectorAll('.tab-content');

    tabs.forEach(function(btn) {{
        btn.addEventListener('click', function() {{
            var target = btn.getAttribute('data-tab');

            tabs.forEach(function(b) {{ b.classList.remove('active'); }});
            panels.forEach(function(p) {{ p.classList.remove('active'); }});

            btn.classList.add('active');
            document.getElementById('tab-' + target).classList.add('active');

            // Lazy render charts
            renderChartsForTab(target);

            // Resize charts (fixes hidden tab sizing)
            setTimeout(function() {{
                window.dispatchEvent(new Event('resize'));
            }}, 50);
        }});
    }});

    // ===== Table sorting =====
    document.querySelectorAll('.data-table').forEach(function(table) {{
        table.querySelectorAll('th').forEach(function(th, colIdx) {{
            th.addEventListener('click', function() {{
                var tbody = table.querySelector('tbody');
                if (!tbody) return;
                var rows = Array.from(tbody.querySelectorAll('tr'));
                var dir = th.dataset.sortDir === 'asc' ? 'desc' : 'asc';
                th.dataset.sortDir = dir;

                rows.sort(function(a, b) {{
                    var aText = a.cells[colIdx] ? a.cells[colIdx].textContent.replace(/[$,%]/g, '').trim() : '';
                    var bText = b.cells[colIdx] ? b.cells[colIdx].textContent.replace(/[$,%]/g, '').trim() : '';
                    var aNum = parseFloat(aText);
                    var bNum = parseFloat(bText);
                    var isNum = !isNaN(aNum) && !isNaN(bNum);

                    if (isNum) {{
                        return dir === 'asc' ? aNum - bNum : bNum - aNum;
                    }}
                    return dir === 'asc' ? aText.localeCompare(bText) : bText.localeCompare(aText);
                }});

                rows.forEach(function(r) {{ tbody.appendChild(r); }});
            }});
        }});
    }});

    // ===== Initial render =====
    renderChartsForTab('dashboard');
}})();
"""
