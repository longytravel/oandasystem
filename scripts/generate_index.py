"""Generate a cross-run leaderboard index.html for all pipeline runs.

Scans results/pipelines/*/state.json and builds a sortable, filterable
master dashboard. Each row links to the individual run's report.html.

Features:
  - Checkbox selection to compare multiple runs
  - Overlaid equity curves with R² annotations
  - Side-by-side metrics comparison table

Usage:
    python scripts/generate_index.py
    python scripts/generate_index.py --output results/pipelines/index.html
    python scripts/generate_index.py --inject-backlink
"""
import json
import os
import argparse
from datetime import datetime
from pathlib import Path


def collect_runs(pipelines_dir: str) -> list:
    """Scan all pipeline run directories and collect metadata."""
    runs = []

    for run_name in sorted(os.listdir(pipelines_dir)):
        run_dir = os.path.join(pipelines_dir, run_name)
        state_file = os.path.join(run_dir, 'state.json')

        if not os.path.isdir(run_dir) or not os.path.exists(state_file):
            continue

        try:
            with open(state_file) as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Extract key info
        pair = state.get('pair', '?')
        tf = state.get('timeframe', '?')
        strategy = state.get('strategy_name', '?')
        description = state.get('description', '')
        score = state.get('final_score') or 0
        rating = state.get('final_rating', 'RED')
        created = state.get('created_at', '')[:19]
        best = state.get('best_candidate', {}) or {}

        # Auto-generate description from config if not present
        if not description:
            cfg = state.get('config', {})
            if cfg:
                ml_cfg = cfg.get('ml_exit', {})
                ml_part = f"ML {ml_cfg.get('policy_mode', 'dual_model')}" if ml_cfg.get('enabled') else "no ML"
                data_cfg = cfg.get('data', {})
                wf_cfg = cfg.get('walkforward', {})
                holdout = data_cfg.get('holdout_months', 0)
                holdout_part = f" hold={holdout}m" if holdout > 0 else ""
                description = (f"{strategy} | "
                              f"{data_cfg.get('years', '?')}yr "
                              f"{wf_cfg.get('train_months', '?')}m/{wf_cfg.get('test_months', '?')}m | "
                              f"{ml_part}{holdout_part}")

        # Stage progress
        stages = state.get('stages', {})
        stage_order = ['data', 'optimization', 'walkforward', 'stability',
                       'montecarlo', 'confidence', 'report']
        completed_stages = 0
        last_stage = 'none'
        failed_stage = None
        total_time = 0

        for sname in stage_order:
            sdata = stages.get(sname, {})
            status = sdata.get('status', 'pending')
            total_time += sdata.get('duration_seconds', 0)
            if status == 'completed':
                completed_stages += 1
                last_stage = sname
            elif status == 'failed':
                failed_stage = sname
                break

        has_report = os.path.exists(os.path.join(run_dir, 'report.html'))
        report_data_path = os.path.join(run_dir, 'report_data.json')
        has_report_data = os.path.exists(report_data_path)

        # Load net profit from report_data.json if available
        net_profit = None
        if has_report_data:
            try:
                with open(report_data_path, encoding='utf-8') as f:
                    rd = json.load(f)
                net_profit = (rd.get('trade_summary') or {}).get('total_net_profit')
            except (json.JSONDecodeError, OSError):
                pass

        if failed_stage:
            run_status = 'FAILED'
        elif completed_stages == 7:
            run_status = 'COMPLETE'
        else:
            run_status = 'PARTIAL'

        # Optimization summary
        opt_summary = stages.get('optimization', {}).get('summary', {})
        n_candidates = opt_summary.get('n_candidates', 0)

        runs.append({
            'run_id': run_name,
            'pair': pair,
            'timeframe': tf,
            'strategy': strategy,
            'description': description,
            'score': score,
            'rating': rating,
            'status': run_status,
            'created': created,
            'stages_done': completed_stages,
            'last_stage': last_stage,
            'failed_stage': failed_stage,
            'total_time_min': total_time / 60,
            'has_report': has_report,
            'has_report_data': has_report_data,
            'n_candidates': n_candidates,
            'back_sharpe': best.get('back_sharpe', 0),
            'forward_sharpe': best.get('forward_sharpe', 0),
            'back_trades': best.get('back_trades', 0),
            'forward_trades': best.get('forward_trades', 0),
            'back_max_dd': best.get('back_max_dd', 0),
            'forward_back_ratio': best.get('forward_back_ratio', 0),
            'win_rate': best.get('back_win_rate', 0),
            'back_return': best.get('back_return', 0),
            'forward_return': best.get('forward_return', 0),
            'net_profit': net_profit,
        })

    return runs


def collect_compare_data(pipelines_dir: str, runs: list) -> dict:
    """Load comparison data (equity curves + metrics) from report_data.json files."""
    compare = {}
    for r in runs:
        if not r['has_report_data']:
            continue
        path = os.path.join(pipelines_dir, r['run_id'], 'report_data.json')
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        best = data.get('best_candidate', {}) or {}
        back_eq = data.get('back_equity', [])
        fwd_eq = data.get('forward_equity', [])

        # Skip runs with no equity data
        if not back_eq and not fwd_eq:
            continue

        stab_overall = (best.get('stability') or {}).get('overall') or {}
        ts = data.get('trade_summary') or {}

        compare[r['run_id']] = {
            'meta': data.get('meta', {}),
            'decision': data.get('decision', {}),
            'back_equity': back_eq,
            'forward_equity': fwd_eq,
            'initial_capital': (data.get('config') or {}).get('initial_capital', 10000),
            'best_rank': best.get('rank'),
            'back_r_squared': best.get('back_r_squared', 0),
            'forward_r_squared': best.get('forward_r_squared', 0),
            'back_return': best.get('back_return', 0),
            'forward_return': best.get('forward_return', 0),
            'back_trades': best.get('back_trades', 0),
            'forward_trades': best.get('forward_trades', 0),
            'back_sharpe': best.get('back_sharpe', 0),
            'forward_sharpe': best.get('forward_sharpe', 0),
            'back_max_dd': best.get('back_max_dd', 0),
            'back_win_rate': best.get('back_win_rate', 0),
            'back_profit_factor': best.get('back_profit_factor', 0),
            'win_rate': ts.get('win_rate', best.get('back_win_rate', 0)),
            'profit_factor': ts.get('profit_factor', best.get('back_profit_factor', 0)),
            'stability_mean': stab_overall.get('mean_stability', 0),
        }

    return compare


def build_index_html(runs: list, compare_data: dict = None) -> str:
    """Build the leaderboard HTML."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Collect unique values for filters
    pairs = sorted(set(r['pair'] for r in runs))
    timeframes = sorted(set(r['timeframe'] for r in runs))
    strategies = sorted(set(r['strategy'] for r in runs))

    # Build table rows - column 0 is now checkbox, data columns start at 1
    rows_html = ''
    for r in runs:
        # Colors
        if r['rating'] == 'GREEN':
            rating_bg = '#065f46'
            rating_fg = '#10b981'
        elif r['rating'] == 'YELLOW':
            rating_bg = '#78350f'
            rating_fg = '#f59e0b'
        else:
            rating_bg = '#7f1d1d'
            rating_fg = '#ef4444'

        if r['status'] == 'FAILED':
            status_html = '<span style="color:#ef4444">FAILED</span>'
        elif r['status'] == 'PARTIAL':
            status_html = f'<span style="color:#5a6a80">{r["stages_done"]}/7</span>'
        else:
            status_html = '<span style="color:#10b981">OK</span>'

        # Link to report
        link = f'{r["run_id"]}/report.html' if r['has_report'] else '#'
        link_attr = '' if r['has_report'] else ' style="pointer-events:none;opacity:0.4"'

        score_display = f'{r["score"]:.0f}' if r['score'] > 0 else '-'

        # Checkbox - only enabled for runs with report_data.json
        cb_disabled = '' if r['has_report_data'] else ' disabled'
        cb_opacity = '' if r['has_report_data'] else ' style="opacity:0.3"'

        desc_escaped = r['description'].replace('"', '&quot;').replace('<', '&lt;')

        rows_html += f"""<tr data-pair="{r['pair']}" data-tf="{r['timeframe']}"
             data-strategy="{r['strategy']}" data-status="{r['status']}"
             data-rating="{r['rating']}" data-run="{r['run_id']}">
            <td{cb_opacity}><input type="checkbox" class="compare-check" data-run="{r['run_id']}"{cb_disabled}></td>
            <td><a href="{link}"{link_attr}>{r['pair']}</a></td>
            <td>{r['timeframe']}</td>
            <td class="strategy-col">{r['strategy']}</td>
            <td class="desc-col" title="{desc_escaped}">{desc_escaped}</td>
            <td class="score-col" style="color:{rating_fg}"><strong>{score_display}</strong></td>
            <td><span style="background:{rating_bg};color:{rating_fg};padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:600">{r['rating']}</span></td>
            <td>{status_html}</td>
            <td>{r['created']}</td>
            <td style="color:{'#10b981' if (r['net_profit'] or 0) > 0 else ('#ef4444' if r['net_profit'] is not None else '#5a6a80')}">{('$' + f'{r["net_profit"]:,.0f}') if r['net_profit'] is not None else '-'}</td>
            <td>{r['back_sharpe']:.2f}</td>
            <td>{r['forward_sharpe']:.2f}</td>
            <td>{r['back_trades']}</td>
            <td>{r['forward_trades']}</td>
            <td>{r['back_max_dd']:.1f}%</td>
            <td>{r['forward_back_ratio']:.2f}</td>
            <td>{r['win_rate']*100:.0f}%</td>
            <td>{r['total_time_min']:.1f}m</td>
            <td class="runid-col"><a href="{link}"{link_attr}>{r['run_id']}</a></td>
        </tr>
"""

    # Filter buttons
    pair_btns = ''.join(f'<button class="filter-btn" data-filter="pair" data-value="{p}">{p}</button>' for p in pairs)
    tf_btns = ''.join(f'<button class="filter-btn" data-filter="tf" data-value="{t}">{t}</button>' for t in timeframes)
    rating_btns = ''.join(
        f'<button class="filter-btn" data-filter="rating" data-value="{r}">{r}</button>'
        for r in ['GREEN', 'YELLOW', 'RED']
    )
    status_btns = ''.join(
        f'<button class="filter-btn" data-filter="status" data-value="{s}">{s}</button>'
        for s in ['COMPLETE', 'PARTIAL', 'FAILED']
    )

    # Summary stats
    total = len(runs)
    green = sum(1 for r in runs if r['rating'] == 'GREEN')
    yellow = sum(1 for r in runs if r['rating'] == 'YELLOW' and r['score'] > 0)
    red = sum(1 for r in runs if r['status'] == 'COMPLETE' and r['score'] == 0)
    failed = sum(1 for r in runs if r['status'] == 'FAILED')
    partial = sum(1 for r in runs if r['status'] == 'PARTIAL')
    best_run = max(runs, key=lambda r: r['score']) if runs else None
    best_str = f'{best_run["pair"]} {best_run["timeframe"]} - {best_run["score"]:.0f}' if best_run and best_run['score'] > 0 else 'None'

    # Serialize comparison data for embedding
    compare_data_json = json.dumps(compare_data or {}, separators=(',', ':'))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Leaderboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', system-ui, sans-serif;
            background: #0a0e1a;
            color: #e2e8f0;
            line-height: 1.5;
            min-height: 100vh;
            padding-bottom: 70px;
        }}

        .page-header {{
            padding: 24px 32px;
            border-bottom: 1px solid #1e293b;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
        }}

        .page-title {{
            font-size: 1.3rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}

        .page-subtitle {{
            font-size: 0.78rem;
            color: #5a6a80;
        }}

        .summary-strip {{
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
        }}

        .summary-item {{
            text-align: center;
        }}

        .summary-num {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.4rem;
            font-weight: 700;
        }}

        .summary-label {{
            font-size: 0.68rem;
            color: #5a6a80;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        /* Filters */
        .filter-bar {{
            padding: 12px 32px;
            border-bottom: 1px solid #1e293b;
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
            background: #0d1320;
        }}

        .filter-group {{
            display: flex;
            align-items: center;
            gap: 4px;
        }}

        .filter-label {{
            font-size: 0.7rem;
            color: #5a6a80;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-right: 4px;
        }}

        .filter-btn {{
            background: #111827;
            border: 1px solid #1e293b;
            color: #8896ab;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            cursor: pointer;
            transition: all 0.15s;
        }}

        .filter-btn:hover {{ border-color: #3b82f6; color: #e2e8f0; }}
        .filter-btn.active {{ background: #1e3a5f; border-color: #3b82f6; color: #3b82f6; }}

        .filter-btn.clear-btn {{
            background: transparent;
            border-color: transparent;
            color: #5a6a80;
            font-family: 'Inter', sans-serif;
        }}
        .filter-btn.clear-btn:hover {{ color: #ef4444; }}

        /* Table */
        .table-container {{
            padding: 0 16px;
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
            margin-top: 8px;
        }}

        th {{
            padding: 10px 10px;
            text-align: left;
            font-weight: 500;
            color: #5a6a80;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-size: 0.68rem;
            border-bottom: 1px solid #1e293b;
            cursor: pointer;
            user-select: none;
            white-space: nowrap;
            position: sticky;
            top: 0;
            background: #0a0e1a;
            z-index: 10;
        }}

        th:hover {{ color: #8896ab; }}
        th.sort-asc::after {{ content: ' \\25B2'; font-size: 0.6rem; }}
        th.sort-desc::after {{ content: ' \\25BC'; font-size: 0.6rem; }}
        th.no-sort {{ cursor: default; }}
        th.no-sort:hover {{ color: #5a6a80; }}

        td {{
            padding: 8px 10px;
            border-bottom: 1px solid rgba(30,41,59,0.4);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            white-space: nowrap;
        }}

        tr:hover td {{ background: rgba(255,255,255,0.02); }}
        tr.hidden {{ display: none; }}

        .score-col {{ font-size: 1rem; }}
        .strategy-col {{ font-family: 'Inter', sans-serif; color: #8896ab; font-size: 0.75rem; }}
        .desc-col {{ font-family: 'Inter', sans-serif; color: #6b7f99; font-size: 0.7rem; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .runid-col {{ font-size: 0.68rem; color: #5a6a80; }}

        a {{ color: #3b82f6; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}

        .footer {{
            text-align: center;
            padding: 16px;
            color: #5a6a80;
            font-size: 0.7rem;
            border-top: 1px solid #1e293b;
            margin-top: 20px;
        }}

        /* Checkbox styling */
        .compare-check {{
            width: 16px;
            height: 16px;
            cursor: pointer;
            accent-color: #3b82f6;
        }}

        /* Compare toolbar - fixed bottom */
        .compare-toolbar {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #111827;
            border-top: 2px solid #3b82f6;
            padding: 12px 32px;
            display: flex;
            align-items: center;
            gap: 16px;
            z-index: 1000;
            transform: translateY(100%);
            transition: transform 0.25s ease;
        }}

        .compare-toolbar.visible {{
            transform: translateY(0);
        }}

        .compare-toolbar .count {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: #3b82f6;
            font-weight: 600;
        }}

        .compare-toolbar button {{
            padding: 8px 20px;
            border-radius: 6px;
            font-family: 'Inter', sans-serif;
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
            border: none;
        }}

        .compare-toolbar .btn-compare {{
            background: #3b82f6;
            color: white;
        }}
        .compare-toolbar .btn-compare:hover {{ background: #2563eb; }}

        .compare-toolbar .btn-clear {{
            background: transparent;
            color: #5a6a80;
            border: 1px solid #1e293b;
        }}
        .compare-toolbar .btn-clear:hover {{ color: #ef4444; border-color: #ef4444; }}

        /* Comparison section */
        #compare-section {{
            display: none;
            padding: 24px 32px;
            border-top: 2px solid #3b82f6;
            margin-top: 16px;
        }}

        #compare-section.visible {{
            display: block;
        }}

        .compare-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }}

        .compare-title {{
            font-size: 1.1rem;
            font-weight: 700;
            color: #3b82f6;
        }}

        .compare-close {{
            background: transparent;
            border: 1px solid #1e293b;
            color: #5a6a80;
            padding: 6px 14px;
            border-radius: 4px;
            cursor: pointer;
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
        }}
        .compare-close:hover {{ color: #ef4444; border-color: #ef4444; }}

        .compare-chart-panel {{
            background: #111827;
            border: 1px solid #1e293b;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
        }}

        .compare-chart-header {{
            padding: 12px 16px;
            border-bottom: 1px solid #1e293b;
            font-size: 0.82rem;
            font-weight: 600;
            color: #8896ab;
        }}

        .compare-loading {{
            text-align: center;
            padding: 40px;
            color: #5a6a80;
            font-size: 0.85rem;
        }}

        /* Metrics comparison table */
        .compare-metrics-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
        }}

        .compare-metrics-table th {{
            position: static;
            padding: 10px 14px;
            text-align: left;
            font-weight: 600;
            color: #5a6a80;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            border-bottom: 1px solid #1e293b;
            background: #0d1320;
            cursor: default;
        }}
        .compare-metrics-table th:hover {{ color: #5a6a80; }}

        .compare-metrics-table td {{
            padding: 8px 14px;
            border-bottom: 1px solid rgba(30,41,59,0.4);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
        }}

        .compare-metrics-table .metric-label-col {{
            color: #8896ab;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            font-size: 0.78rem;
            width: 140px;
        }}

        .compare-metrics-table .best-val {{
            color: #10b981;
            font-weight: 600;
        }}
    </style>
</head>
<body>

<div class="page-header">
    <div>
        <div class="page-title">Pipeline Leaderboard</div>
        <div class="page-subtitle">All pipeline runs &middot; Updated {now}</div>
    </div>
    <div class="summary-strip">
        <div class="summary-item">
            <div class="summary-num">{total}</div>
            <div class="summary-label">Total Runs</div>
        </div>
        <div class="summary-item">
            <div class="summary-num" style="color:#10b981">{green}</div>
            <div class="summary-label">Green</div>
        </div>
        <div class="summary-item">
            <div class="summary-num" style="color:#f59e0b">{yellow}</div>
            <div class="summary-label">Yellow</div>
        </div>
        <div class="summary-item">
            <div class="summary-num" style="color:#ef4444">{red + failed}</div>
            <div class="summary-label">Red/Failed</div>
        </div>
        <div class="summary-item">
            <div class="summary-num" style="color:#5a6a80">{partial}</div>
            <div class="summary-label">Partial</div>
        </div>
        <div class="summary-item">
            <div class="summary-num" style="color:#3b82f6">{best_str}</div>
            <div class="summary-label">Best Run</div>
        </div>
    </div>
</div>

<div class="filter-bar">
    <div class="filter-group">
        <span class="filter-label">Pair</span>
        <button class="filter-btn active" data-filter="pair" data-value="ALL">All</button>
        {pair_btns}
    </div>
    <div class="filter-group">
        <span class="filter-label">TF</span>
        <button class="filter-btn active" data-filter="tf" data-value="ALL">All</button>
        {tf_btns}
    </div>
    <div class="filter-group">
        <span class="filter-label">Rating</span>
        <button class="filter-btn active" data-filter="rating" data-value="ALL">All</button>
        {rating_btns}
    </div>
    <div class="filter-group">
        <span class="filter-label">Status</span>
        <button class="filter-btn active" data-filter="status" data-value="ALL">All</button>
        {status_btns}
    </div>
</div>

<div class="table-container">
    <table id="leaderboard">
        <thead>
            <tr>
                <th class="no-sort" style="width:36px;cursor:default">&nbsp;</th>
                <th data-col="1">Pair</th>
                <th data-col="2">TF</th>
                <th data-col="3">Strategy</th>
                <th data-col="4">Description</th>
                <th data-col="5">Score</th>
                <th data-col="6">Rating</th>
                <th data-col="7">Status</th>
                <th data-col="8">Date</th>
                <th data-col="9">Profit</th>
                <th data-col="10">Back Sh</th>
                <th data-col="11">Fwd Sh</th>
                <th data-col="12">Back Tr</th>
                <th data-col="13">Fwd Tr</th>
                <th data-col="14">Max DD</th>
                <th data-col="15">F/B Ratio</th>
                <th data-col="16">Win %</th>
                <th data-col="17">Time</th>
                <th data-col="18">Run ID</th>
            </tr>
        </thead>
        <tbody>
{rows_html}
        </tbody>
    </table>
</div>

<!-- ===== COMPARISON SECTION ===== -->
<div id="compare-section">
    <div class="compare-header">
        <div class="compare-title">Strategy Comparison</div>
        <button class="compare-close" onclick="closeComparison()">Close</button>
    </div>
    <div class="compare-chart-panel">
        <div class="compare-chart-header">Equity Curves Overlay</div>
        <div id="compare-chart" style="height:450px"></div>
    </div>
    <div class="compare-chart-panel">
        <div class="compare-chart-header">Metrics Comparison</div>
        <div id="compare-metrics" style="padding:0"></div>
    </div>
</div>

<!-- ===== COMPARE TOOLBAR ===== -->
<div class="compare-toolbar" id="compare-toolbar">
    <span class="count"><span id="compare-count">0</span> selected</span>
    <button class="btn-compare" onclick="runComparison()">Compare</button>
    <button class="btn-clear" onclick="clearSelection()">Clear</button>
</div>

<div class="footer">
    OANDA Trading System &middot; {total} pipeline runs &middot; Generated {now}
</div>

<script>
var COMPARE_DATA = {compare_data_json};
(function() {{
    var COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

    // ===== Sorting =====
    var table = document.getElementById('leaderboard');
    var headers = table.querySelectorAll('th[data-col]');
    var currentSort = {{ col: -1, dir: 'desc' }};

    headers.forEach(function(th) {{
        th.addEventListener('click', function() {{
            var col = parseInt(th.dataset.col);
            var dir = (currentSort.col === col && currentSort.dir === 'desc') ? 'asc' : 'desc';
            currentSort = {{ col: col, dir: dir }};

            table.querySelectorAll('th').forEach(function(h) {{ h.classList.remove('sort-asc', 'sort-desc'); }});
            th.classList.add('sort-' + dir);

            var tbody = table.querySelector('tbody');
            var rows = Array.from(tbody.querySelectorAll('tr'));

            rows.sort(function(a, b) {{
                var aText = a.cells[col] ? a.cells[col].textContent.replace(/[$,%m]/g, '').trim() : '';
                var bText = b.cells[col] ? b.cells[col].textContent.replace(/[$,%m]/g, '').trim() : '';

                // ISO date detection (e.g. 2026-02-13T15:42:33)
                if (/^\d{{4}}-\d{{2}}-\d{{2}}T/.test(aText) && /^\d{{4}}-\d{{2}}-\d{{2}}T/.test(bText)) {{
                    return dir === 'asc' ? aText.localeCompare(bText) : bText.localeCompare(aText);
                }}

                var aNum = parseFloat(aText);
                var bNum = parseFloat(bText);

                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return dir === 'asc' ? aNum - bNum : bNum - aNum;
                }}
                return dir === 'asc' ? aText.localeCompare(bText) : bText.localeCompare(aText);
            }});

            rows.forEach(function(r) {{ tbody.appendChild(r); }});
        }});
    }});

    // ===== Filtering =====
    var activeFilters = {{ pair: 'ALL', tf: 'ALL', rating: 'ALL', status: 'ALL' }};

    document.querySelectorAll('.filter-btn').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
            var filterType = btn.dataset.filter;
            var value = btn.dataset.value;

            btn.parentElement.querySelectorAll('.filter-btn').forEach(function(b) {{
                b.classList.remove('active');
            }});
            btn.classList.add('active');

            activeFilters[filterType] = value;
            applyFilters();
        }});
    }});

    function applyFilters() {{
        var rows = table.querySelectorAll('tbody tr');
        rows.forEach(function(row) {{
            var show = true;
            if (activeFilters.pair !== 'ALL' && row.dataset.pair !== activeFilters.pair) show = false;
            if (activeFilters.tf !== 'ALL' && row.dataset.tf !== activeFilters.tf) show = false;
            if (activeFilters.rating !== 'ALL' && row.dataset.rating !== activeFilters.rating) show = false;
            if (activeFilters.status !== 'ALL' && row.dataset.status !== activeFilters.status) show = false;

            row.classList.toggle('hidden', !show);
        }});
    }}

    // ===== Checkbox comparison =====
    var toolbar = document.getElementById('compare-toolbar');
    var countEl = document.getElementById('compare-count');

    function updateToolbar() {{
        var checked = document.querySelectorAll('.compare-check:checked');
        var n = checked.length;
        countEl.textContent = n;
        toolbar.classList.toggle('visible', n >= 2);
    }}

    document.querySelectorAll('.compare-check').forEach(function(cb) {{
        cb.addEventListener('change', updateToolbar);
    }});

    // Expose to global scope for onclick handlers
    window.clearSelection = function() {{
        document.querySelectorAll('.compare-check:checked').forEach(function(cb) {{
            cb.checked = false;
        }});
        updateToolbar();
    }};

    window.closeComparison = function() {{
        document.getElementById('compare-section').classList.remove('visible');
    }};

    window.runComparison = function() {{
        var checked = document.querySelectorAll('.compare-check:checked');
        var runIds = Array.from(checked).map(function(cb) {{ return cb.dataset.run; }});

        if (runIds.length < 2) return;

        var section = document.getElementById('compare-section');
        section.classList.add('visible');

        var chartDiv = document.getElementById('compare-chart');
        var metricsDiv = document.getElementById('compare-metrics');
        metricsDiv.innerHTML = '';

        // Look up embedded data
        var allData = [];
        for (var i = 0; i < runIds.length && i < 6; i++) {{
            var d = COMPARE_DATA[runIds[i]];
            if (d) allData.push({{ runId: runIds[i], data: d }});
        }}

        if (allData.length < 2) {{
            chartDiv.innerHTML = '<div class="compare-loading">Not enough runs with equity data (need 2+).</div>';
            return;
        }}

        buildEquityOverlay(chartDiv, allData);
        buildMetricsTable(metricsDiv, allData);

        section.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }};

    function buildEquityOverlay(chartDiv, allData) {{
        var traces = [];
        var annotations = [];
        var shapes = [];

        for (var i = 0; i < allData.length; i++) {{
            var d = allData[i].data;
            var color = COLORS[i % COLORS.length];
            var meta = d.meta || {{}};
            var score = (d.decision || {{}}).score || 0;
            var backR2 = d.back_r_squared || 0;
            var fwdR2 = d.forward_r_squared || 0;
            var label = meta.strategy + ' (' + score.toFixed(0) + ')';

            var backEquity = d.back_equity || [];
            var fwdEquity = d.forward_equity || [];
            var initialCap = d.initial_capital || 10000;

            // Back equity trace (solid)
            if (backEquity.length > 0) {{
                var bx = backEquity.map(function(p) {{ return p.timestamp; }});
                var by = backEquity.map(function(p) {{ return p.equity - initialCap; }});

                traces.push({{
                    x: bx, y: by,
                    type: 'scatter', mode: 'lines',
                    name: label + ' Back R\\u00B2=' + backR2.toFixed(3),
                    line: {{ color: color, width: 2 }},
                    legendgroup: 'run' + i,
                }});

                // Vertical divider at back/forward boundary
                if (fwdEquity.length > 0) {{
                    var splitTime = bx[bx.length - 1];
                    shapes.push({{
                        type: 'line',
                        x0: splitTime, x1: splitTime,
                        y0: 0, y1: 1,
                        yref: 'paper',
                        line: {{ color: color, width: 1, dash: 'dot' }},
                        opacity: 0.4,
                    }});
                }}
            }}

            // Forward equity trace (dashed)
            if (fwdEquity.length > 0) {{
                var fx = fwdEquity.map(function(p) {{ return p.timestamp; }});
                var fy = fwdEquity.map(function(p) {{ return p.equity - initialCap; }});

                traces.push({{
                    x: fx, y: fy,
                    type: 'scatter', mode: 'lines',
                    name: label + ' Fwd R\\u00B2=' + fwdR2.toFixed(3),
                    line: {{ color: color, width: 2, dash: 'dash' }},
                    legendgroup: 'run' + i,
                }});

                // R2 annotation at curve end
                annotations.push({{
                    x: fx[fx.length - 1],
                    y: fy[fy.length - 1],
                    text: 'R\\u00B2=' + fwdR2.toFixed(3),
                    showarrow: true,
                    arrowhead: 0,
                    arrowcolor: color,
                    ax: 40,
                    ay: -20 - (i * 18),
                    font: {{ color: color, size: 11, family: 'JetBrains Mono, monospace' }},
                    bgcolor: 'rgba(10,14,26,0.8)',
                    bordercolor: color,
                    borderwidth: 1,
                    borderpad: 3,
                }});
            }}
        }}

        var layout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#8896ab', family: 'Inter, system-ui, sans-serif', size: 12 }},
            xaxis: {{
                gridcolor: 'rgba(42,58,82,0.4)',
                zerolinecolor: '#2a3a52',
                tickfont: {{ color: '#5a6a80' }},
            }},
            yaxis: {{
                title: 'Profit ($)',
                gridcolor: 'rgba(42,58,82,0.4)',
                zerolinecolor: '#2a3a52',
                tickfont: {{ color: '#5a6a80' }},
            }},
            margin: {{ l: 60, r: 30, t: 30, b: 50 }},
            legend: {{
                orientation: 'h',
                y: -0.15,
                x: 0.5,
                xanchor: 'center',
                font: {{ size: 11 }},
            }},
            hoverlabel: {{
                bgcolor: '#111827',
                bordercolor: '#2a3a52',
                font: {{ color: '#e2e8f0', size: 12 }},
            }},
            shapes: shapes,
            annotations: annotations,
        }};

        Plotly.newPlot(chartDiv, traces, layout, {{ responsive: true, displayModeBar: false }});
    }}

    function buildMetricsTable(container, allData) {{
        var metrics = [
            {{ key: 'score', label: 'Score', fmt: function(v) {{ return v.toFixed(1); }}, higher: true }},
            {{ key: 'back_return', label: 'Back Return %', fmt: function(v) {{ return v.toFixed(1) + '%'; }}, higher: true }},
            {{ key: 'forward_return', label: 'Fwd Return %', fmt: function(v) {{ return v.toFixed(1) + '%'; }}, higher: true }},
            {{ key: 'back_trades', label: 'Back Trades', fmt: function(v) {{ return v.toFixed(0); }}, higher: true }},
            {{ key: 'forward_trades', label: 'Fwd Trades', fmt: function(v) {{ return v.toFixed(0); }}, higher: true }},
            {{ key: 'win_rate', label: 'Win Rate', fmt: function(v) {{ return (v * 100).toFixed(1) + '%'; }}, higher: true }},
            {{ key: 'profit_factor', label: 'Profit Factor', fmt: function(v) {{ return v.toFixed(2); }}, higher: true }},
            {{ key: 'back_sharpe', label: 'Back Sharpe', fmt: function(v) {{ return v.toFixed(2); }}, higher: true }},
            {{ key: 'forward_sharpe', label: 'Fwd Sharpe', fmt: function(v) {{ return v.toFixed(2); }}, higher: true }},
            {{ key: 'back_r2', label: 'Back R\\u00B2', fmt: function(v) {{ return v.toFixed(3); }}, higher: true }},
            {{ key: 'forward_r2', label: 'Fwd R\\u00B2', fmt: function(v) {{ return v.toFixed(3); }}, higher: true }},
            {{ key: 'back_max_dd', label: 'Max DD %', fmt: function(v) {{ return v.toFixed(1) + '%'; }}, higher: false }},
            {{ key: 'stability', label: 'Stability %', fmt: function(v) {{ return v.toFixed(1) + '%'; }}, higher: true }},
        ];

        // Extract values per run (flat embedded structure)
        var runMetrics = allData.map(function(item, idx) {{
            var d = item.data;
            var meta = d.meta || {{}};
            return {{
                label: meta.strategy + ' (#' + (d.best_rank || '?') + ')',
                color: COLORS[idx % COLORS.length],
                score: (d.decision || {{}}).score || 0,
                back_return: d.back_return || 0,
                forward_return: d.forward_return || 0,
                back_trades: d.back_trades || 0,
                forward_trades: d.forward_trades || 0,
                win_rate: d.win_rate || 0,
                profit_factor: d.profit_factor || 0,
                back_sharpe: d.back_sharpe || 0,
                forward_sharpe: d.forward_sharpe || 0,
                back_r2: d.back_r_squared || 0,
                forward_r2: d.forward_r_squared || 0,
                back_max_dd: d.back_max_dd || 0,
                stability: (d.stability_mean || 0) * 100,
            }};
        }});

        // Build table HTML
        var headerCells = '<th class="metric-label-col">Metric</th>';
        runMetrics.forEach(function(rm) {{
            headerCells += '<th style="color:' + rm.color + '">' + rm.label + '</th>';
        }});

        var bodyRows = '';
        metrics.forEach(function(m) {{
            var vals = runMetrics.map(function(rm) {{ return rm[m.key]; }});
            var bestVal = m.higher
                ? Math.max.apply(null, vals)
                : Math.min.apply(null, vals);

            bodyRows += '<tr><td class="metric-label-col">' + m.label + '</td>';
            runMetrics.forEach(function(rm) {{
                var v = rm[m.key];
                var cls = (v === bestVal && runMetrics.length > 1) ? ' class="best-val"' : '';
                bodyRows += '<td' + cls + '>' + m.fmt(v) + '</td>';
            }});
            bodyRows += '</tr>';
        }});

        container.innerHTML = '<table class="compare-metrics-table"><thead><tr>' +
            headerCells + '</tr></thead><tbody>' + bodyRows + '</tbody></table>';
    }}

    // Default sort: score descending (col 5 now)
    var scoreHeader = table.querySelector('th[data-col="5"]');
    if (scoreHeader) scoreHeader.click();
}})();
</script>

</body>
</html>"""


def inject_backlinks(pipelines_dir: str):
    """Inject breadcrumb back-link into existing report.html files."""
    breadcrumb_html = '<div class="breadcrumb" style="padding:8px 32px;border-bottom:1px solid #1e293b;background:#0d1320"><a href="../index.html" style="color:#3b82f6;text-decoration:none;font-size:0.8rem;font-weight:500">&larr; Back to Leaderboard</a></div>'

    count = 0
    for run_name in sorted(os.listdir(pipelines_dir)):
        report_path = os.path.join(pipelines_dir, run_name, 'report.html')
        if not os.path.isfile(report_path):
            continue

        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                html = f.read()
        except OSError:
            continue

        # Skip if already has breadcrumb
        if 'class="breadcrumb"' in html:
            continue

        # Insert after <body> tag
        new_html = html.replace('<body>', '<body>\n' + breadcrumb_html, 1)
        if new_html == html:
            continue

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(new_html)
        count += 1
        print(f"  Injected backlink: {run_name}/report.html")

    return count


def main():
    parser = argparse.ArgumentParser(description='Generate pipeline leaderboard index')
    parser.add_argument('--output', '-o', default=None,
                        help='Output path (default: results/pipelines/index.html)')
    parser.add_argument('--pipelines-dir', default=None,
                        help='Pipelines directory (default: results/pipelines)')
    parser.add_argument('--inject-backlink', action='store_true',
                        help='Inject back-to-leaderboard link into existing report.html files')
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    pipelines_dir = args.pipelines_dir or str(project_root / 'results' / 'pipelines')
    output_path = args.output or os.path.join(pipelines_dir, 'index.html')

    if not os.path.exists(pipelines_dir):
        print(f"No pipelines directory found at {pipelines_dir}")
        return

    runs = collect_runs(pipelines_dir)
    print(f"Found {len(runs)} pipeline runs")

    compare_data = collect_compare_data(pipelines_dir, runs)
    print(f"Loaded comparison data for {len(compare_data)} runs")

    html = build_index_html(runs, compare_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Leaderboard saved to: {output_path}")

    if args.inject_backlink:
        count = inject_backlinks(pipelines_dir)
        print(f"Injected backlinks into {count} existing reports")

    print(f"Open: file://{os.path.abspath(output_path)}")


if __name__ == '__main__':
    main()
