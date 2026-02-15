"""Plotly chart generators for the pipeline report.

All functions return a dict with 'data' and 'layout' keys suitable
for JSON serialization and Plotly.newPlot() in the browser.
"""
import json
import numpy as np
from typing import Dict, Any, List, Optional

from pipeline.report import style


def _base_layout(**overrides) -> Dict:
    """Create a base Plotly layout merged with overrides."""
    layout = {**style.PLOTLY_LAYOUT}
    for k, v in overrides.items():
        if isinstance(v, dict) and k in layout and isinstance(layout[k], dict):
            layout[k] = {**layout[k], **v}
        else:
            layout[k] = v
    return layout


def chart_to_json(data: List[Dict], layout: Dict) -> str:
    """Serialize chart data + layout to JSON for embedding."""
    return json.dumps({'data': data, 'layout': layout}, default=str)


# ═══════════════════════════════════════════════════════════════
#  DASHBOARD TAB
# ═══════════════════════════════════════════════════════════════

def confidence_gauge(score: float, rating: str) -> str:
    """Semicircle gauge for confidence score."""
    color = style.RATING_COLORS.get(rating, style.BLUE)

    data = [{
        'type': 'indicator',
        'mode': 'gauge+number',
        'value': score,
        'gauge': {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': style.TEXT_MUTED},
            'bar': {'color': color, 'thickness': 0.7},
            'bgcolor': style.BG_INPUT,
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': style.RED_DIM},
                {'range': [40, 70], 'color': style.YELLOW_DIM},
                {'range': [70, 100], 'color': style.GREEN_DIM},
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 2},
                'thickness': 0.8,
                'value': score,
            },
        },
        'number': {'font': {'size': 42, 'color': color, 'family': 'JetBrains Mono, monospace'}},
    }]

    layout = _base_layout(
        height=200,
        margin={'l': 30, 'r': 30, 't': 30, 'b': 10},
    )

    return chart_to_json(data, layout)


def mini_equity_sparkline(equity_pts: List[Dict], initial_capital: float = 10000) -> str:
    """Small profit sparkline for the dashboard (from zero)."""
    if not equity_pts:
        return chart_to_json([], _base_layout(height=80))

    y = [pt['equity'] - initial_capital for pt in equity_pts]
    x = list(range(len(y)))

    final_color = style.GREEN if y[-1] >= y[0] else style.RED

    data = [{
        'type': 'scatter',
        'x': x,
        'y': y,
        'mode': 'lines',
        'line': {'color': final_color, 'width': 1.5},
        'fill': 'tozeroy',
        'fillcolor': f'rgba({_hex_to_rgb(final_color)},0.08)',
        'hoverinfo': 'skip',
    }]

    layout = _base_layout(
        height=80,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        xaxis={'visible': False, 'gridcolor': 'rgba(0,0,0,0)'},
        yaxis={'visible': False, 'gridcolor': 'rgba(0,0,0,0)'},
        showlegend=False,
    )

    return chart_to_json(data, layout)


def score_breakdown_bar(confidence: Dict) -> str:
    """Horizontal stacked bar showing confidence score components."""
    components = [
        ('BT Quality', confidence.get('backtest_quality_score', 0),
         confidence.get('weights', {}).get('backtest_quality', 0.15), style.CYAN),
        ('Forward/Back', confidence.get('forward_back_score', 0),
         confidence.get('weights', {}).get('forward_back', 0.15), style.BLUE),
        ('Walk-Forward', confidence.get('walkforward_score', 0),
         confidence.get('weights', {}).get('walkforward', 0.25), style.GREEN),
        ('Stability', confidence.get('stability_score', 0),
         confidence.get('weights', {}).get('stability', 0.15), style.YELLOW),
        ('Monte Carlo', confidence.get('montecarlo_score', 0),
         confidence.get('weights', {}).get('montecarlo', 0.15), style.ORANGE),
        ('Quality Score', confidence.get('sharpe_score', 0),
         confidence.get('weights', {}).get('sharpe', 0.15), style.PURPLE),
    ]

    data = []
    for name, score, weight, color in components:
        weighted = score * weight
        data.append({
            'type': 'bar',
            'y': ['Score'],
            'x': [weighted],
            'name': f'{name} ({weighted:.1f})',
            'orientation': 'h',
            'marker': {'color': color},
            'hovertemplate': f'{name}: {score:.0f}/100 x {weight:.0%} = {weighted:.1f}<extra></extra>',
        })

    layout = _base_layout(
        height=80,
        barmode='stack',
        showlegend=True,
        legend={'orientation': 'h', 'y': -0.5, 'x': 0, 'font': {'size': 10}},
        margin={'l': 0, 'r': 10, 't': 5, 'b': 5},
        xaxis={'range': [0, 100], 'visible': False, 'gridcolor': 'rgba(0,0,0,0)'},
        yaxis={'visible': False, 'gridcolor': 'rgba(0,0,0,0)'},
    )

    return chart_to_json(data, layout)


# ═══════════════════════════════════════════════════════════════
#  PERFORMANCE TAB
# ═══════════════════════════════════════════════════════════════

def equity_curve(back_equity: List[Dict], forward_equity: List[Dict],
                 initial_capital: float) -> str:
    """Dual-section profit curve starting from zero."""
    data = []

    if back_equity:
        back_x = [pt.get('timestamp', pt['trade_num']) for pt in back_equity]
        back_y = [pt['equity'] - initial_capital for pt in back_equity]

        data.append({
            'type': 'scatter',
            'x': back_x,
            'y': back_y,
            'name': 'Backtest',
            'mode': 'lines',
            'line': {'color': style.BLUE, 'width': 1.5},
            'fill': 'tozeroy',
            'fillcolor': f'rgba({_hex_to_rgb(style.BLUE)},0.05)',
            'hovertemplate': 'Profit: $%{y:,.0f}<extra>Backtest</extra>',
        })

    if forward_equity:
        fwd_x = [pt.get('timestamp', pt['trade_num']) for pt in forward_equity]
        fwd_y = [pt['equity'] - initial_capital for pt in forward_equity]

        data.append({
            'type': 'scatter',
            'x': fwd_x,
            'y': fwd_y,
            'name': 'Forward Test',
            'mode': 'lines',
            'line': {'color': style.GREEN, 'width': 2},
            'fill': 'tozeroy',
            'fillcolor': f'rgba({_hex_to_rgb(style.GREEN)},0.08)',
            'hovertemplate': 'Profit: $%{y:,.0f}<extra>Forward</extra>',
        })

    # Divider line between back and forward
    if back_equity and forward_equity:
        div_x = back_equity[-1].get('timestamp', back_equity[-1]['trade_num'])
        all_profit = ([pt['equity'] - initial_capital for pt in back_equity] +
                      [pt['equity'] - initial_capital for pt in forward_equity])
        y_min = min(all_profit) * 1.1 if min(all_profit) < 0 else -10
        y_max = max(all_profit) * 1.1
        data.append({
            'type': 'scatter',
            'x': [div_x, div_x],
            'y': [y_min, y_max],
            'mode': 'lines',
            'line': {'color': style.YELLOW, 'width': 1, 'dash': 'dash'},
            'name': 'Back/Forward Split',
            'hoverinfo': 'skip',
        })

    layout = _base_layout(
        height=360,
        title={'text': 'Profit Curve', 'font': {'size': 13}},
        showlegend=True,
        legend={'orientation': 'h', 'y': 1.08, 'font': {'size': 10}},
        yaxis={'title': 'Profit ($)', 'tickprefix': '$', 'zeroline': True,
               'zerolinecolor': style.BORDER_LIGHT, 'zerolinewidth': 1},
    )

    return chart_to_json(data, layout)


def drawdown_chart(dd_curve: List[Dict]) -> str:
    """Underwater drawdown chart."""
    if not dd_curve:
        return chart_to_json([], _base_layout(height=200))

    x = [pt.get('timestamp', pt['trade_num']) for pt in dd_curve]
    y = [-pt['drawdown'] for pt in dd_curve]  # Negative for underwater

    data = [{
        'type': 'scatter',
        'x': x,
        'y': y,
        'mode': 'lines',
        'fill': 'tozeroy',
        'line': {'color': style.RED, 'width': 1},
        'fillcolor': f'rgba({_hex_to_rgb(style.RED)},0.15)',
        'name': 'Drawdown',
        'hovertemplate': 'DD: %{y:.2f}%<extra></extra>',
    }]

    layout = _base_layout(
        height=200,
        title={'text': 'Drawdown', 'font': {'size': 13}},
        showlegend=False,
        yaxis={'title': 'Drawdown %', 'ticksuffix': '%'},
    )

    return chart_to_json(data, layout)


def monthly_heatmap(monthly_returns: Dict[str, Dict[str, float]]) -> str:
    """Monthly returns heatmap (year rows, month columns)."""
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    if not monthly_returns:
        return chart_to_json([], _base_layout(height=200))

    years = sorted(monthly_returns.keys())

    z = []
    text = []
    for year in years:
        row = []
        text_row = []
        for month in month_names:
            val = monthly_returns.get(year, {}).get(month, None)
            row.append(val if val is not None else 0)
            text_row.append(f'${val:.0f}' if val is not None else '-')
        z.append(row)
        text.append(text_row)

    data = [{
        'type': 'heatmap',
        'x': month_names,
        'y': years,
        'z': z,
        'text': text,
        'texttemplate': '%{text}',
        'textfont': {'size': 10},
        'colorscale': [
            [0, style.RED_DIM],
            [0.5, style.BG_INPUT],
            [1, style.GREEN_DIM],
        ],
        'showscale': False,
        'hovertemplate': '%{y} %{x}: $%{z:.0f}<extra></extra>',
    }]

    layout = _base_layout(
        height=max(120, len(years) * 40 + 60),
        title={'text': 'Monthly P&L', 'font': {'size': 13}},
        margin={'l': 50, 'r': 20, 't': 40, 'b': 30},
        xaxis={'side': 'top', 'tickfont': {'size': 10}},
        yaxis={'autorange': 'reversed'},
    )

    return chart_to_json(data, layout)


def yearly_bars(monthly_returns: Dict[str, Dict[str, float]]) -> str:
    """Yearly total return bars."""
    if not monthly_returns:
        return chart_to_json([], _base_layout(height=200))

    years = sorted(monthly_returns.keys())
    totals = [sum(monthly_returns[y].values()) for y in years]
    colors = [style.GREEN if t >= 0 else style.RED for t in totals]

    data = [{
        'type': 'bar',
        'x': years,
        'y': totals,
        'marker': {'color': colors},
        'hovertemplate': '%{x}: $%{y:.0f}<extra></extra>',
    }]

    layout = _base_layout(
        height=200,
        title={'text': 'Yearly P&L', 'font': {'size': 13}},
        showlegend=False,
        yaxis={'title': 'P&L ($)', 'tickprefix': '$'},
    )

    return chart_to_json(data, layout)


# ═══════════════════════════════════════════════════════════════
#  TRADES TAB
# ═══════════════════════════════════════════════════════════════

def pnl_scatter(trades: List[Dict]) -> str:
    """Scatter plot of individual trade PnLs."""
    if not trades:
        return chart_to_json([], _base_layout(height=300))

    x = list(range(len(trades)))
    y = [t['pnl'] for t in trades]
    colors = [style.GREEN if p > 0 else style.RED for p in y]

    data = [{
        'type': 'bar',
        'x': x,
        'y': y,
        'marker': {'color': colors},
        'hovertemplate': 'Trade #%{x}<br>P&L: $%{y:.2f}<extra></extra>',
    }]

    layout = _base_layout(
        height=300,
        title={'text': 'Trade P&L Distribution', 'font': {'size': 13}},
        showlegend=False,
        xaxis={'title': 'Trade #'},
        yaxis={'title': 'P&L ($)', 'tickprefix': '$'},
    )

    return chart_to_json(data, layout)


def exit_reason_donut(trades: List[Dict]) -> str:
    """Donut chart of exit reasons (win/loss breakdown)."""
    if not trades:
        return chart_to_json([], _base_layout(height=250))

    long_wins = len([t for t in trades if t.get('direction') == 'long' and t['pnl'] > 0])
    long_losses = len([t for t in trades if t.get('direction') == 'long' and t['pnl'] <= 0])
    short_wins = len([t for t in trades if t.get('direction') == 'short' and t['pnl'] > 0])
    short_losses = len([t for t in trades if t.get('direction') == 'short' and t['pnl'] <= 0])

    labels = ['Long Wins', 'Long Losses', 'Short Wins', 'Short Losses']
    values = [long_wins, long_losses, short_wins, short_losses]
    colors = [style.GREEN, style.RED, style.CYAN, style.ORANGE]

    # Filter out zero values
    filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not filtered:
        return chart_to_json([], _base_layout(height=250))

    data = [{
        'type': 'pie',
        'labels': [f[0] for f in filtered],
        'values': [f[1] for f in filtered],
        'hole': 0.55,
        'marker': {'colors': [f[2] for f in filtered]},
        'textinfo': 'label+value',
        'textfont': {'size': 10, 'color': style.TEXT_PRIMARY},
        'hovertemplate': '%{label}: %{value} (%{percent})<extra></extra>',
    }]

    layout = _base_layout(
        height=250,
        title={'text': 'Win/Loss by Direction', 'font': {'size': 13}},
        showlegend=False,
        margin={'l': 10, 'r': 10, 't': 40, 'b': 10},
    )

    return chart_to_json(data, layout)


def hourly_distribution(trades: List[Dict]) -> str:
    """Bar chart of trade count by hour of day."""
    if not trades:
        return chart_to_json([], _base_layout(height=250))

    hours = [0] * 24
    for t in trades:
        ts = t.get('entry_time', '')
        if ts and len(ts) >= 13:
            try:
                h = int(ts[11:13])
                hours[h] += 1
            except ValueError:
                pass

    data = [{
        'type': 'bar',
        'x': list(range(24)),
        'y': hours,
        'marker': {'color': style.BLUE},
        'hovertemplate': 'Hour %{x}: %{y} trades<extra></extra>',
    }]

    layout = _base_layout(
        height=250,
        title={'text': 'Trades by Hour (UTC)', 'font': {'size': 13}},
        showlegend=False,
        xaxis={'title': 'Hour', 'dtick': 2},
        yaxis={'title': 'Count'},
    )

    return chart_to_json(data, layout)


def daily_distribution(trades: List[Dict]) -> str:
    """Bar chart of trade count by day of week."""
    if not trades:
        return chart_to_json([], _base_layout(height=250))

    from datetime import datetime as dt

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    days = [0] * 7
    for t in trades:
        ts = t.get('entry_time', '')
        if ts and len(ts) >= 10:
            try:
                d = dt.fromisoformat(ts.replace('+00:00', '').replace('Z', '')[:19])
                days[d.weekday()] += 1
            except (ValueError, IndexError):
                pass

    data = [{
        'type': 'bar',
        'x': day_names,
        'y': days,
        'marker': {'color': style.PURPLE},
        'hovertemplate': '%{x}: %{y} trades<extra></extra>',
    }]

    layout = _base_layout(
        height=250,
        title={'text': 'Trades by Day of Week', 'font': {'size': 13}},
        showlegend=False,
        yaxis={'title': 'Count'},
    )

    return chart_to_json(data, layout)


# ═══════════════════════════════════════════════════════════════
#  WALK-FORWARD TAB
# ═══════════════════════════════════════════════════════════════

def walkforward_bars(window_results: List[Dict]) -> str:
    """Grouped bar chart: Sharpe per WF window."""
    if not window_results:
        return chart_to_json([], _base_layout(height=300))

    windows = [f"W{wr.get('window', i+1)}" for i, wr in enumerate(window_results)]

    sharpes = [wr.get('sharpe', 0) for wr in window_results]
    returns = [wr.get('return', 0) for wr in window_results]
    passed = [wr.get('passed', False) for wr in window_results]
    colors = [style.GREEN if p else style.RED for p in passed]

    data = [
        {
            'type': 'bar',
            'x': windows,
            'y': sharpes,
            'name': 'Sharpe Ratio',
            'marker': {'color': colors},
            'hovertemplate': '%{x}<br>Sharpe: %{y:.2f}<extra></extra>',
        },
        {
            'type': 'scatter',
            'x': windows,
            'y': returns,
            'name': 'Return %',
            'yaxis': 'y2',
            'mode': 'lines+markers',
            'line': {'color': style.CYAN, 'width': 2},
            'marker': {'size': 6},
            'hovertemplate': '%{x}<br>Return: %{y:.1f}%<extra></extra>',
        },
    ]

    layout = _base_layout(
        height=300,
        title={'text': 'Walk-Forward Window Results', 'font': {'size': 13}},
        showlegend=True,
        legend={'orientation': 'h', 'y': 1.08, 'font': {'size': 10}},
        yaxis={'title': 'Sharpe Ratio'},
        yaxis2={
            'title': 'Return %',
            'overlaying': 'y',
            'side': 'right',
            'ticksuffix': '%',
            'gridcolor': 'rgba(0,0,0,0)',
            'tickfont': {'color': style.CYAN},
            'titlefont': {'color': style.CYAN},
        },
    )

    return chart_to_json(data, layout)


def walkforward_consistency(window_results: List[Dict]) -> str:
    """Horizontal bar: each window's Quality Score vs pass threshold."""
    if not window_results:
        return chart_to_json([], _base_layout(height=250))

    windows = [f"W{wr.get('window', i+1)}" for i, wr in enumerate(window_results)]
    quality_vals = [wr.get('quality_score', 0) for wr in window_results]
    passed = [wr.get('passed', False) for wr in window_results]
    colors = [style.GREEN if p else style.RED for p in passed]

    data = [{
        'type': 'bar',
        'y': windows,
        'x': quality_vals,
        'orientation': 'h',
        'marker': {'color': colors},
        'hovertemplate': '%{y}: Quality = %{x:.2f}<extra></extra>',
    }]

    layout = _base_layout(
        height=max(200, len(windows) * 40 + 60),
        title={'text': 'Quality Score per Window', 'font': {'size': 13}},
        showlegend=False,
        xaxis={'title': 'Quality Score'},
        yaxis={'autorange': 'reversed'},
    )

    return chart_to_json(data, layout)


# ═══════════════════════════════════════════════════════════════
#  MONTE CARLO TAB
# ═══════════════════════════════════════════════════════════════

def mc_return_histogram(raw_returns: List[float], mc_results: Dict) -> str:
    """Return distribution histogram with percentile lines."""
    if not raw_returns:
        return chart_to_json([], _base_layout(height=300))

    pct_5 = mc_results.get('pct_5_return', 0)
    pct_50 = mc_results.get('pct_50_return', 0)
    original = mc_results.get('original_return', 0)

    data = [
        {
            'type': 'histogram',
            'x': raw_returns,
            'nbinsx': min(50, max(10, len(set(raw_returns)) // 5)),
            'marker': {'color': style.BLUE, 'line': {'color': style.BLUE_DIM, 'width': 0.5}},
            'opacity': 0.8,
            'name': 'MC Returns',
            'hovertemplate': 'Return: %{x:.1f}%<br>Count: %{y}<extra></extra>',
        },
    ]

    # Add percentile lines as shapes in layout
    shapes = [
        {'type': 'line', 'x0': pct_5, 'x1': pct_5, 'y0': 0, 'y1': 1,
         'yref': 'paper', 'line': {'color': style.RED, 'width': 2, 'dash': 'dash'}},
        {'type': 'line', 'x0': pct_50, 'x1': pct_50, 'y0': 0, 'y1': 1,
         'yref': 'paper', 'line': {'color': style.YELLOW, 'width': 2, 'dash': 'dash'}},
        {'type': 'line', 'x0': original, 'x1': original, 'y0': 0, 'y1': 1,
         'yref': 'paper', 'line': {'color': style.GREEN, 'width': 2}},
    ]

    annotations = [
        {'x': pct_5, 'y': 1.02, 'yref': 'paper', 'text': f'5th: {pct_5:.1f}%',
         'showarrow': False, 'font': {'color': style.RED, 'size': 10}},
        {'x': original, 'y': 1.06, 'yref': 'paper', 'text': f'Actual: {original:.1f}%',
         'showarrow': False, 'font': {'color': style.GREEN, 'size': 10}},
    ]

    layout = _base_layout(
        height=300,
        title={'text': 'MC Return Distribution', 'font': {'size': 13}},
        showlegend=False,
        shapes=shapes,
        annotations=annotations,
        xaxis={'title': 'Return %', 'ticksuffix': '%'},
        yaxis={'title': 'Frequency'},
    )

    return chart_to_json(data, layout)


def mc_dd_histogram(raw_max_dds: List[float], mc_results: Dict) -> str:
    """Max drawdown distribution histogram."""
    if not raw_max_dds:
        return chart_to_json([], _base_layout(height=300))

    pct_95 = mc_results.get('pct_95_dd', 0)
    original_dd = mc_results.get('original_max_dd', 0)

    data = [{
        'type': 'histogram',
        'x': raw_max_dds,
        'nbinsx': min(50, max(10, len(set(raw_max_dds)) // 5)),
        'marker': {'color': style.ORANGE, 'line': {'color': style.RED_DIM, 'width': 0.5}},
        'opacity': 0.8,
        'name': 'MC Max DDs',
        'hovertemplate': 'Max DD: %{x:.1f}%<br>Count: %{y}<extra></extra>',
    }]

    shapes = [
        {'type': 'line', 'x0': pct_95, 'x1': pct_95, 'y0': 0, 'y1': 1,
         'yref': 'paper', 'line': {'color': style.RED, 'width': 2, 'dash': 'dash'}},
        {'type': 'line', 'x0': original_dd, 'x1': original_dd, 'y0': 0, 'y1': 1,
         'yref': 'paper', 'line': {'color': style.GREEN, 'width': 2}},
    ]

    annotations = [
        {'x': pct_95, 'y': 1.02, 'yref': 'paper', 'text': f'95th: {pct_95:.1f}%',
         'showarrow': False, 'font': {'color': style.RED, 'size': 10}},
        {'x': original_dd, 'y': 1.06, 'yref': 'paper', 'text': f'Actual: {original_dd:.1f}%',
         'showarrow': False, 'font': {'color': style.GREEN, 'size': 10}},
    ]

    layout = _base_layout(
        height=300,
        title={'text': 'MC Max Drawdown Distribution', 'font': {'size': 13}},
        showlegend=False,
        shapes=shapes,
        annotations=annotations,
        xaxis={'title': 'Max Drawdown %', 'ticksuffix': '%'},
        yaxis={'title': 'Frequency'},
    )

    return chart_to_json(data, layout)


# ═══════════════════════════════════════════════════════════════
#  STABILITY TAB
# ═══════════════════════════════════════════════════════════════

def stability_bars(candidate: Dict) -> str:
    """Horizontal bar chart of parameter stability ratios."""
    stability = candidate.get('stability', {})
    per_param = stability.get('per_param', stability.get('params', {}))

    if not per_param:
        return chart_to_json([], _base_layout(height=200))

    params = sorted(per_param.keys())
    ratios = [per_param[p].get('stability_ratio', 0) for p in params]
    colors = [
        style.GREEN if r >= 0.8 else (style.YELLOW if r >= 0.5 else style.RED)
        for r in ratios
    ]

    data = [{
        'type': 'bar',
        'y': params,
        'x': ratios,
        'orientation': 'h',
        'marker': {'color': colors},
        'hovertemplate': '%{y}: %{x:.1%}<extra></extra>',
    }]

    # Threshold line at 0.6
    shapes = [{
        'type': 'line', 'x0': 0.6, 'x1': 0.6, 'y0': -0.5, 'y1': len(params) - 0.5,
        'line': {'color': style.YELLOW, 'width': 1, 'dash': 'dash'},
    }]

    layout = _base_layout(
        height=max(200, len(params) * 28 + 60),
        title={'text': 'Parameter Stability Ratios', 'font': {'size': 13}},
        showlegend=False,
        shapes=shapes,
        xaxis={'title': 'Stability Ratio', 'range': [0, 1.05], 'tickformat': '.0%'},
        yaxis={'autorange': 'reversed'},
        margin={'l': 130, 'r': 20, 't': 40, 'b': 40},
    )

    return chart_to_json(data, layout)


# ═══════════════════════════════════════════════════════════════
#  ML EXIT TAB
# ═══════════════════════════════════════════════════════════════

def ml_feature_importance(ml_data: Dict) -> str:
    """Horizontal bar chart of ML feature importances."""
    importances = ml_data.get('feature_importances', {})

    if not importances:
        return chart_to_json([], _base_layout(height=350))

    # Sort by importance descending
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    feature_names = [item[0] for item in sorted_items]
    importance_vals = [item[1] for item in sorted_items]

    # Reverse for horizontal bar (plotly renders bottom-to-top)
    feature_names = feature_names[::-1]
    importance_vals = importance_vals[::-1]

    data = [{
        'type': 'bar',
        'y': feature_names,
        'x': importance_vals,
        'orientation': 'h',
        'marker': {'color': style.GREEN},
        'hovertemplate': '%{y}: %{x:.3f}<extra></extra>',
    }]

    layout = _base_layout(
        height=max(200, len(feature_names) * 35 + 80),
        title={'text': 'Feature Importance', 'font': {'size': 13}},
        showlegend=False,
        xaxis={'title': 'Importance'},
        yaxis={'autorange': True},
        margin={'l': 160, 'r': 20, 't': 40, 'b': 40},
    )

    return chart_to_json(data, layout)


def ml_training_metrics_per_window(ml_data: Dict) -> str:
    """Line chart showing ML model performance across walk-forward windows."""
    window_metrics = ml_data.get('window_ml_metrics', [])

    if not window_metrics:
        return chart_to_json([], _base_layout(height=300))

    windows = [f"W{m.get('window', i+1)}" for i, m in enumerate(window_metrics)]

    # Extract hold model RMSE and risk model AUC per window
    hold_rmse = []
    risk_auc = []
    n_exit_signals = []
    for m in window_metrics:
        tm = m.get('training_metrics', {})
        hold_rmse.append(tm.get('hold_value', {}).get('val_rmse', 0))
        risk_auc.append(tm.get('adverse_risk', {}).get('val_auc', 0))
        n_exit_signals.append(m.get('n_exit_signals', 0))

    data = [
        {
            'type': 'scatter',
            'x': windows,
            'y': hold_rmse,
            'name': 'Hold RMSE',
            'mode': 'lines+markers',
            'line': {'color': style.BLUE, 'width': 2},
            'marker': {'size': 7},
            'hovertemplate': '%{x}<br>RMSE: %{y:.3f}<extra>Hold Model</extra>',
        },
        {
            'type': 'scatter',
            'x': windows,
            'y': risk_auc,
            'name': 'Risk AUC',
            'yaxis': 'y2',
            'mode': 'lines+markers',
            'line': {'color': style.ORANGE, 'width': 2},
            'marker': {'size': 7},
            'hovertemplate': '%{x}<br>AUC: %{y:.3f}<extra>Risk Model</extra>',
        },
        {
            'type': 'bar',
            'x': windows,
            'y': n_exit_signals,
            'name': 'Exit Signals',
            'yaxis': 'y3',
            'marker': {'color': style.PURPLE, 'opacity': 0.3},
            'hovertemplate': '%{x}<br>Exits: %{y}<extra></extra>',
        },
    ]

    layout = _base_layout(
        height=300,
        title={'text': 'ML Model Performance per Window', 'font': {'size': 13}},
        showlegend=True,
        legend={'orientation': 'h', 'y': 1.15, 'font': {'size': 10}},
        yaxis={'title': 'RMSE', 'titlefont': {'color': style.BLUE},
               'tickfont': {'color': style.BLUE}},
        yaxis2={
            'title': 'AUC',
            'overlaying': 'y',
            'side': 'right',
            'titlefont': {'color': style.ORANGE},
            'tickfont': {'color': style.ORANGE},
            'gridcolor': 'rgba(0,0,0,0)',
            'range': [0, 1.05],
        },
        yaxis3={
            'overlaying': 'y',
            'visible': False,
        },
    )

    return chart_to_json(data, layout)


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#10b981' to '16,185,129'."""
    h = hex_color.lstrip('#')
    return ','.join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))
