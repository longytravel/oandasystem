"""Dark theme CSS constants for pipeline reports."""

# Color palette
BG_PRIMARY = '#0a0e1a'
BG_CARD = '#111827'
BG_CARD_HOVER = '#1a2237'
BG_INPUT = '#0d1320'
BORDER = '#1e293b'
BORDER_LIGHT = '#2a3a52'

TEXT_PRIMARY = '#e2e8f0'
TEXT_SECONDARY = '#8896ab'
TEXT_MUTED = '#5a6a80'

GREEN = '#10b981'
GREEN_DIM = '#065f46'
RED = '#ef4444'
RED_DIM = '#7f1d1d'
YELLOW = '#f59e0b'
YELLOW_DIM = '#78350f'
BLUE = '#3b82f6'
BLUE_DIM = '#1e3a5f'
PURPLE = '#8b5cf6'
CYAN = '#06b6d4'
ORANGE = '#f97316'

# Rating colors
RATING_COLORS = {
    'GREEN': GREEN,
    'YELLOW': YELLOW,
    'RED': RED,
}

# Plotly chart theme
PLOTLY_LAYOUT = {
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'font': {'color': TEXT_SECONDARY, 'family': 'Inter, system-ui, sans-serif', 'size': 12},
    'xaxis': {
        'gridcolor': 'rgba(42,58,82,0.4)',
        'zerolinecolor': BORDER_LIGHT,
        'tickfont': {'color': TEXT_MUTED},
    },
    'yaxis': {
        'gridcolor': 'rgba(42,58,82,0.4)',
        'zerolinecolor': BORDER_LIGHT,
        'tickfont': {'color': TEXT_MUTED},
    },
    'margin': {'l': 50, 'r': 20, 't': 40, 'b': 40},
    'hoverlabel': {
        'bgcolor': BG_CARD,
        'bordercolor': BORDER_LIGHT,
        'font': {'color': TEXT_PRIMARY, 'size': 12},
    },
}


def get_css() -> str:
    """Return the complete CSS for the report."""
    return f"""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    * {{ margin: 0; padding: 0; box-sizing: border-box; }}

    :root {{
        --bg-primary: {BG_PRIMARY};
        --bg-card: {BG_CARD};
        --bg-card-hover: {BG_CARD_HOVER};
        --bg-input: {BG_INPUT};
        --border: {BORDER};
        --border-light: {BORDER_LIGHT};
        --text-primary: {TEXT_PRIMARY};
        --text-secondary: {TEXT_SECONDARY};
        --text-muted: {TEXT_MUTED};
        --green: {GREEN};
        --green-dim: {GREEN_DIM};
        --red: {RED};
        --red-dim: {RED_DIM};
        --yellow: {YELLOW};
        --blue: {BLUE};
        --purple: {PURPLE};
        --cyan: {CYAN};
        --orange: {ORANGE};
    }}

    html {{ scroll-behavior: smooth; }}

    body {{
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        background: var(--bg-primary);
        color: var(--text-primary);
        line-height: 1.5;
        min-height: 100vh;
    }}

    /* ===== BREADCRUMB ===== */
    .breadcrumb {{
        padding: 8px 32px;
        border-bottom: 1px solid var(--border);
        background: var(--bg-input);
    }}

    .breadcrumb a {{
        color: var(--blue);
        text-decoration: none;
        font-size: 0.8rem;
        font-weight: 500;
    }}

    .breadcrumb a:hover {{
        text-decoration: underline;
    }}

    /* ===== HEADER ===== */
    .report-header {{
        background: linear-gradient(180deg, rgba(17,24,39,0.95) 0%, var(--bg-primary) 100%);
        border-bottom: 1px solid var(--border);
        padding: 24px 32px 0;
        position: sticky;
        top: 0;
        z-index: 100;
        backdrop-filter: blur(12px);
    }}

    .header-top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
    }}

    .header-title {{
        display: flex;
        align-items: center;
        gap: 16px;
    }}

    .pair-badge {{
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }}

    .tf-badge {{
        background: var(--border);
        color: var(--text-secondary);
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
    }}

    .score-pill {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}

    .score-number {{
        font-size: 2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.03em;
    }}

    .score-label {{
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .rating-badge {{
        padding: 5px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.04em;
    }}

    /* ===== TAB NAVIGATION ===== */
    .tab-nav {{
        display: flex;
        gap: 2px;
        overflow-x: auto;
        scrollbar-width: none;
    }}
    .tab-nav::-webkit-scrollbar {{ display: none; }}

    .tab-btn {{
        padding: 10px 18px;
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        color: var(--text-muted);
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        font-weight: 500;
        cursor: pointer;
        white-space: nowrap;
        transition: all 0.15s ease;
    }}

    .tab-btn:hover {{
        color: var(--text-secondary);
        background: rgba(255,255,255,0.02);
    }}

    .tab-btn.active {{
        color: var(--text-primary);
        border-bottom-color: var(--blue);
    }}

    /* ===== TAB CONTENT ===== */
    .tab-content {{
        display: none;
        padding: 28px 32px;
        animation: fadeIn 0.2s ease;
    }}

    .tab-content.active {{ display: block; }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(4px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* ===== METRIC CARDS ===== */
    .metrics-row {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 12px;
        margin-bottom: 24px;
    }}

    .metric-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 14px 16px;
    }}

    .metric-label {{
        font-size: 0.72rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }}

    .metric-value {{
        font-size: 1.3rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
    }}

    .metric-sub {{
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 2px;
    }}

    /* ===== CHART PANELS ===== */
    .chart-panel {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        margin-bottom: 16px;
        overflow: hidden;
    }}

    .chart-panel-header {{
        padding: 12px 16px;
        border-bottom: 1px solid var(--border);
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}

    .chart-panel-body {{
        padding: 8px;
    }}

    .chart-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
    }}

    @media (max-width: 900px) {{
        .chart-grid {{ grid-template-columns: 1fr; }}
    }}

    /* ===== TABLES ===== */
    .data-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.8rem;
    }}

    .data-table th {{
        padding: 10px 12px;
        text-align: left;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 0.7rem;
        border-bottom: 1px solid var(--border);
        cursor: pointer;
        user-select: none;
        background: var(--bg-card);
        position: sticky;
        top: 0;
    }}

    .data-table th:hover {{ color: var(--text-secondary); }}

    .data-table td {{
        padding: 8px 12px;
        border-bottom: 1px solid rgba(30,41,59,0.5);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }}

    .data-table tr:hover td {{ background: rgba(255,255,255,0.015); }}

    .positive {{ color: var(--green); }}
    .negative {{ color: var(--red); }}

    /* ===== PARAMS GRID ===== */
    .params-section {{
        margin-bottom: 20px;
    }}

    .params-section-title {{
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
        padding-left: 4px;
    }}

    .params-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 6px;
    }}

    .param-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 10px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 5px;
    }}

    .param-key {{
        font-size: 0.78rem;
        color: var(--text-muted);
    }}

    .param-val {{
        font-size: 0.82rem;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-primary);
    }}

    /* ===== CONFIDENCE BAR ===== */
    .conf-bar-container {{
        margin: 16px 0;
    }}

    .conf-bar {{
        display: flex;
        height: 28px;
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 8px;
    }}

    .conf-segment {{
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.68rem;
        font-weight: 600;
        color: white;
        transition: opacity 0.2s;
        cursor: default;
    }}

    .conf-segment:hover {{ opacity: 0.85; }}

    .conf-legend {{
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
    }}

    .conf-legend-item {{
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.72rem;
        color: var(--text-muted);
    }}

    .conf-legend-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }}

    /* ===== STABILITY BARS ===== */
    .stability-bar-container {{
        margin-bottom: 10px;
    }}

    .stability-bar-label {{
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        margin-bottom: 3px;
    }}

    .stability-bar-track {{
        height: 6px;
        background: var(--border);
        border-radius: 3px;
        overflow: hidden;
    }}

    .stability-bar-fill {{
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }}

    /* ===== RISK CARDS ===== */
    .risk-card {{
        display: flex;
        gap: 14px;
        padding: 14px 16px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        margin-bottom: 10px;
    }}

    .risk-icon {{
        width: 36px;
        height: 36px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
    }}

    .risk-title {{
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 2px;
    }}

    .risk-desc {{
        font-size: 0.78rem;
        color: var(--text-muted);
        line-height: 1.4;
    }}

    /* ===== FOOTER ===== */
    .report-footer {{
        text-align: center;
        padding: 20px 32px;
        color: var(--text-muted);
        font-size: 0.72rem;
        border-top: 1px solid var(--border);
        margin-top: 20px;
    }}

    /* ===== SCROLLBAR ===== */
    .table-scroll {{
        max-height: 480px;
        overflow-y: auto;
    }}

    .table-scroll::-webkit-scrollbar {{ width: 6px; }}
    .table-scroll::-webkit-scrollbar-track {{ background: var(--bg-card); }}
    .table-scroll::-webkit-scrollbar-thumb {{ background: var(--border-light); border-radius: 3px; }}
    .table-scroll::-webkit-scrollbar-thumb:hover {{ background: var(--text-muted); }}
    """
