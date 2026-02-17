"""
Read instance data from filesystem for the dashboard.

Single data access layer - reads health, positions, config (expectations),
and trade history from instance directories.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class LivePerformance:
    """Computed live trading performance stats."""
    win_rate: float = 0.0
    profit_factor: float = 0.0
    trades_total: int = 0
    trades_per_month: float = 0.0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    first_trade: Optional[str] = None
    last_trade: Optional[str] = None


@dataclass
class InstanceStatus:
    """Everything needed to render one dashboard row."""
    # Strategy config
    id: str = ""
    strategy: str = ""
    pair: str = ""
    timeframe: str = ""
    risk_pct: float = 1.0
    enabled: bool = True
    description: str = ""
    from_run: str = ""

    # Health
    status: str = "unknown"
    heartbeat_age_seconds: float = -1
    positions: int = 0
    errors: int = 0
    last_signal: str = ""

    # Service
    service_status: str = "unknown"

    # P&L from position state
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_wins: int = 0
    daily_losses: int = 0

    # Backtest expectations (from config.json)
    expectations: dict = field(default_factory=dict)

    # Live performance (computed from trade_history)
    live_performance: Optional[LivePerformance] = None

    # Heartbeat threshold (timeframe-aware)
    max_heartbeat_age: float = 300

    # Data availability
    has_health: bool = False
    has_config: bool = False
    has_trades: bool = False

    # Display enrichment (populated by collect_instance)
    strategy_display: str = ""
    strategy_tag: str = ""
    auto_comment: str = ""


def compute_live_performance(trades: list) -> LivePerformance:
    """Compute live performance stats from trade history."""
    perf = LivePerformance()
    if not trades:
        return perf

    perf.trades_total = len(trades)
    wins = sum(1 for t in trades if t.get("realized_pnl", 0) > 0)
    perf.win_rate = wins / len(trades) if trades else 0

    gross_profit = sum(t["realized_pnl"] for t in trades if t.get("realized_pnl", 0) > 0)
    gross_loss = abs(sum(t["realized_pnl"] for t in trades if t.get("realized_pnl", 0) < 0))
    perf.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    pnls = [t.get("realized_pnl", 0) for t in trades]
    perf.avg_pnl = sum(pnls) / len(pnls) if pnls else 0
    perf.total_pnl = sum(pnls)

    # Trades per month
    dates = []
    for t in trades:
        for key in ("exit_time", "entry_time"):
            if t.get(key):
                try:
                    dates.append(datetime.fromisoformat(t[key]))
                except (ValueError, TypeError):
                    pass
                break
    if len(dates) >= 2:
        dates.sort()
        perf.first_trade = dates[0].strftime("%Y-%m-%d %H:%M")
        perf.last_trade = dates[-1].strftime("%Y-%m-%d %H:%M")
        span_days = (dates[-1] - dates[0]).total_seconds() / 86400
        if span_days > 1:
            perf.trades_per_month = len(trades) / (span_days / 30.44)
    elif dates:
        perf.first_trade = dates[0].strftime("%Y-%m-%d %H:%M")
        perf.last_trade = perf.first_trade

    # Max drawdown from cumulative P&L curve
    if pnls:
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        # Express as % of peak (or initial if peak is 0)
        perf.max_drawdown_pct = (max_dd / peak * 100) if peak > 0 else 0

    return perf


def _read_json_safe(path: Path) -> Optional[dict]:
    """Read a JSON file, returning None on any error."""
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError, OSError):
        pass
    return None


def read_trade_history(instance_dir: Path, limit: int = 50) -> list:
    """Read trade history from an instance directory."""
    data = _read_json_safe(instance_dir / "state" / "trade_history.json")
    if isinstance(data, list):
        return data[-limit:] if limit else data
    if isinstance(data, dict):
        trades = data.get("trades", data.get("trade_history", []))
        return trades[-limit:] if limit else trades
    return []


HEARTBEAT_THRESHOLDS = {
    'M1': 180, 'M5': 180, 'M15': 900, 'M30': 600,
    'H1': 5400, 'H4': 18000, 'D': 90000,
}

# Human-readable strategy name mapping
STRATEGY_DISPLAY_NAMES = {
    'rsi_v3': 'RSI Divergence V3',
    'rsi_v1': 'RSI Divergence V1',
    'rsi_v4': 'RSI Divergence V4',
    'rsi_v5': 'RSI Divergence V5',
    'ema_cross_ml': 'EMA Cross V6',
    'fair_price_ma': 'Fair Price MA',
    'rsi_fast': 'RSI Fast',
    'donchian_breakout': 'Donchian Breakout',
    'bollinger_squeeze': 'Bollinger Squeeze',
    'london_breakout': 'London Breakout',
    'stochastic_adx': 'Stochastic ADX',
    'RSI_Divergence_v3': 'RSI Divergence V3',
    'RSI_Divergence_v1': 'RSI Divergence V1',
    'EMA_Cross_v6': 'EMA Cross V6',
    'Fair_Price_MA': 'Fair Price MA',
    'Donchian_Breakout': 'Donchian Breakout',
    'Bollinger_Squeeze': 'Bollinger Squeeze',
    'London_Breakout': 'London Breakout',
    'Stochastic_ADX': 'Stochastic ADX',
}

# Strategy type tags for visual badges
STRATEGY_TAGS = {
    'rsi_v3': ('RSI', 'mean-rev'),
    'rsi_v1': ('RSI', 'mean-rev'),
    'rsi_v4': ('RSI', 'mean-rev'),
    'rsi_v5': ('RSI', 'mean-rev'),
    'ema_cross_ml': ('EMA', 'trend'),
    'fair_price_ma': ('FPMA', 'value'),
    'rsi_fast': ('RSI', 'momentum'),
    'donchian_breakout': ('DCH', 'trend'),
    'bollinger_squeeze': ('BB', 'volatility'),
    'london_breakout': ('LDN', 'session'),
    'stochastic_adx': ('STCH', 'momentum'),
}


def get_strategy_display_name(strategy_id: str) -> str:
    """Get human-readable strategy name."""
    return STRATEGY_DISPLAY_NAMES.get(strategy_id, strategy_id)


def generate_instance_comment(inst: 'InstanceStatus') -> str:
    """Generate a rich description from instance metrics.

    Format: "Score RATING | X.X trades/mo | WR XX% | PF X.XX | DD X.X%"
    Falls back to basic info if no live performance data.
    """
    parts = []

    # Score and rating from expectations/config
    if inst.expectations:
        # Live performance stats (from actual trades)
        lp = inst.live_performance
        if lp and lp.trades_total > 0:
            parts.append(f"{lp.trades_total} trades")
            if lp.trades_per_month > 0:
                parts.append(f"{lp.trades_per_month:.1f}/mo")
            parts.append(f"WR {lp.win_rate * 100:.0f}%")
            if lp.profit_factor < 999:
                parts.append(f"PF {lp.profit_factor:.2f}")
            parts.append(f"PnL {lp.total_pnl:+.2f}")
        else:
            # Fall back to expected metrics
            exp = inst.expectations
            if exp.get('avg_trades_per_month'):
                parts.append(f"exp {exp['avg_trades_per_month']:.1f}/mo")
            if exp.get('win_rate'):
                parts.append(f"WR {exp['win_rate'] * 100:.0f}%")
            if exp.get('profit_factor'):
                parts.append(f"PF {exp['profit_factor']:.2f}")
            if exp.get('max_drawdown_pct'):
                parts.append(f"DD {exp['max_drawdown_pct']:.0f}%")

    if not parts:
        return inst.description or ""

    return " | ".join(parts)


def collect_instance(strategy: dict, instances_dir: Path) -> InstanceStatus:
    """Collect all data for a single instance."""
    sid = strategy["id"]
    tf = strategy.get("timeframe", "")
    inst = InstanceStatus(
        id=sid,
        strategy=strategy.get("strategy", ""),
        pair=strategy.get("pair", ""),
        timeframe=tf,
        risk_pct=strategy.get("risk_pct", 1.0),
        enabled=strategy.get("enabled", True),
        description=strategy.get("description", ""),
        from_run=strategy.get("from_run", ""),
        max_heartbeat_age=HEARTBEAT_THRESHOLDS.get(tf, 300),
    )

    instance_dir = instances_dir / sid

    # Health
    health = _read_json_safe(instance_dir / "health.json")
    if health:
        inst.has_health = True
        inst.status = health.get("status", "unknown")
        inst.positions = health.get("positions", 0)
        inst.errors = health.get("errors", 0)
        inst.last_signal = health.get("last_signal", "")

        ts_str = health.get("timestamp")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                inst.heartbeat_age_seconds = (datetime.now(timezone.utc) - ts).total_seconds()
            except (ValueError, TypeError):
                pass

    # Config (for expectations)
    config = _read_json_safe(instance_dir / "config.json")
    if config:
        inst.has_config = True
        inst.expectations = config.get("expectations", {})

    # Position state (for P&L)
    pos_state = _read_json_safe(instance_dir / "state" / "position_state.json")
    if pos_state:
        inst.unrealized_pnl = pos_state.get("unrealized_pnl", 0)
        daily = pos_state.get("daily_stats") or {}
        inst.daily_pnl = daily.get("gross_profit", 0) + daily.get("gross_loss", 0)
        inst.daily_trades = daily.get("trades_closed", 0)
        inst.daily_wins = daily.get("wins", 0)
        inst.daily_losses = daily.get("losses", 0)

    # Trade history
    all_trades = read_trade_history(instance_dir, limit=0)
    if all_trades:
        inst.has_trades = True
        inst.live_performance = compute_live_performance(all_trades)

    # Enrich with display name and auto-comment
    inst.strategy_display = get_strategy_display_name(inst.strategy)
    inst.strategy_tag = STRATEGY_TAGS.get(inst.strategy, ('', ''))[1]
    inst.auto_comment = generate_instance_comment(inst)

    return inst


def collect_all_instances(strategies_file: Path, instances_dir: Path) -> list:
    """Collect status for all instances defined in strategies.json."""
    data = _read_json_safe(strategies_file)
    if not data:
        return []

    strategies = data.get("strategies", [])
    return [collect_instance(s, instances_dir) for s in strategies]


def read_scan_progress(results_dir: Path) -> dict:
    """Read scan progress from scan_progress.json."""
    return _read_json_safe(results_dir / "scan_progress.json") or {"status": "no_scan", "results": []}


def get_daily_summary(instances: list) -> dict:
    """Aggregate summary across all instances."""
    total_pnl = sum(i.daily_pnl for i in instances)
    total_unrealized = sum(i.unrealized_pnl for i in instances)
    total_trades = sum(i.daily_trades for i in instances)
    open_positions = sum(i.positions for i in instances)
    running = sum(1 for i in instances if i.status == "running" and i.heartbeat_age_seconds < i.max_heartbeat_age)
    errors = sum(1 for i in instances if i.errors > 0 or i.status == "error")

    return {
        "daily_pnl": round(total_pnl, 2),
        "unrealized_pnl": round(total_unrealized, 2),
        "total_trades": total_trades,
        "open_positions": open_positions,
        "running": running,
        "total": len(instances),
        "errors": errors,
    }
