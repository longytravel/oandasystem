"""
FastAPI dashboard application for monitoring trading instances.

Endpoints:
    GET  /                              Main dashboard page
    GET  /api/status                    JSON status for auto-refresh
    GET  /api/trades/{instance_id}      Recent closed trades
    GET  /api/performance/{instance_id} Live vs backtest comparison
    GET  /api/export/{instance_id}      CSV download of all trades
    POST /api/service/{instance_id}/start    Start service
    POST /api/service/{instance_id}/stop     Stop service
    POST /api/service/{instance_id}/restart  Restart service

No auth required (VPS is private network).
# TODO: Add HTTPBasic or API key auth for POST endpoints if exposed publicly.
"""
import csv
import io
import json
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from dashboard.data_reader import (
    collect_all_instances,
    compute_live_performance,
    get_daily_summary,
    read_scan_progress,
    read_trade_history,
    _read_json_safe,
)
from dashboard.service_control import (
    get_service_status,
    start_service,
    stop_service,
    restart_service,
)

ROOT = Path(__file__).parent.parent
STRATEGIES_FILE = ROOT / "deploy" / "strategies.json"
INSTANCES_DIR = ROOT / "instances"
NSSM = ROOT / "deploy" / "nssm" / "nssm.exe"
TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="OANDA Trading Dashboard")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _load_dashboard_config() -> dict:
    """Load dashboard config from strategies.json."""
    data = _read_json_safe(STRATEGIES_FILE)
    if data:
        return data.get("dashboard", {})
    return {}


def _instance_to_dict(inst) -> dict:
    """Convert InstanceStatus to JSON-safe dict."""
    d = asdict(inst)
    if d.get("live_performance") is None:
        d["live_performance"] = None
    # Format heartbeat age
    age = d.get("heartbeat_age_seconds", -1)
    if age >= 0:
        if age < 60:
            d["heartbeat_display"] = f"{int(age)}s ago"
        elif age < 3600:
            d["heartbeat_display"] = f"{int(age / 60)}m ago"
        else:
            d["heartbeat_display"] = f"{age / 3600:.1f}h ago"
        d["heartbeat_ok"] = age < d.get("max_heartbeat_age", 300)
    else:
        d["heartbeat_display"] = "N/A"
        d["heartbeat_ok"] = False

    # Service status
    if NSSM.exists():
        d["service_status"] = get_service_status(NSSM, inst.id)
    else:
        d["service_status"] = "nssm_not_found"

    return d


def _compare_metric(live_val, expected_val, higher_is_better=True) -> str:
    """Compare live vs expected, return status: ok/warning/alert."""
    if expected_val == 0 or live_val is None or expected_val is None:
        return "ok"
    if higher_is_better:
        ratio = live_val / expected_val
    else:
        ratio = expected_val / live_val if live_val > 0 else 0
    diff = abs(1 - ratio)
    if diff <= 0.10:
        return "ok"
    if diff <= 0.25:
        return "warning"
    return "alert"


# ── HTML Page ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Render the main dashboard page."""
    instances = collect_all_instances(STRATEGIES_FILE, INSTANCES_DIR)
    instance_dicts = [_instance_to_dict(i) for i in instances]
    summary = get_daily_summary(instances)
    dash_config = _load_dashboard_config()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "instances": instance_dicts,
        "summary": summary,
        "refresh_seconds": dash_config.get("refresh_seconds", 15),
    })


# ── JSON API ───────────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    """JSON status for auto-refresh."""
    instances = collect_all_instances(STRATEGIES_FILE, INSTANCES_DIR)
    instance_dicts = [_instance_to_dict(i) for i in instances]
    summary = get_daily_summary(instances)
    return {"instances": instance_dicts, "summary": summary}


@app.get("/api/trades/{instance_id}")
async def api_trades(instance_id: str, limit: int = 50):
    """Recent closed trades for an instance."""
    instance_dir = INSTANCES_DIR / instance_id
    if not instance_dir.exists():
        return JSONResponse({"error": "Instance not found"}, status_code=404)
    trades = read_trade_history(instance_dir, limit=limit)
    # Return newest first
    trades.reverse()
    return {"instance_id": instance_id, "trades": trades, "total": len(trades)}


@app.get("/api/performance/{instance_id}")
async def api_performance(instance_id: str):
    """Live vs backtest comparison data."""
    instance_dir = INSTANCES_DIR / instance_id

    # Load expectations from config
    config = _read_json_safe(instance_dir / "config.json")
    expected = config.get("expectations", {}) if config else {}

    # Compute live stats from trade history
    all_trades = read_trade_history(instance_dir, limit=0)
    live_perf = compute_live_performance(all_trades)

    live = {
        "win_rate": round(live_perf.win_rate, 4),
        "profit_factor": round(live_perf.profit_factor, 3),
        "trades": live_perf.trades_total,
        "trades_per_month": round(live_perf.trades_per_month, 1),
        "avg_pnl": round(live_perf.avg_pnl, 2),
        "total_pnl": round(live_perf.total_pnl, 2),
        "max_drawdown_pct": round(live_perf.max_drawdown_pct, 2),
    }

    comparison = {}
    metrics = [
        ("win_rate", True),
        ("profit_factor", True),
        ("trades_per_month", True),  # higher_is_better set loosely
    ]
    for metric, higher in metrics:
        exp_key = metric if metric != "trades_per_month" else "avg_trades_per_month"
        exp_val = expected.get(exp_key)
        live_val = live.get(metric)
        if exp_val and live_val:
            comparison[f"{metric}_diff"] = round(live_val - exp_val, 4)
            comparison[f"{metric}_status"] = _compare_metric(live_val, exp_val, higher)
        else:
            comparison[f"{metric}_status"] = "no_data"

    return {
        "instance_id": instance_id,
        "live": live,
        "expected": expected,
        "comparison": comparison,
    }


@app.get("/api/export/{instance_id}")
async def api_export(instance_id: str):
    """CSV download of all trades."""
    instance_dir = INSTANCES_DIR / instance_id
    all_trades = read_trade_history(instance_dir, limit=0)

    if not all_trades:
        return JSONResponse({"error": "No trades found"}, status_code=404)

    columns = [
        "trade_id", "instrument", "direction", "units",
        "entry_price", "entry_time", "exit_price", "exit_time",
        "exit_reason", "realized_pnl", "stop_loss", "take_profit",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for trade in all_trades:
        writer.writerow(trade)

    output.seek(0)
    filename = f"trades_{instance_id}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── Scan Progress ─────────────────────────────────────────

@app.get("/api/scan")
async def api_scan():
    """Current/recent scan progress."""
    return read_scan_progress(ROOT / "results")


# ── Service Control ────────────────────────────────────────

@app.post("/api/service/{instance_id}/start")
async def api_service_start(instance_id: str):
    """Start a service."""
    if not NSSM.exists():
        return JSONResponse({"ok": False, "message": "NSSM not found"}, status_code=500)
    ok, msg = start_service(NSSM, instance_id)
    return {"ok": ok, "message": msg}


@app.post("/api/service/{instance_id}/stop")
async def api_service_stop(instance_id: str):
    """Stop a service."""
    if not NSSM.exists():
        return JSONResponse({"ok": False, "message": "NSSM not found"}, status_code=500)
    ok, msg = stop_service(NSSM, instance_id)
    return {"ok": ok, "message": msg}


@app.post("/api/service/{instance_id}/restart")
async def api_service_restart(instance_id: str):
    """Restart a service."""
    if not NSSM.exists():
        return JSONResponse({"ok": False, "message": "NSSM not found"}, status_code=500)
    ok, msg = restart_service(NSSM, instance_id)
    return {"ok": ok, "message": msg}
