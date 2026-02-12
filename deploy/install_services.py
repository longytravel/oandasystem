#!/usr/bin/env python
"""
Install trading strategies as Windows services via NSSM.

Reads deploy/strategies.json, creates instance dirs, installs services.
Called by setup.bat - not meant to be run directly.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
NSSM = ROOT / "deploy" / "nssm" / "nssm.exe"
STRATEGIES_FILE = ROOT / "deploy" / "strategies.json"
INSTANCES_DIR = ROOT / "instances"


def find_python() -> str:
    """Find the Python executable path."""
    return sys.executable


def run_nssm(*args, check=False) -> subprocess.CompletedProcess:
    """Run an NSSM command."""
    cmd = [str(NSSM)] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True)


def service_exists(name: str) -> bool:
    """Check if a Windows service exists."""
    result = run_nssm("status", name)
    return result.returncode != 3  # 3 = service not found


def wait_for_removal(svc_name: str, timeout: float = 10.0) -> bool:
    """Wait until Windows fully removes the service."""
    start = time.time()
    while time.time() - start < timeout:
        if not service_exists(svc_name):
            return True
        time.sleep(0.5)
    return False


def remove_service(svc_name: str):
    """Stop and remove a service, waiting for Windows to release it."""
    run_nssm("stop", svc_name)
    run_nssm("remove", svc_name, "confirm")
    if not wait_for_removal(svc_name):
        print(f"  [WARN] Service {svc_name} slow to remove, retrying...")
        time.sleep(2)
        run_nssm("remove", svc_name, "confirm")
        wait_for_removal(svc_name, timeout=5.0)


def install_strategy(strategy: dict, python_path: str) -> bool:
    """Install a single strategy as a Windows service."""
    sid = strategy["id"]
    svc_name = f"OandaTrader_{sid}"
    instance_dir = INSTANCES_DIR / sid

    print(f"\n--- {sid} ---")

    if not strategy.get("enabled", True):
        # If disabled but service exists, stop and remove it
        if service_exists(svc_name):
            print(f"  Stopping disabled strategy...")
            remove_service(svc_name)
            print(f"  [OK] Service removed (disabled in strategies.json)")
        else:
            print(f"  [SKIP] Disabled")
        return True

    # Create instance directories
    (instance_dir / "state").mkdir(parents=True, exist_ok=True)
    (instance_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Check for config.json
    config_file = instance_dir / "config.json"
    if not config_file.exists():
        # Try to find it in config/ dir
        pair = strategy["pair"]
        timeframe = strategy["timeframe"]
        source = ROOT / "config" / f"live_{pair}_{timeframe}.json"
        if source.exists():
            shutil.copy2(source, config_file)
            print(f"  [OK] Copied config from {source.name}")
        else:
            print(f"  [ERROR] No config.json found!")
            print(f"    Put it at: {config_file}")
            print(f"    Or export: python scripts\\export_params.py --run RUN_ID --output {config_file}")
            return False

    # Build command args
    script = ROOT / "scripts" / "run_live.py"
    risk = strategy.get("risk_pct", 1.0)
    app_args = (
        f'"{script}" '
        f"--strategy {strategy['strategy']} "
        f"--pair {strategy['pair']} "
        f"--timeframe {strategy['timeframe']} "
        f'--params-file "{config_file}" '
        f'--instance-dir "{instance_dir}" '
        f"--instance-id {sid} "
        f"--risk {risk} "
        f"--yes"
    )

    # Remove existing service if present
    if service_exists(svc_name):
        print(f"  Updating existing service...")
        remove_service(svc_name)

    # Install
    result = run_nssm("install", svc_name, python_path, app_args)
    if result.returncode != 0:
        print(f"  [ERROR] Install failed: {result.stderr}")
        return False

    # Configure
    run_nssm("set", svc_name, "AppDirectory", str(ROOT))
    run_nssm("set", svc_name, "AppStdout", str(instance_dir / "logs" / "stdout.log"))
    run_nssm("set", svc_name, "AppStderr", str(instance_dir / "logs" / "stderr.log"))
    run_nssm("set", svc_name, "AppRotateFiles", "1")
    run_nssm("set", svc_name, "AppRotateBytes", "10485760")  # 10MB
    run_nssm("set", svc_name, "AppRestartDelay", "30000")  # 30s
    run_nssm("set", svc_name, "Start", "SERVICE_AUTO_START")
    desc = f"OANDA {strategy['strategy'].upper()} - {strategy['pair']} {strategy['timeframe']}"
    run_nssm("set", svc_name, "Description", desc)

    # Start
    result = run_nssm("start", svc_name)
    if result.returncode != 0:
        print(f"  [WARN] Start may have failed: {result.stderr.strip()}")
        print(f"    Check logs: type {instance_dir}\\logs\\stderr.log")
    else:
        print(f"  [OK] Service installed and started")

    print(f"    Service:  {svc_name}")
    print(f"    Strategy: {strategy['strategy']} {strategy['pair']} {strategy['timeframe']}")
    print(f"    Risk:     {risk}%")
    print(f"    Logs:     {instance_dir}\\logs\\")

    return True


def install_dashboard(config: dict, python_path: str) -> bool:
    """Install the dashboard as a Windows service."""
    svc_name = "OandaTrader_Dashboard"
    dash_config = config.get("dashboard", {})
    port = dash_config.get("port", 8080)

    print(f"\n--- Dashboard ---")

    script = ROOT / "scripts" / "run_dashboard.py"
    app_args = f'"{script}" --port {port} --host 0.0.0.0'

    # Remove existing service if present
    if service_exists(svc_name):
        print(f"  Updating existing dashboard service...")
        remove_service(svc_name)

    # Install
    result = run_nssm("install", svc_name, python_path, app_args)
    if result.returncode != 0:
        print(f"  [ERROR] Dashboard install failed: {result.stderr}")
        return False

    # Configure
    run_nssm("set", svc_name, "AppDirectory", str(ROOT))
    log_dir = INSTANCES_DIR / "_dashboard" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_nssm("set", svc_name, "AppStdout", str(log_dir / "stdout.log"))
    run_nssm("set", svc_name, "AppStderr", str(log_dir / "stderr.log"))
    run_nssm("set", svc_name, "AppRotateFiles", "1")
    run_nssm("set", svc_name, "AppRotateBytes", "10485760")
    run_nssm("set", svc_name, "AppRestartDelay", "10000")
    run_nssm("set", svc_name, "Start", "SERVICE_AUTO_START")
    run_nssm("set", svc_name, "Description", f"OANDA Trading Dashboard (port {port})")

    # Add firewall rule
    try:
        fw_cmd = [
            "netsh", "advfirewall", "firewall", "add", "rule",
            f"name=OANDA Dashboard (port {port})",
            "dir=in", "action=allow", "protocol=TCP",
            f"localport={port}",
        ]
        subprocess.run(fw_cmd, capture_output=True, text=True, timeout=10)
    except Exception:
        print(f"  [WARN] Could not add firewall rule (run as admin)")

    # Start
    result = run_nssm("start", svc_name)
    if result.returncode != 0:
        print(f"  [WARN] Dashboard start may have failed: {result.stderr.strip()}")
    else:
        print(f"  [OK] Dashboard installed and started")

    print(f"    Service: {svc_name}")
    print(f"    URL:     http://0.0.0.0:{port}")
    print(f"    Logs:    {log_dir}")

    return True


def main():
    if not NSSM.exists():
        print("ERROR: NSSM not found at", NSSM)
        sys.exit(1)

    if not STRATEGIES_FILE.exists():
        print("ERROR: strategies.json not found at", STRATEGIES_FILE)
        sys.exit(1)

    with open(STRATEGIES_FILE) as f:
        config = json.load(f)

    strategies = config.get("strategies", [])
    if not strategies:
        print("No strategies defined in strategies.json")
        sys.exit(1)

    python_path = find_python()
    print(f"Python: {python_path}")
    print(f"Strategies: {len(strategies)}")

    ok = 0
    failed = 0
    skipped = 0

    for s in strategies:
        if not s.get("enabled", True):
            install_strategy(s, python_path)  # handles cleanup
            skipped += 1
        elif install_strategy(s, python_path):
            ok += 1
        else:
            failed += 1

    # Install dashboard
    if config.get("dashboard"):
        install_dashboard(config, python_path)

    print(f"\n{'=' * 40}")
    print(f"Results: {ok} running, {skipped} disabled, {failed} failed")

    if config.get("dashboard"):
        port = config["dashboard"].get("port", 8080)
        print(f"Dashboard: http://YOUR_VPS_IP:{port}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
