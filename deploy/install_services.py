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
            run_nssm("stop", svc_name)
            run_nssm("remove", svc_name, "confirm")
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
        run_nssm("stop", svc_name)
        run_nssm("remove", svc_name, "confirm")

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

    print(f"\n{'=' * 40}")
    print(f"Results: {ok} running, {skipped} disabled, {failed} failed")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
