"""
Thin wrapper over NSSM subprocess calls for service control.

All commands have 10s timeout. Service name convention: OandaTrader_{instance_id}
"""
import re
import subprocess
from pathlib import Path
from typing import Tuple


def _run_nssm(nssm_path: Path, *args, timeout: int = 10) -> subprocess.CompletedProcess:
    """Run an NSSM command with timeout."""
    cmd = [str(nssm_path)] + list(args)
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, returncode=-1, stdout="", stderr="Timeout")
    except FileNotFoundError:
        return subprocess.CompletedProcess(cmd, returncode=-1, stdout="", stderr="NSSM not found")


def _svc_name(instance_id: str) -> str:
    if not re.match(r'^[a-zA-Z0-9_-]+$', instance_id):
        raise ValueError(f"Invalid instance ID: {instance_id!r}")
    return f"OandaTrader_{instance_id}"


def get_service_status(nssm_path: Path, instance_id: str) -> str:
    """Get service status string."""
    result = _run_nssm(nssm_path, "status", _svc_name(instance_id))
    if result.returncode == 3:
        return "not_installed"
    stdout = result.stdout.strip()
    if "SERVICE_RUNNING" in stdout:
        return "SERVICE_RUNNING"
    if "SERVICE_STOPPED" in stdout:
        return "SERVICE_STOPPED"
    if "SERVICE_PAUSED" in stdout:
        return "SERVICE_PAUSED"
    return stdout or "unknown"


def start_service(nssm_path: Path, instance_id: str) -> Tuple[bool, str]:
    """Start a service. Returns (success, message)."""
    result = _run_nssm(nssm_path, "start", _svc_name(instance_id))
    if result.returncode == 0:
        return True, "Service started"
    return False, result.stderr.strip() or result.stdout.strip() or "Start failed"


def stop_service(nssm_path: Path, instance_id: str) -> Tuple[bool, str]:
    """Stop a service. Returns (success, message)."""
    result = _run_nssm(nssm_path, "stop", _svc_name(instance_id))
    if result.returncode == 0:
        return True, "Service stopped"
    return False, result.stderr.strip() or result.stdout.strip() or "Stop failed"


def restart_service(nssm_path: Path, instance_id: str) -> Tuple[bool, str]:
    """Restart a service. Returns (success, message)."""
    result = _run_nssm(nssm_path, "restart", _svc_name(instance_id))
    if result.returncode == 0:
        return True, "Service restarted"
    return False, result.stderr.strip() or result.stdout.strip() or "Restart failed"
