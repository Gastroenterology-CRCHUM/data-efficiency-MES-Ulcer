"""Common utilities."""

from __future__ import annotations

import concurrent.futures
import contextlib
import signal
import subprocess
from pathlib import Path

import torch


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    pass


def get_device(device_id: int = 0) -> torch.device:
    """Return a safe device handle. Pass device_id < 0 to force CPU."""
    if device_id < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    if device_id >= torch.cuda.device_count():
        raise ValueError(
            f"GPU {device_id} not available. Available GPUs: {torch.cuda.device_count()}"
        )
    return torch.device(f"cuda:{device_id}")


def get_git_sha(repo_dir: Path | str | None = None, short: bool = False) -> str | None:
    """Return the current git commit SHA (with a ``-dirty`` suffix), or None.

    Args:
        repo_dir: Directory inside the repository (default: this package's repo).
        short:    Return the abbreviated SHA.

    Returns:
        e.g. ``"3f2c1ab...-dirty"``; None when git or the repository is unavailable.
    """
    cwd = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parents[2]
    cmd = ["git", "rev-parse", "--short" if short else "--verify", "HEAD"]
    try:
        sha = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not sha:
        return None
    return f"{sha}-dirty" if dirty else sha


def ignore_sigint() -> None:
    """Process-pool initializer: Ctrl+C is left to the parent, which stops the workers."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def sigterm_as_interrupt() -> None:
    """Raise KeyboardInterrupt on SIGTERM, so ``kill <pid>`` stops like Ctrl+C."""

    def _raise(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _raise)


def _descendants(pid: int) -> list:
    """Child processes of ``pid`` (e.g. DataLoader workers); [] without psutil."""
    try:
        import psutil

        return psutil.Process(pid).children(recursive=True)
    except Exception:  # noqa: BLE001  (psutil missing, process already gone)
        return []


def terminate_pool(executor: concurrent.futures.ProcessPoolExecutor) -> None:
    """Cancel the pending jobs and kill the running ones (no public API before Python 3.14).

    Their children are killed first, so no orphan is left half-started.
    """
    for proc in list((executor._processes or {}).values()):
        for child in _descendants(proc.pid):
            with contextlib.suppress(Exception):
                child.kill()
        proc.kill()
    executor.shutdown(wait=True, cancel_futures=True)
