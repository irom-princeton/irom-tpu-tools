"""Persistent job state stored under ~/.tpu-jobs/<name>/."""

from __future__ import annotations

import json
import os
import signal
from dataclasses import asdict, dataclass
from pathlib import Path
from time import sleep
from typing import Literal


JOBS_DIR = Path.home() / ".tpu-jobs"

TPUVersion = Literal["v4", "v5", "v6"]


@dataclass
class JobConfig:
    """Everything needed to recreate a TPU and relaunch training.

    `repo` is "owner/name" for a repo-backed job, or "" for a bare TPU
    (no clone, no setup_cmd executed inside a repo).
    """

    name: str
    version: TPUVersion
    tpu_num: int
    command: str
    branch: str
    setup_cmd: str
    repo: str = ""
    topology: str | None = None

    def save(self) -> Path:
        d = JOBS_DIR / self.name
        d.mkdir(parents=True, exist_ok=True)
        p = d / "config.json"
        p.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return p

    @staticmethod
    def load(name: str) -> JobConfig:
        p = JOBS_DIR / name / "config.json"
        if not p.exists():
            raise FileNotFoundError(f"No job config for '{name}' at {p}")
        data = json.loads(p.read_text())
        # Drop unknown keys so older/newer configs stay loadable
        allowed = {f for f in JobConfig.__dataclass_fields__}
        data = {k: v for k, v in data.items() if k in allowed}
        return JobConfig(**data)

    @property
    def repo_owner(self) -> str:
        return self.repo.split("/", 1)[0] if "/" in self.repo else ""

    @property
    def repo_name(self) -> str:
        return self.repo.split("/", 1)[1] if "/" in self.repo else ""

    @staticmethod
    def all_names() -> list[str]:
        if not JOBS_DIR.is_dir():
            return []
        return sorted(
            d.name for d in JOBS_DIR.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        )


def _pid_file(name: str) -> Path:
    return JOBS_DIR / name / "watch.pid"


def log_path(name: str) -> Path:
    return JOBS_DIR / name / "watch.log"


def save_pid(name: str, pid: int) -> None:
    _pid_file(name).write_text(str(pid))


def read_pid(name: str) -> int | None:
    p = _pid_file(name)
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except (ValueError, OSError):
        return None


def _pid_is_watcher(pid: int) -> bool:
    """Verify the PID actually belongs to a watcher process, not a recycled PID."""
    try:
        cmdline = (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\x00", b" ")
            .decode(errors="replace")
        )
        return "irom" in cmdline
    except OSError:
        return False


def is_watcher_running(name: str) -> bool:
    pid = read_pid(name)
    if pid is None:
        return False
    try:
        os.kill(pid, 0)  # signal 0 = check existence
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # process exists but we can't read it
    return _pid_is_watcher(pid)


def stop_watcher(name: str) -> bool:
    """Stop the watcher daemon. Returns True if it was running."""
    pid = read_pid(name)
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _pid_file(name).unlink(missing_ok=True)
        return False
    # Wait up to 5 s for graceful exit, then SIGKILL
    for _ in range(10):
        sleep(0.5)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        _pid_file(name).unlink(missing_ok=True)
    except OSError:
        pass
    return True


def _preempted_file(name: str) -> Path:
    return JOBS_DIR / name / "last_preempted"


def _preemption_count_file(name: str) -> Path:
    return JOBS_DIR / name / "preemption_count"


def _running_since_file(name: str) -> Path:
    return JOBS_DIR / name / "running_since"


def _now_ts() -> str:
    from datetime import datetime

    return datetime.now().astimezone().strftime("%m%d-%H:%M:%S")


def record_preemption(name: str) -> None:
    """Write the current timestamp as the last preemption time, increment count, clear running_since."""
    _preempted_file(name).write_text(_now_ts())
    _running_since_file(name).unlink(missing_ok=True)
    # Increment count
    count = preemption_count(name) + 1
    _preemption_count_file(name).write_text(str(count))


def record_running(name: str) -> None:
    """Write the current timestamp as when the TPU became ready."""
    _running_since_file(name).write_text(_now_ts())


def preemption_count(name: str) -> int:
    """Read the total number of preemptions."""
    p = _preemption_count_file(name)
    if not p.exists():
        return 0
    try:
        return int(p.read_text().strip())
    except (ValueError, OSError):
        return 0


def last_preempted(name: str) -> str | None:
    """Read the last preemption timestamp, or None if never preempted."""
    p = _preempted_file(name)
    if not p.exists():
        return None
    try:
        return p.read_text().strip() or None
    except OSError:
        return None


def running_since(name: str) -> str | None:
    """Read when the TPU last became ready, or None."""
    p = _running_since_file(name)
    if not p.exists():
        return None
    try:
        return p.read_text().strip() or None
    except OSError:
        return None


def remove_job(name: str) -> None:
    """Stop watcher and remove all job state."""
    stop_watcher(name)
    d = JOBS_DIR / name
    if d.is_dir():
        import shutil
        shutil.rmtree(d)
