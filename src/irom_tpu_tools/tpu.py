from __future__ import annotations

import base64
from dataclasses import dataclass
import os
import re
import shlex
from typing import Literal

from .config import TPUEnvConfig
from .ssh import SSHOptions
from .ssh import gcloud_tpu_ssh
from .ssh import gcloud_tpu_ssh_stream
from .ssh import run_streaming
from .ssh import run_with_timeout


def _ts() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


DescribeRC = Literal[0, 1, 2]
TPUVersion = Literal["v4", "v5", "v6"]


def resolve_tpu(
    name: str,
    project: str,
    zones: dict[str, str],
    timeout_s: int = 20,
) -> tuple[TPUVersion, str]:
    """Find which version/zone a TPU lives in by querying all configured zones.

    Returns (version, zone).  Raises RuntimeError if the TPU is not found.
    """
    for version, zone in zones.items():
        rc, state = _gcloud_describe_state(project, zone, name, timeout_s)
        if rc == 0 and state not in ("NOT_FOUND",):
            return version, zone  # type: ignore[return-value]
    configured = ", ".join(f"{v}={z}" for v, z in zones.items())
    raise RuntimeError(f"TPU '{name}' not found in any configured zone ({configured})")


def _gcloud_describe_state(project: str, zone: str, name: str, timeout_s: int) -> tuple[DescribeRC, str]:
    proc = run_with_timeout(
        timeout_s,
        int(os.environ.get("SSH_KILL_AFTER", 5)),
        [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            name,
            "--zone",
            zone,
            "--project",
            project,
            "--format",
            "value(state)",
        ],
    )
    if proc.returncode == 0:
        return 0, (proc.stdout.strip() or "UNKNOWN")
    out = (proc.stderr or proc.stdout or "").lower()
    if re.search(r"not\s*found|404", out):
        return 0, "NOT_FOUND"
    if re.search(r"permission_denied|forbidden|403", out):
        return 0, "PERMISSION_DENIED"
    if re.search(r"invalid value for \[--zone\]|argument --zone", out):
        return 2, "INVALID_ZONE"
    return 1, out.strip().splitlines()[-1] if out else "ERROR"


@dataclass
class TPUManager:
    env: TPUEnvConfig
    ssh: SSHOptions = SSHOptions()
    describe_timeout_s: int = int(os.environ.get("DESCRIBE_TIMEOUT", 20))
    sleep_secs: int = int(os.environ.get("SLEEP_SECS", 20))

    # --- resolved overrides (set by CLI after resolve) ---
    _name: str | None = None
    _version: TPUVersion | None = None
    _zone: str | None = None

    def for_tpu(self, name: str, version: TPUVersion, zone: str) -> "TPUManager":
        """Return a copy of this manager pinned to a specific TPU."""
        from dataclasses import replace

        return replace(self, _name=name, _version=version, _zone=zone)

    @property
    def tpu_name(self) -> str:
        return self._name or self.env.tpu_name

    def _zone_for(self, version: TPUVersion) -> str:
        if self._zone and (self._version == version or version is None):
            return self._zone
        return {
            "v4": self.env.tpu_zone_v4,
            "v5": self.env.tpu_zone_v5,
            "v6": self.env.tpu_zone_v6,
        }[version]

    def _bucket_for(self, version: TPUVersion) -> str:
        return {
            "v4": self.env.tpu_bucket_v4,
            "v5": self.env.tpu_bucket_v5,
            "v6": self.env.tpu_bucket_v6,
        }[version]

    @property
    def version(self) -> TPUVersion:
        if self._version is None:
            raise RuntimeError("TPU version not resolved; pass a TPU name or version")
        return self._version

    def resolve(self, name: str | None = None) -> "TPUManager":
        """Resolve a TPU name to its version/zone and return a pinned manager."""
        tpu_name = name or self.tpu_name
        if not tpu_name:
            raise RuntimeError("No TPU name provided and TPU_NAME not set")
        ver, zone = resolve_tpu(tpu_name, self.env.tpu_project, self.env.zones, self.describe_timeout_s)
        return self.for_tpu(tpu_name, ver, zone)

    def describe(self, version: TPUVersion) -> str:
        rc, state = _gcloud_describe_state(
            self.env.tpu_project, self._zone_for(version), self.tpu_name, self.describe_timeout_s
        )
        if rc == 2:
            raise RuntimeError(f"Invalid zone for {version}: {self._zone_for(version)}")
        if rc != 0:
            print(f"{_ts()} - Describe error: {state}")
            return "ERROR"
        return state

    def delete(self, version: Literal["v4", "v5", "v6"]) -> bool:
        zone = self._zone_for(version)
        rc = run_streaming(
            [
                "gcloud",
                "alpha",
                "compute",
                "tpus",
                "tpu-vm",
                "delete",
                self.tpu_name,
                "--zone",
                zone,
                "--project",
                self.env.tpu_project,
                "--quiet",
            ]
        )
        return rc == 0

    def create(self, version: Literal["v4", "v5", "v6"], *, tpu_num: int, topology: str | None = None) -> bool:
        zone = self._zone_for(version)
        sa = self.env.service_account_for_zone(zone)
        common = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "create",
            self.tpu_name,
            "--zone",
            zone,
            "--project",
            self.env.tpu_project,
            "--service-account",
            sa,
            "--spot",
        ]
        if version == "v4":
            if not topology:
                raise ValueError("topology is required for v4")
            args = [*common, "--type", "v4", "--topology", topology, "--version", "tpu-ubuntu2204-base"]
        elif version == "v5":
            accel = {8: "v5litepod-8", 16: "v5litepod-16", 32: "v5litepod-32", 64: "v5litepod-64"}.get(tpu_num)
            if not accel:
                raise ValueError("Unsupported TPU_NUM for v5: expected 16/32/64")
            args = [*common, "--accelerator-type", accel, "--version", "v2-alpha-tpuv5-lite"]
        else:  # v6
            args = [*common, "--accelerator-type", f"v6e-{tpu_num}", "--version", "v2-alpha-tpuv6e"]

        rc = run_streaming(args)
        return rc == 0

    def tmux(self, version: Literal["v4", "v5", "v6"], *, cmd: str, session: str = "tpu") -> bool:
        # Ensure tmux exists and start/send in a session across all workers
        line = f"set -eo pipefail; export PYTHONUNBUFFERED=1; {cmd} 2>&1 | tee -a $LOG"
        remote = (
            "command -v tmux >/dev/null || (sudo apt-get update && sudo apt-get install -y tmux);"
            f"mkdir -p $HOME/{self.env.gh_repo_name}/logs;"
            "TS=$(date +%Y%m%d-%H%M%S);"
            f"LOG=$HOME/{self.env.gh_repo_name}/logs/{session}_$TS.log;"
            f"if ! tmux has-session -t {shlex.quote(session)} 2>/dev/null; then "
            f"    tmux new-session -ds {shlex.quote(session)} -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK -e LOG=$LOG; "
            "else "
            f"    tmux set-environment -t {shlex.quote(session)} LOG $LOG; "
            "fi;"
            # Send the command and execute it
            f"tmux send-keys -t {shlex.quote(session)} {shlex.quote(line)} Enter"
        )
        return (
            gcloud_tpu_ssh_stream(
                tpu_name=self.tpu_name,
                project=self.env.tpu_project,
                zone=self._zone_for(version),
                worker="all",
                command=remote,
                ssh=self.ssh,
            )
            == 0
        )

    def raw(self, version: Literal["v4", "v5", "v6"], *, cmd: str, worker: str | None = "all") -> int:
        """Run a raw command on TPU worker(s) without tmux.

        Mirrors `v4 "<cmd>"` style helpers from ~/.tpu_funcs.sh.
        """
        return gcloud_tpu_ssh_stream(
            tpu_name=self.tpu_name,
            project=self.env.tpu_project,
            zone=self._zone_for(version),
            worker=worker if worker is not None else None,
            command=cmd,
            ssh=self.ssh,
        )

    def attach(self, version: Literal["v4", "v5", "v6"], *, session: str = "tpu", worker: int = 0) -> int:
        # Use exec with `tmux new -As` to attach-or-create without running extra commands afterward
        return gcloud_tpu_ssh_stream(
            tpu_name=self.tpu_name,
            project=self.env.tpu_project,
            zone=self._zone_for(version),
            worker=str(worker),
            command=(
                "command -v tmux >/dev/null || (sudo apt-get update && sudo apt-get install -y tmux); "
                f"exec tmux new -As {shlex.quote(session)}"
            ),
            ssh=self.ssh,
            allocate_tty=True,
            no_shell_rc=True,
        )

    def tail_log(self, version: Literal["v4", "v5", "v6"], *, worker: int = 0) -> int:
        # Prefer tmux's LOG environment for the current session; fallback to newest log file.
        session = "tpu"
        cmd = (
            f"SESSION={shlex.quote(session)}; "
            # 1. Try tmux LOG env var
            'LOG_FILE="$(tmux show-environment -t "$SESSION" LOG 2>/dev/null | sed -n "s/^LOG=//p")"; '
            '[ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ] && { tail -n 1000 -f "$LOG_FILE"; exit $?; }; '
            # 2. Try repo-specific logs dir
            f'LOG_DIR=$HOME/{self.env.gh_repo_name}/logs; '
            'if [ -d "$LOG_DIR" ]; then '
            '  F="$(ls -1t "$LOG_DIR" | head -n1 || true)"; '
            '  [ -n "$F" ] && { tail -n 1000 -f "$LOG_DIR/$F"; exit $?; }; '
            'fi; '
            # 3. Try any logs dir under $HOME
            'for d in $HOME/*/logs; do '
            '  [ -d "$d" ] || continue; '
            '  F="$(ls -1t "$d" | head -n1 || true)"; '
            '  [ -n "$F" ] && { echo "[found log in $d]"; tail -n 1000 -f "$d/$F"; exit $?; }; '
            'done; '
            'echo "[ERROR] No log files found"; exit 1'
        )
        rc = gcloud_tpu_ssh_stream(
            tpu_name=self.tpu_name,
            project=self.env.tpu_project,
            zone=self._zone_for(version),
            worker=str(worker),
            command=cmd,
            ssh=self.ssh,
        )
        return rc

    def _tmux_kill_all(self, version: Literal["v4", "v5", "v6"]) -> bool:
        remote = (
            "set -euo pipefail;"
            "if command -v tmux >/dev/null 2>&1; then "
            "tmux ls >/dev/null 2>&1 && tmux kill-server || true; "
            "rm -rf /tmp/tmux-$(id -u) 2>/dev/null || true; fi"
        )
        return (
            gcloud_tpu_ssh_stream(
                tpu_name=self.tpu_name,
                project=self.env.tpu_project,
                zone=self._zone_for(version),
                worker="all",
                command=remote,
                ssh=self.ssh,
            )
            == 0
        )

    def _kill_jax(self, version: Literal["v4", "v5", "v6"]) -> bool:
        remote = (
            "set -euo pipefail;"
            "PIDS=$(pgrep -u $USER -f python || true);"
            "for pid in $PIDS; do "
            "if [ -r \"/proc/$pid/environ\" ] && tr '\\0' '\\n' </proc/$pid/environ 2>/dev/null | grep -qE '(^(JAX_|XLA_|TPU_|LIBTPU))'; then "
            "kill -TERM $pid 2>/dev/null || true; fi; done;"
            "sleep 2;"
            "for pid in $(pgrep -u $USER -f python || true); do "
            "if [ -r \"/proc/$pid/environ\" ] && tr '\\0' '\\n' </proc/$pid/environ 2>/dev/null | grep -qE '(^(JAX_|XLA_|TPU_|LIBTPU))'; then "
            "kill -0 $pid 2>/dev/null && kill -KILL $pid 2>/dev/null || true; fi; done;"
            "pgrep -a -u $USER -f python || true"
        )
        return (
            gcloud_tpu_ssh_stream(
                tpu_name=self.tpu_name,
                project=self.env.tpu_project,
                zone=self._zone_for(version),
                worker="all",
                command=remote,
                ssh=self.ssh,
            )
            == 0
        )

    def _clean_jax_tmp(self, version: Literal["v4", "v5", "v6"]) -> bool:
        remote = (
            'echo "[INFO] Cleaning /tmp…";'
            "find /tmp -maxdepth 1 -user $USER "
            "\\( -name 'jax*' -o -name '.jax*' -o -name 'pjrt*' -o -name 'xla*' "
            "-o -name 'libtpu*' -o -name 'tpu*' -o -name 'coordination-*' -o -name 'jax-mp-*' \\) "
            "-print -exec rm -rf {} + 2>/dev/null || true;"
            'echo "[INFO] Cleaning /dev/shm…";'
            "find /dev/shm -maxdepth 1 -user $USER "
            "\\( -name 'sem.*' -o -name 'psm_*' -o -name 'jax*' -o -name 'xla*' -o -name 'pjrt*' \\) "
            "-print -exec rm -f {} + 2>/dev/null || true"
        )
        return (
            gcloud_tpu_ssh_stream(
                tpu_name=self.tpu_name,
                project=self.env.tpu_project,
                zone=self._zone_for(version),
                worker="all",
                command=remote,
                ssh=self.ssh,
            )
            == 0
        )

    def clean_logs(self, version: Literal["v4", "v5", "v6"]) -> bool:
        """Truncate system log files to free up disk space."""
        remote = (
            "set -euo pipefail;"
            "sudo truncate -s 0 /var/log/syslog 2>/dev/null || true;"
            "sudo truncate -s 0 /var/log/kern.log 2>/dev/null || true;"
            "sudo truncate -s 0 /var/log/syslog.1 2>/dev/null || true;"
            "sudo truncate -s 0 /var/log/kern.log.1 2>/dev/null || true"
        )
        return (
            gcloud_tpu_ssh_stream(
                tpu_name=self.tpu_name,
                project=self.env.tpu_project,
                zone=self._zone_for(version),
                worker="all",
                command=remote,
                ssh=self.ssh,
            )
            == 0
        )

    def _kill_device_holders(self, version: Literal["v4", "v5", "v6"]) -> bool:
        """Kill python3 processes holding TPU device files (v4: /dev/accel0, v6: /dev/vfio/0)."""
        if version == "v4":
            remote = (
                "set -euo pipefail;"
                "sudo lsof /dev/accel0 2>/dev/null | awk '$1==\"python3\" {print $2}' | xargs -r sudo kill -9 || true"
            )
        elif version in {"v5", "v6"}:
            remote = (
                "set -euo pipefail;"
                "sudo lsof /dev/vfio/0 2>/dev/null | awk '$1==\"python3\" {print $2}' | xargs -r sudo kill -9 || true"
            )
        else:  # v5
            return True  # No device-specific killing for v5

        return (
            gcloud_tpu_ssh_stream(
                tpu_name=self.tpu_name,
                project=self.env.tpu_project,
                zone=self._zone_for(version),
                worker="all",
                command=remote,
                ssh=self.ssh,
            )
            == 0
        )

    def nuke_all(self, version: Literal["v4", "v5", "v6"]) -> bool:
        ok = self._tmux_kill_all(version)
        ok = self._kill_jax(version) and ok
        ok = self._kill_device_holders(version) and ok
        ok = self._clean_jax_tmp(version) and ok
        return ok

    def list(self, version: TPUVersion) -> int:
        zone = self._zone_for(version)
        project = self.env.tpu_project
        rc = run_streaming(["gcloud", "compute", "tpus", "tpu-vm", "list", "--zone", zone, "--project", project])
        return rc
