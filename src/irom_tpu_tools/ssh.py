from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import os
import shutil
import subprocess


def _which_timeout() -> str:
    env_val = os.environ.get("TIMEOUT_BIN", "timeout").strip()
    if shutil.which(env_val):
        return env_val
    for candidate in (env_val, "timeout", "gtimeout"):
        path = shutil.which(candidate)
        if path:
            return path
    return "timeout"


@dataclass(frozen=True)
class SSHOptions:
    connect_timeout_s: int = int(os.environ.get("SSH_CONNECT_TIMEOUT", 12))
    alive_interval_s: int = int(os.environ.get("SSH_ALIVE_INTERVAL", 10))
    alive_count_max: int = int(os.environ.get("SSH_ALIVE_COUNT_MAX", 3))
    total_timeout_s: int = int(os.environ.get("SSH_TOTAL_TIMEOUT", 60))
    kill_after_s: int = int(os.environ.get("SSH_KILL_AFTER", 5))
    key_file: str | None = os.environ.get("GCLOUD_SSH_KEY_FILE")
    forward_agent: bool = os.environ.get("SSH_FORWARD_AGENT", "1") != "0"

    def to_ssh_flags(self) -> list[str]:
        flags: list[str] = [
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.connect_timeout_s}",
            "-o",
            f"ServerAliveInterval={self.alive_interval_s}",
            "-o",
            f"ServerAliveCountMax={self.alive_count_max}",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "UserKnownHostsFile=/dev/null",
        ]
        # Forward agent if enabled and an agent socket is present
        if self.forward_agent and os.environ.get("SSH_AUTH_SOCK"):
            flags.append("-A")
        return flags


def run_with_timeout(timeout_s: int, kill_after_s: int, argv: Sequence[str]) -> subprocess.CompletedProcess:
    timeout_bin = _which_timeout()
    cmd = [timeout_bin, "-k", f"{kill_after_s}s", f"{timeout_s}s", *argv]
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def run_streaming(argv: Sequence[str]) -> int:
    """Run a command and stream stdout/stderr directly to the terminal.

    This avoids wrapping with a timeout and does not capture output, matching
    interactive behavior (e.g., tail -f).
    """
    try:
        proc = subprocess.run(list(argv), check=False)
        return proc.returncode
    except KeyboardInterrupt:
        # Propagate a conventional exit code for SIGINT
        return 130


def gcloud_tpu_ssh(
    *,
    tpu_name: str,
    project: str,
    zone: str,
    worker: str | None = None,
    command: str | None = None,
    extra_args: Iterable[str] | None = None,
    ssh: SSHOptions | None = None,
    allocate_tty: bool = False,
    no_shell_rc: bool = False,
) -> subprocess.CompletedProcess:
    ssh = ssh or SSHOptions()
    args: list[str] = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        tpu_name,
        "--project",
        project,
        "--zone",
        zone,
    ]
    # Allow tunneling through IAP if requested (helps when port 22 is blocked)
    if os.environ.get("GCLOUD_TPU_USE_IAP", "").strip() not in {"", "0", "false", "False"}:
        args.append("--tunnel-through-iap")
    if worker is not None:
        args += ["--worker", str(worker)]
    if ssh.key_file and os.path.exists(ssh.key_file):
        args += ["--ssh-key-file", ssh.key_file]
    if worker == "all":
        if command:
            args += ["--command", command]
        return run_with_timeout(ssh.total_timeout_s, ssh.kill_after_s, args)
    if extra_args:
        args.extend(list(extra_args))
    args.append("--")
    args.extend(ssh.to_ssh_flags())
    if allocate_tty:
        args.extend(["-t", "-t"])  # force TTY allocation
    if command:
        if no_shell_rc:
            args += ["bash", "--noprofile", "--norc", "-lc", command]
        else:
            args += ["bash", "-lc", command]
    return run_with_timeout(ssh.total_timeout_s, ssh.kill_after_s, args)


def gcloud_tpu_ssh_stream(
    *,
    tpu_name: str,
    project: str,
    zone: str,
    worker: str | None = None,
    command: str | None = None,
    extra_args: Iterable[str] | None = None,
    ssh: SSHOptions | None = None,
    allocate_tty: bool = False,
    no_shell_rc: bool = False,
) -> int:
    """Run gcloud TPU SSH and stream output live without a timeout wrapper.

    Intended for long-running interactive commands like tail -f.
    """
    ssh = ssh or SSHOptions()
    args: list[str] = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        tpu_name,
        "--project",
        project,
        "--zone",
        zone,
    ]
    # Allow tunneling through IAP if requested (helps when port 22 is blocked)
    if os.environ.get("GCLOUD_TPU_USE_IAP", "").strip() not in {"", "0", "false", "False"}:
        args.append("--tunnel-through-iap")
    if worker is not None:
        args += ["--worker", str(worker)]
    if ssh.key_file and os.path.exists(ssh.key_file):
        args += ["--ssh-key-file", ssh.key_file]
    if worker == "all":
        if command:
            args += ["--command", command]
        return run_streaming(args)
    if extra_args:
        args.extend(list(extra_args))
    args.append("--")
    args.extend(ssh.to_ssh_flags())
    if allocate_tty:
        args.extend(["-t", "-t"])  # force TTY allocation
    if command:
        if no_shell_rc:
            args += ["bash", "--noprofile", "--norc", "-lc", command]
        else:
            args += ["bash", "-lc", command]
    return run_streaming(args)
