from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime
import os
import signal
from string import Template
import sys
from time import sleep

from .config import TPUEnvConfig
from .jobs import JobConfig
from .tpu import TPUManager


def _ts() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _map_v4_topology(tpu_num: int) -> str:
    mapping = {4: "2x2x1", 8: "2x2x2", 16: "2x2x4", 32: "2x4x4"}
    if tpu_num not in mapping:
        raise SystemExit(f"Error: unsupported TPU_NUM '{tpu_num}' (allowed: 4, 8, 16, 32)")
    return mapping[tpu_num]


@dataclass
class WatchConfig:
    version: str  # v4/v5/v6
    force_run: bool
    tpu_num: int
    branch: str
    extra_args: list[str]
    setup_cmd: str = "uv sync"  # commands to run after clone, e.g. "uv sync && uv pip install -e ."


def _build_setup_script(version: str, env: TPUEnvConfig, setup_cmd: str = "uv sync") -> str:
    bucket_env = {
        "v4": env.tpu_bucket_v4,
        "v5": env.tpu_bucket_v5,
        "v6": env.tpu_bucket_v6,
    }[version]
    setup_tpl = Template(r"""set -euo pipefail

            # 1. Set up environment variables
            echo 'export WANDB_API_KEY="${WANDB_API_KEY}"' >> ~/.zshrc
            echo 'export OPENPI_DATA_HOME="${OPENPI_DATA_HOME}"' >> ~/.zshrc
            echo 'export GH_TOKEN="${GH_TOKEN}"' >> ~/.zshrc
            echo 'export GH_OWNER="${GH_OWNER}"' >> ~/.zshrc
            echo 'export GH_REPO="${GH_REPO}"' >> ~/.zshrc
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
            # 2. Download uv
            curl -LsSf https://astral.sh/uv/install.sh | sh

            # 3. Clone the repository and set up deps only if missing
            source ~/.zshrc
            if [ ! -d "${GH_REPO}/.git" ]; then
            git clone --recurse-submodules "https://${GH_TOKEN}@github.com/${GH_OWNER}/${GH_REPO}.git"
            cd ${GH_REPO}
            ${SETUP_CMD}
            fi
            """)
    return setup_tpl.safe_substitute(
        OPENPI_DATA_HOME=f"{bucket_env}/cache",
        GH_TOKEN=env.gh_token,
        WANDB_API_KEY=env.wandb_api_key,
        GH_REPO=env.gh_repo_name,
        GH_OWNER=env.gh_owner,
        SETUP_CMD=setup_cmd,
    )


def build_setup_cmd(version: str, env: TPUEnvConfig, setup_cmd: str = "uv sync") -> str:
    """Build the remote setup command identical to watch()'s setup step.

    Returns a shell command suitable for execution over SSH.
    """
    setup_script = _build_setup_script(version, env, setup_cmd)
    encoded = base64.b64encode(setup_script.encode()).decode().replace("\n", "")
    return f"bash -lc 'echo {encoded} | base64 -d | bash -l -s'"


def run_setup(version: str, env: TPUEnvConfig, *, worker: str | None = "all", setup_cmd: str = "uv sync") -> int:
    """Run the setup step on the TPU worker(s).

    This is exposed so callers can do: `tpu v4 setup`.
    """
    mgr = TPUManager(env)
    remote_cmd = build_setup_cmd(version, env, setup_cmd)
    return mgr.raw(version, cmd=remote_cmd, worker=worker)


def _do_setup_and_training(
    mgr: TPUManager,
    version: str,
    env: TPUEnvConfig,
    *,
    command: str,
    branch: str,
    setup_cmd: str,
) -> bool:
    """Run setup + optionally launch training. Returns True on success."""
    print(f"{_ts()} - Setting up environment and repository...")
    remote_cmd = build_setup_cmd(version, env, setup_cmd)
    rc = mgr.raw(version, cmd=remote_cmd, worker="all")
    if rc != 0:
        print(f"{_ts()} - Setup failed (rc={rc}).")
        return False

    if not command:
        print(f"{_ts()} - Setup complete (no training command specified).")
        return True

    print(f"{_ts()} - Starting training...")
    train_cmd = (
        f"source ~/.zshrc && cd {env.gh_repo_name} && "
        f"git fetch origin && git checkout {branch} && git pull origin {branch} && "
        f"{command}"
    )
    if not mgr.tmux(version, cmd=train_cmd, session="tpu"):
        print(f"{_ts()} - Launch failed/SSH timed out.")
        return False

    print(f"{_ts()} - Training started successfully!")
    return True


def create_and_launch(job: JobConfig, env: TPUEnvConfig) -> bool:
    """Create a TPU, run setup, and launch training. Returns True on success.

    Used by `tpu create` for the initial provisioning.
    """
    mgr = TPUManager(env).for_tpu(job.name, job.version, env.zones[job.version])

    print(f"{_ts()} - Creating TPU '{job.name}'...")
    topo = job.topology or (_map_v4_topology(job.tpu_num) if job.version == "v4" else None)
    if not mgr.create(job.version, tpu_num=job.tpu_num, topology=topo):
        print(f"{_ts()} - Create failed/timed out.")
        return False

    print(f"{_ts()} - Waiting for TPU to be READY...")
    sleep(10)

    return _do_setup_and_training(
        mgr, job.version, env,
        command=job.command, branch=job.branch, setup_cmd=job.setup_cmd,
    )


def watch_loop(job: JobConfig, env: TPUEnvConfig) -> None:
    """Background watcher loop: monitor TPU state and recover from preemptions.

    This runs as a daemon — it never returns unless signaled.
    """
    mgr = TPUManager(env).for_tpu(job.name, job.version, env.zones[job.version])

    print(f"{_ts()} - Watcher started for TPU '{job.name}' ({job.version})")
    print(f"{_ts()} - Command: {job.command}")
    print(f"{_ts()} - Branch: {job.branch}")
    sys.stdout.flush()

    def handle_sig(signum, frame):
        print(f"{_ts()} - Watcher caught signal {signum}, exiting.")
        sys.stdout.flush()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    from .jobs import record_preemption, record_running

    recorded_ready = False

    while True:
        try:
            state = mgr.describe(job.version)
        except Exception as exc:
            print(f"{_ts()} - Describe error: {exc}")
            sys.stdout.flush()
            sleep(mgr.sleep_secs)
            continue

        print(f"{_ts()} - TPU '{job.name}' state: {state}")
        sys.stdout.flush()

        if state == "READY":
            if not recorded_ready:
                record_running(job.name)
                recorded_ready = True
            sleep(mgr.sleep_secs)
            continue

        if state in {"PREEMPTED", "STOPPED", "NOT_FOUND"}:
            recorded_ready = False
            if state == "PREEMPTED":
                record_preemption(job.name)
                print(f"{_ts()} - Preemption recorded.")
            print(f"{_ts()} - Creating/recovering TPU...")
            if state != "NOT_FOUND" and not mgr.delete(job.version):
                print(f"{_ts()} - Delete failed/timed out.")
                sys.stdout.flush()
                sleep(mgr.sleep_secs)
                continue

            topo = job.topology or (_map_v4_topology(job.tpu_num) if job.version == "v4" else None)
            print(f"{_ts()} - Creating TPU...")
            if not mgr.create(job.version, tpu_num=job.tpu_num, topology=topo):
                print(f"{_ts()} - Create failed/timed out, will retry.")
                sys.stdout.flush()
                sleep(mgr.sleep_secs)
                continue

            print(f"{_ts()} - Waiting for TPU to be READY...")
            sleep(10)

            ok = _do_setup_and_training(
                mgr, job.version, env,
                command=job.command, branch=job.branch, setup_cmd=job.setup_cmd,
            )
            if ok:
                record_running(job.name)
                recorded_ready = True
                print(f"{_ts()} - TPU ready, training launched.")
            else:
                print(f"{_ts()} - Setup/launch failed, will retry next cycle.")
            sys.stdout.flush()

        elif state == "PERMISSION_DENIED":
            print(f"{_ts()} - PERMISSION_DENIED. Check IAM/API enablement.")
            sys.stdout.flush()

        sleep(mgr.sleep_secs)


def spawn_watcher(job: JobConfig, env: TPUEnvConfig) -> int:
    """Fork a background watcher daemon. Returns the daemon PID."""
    from .jobs import log_path, save_pid

    log_file = log_path(job.name)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    pid = os.fork()
    if pid > 0:
        # Parent — record daemon PID and return
        save_pid(job.name, pid)
        return pid

    # Child — become a daemon
    os.setsid()

    # Redirect stdout/stderr to log file
    fd = os.open(str(log_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)  # stdout
    os.dup2(fd, 2)  # stderr
    os.close(fd)
    # Close stdin
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)

    try:
        watch_loop(job, env)
    except SystemExit:
        pass
    except Exception as exc:
        print(f"{_ts()} - Watcher crashed: {exc}")
    finally:
        os._exit(0)


# ---------- legacy watch (kept for `tpu watch` backwards compat) ----------


def watch_and_run(cfg: WatchConfig, env: TPUEnvConfig) -> None:
    mgr = TPUManager(env)

    print("Starting TPU auto-launcher with:")
    print(f"  TPU Name: {env.tpu_name}")
    zone = getattr(env, f"tpu_zone_{cfg.version}")
    print(f"  Zone: {zone}")
    print(f"  Project: {env.tpu_project}")
    effective_sa = env.service_account_for_zone(zone)
    print(f"  Service Account: {effective_sa}")
    print(f"  Repo Name: {env.gh_repo_name}")
    bucket = getattr(env, f"tpu_bucket_{cfg.version}")
    print(f"  Bucket: {bucket}")
    print(f"  TPU Num: {cfg.tpu_num}")
    if cfg.version == "v4":
        print(f"  Topology: {_map_v4_topology(cfg.tpu_num)}")
    print(f"  Branch: {cfg.branch}")
    print(f"  Setup cmd: {cfg.setup_cmd}")
    print(f"  Force run: {cfg.force_run}")
    if cfg.extra_args:
        print(f"  Command: {' '.join(cfg.extra_args)}")
    print()

    def handle_sig(signum, frame):
        print(f"{_ts()} - Caught signal, exiting.")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    while True:
        print(f"{_ts()} - Checking TPU state...")
        try:
            state = mgr.describe(cfg.version)
        except Exception as exc:
            print(str(exc))
            sleep(mgr.sleep_secs)
            continue

        print(f"{_ts()} - TPU {env.tpu_name} state: {state}")

        run_setup_and_training = False

        if state in {"NOT_FOUND", "PREEMPTED", "STOPPED"}:
            print(f"{_ts()} - Need to (re)create TPU...")
            if state != "NOT_FOUND" and not mgr.delete(cfg.version):
                print(f"{_ts()} - Delete failed/timed out.")
                sleep(mgr.sleep_secs)
                continue
            print(f"{_ts()} - Creating new TPU...")
            topo = _map_v4_topology(cfg.tpu_num) if cfg.version == "v4" else None
            if not mgr.create(cfg.version, tpu_num=cfg.tpu_num, topology=topo):
                print(f"{_ts()} - Create failed/timed out.")
                sleep(mgr.sleep_secs)
                continue
            print(f"{_ts()} - Waiting for TPU to be READY...")
            sleep(10)
            run_setup_and_training = True
        elif state == "PERMISSION_DENIED":
            print(f"{_ts()} - PERMISSION_DENIED from describe. Check IAM/API enablement.")
            sleep(mgr.sleep_secs)
            continue
        elif state == "READY":
            run_setup_and_training = cfg.force_run
        else:
            print(f"{_ts()} - TPU in state: {state} (not actionable now).")
            sleep(mgr.sleep_secs)
            continue

        if run_setup_and_training:
            if not cfg.extra_args:
                print(f"{_ts()} - No command provided. Pass the command to run after the branch name.")
                sleep(mgr.sleep_secs)
                continue
            ok = _do_setup_and_training(
                mgr, cfg.version, env,
                command=" ".join(cfg.extra_args), branch=cfg.branch, setup_cmd=cfg.setup_cmd,
            )
            if not ok:
                sleep(mgr.sleep_secs)
                continue
            if cfg.force_run:
                print(f"{_ts()} - Force run requested; exiting.")
                return

        sleep(mgr.sleep_secs)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tpu-tools watch")
    p.add_argument("version", choices=["v4", "v5", "v6"], help="TPU version to target")
    p.add_argument("--force", "-f", action="store_true", help="Force setup and training even if READY")
    p.add_argument("--tpu-num", "-n", type=int, default=8, help="TPU chips (v4: 4/8/16/32; v5:16/32/64; v6:any)")
    p.add_argument("--setup-cmd", "-s", default="uv sync", help='Commands to run after clone (e.g. "uv sync && uv pip install -e .")')
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_arg_parser()
    ns, extra = ap.parse_known_args(argv)
    # Normalize extras: drop a leading '--' sentinel if present
    if extra and extra[0] == "--":
        extra = extra[1:]
    # Extract branch name if present (first non-flag argument in extra)
    branch = "main"
    if extra and not extra[0].startswith("-"):
        branch = extra[0]
        extra = extra[1:]
    cfg = WatchConfig(version=ns.version, force_run=ns.force, tpu_num=ns.tpu_num, branch=branch, extra_args=extra, setup_cmd=ns.setup_cmd)
    env = TPUEnvConfig.from_env()
    watch_and_run(cfg, env)
    return 0
