from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime
import signal
from string import Template
import sys
from time import sleep

from .config import TPUEnvConfig
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


def watch_and_run(cfg: WatchConfig, env: TPUEnvConfig) -> None:
    mgr = TPUManager(env)

    print("Starting TPU auto-launcher with:")
    print(f"  TPU Name: {env.tpu_name}")
    print(f"  Zone: {getattr(env, f'tpu_zone_{cfg.version}')}")
    print(f"  Project: {env.tpu_project}")
    print(f"  Service Account: {env.tpu_service_account}")
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
            print(f"{_ts()} - Setting up environment and repository...")
            rc = run_setup(cfg.version, env, worker="all", setup_cmd=cfg.setup_cmd)
            if rc != 0:
                print(f"{_ts()} - Setup failed (rc={rc}). See above for remote logs. Back to state check.")
                sleep(mgr.sleep_secs)
                continue

            print(f"{_ts()} - Starting training...")
            if not cfg.extra_args:
                print(f"{_ts()} - No command provided. Pass the command to run after the branch name.")
                sleep(mgr.sleep_secs)
                continue
            run_cmd = " ".join(cfg.extra_args)
            train_cmd = (
                f"source ~/.zshrc && cd {env.gh_repo_name} && "
                f"git fetch origin && git checkout {cfg.branch} && git pull origin {cfg.branch} && "
                f"{run_cmd}"
            )
            if not mgr.tmux(cfg.version, cmd=train_cmd, session="tpu"):
                print(f"{_ts()} - Launch failed/SSH timed out. Back to state check.")
                sleep(mgr.sleep_secs)
                continue

            print(f"{_ts()} - Training started successfully!")
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
