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


def _build_setup_script(version: str, env: TPUEnvConfig) -> str:
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
            echo 'export READ_ONLY=true' >> ~/.zshrc
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
            echo 'git config --global credential."https://github.com".helper ""' >> ~/.zshrc
            echo 'export GIT_TERMINAL_PROMPT=0' >> ~/.zshrc
            # 2. Download uv, or conda
            curl -LsSf https://astral.sh/uv/install.sh | sh

            # 3. Set up Github permissions
            source ~/.zshrc

            KEY_TITLE="${GH_REPO}-deploy-$(hostname)"
            KEY_PATH="$HOME/.ssh/${GH_REPO}_deploy"

            mkdir -p ~/.ssh && chmod 700 ~/.ssh
            # Generate key only if not present (idempotent)
            if [ ! -f "$KEY_PATH" ]; then
            ssh-keygen -t ed25519 -N "" -f "$KEY_PATH" -C "$KEY_TITLE" >/dev/null
            fi

            ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null || true
            chmod 600 ~/.ssh/known_hosts

            PUB_KEY="$(cat "${KEY_PATH}.pub")"

            if [ "${READ_ONLY}" = "true" ]; then RO=true; else RO=false; fi
            JSON=$(printf '{"title":"%s","key":"%s","read_only":%s}' "$KEY_TITLE" "$PUB_KEY" "$RO")

            set +e
            HTTP=$(curl -sS -o /tmp/deploykey.out -w "%{http_code}" -X POST \
            -H "Authorization: token ${GH_TOKEN}" \
            -H "Accept: application/vnd.github+json" \
            "https://api.github.com/repos/${GH_OWNER}/${GH_REPO}/keys" \
            -d "$JSON")
            set -e
            if [ "$HTTP" != "201" ] && [ "$HTTP" != "422" ]; then
            echo "Deploy key API error (HTTP $HTTP):"
            cat /tmp/deploykey.out
            exit 1
            fi

            # Append SSH config only if missing (idempotent)
            if ! grep -q "^Host github-${GH_REPO}$" ~/.ssh/config 2>/dev/null; then
            {
            echo "Host github-${GH_REPO}"
            echo "  HostName github.com"
            echo "  User git"
            echo "  IdentityFile ${KEY_PATH}"
            echo "  IdentitiesOnly yes"
            } >> ~/.ssh/config
            fi
            chmod 600 ~/.ssh/config

            # if ! dpkg -s libgl1 libglib2.0-0 >/dev/null 2>&1; then
            #     # Free dpkg/apt locks by stopping services and killing lock holders
            #     for i in $(seq 1 5); do
            #         set +e
            #         sudo systemctl stop unattended-upgrades.service apt-daily.service apt-daily-upgrade.service 2>/dev/null
            #         sudo systemctl kill --kill-who=all unattended-upgrades.service 2>/dev/null
            #         for lock in /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock /var/cache/apt/archives/lock; do
            #             pids=$(sudo lsof -t "$lock" 2>/dev/null | sort -u | xargs -r echo)
            #             if [ -n "$pids" ]; then sudo kill -9 $pids 2>/dev/null; fi
            #         done
            #         sudo rm -f /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock /var/cache/apt/archives/lock
            #         sudo dpkg --configure -a >/dev/null 2>&1

            #         sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=30 update -y
            #         rc_update=$?
            #         sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=30 install -y --no-install-recommends libgl1 libglib2.0-0
            #         rc_install=$?
            #         set -e
            #         if [ "$rc_update" -eq 0 ] && [ "$rc_install" -eq 0 ]; then
            #             break
            #         fi
            #         echo "dpkg/apt locked or failed (update=$rc_update install=$rc_install); retrying in 10s ($i/5)..."
            #         sleep 10
            #     done
            #     dpkg -s libgl1 libglib2.0-0 >/dev/null 2>&1 || { echo "Failed to install libgl1 libglib2.0-0"; exit 1; }
            # fi

            # 4. Clone the repository and set up deps only if missing
            if [ ! -d "${GH_REPO}/.git" ]; then
            git clone --recurse-submodules "git@github-${GH_REPO}:lihzha/${GH_REPO}.git" || true
            cd ${GH_REPO}
            uv sync
            fi
            """)
    return setup_tpl.safe_substitute(
        OPENPI_DATA_HOME=f"{bucket_env}/cache",
        GH_TOKEN=env.gh_token,
        WANDB_API_KEY=env.wandb_api_key,
        GH_REPO=env.gh_repo_name,
        GH_OWNER=env.gh_owner,
    )


def build_setup_cmd(version: str, env: TPUEnvConfig) -> str:
    """Build the remote setup command identical to watch()'s setup step.

    Returns a shell command suitable for execution over SSH.
    """
    setup_script = _build_setup_script(version, env)
    encoded = base64.b64encode(setup_script.encode()).decode().replace("\n", "")
    return f"bash -lc 'echo {encoded} | base64 -d | bash -l -s'"


def run_setup(version: str, env: TPUEnvConfig, *, worker: str | None = "all") -> int:
    """Run the setup step on the TPU worker(s).

    This is exposed so callers can do: `tpu v4 setup`.
    """
    mgr = TPUManager(env)
    setup_cmd = build_setup_cmd(version, env)
    return mgr.raw(version, cmd=setup_cmd, worker=worker)


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
    print(f"  Force run: {cfg.force_run}")
    if cfg.extra_args:
        print(f"  Extra args: {' '.join(cfg.extra_args)}")
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
            rc = run_setup(cfg.version, env, worker="all")
            if rc != 0:
                print(f"{_ts()} - Setup failed (rc={rc}). See above for remote logs. Back to state check.")
                sleep(mgr.sleep_secs)
                continue

            print(f"{_ts()} - Starting training...")
            extra = " ".join(cfg.extra_args) if cfg.extra_args else ""
            # Add set -x to echo commands in the training pipeline and preserve stderr/stdout
            train_cmd = (
                f"source ~/.zshrc && cd {env.gh_repo_name} && "
                f"git fetch origin && git checkout {cfg.branch} && git pull origin {cfg.branch} && "
                "XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 "
                f"uv run --group tpu scripts/train.py {extra}"
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
    cfg = WatchConfig(version=ns.version, force_run=ns.force, tpu_num=ns.tpu_num, branch=branch, extra_args=extra)
    env = TPUEnvConfig.from_env()
    watch_and_run(cfg, env)
    return 0
