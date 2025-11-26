from __future__ import annotations

import argparse
import signal
import sys
from time import sleep

from .config import TPUEnvConfig
from .tpu import TPUManager
from .watch import _ts, _map_v4_topology, WatchConfig, run_setup


def watch_and_run_vis(cfg: WatchConfig, env: TPUEnvConfig) -> None:
    mgr = TPUManager(env)

    print("Starting TPU visualization launcher with:")
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

        run_setup_and_vis = False

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
            run_setup_and_vis = True
        elif state == "PERMISSION_DENIED":
            print(f"{_ts()} - PERMISSION_DENIED from describe. Check IAM/API enablement.")
            sleep(mgr.sleep_secs)
            continue
        elif state == "READY":
            run_setup_and_vis = cfg.force_run
        else:
            print(f"{_ts()} - TPU in state: {state} (not actionable now).")
            sleep(mgr.sleep_secs)
            continue

        if run_setup_and_vis:
            print(f"{_ts()} - Setting up environment and repository...")
            rc = run_setup(cfg.version, env, worker="all")
            if rc != 0:
                print(f"{_ts()} - Setup failed (rc={rc}). See above for remote logs. Back to state check.")
                sleep(mgr.sleep_secs)
                continue

            print(f"{_ts()} - Starting visualization...")
            extra = " ".join(cfg.extra_args) if cfg.extra_args else ""
            # Add set -x to echo commands in the visualization pipeline and preserve stderr/stdout
            vis_cmd = (
                f"source ~/.zshrc && cd {env.gh_repo_name} && "
                f"git fetch origin && git checkout {cfg.branch} && git pull origin {cfg.branch} && "
                "XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 "
                f"uv run --group tpu scripts/vis_token_distribution.py {extra}"
            )
            if not mgr.tmux(cfg.version, cmd=vis_cmd, session="tpu"):
                print(f"{_ts()} - Launch failed/SSH timed out. Back to state check.")
                sleep(mgr.sleep_secs)
                continue

            print(f"{_ts()} - Visualization started successfully!")
            if cfg.force_run:
                print(f"{_ts()} - Force run requested; exiting.")
                return

        sleep(mgr.sleep_secs)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tpu-tools watch_vis")
    p.add_argument("version", choices=["v4", "v5", "v6"], help="TPU version to target")
    p.add_argument("--force", "-f", action="store_true", help="Force setup and visualization even if READY")
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
    watch_and_run_vis(cfg, env)
    return 0
