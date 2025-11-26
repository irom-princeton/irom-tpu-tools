from __future__ import annotations

import argparse
import sys

from .config import TPUEnvConfig
from .tpu import TPUManager


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("version", choices=["v4", "v5", "v6"], help="TPU version to target")


def build_parser() -> argparse.ArgumentParser:
    prog_name = (sys.argv[0].rsplit("/", 1)[-1] or "tpu") if getattr(sys, "argv", None) else "tpu"
    ap = argparse.ArgumentParser(prog=prog_name, description="Unified TPU utilities for v4/v5/v6")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_watch = sub.add_parser("watch", help="Watch TPU state and (re)create + run training")
    p_watch.add_argument("version", choices=["v4", "v5", "v6"], help="TPU version to target")
    p_watch.add_argument("--force", "-f", action="store_true", help="Force setup and training even if READY")
    p_watch.add_argument("--tpu-num", "-n", type=int, default=8, help="TPU chips")

    p_watch_vis = sub.add_parser("watch_vis", help="Watch TPU state and (re)create + run visualization")
    p_watch_vis.add_argument("version", choices=["v4", "v5", "v6"], help="TPU version to target")
    p_watch_vis.add_argument("--force", "-f", action="store_true", help="Force setup and visualization even if READY")
    p_watch_vis.add_argument("--tpu-num", "-n", type=int, default=8, help="TPU chips")

    p_list = sub.add_parser("list", help="List TPUs in zone")
    _add_common(p_list)

    p_delete = sub.add_parser("delete", help="Delete this TPU (by TPU_NAME env)")
    _add_common(p_delete)

    p_stop = sub.add_parser("stop", help="Stop this TPU (preserve allocation)")
    _add_common(p_stop)

    p_start = sub.add_parser("start", help="Start this TPU (previously stopped)")
    _add_common(p_start)

    p_delete_name = sub.add_parser("delete-name", help="Delete a TPU by explicit name")
    _add_common(p_delete_name)
    p_delete_name.add_argument("name", help="TPU name to delete")

    p_tmux = sub.add_parser("tmux", help="Run a tmux command on all workers")
    _add_common(p_tmux)
    p_tmux.add_argument("--session", default="tpu")
    p_tmux.add_argument("rest", nargs=argparse.REMAINDER, help="Command to run in tmux session")

    p_attach = sub.add_parser("attach", help="Attach to tmux on a worker")
    _add_common(p_attach)
    p_attach.add_argument("--session", default="tpu")
    p_attach.add_argument("--worker", type=int, default=0)

    p_ls = sub.add_parser("tmux-ls", help="List tmux sessions on all workers")
    _add_common(p_ls)

    p_tail = sub.add_parser("tail", help="Show last 50 lines of latest tmux log on a worker")
    _add_common(p_tail)
    p_tail.add_argument("--worker", type=int, default=0)

    p_kill = sub.add_parser("tmux-kill-all", help="Kill tmux server on all workers")
    _add_common(p_kill)

    p_kill_jax = sub.add_parser("kill-jax", help="Kill JAX/XLA processes on all workers")
    _add_common(p_kill_jax)

    p_clean = sub.add_parser("clean-tmp", help="Clean JAX/XLA tmp files on all workers")
    _add_common(p_clean)

    p_nuke = sub.add_parser("nuke", help="Kill tmux, JAX, and clean tmp on all workers")
    _add_common(p_nuke)

    # Raw SSH commands without tmux. Used for debugging.
    p_v4 = sub.add_parser("v4", help="Run raw command on v4 workers (no tmux)")
    p_v4.add_argument("--worker", type=int, default=None, help="Worker index (default: all)")
    p_v4.add_argument("rest", nargs=argparse.REMAINDER, help="Command to run remotely")
    p_v5 = sub.add_parser("v5", help="Run raw command on v5 workers (no tmux)")
    p_v5.add_argument("--worker", type=int, default=None, help="Worker index (default: all)")
    p_v5.add_argument("rest", nargs=argparse.REMAINDER, help="Command to run remotely")
    p_v6 = sub.add_parser("v6", help="Run raw command on v6 workers (no tmux)")
    p_v6.add_argument("--worker", type=int, default=None, help="Worker index (default: all)")
    p_v6.add_argument("rest", nargs=argparse.REMAINDER, help="Command to run remotely")

    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    ns, unknown = ap.parse_known_args(argv)
    if ns.cmd == "watch":
        from .watch import main as _watch_main

        return _watch_main([ns.version, *((ns.force and ["--force"]) or []), "-n", str(ns.tpu_num), *unknown])
    if ns.cmd == "watch_vis":
        from .watch_vis import main as _watch_vis_main

        return _watch_vis_main([ns.version, *((ns.force and ["--force"]) or []), "-n", str(ns.tpu_num), *unknown])

    env = TPUEnvConfig.from_env()
    mgr = TPUManager(env)

    if ns.cmd == "list":
        return mgr.list(ns.version)
    if ns.cmd == "delete":
        ok = mgr.delete(ns.version)
        return 0 if ok else 1
    if ns.cmd == "stop":
        ok = mgr.stop(ns.version)
        return 0 if ok else 1
    if ns.cmd == "start":
        ok = mgr.start(ns.version)
        return 0 if ok else 1
    if ns.cmd == "delete-name":
        return mgr.delete_by_name(ns.version, ns.name)
    if ns.cmd == "tmux":
        cmd = " ".join(ns.rest) if getattr(ns, "rest", None) else ""
        ok = mgr.tmux(ns.version, cmd=cmd, session=ns.session)
        return 0 if ok else 1
    if ns.cmd == "attach":
        return mgr.attach(ns.version, session=ns.session, worker=ns.worker)
    if ns.cmd == "tmux-ls":
        ok = mgr.tmux_ls(ns.version)
        return 0 if ok else 1
    if ns.cmd == "tail":
        return mgr.tail_log(ns.version, worker=ns.worker)
    if ns.cmd == "tmux-kill-all":
        ok = mgr.tmux_kill_all(ns.version)
        return 0 if ok else 1
    if ns.cmd == "kill-jax":
        ok = mgr.kill_jax(ns.version)
        return 0 if ok else 1
    if ns.cmd == "clean-tmp":
        ok = mgr.clean_jax_tmp(ns.version)
        return 0 if ok else 1
    if ns.cmd == "nuke":
        ok = mgr.nuke_all(ns.version)
        return 0 if ok else 1
    if ns.cmd in {"v4", "v5", "v6"}:
        # Special case: allow `tpu v4 setup` to run the watch setup step
        if getattr(ns, "rest", None) and len(ns.rest) >= 1 and ns.rest[0] == "setup":
            from .watch import run_setup

            worker = None if getattr(ns, "worker", None) is None else str(ns.worker)
            return run_setup(ns.cmd, env, worker=(worker or "all"))
        # Otherwise, treat as a raw remote command
        cmd = " ".join(ns.rest) if getattr(ns, "rest", None) else ""
        worker = None if getattr(ns, "worker", None) is None else str(ns.worker)
        return mgr.raw(ns.cmd, cmd=cmd, worker=(worker or "all"))
    ap.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
