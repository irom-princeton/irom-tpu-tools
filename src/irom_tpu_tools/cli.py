from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .config import TPUEnvConfig
from .jobs import JobConfig, is_watcher_running, last_preempted, log_path, preemption_count, remove_job, running_since, stop_watcher
from .tpu import TPUManager


def _add_name_arg(p: argparse.ArgumentParser) -> None:
    """Add optional TPU name positional; falls back to TPU_NAME env."""
    p.add_argument("name", nargs="?", default=None, help="TPU name (default: $TPU_NAME env var)")


def _print_commands() -> None:
    """Print a nicely formatted, color-coded cheat-sheet of all commands."""
    from rich.console import Console
    from rich.text import Text

    c = Console()
    c.print()
    c.print(Text("  ⚡ TPU Tools — Command Reference", style="bold bright_cyan"))
    c.print(Text("  ─" * 28, style="dim"))
    c.print()

    sections = [
        (
            "🚀 Lifecycle",
            [
                ("tpu create v4 -n 8 --name my-tpu -- python train.py", "Create TPU, setup, launch training, start background watcher"),
                ("tpu delete my-tpu", "Delete TPU and stop its background watcher"),
                ("tpu stop my-tpu", "Stop TPU (preserve allocation, can restart later)"),
                ("tpu start my-tpu", "Restart a stopped TPU"),
            ],
        ),
        (
            "📋 Monitoring",
            [
                ("tpu list", "List all TPUs across all zones with watcher status"),
                ("tpu list v4", "List TPUs in a specific zone"),
                ("tpu status", "Show status of all managed jobs"),
                ("tpu status my-tpu", "Show status of a specific job"),
                ("tpu logs my-tpu", "View background watcher logs"),
                ("tpu logs my-tpu -f", "Follow watcher logs in real time"),
            ],
        ),
        (
            "🔗 Connect",
            [
                ("tpu attach my-tpu", "Attach to tmux session on worker 0"),
                ("tpu attach my-tpu --worker 1", "Attach to a specific worker"),
                ("tpu tail my-tpu", "Tail the training log on the TPU"),
                ("tpu tmux-ls my-tpu", "List tmux sessions on all workers"),
            ],
        ),
        (
            "🧹 Cleanup",
            [
                ("tpu nuke my-tpu", "Kill tmux + JAX processes + clean tmp (full reset)"),
                ("tpu kill-jax my-tpu", "Kill only JAX/XLA processes"),
                ("tpu tmux-kill-all my-tpu", "Kill tmux server on all workers"),
                ("tpu clean-tmp my-tpu", "Clean JAX/XLA temp files"),
                ("tpu clean my-tpu", "Truncate system logs to free disk"),
            ],
        ),
        (
            "🔧 Advanced",
            [
                ("tpu v4 -- ls -la", "Run raw SSH command on all v4 workers"),
                ("tpu v4 --worker 0 -- nvidia-smi", "Run raw command on a specific worker"),
                ("tpu v4 setup", "Re-run the setup step on v4 workers"),
                ("tpu watch v4 -n 8 -f", "[Legacy] Foreground watch loop"),
            ],
        ),
    ]

    for header, cmds in sections:
        c.print(Text(f"  {header}", style="bold yellow"))
        c.print()
        for cmd, desc in cmds:
            line = Text("    ")
            line.append(cmd, style="bold green")
            # Pad to align descriptions
            padding = max(1, 56 - len(cmd))
            line.append(" " * padding)
            line.append(desc, style="dim")
            c.print(line)
        c.print()

    c.print(Text("  💡 Tip: ", style="bold bright_magenta"), end="")
    c.print(Text("Most commands auto-detect the TPU zone — just pass the name!", style="bright_magenta"))
    c.print()


def build_parser() -> argparse.ArgumentParser:
    prog_name = (sys.argv[0].rsplit("/", 1)[-1] or "tpu") if getattr(sys, "argv", None) else "tpu"
    ap = argparse.ArgumentParser(prog=prog_name, description="Unified TPU utilities for v4/v5/v6")
    ap.add_argument("--commands", action="store_true", help="Show example commands with explanations")
    sub = ap.add_subparsers(dest="cmd", required=False)

    # --- create: provision + setup + launch + background watcher ---
    p_create = sub.add_parser("create", help="Create TPU, run setup, launch training, start background watcher")
    p_create.add_argument("version", choices=["v4", "v5", "v6"], help="TPU version")
    p_create.add_argument("--name", default=None, help="TPU name (default: $TPU_NAME env var)")
    p_create.add_argument("--tpu-num", "-n", type=int, default=8, help="TPU chips")
    p_create.add_argument("--branch", "-b", default="main", help="Git branch to checkout")
    p_create.add_argument("--setup-cmd", "-s", default="uv sync", help="Setup command after clone")

    # --- watch (legacy, kept for backwards compat) ---
    p_watch = sub.add_parser("watch", help="[Legacy] Watch TPU state in foreground")
    p_watch.add_argument("version", choices=["v4", "v5", "v6"], help="TPU version to target")
    p_watch.add_argument("--force", "-f", action="store_true", help="Force setup and training even if READY")
    p_watch.add_argument("--tpu-num", "-n", type=int, default=8, help="TPU chips")

    # --- list (optional version filter, shows watcher status) ---
    p_list = sub.add_parser("list", help="List TPUs with watcher status")
    p_list.add_argument("version", nargs="?", choices=["v4", "v5", "v6"], default=None, help="Filter by version (omit for all)")

    # --- status: show all managed jobs ---
    p_status = sub.add_parser("status", help="Show status of managed TPU jobs")
    p_status.add_argument("name", nargs="?", default=None, help="Specific job name (omit for all)")

    # --- logs: tail watcher log ---
    p_logs = sub.add_parser("logs", help="Tail the watcher log for a job")
    _add_name_arg(p_logs)
    p_logs.add_argument("--lines", "-n", type=int, default=50, help="Number of lines to show")
    p_logs.add_argument("--follow", "-f", action="store_true", help="Follow log output")

    # --- per-TPU commands: take optional name, auto-detect version/zone ---
    p_delete = sub.add_parser("delete", help="Delete a TPU (also stops watcher)")
    _add_name_arg(p_delete)

    p_stop = sub.add_parser("stop", help="Stop a TPU (preserve allocation)")
    _add_name_arg(p_stop)

    p_start = sub.add_parser("start", help="Start a stopped TPU")
    _add_name_arg(p_start)

    p_tmux = sub.add_parser("tmux", help="Run a tmux command on all workers")
    _add_name_arg(p_tmux)
    p_tmux.add_argument("--session", default="tpu")
    p_tmux.add_argument("rest", nargs=argparse.REMAINDER, help="Command to run in tmux session")

    p_attach = sub.add_parser("attach", help="Attach to tmux on a worker")
    _add_name_arg(p_attach)
    p_attach.add_argument("--session", default="tpu")
    p_attach.add_argument("--worker", type=int, default=0)

    p_ls = sub.add_parser("tmux-ls", help="List tmux sessions on all workers")
    _add_name_arg(p_ls)

    p_tail = sub.add_parser("tail", help="Show last 50 lines of latest tmux log on a worker")
    _add_name_arg(p_tail)
    p_tail.add_argument("--worker", type=int, default=0)

    p_kill = sub.add_parser("tmux-kill-all", help="Kill tmux server on all workers")
    _add_name_arg(p_kill)

    p_kill_jax = sub.add_parser("kill-jax", help="Kill JAX/XLA processes on all workers")
    _add_name_arg(p_kill_jax)

    p_clean = sub.add_parser("clean-tmp", help="Clean JAX/XLA tmp files on all workers")
    _add_name_arg(p_clean)

    p_clean_logs = sub.add_parser("clean", help="Truncate system logs on all workers")
    _add_name_arg(p_clean_logs)

    p_nuke = sub.add_parser("nuke", help="Kill tmux, JAX, and clean tmp on all workers")
    _add_name_arg(p_nuke)

    # --- raw SSH commands (version is the subcommand itself) ---
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


def _resolve_mgr(env: TPUEnvConfig, name: str | None) -> TPUManager:
    """Create a TPUManager and resolve the TPU name to its version/zone."""
    mgr = TPUManager(env)
    tpu_name = name or env.tpu_name
    if not tpu_name:
        raise SystemExit("Error: no TPU name provided and TPU_NAME is not set")
    print(f"Resolving TPU '{tpu_name}'...")
    try:
        resolved = mgr.resolve(tpu_name)
    except RuntimeError as e:
        raise SystemExit(f"Error: {e}") from None
    print(f"Found: {tpu_name} -> {resolved.version} ({resolved._zone})")
    return resolved


# ---- list with watcher status ----


def _list_tpus_in_zone(project: str, zone: str) -> list[dict]:
    """Query gcloud for TPUs in a zone, return parsed JSON list."""
    proc = subprocess.run(
        [
            "gcloud", "compute", "tpus", "tpu-vm", "list",
            "--zone", zone, "--project", project,
            "--format=json(name,state,acceleratorType)",
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []


_STATE_DISPLAY = {
    "NOT_FOUND": "-",
    "UNKNOWN": "?",
}


def _print_tpu_table(rows: list[dict]) -> None:
    """Print a formatted table of TPU rows."""
    if not rows:
        print("  (none)")
        return
    # Remap internal states to friendlier display names
    for r in rows:
        r["state"] = _STATE_DISPLAY.get(r.get("state", ""), r.get("state", ""))
    # Column widths
    headers = ["NAME", "STATE", "ACCELERATOR", "WATCHER", "RUNNING SINCE", "#PREEMPTIONS", "LAST PREEMPTED"]
    keys = ["name", "state", "accel", "watcher", "running", "pcount", "preempted"]
    widths = [len(h) for h in headers]
    for r in rows:
        for i, k in enumerate(keys):
            widths[i] = max(widths[i], len(r.get(k, "")))

    fmt = "  " + "  ".join(f"{{:<{{w{i}}}}}" for i in range(len(headers)))
    kw = {f"w{i}": w for i, w in enumerate(widths)}
    print(fmt.format(*headers, **kw))
    print(fmt.format(*["-" * w for w in widths], **kw))
    for r in rows:
        print(fmt.format(*[r.get(k, "") for k in keys], **kw))


_DINO = r"""
           ___
          / `_)
   .-^^^-/ /
__/       /
<__.|_|-|_|   < hello from irom dino
"""


def _do_list(env: TPUEnvConfig, version: str | None) -> int:
    """List TPUs with watcher status."""
    print(_DINO)
    project = env.tpu_project
    zones = {version: env.zones[version]} if version else env.zones

    for ver, zone in zones.items():
        print(f"--- {ver} ({zone}) ---")
        tpus = _list_tpus_in_zone(project, zone)
        rows = []
        for t in tpus:
            name = t.get("name", "").rsplit("/", 1)[-1]  # strip resource path
            accel = t.get("acceleratorType", "").rsplit("/", 1)[-1]
            state = t.get("state", "UNKNOWN")
            watcher = "running" if is_watcher_running(name) else "-"
            running = running_since(name) or "-"
            pcount = str(preemption_count(name))
            preempted = last_preempted(name) or "-"
            rows.append({"name": name, "state": state, "accel": accel, "watcher": watcher, "running": running, "pcount": pcount, "preempted": preempted})
        _print_tpu_table(rows)
        print()
    return 0


# ---- status ----


def _do_status(env: TPUEnvConfig, name: str | None) -> int:
    """Show status of managed jobs."""
    names = [name] if name else JobConfig.all_names()
    if not names:
        print("No managed jobs. Use `tpu create` to start one.")
        return 0

    rows = []
    for n in names:
        try:
            job = JobConfig.load(n)
        except FileNotFoundError:
            if name:
                print(f"No managed job named '{n}'.")
                return 1
            continue

        watcher = "running" if is_watcher_running(n) else "stopped"

        # Query TPU state
        mgr = TPUManager(env).for_tpu(n, job.version, env.zones[job.version])
        try:
            state = mgr.describe(job.version)
        except Exception:
            state = "UNKNOWN"

        running = running_since(n) or "-"
        pcount = str(preemption_count(n))
        preempted = last_preempted(n) or "-"
        rows.append({
            "name": n,
            "state": state,
            "accel": f"{job.version}-{job.tpu_num}",
            "watcher": watcher,
            "running": running,
            "pcount": pcount,
            "preempted": preempted,
        })

    _print_tpu_table(rows)
    return 0


# ---- create ----


def _do_create(ns: argparse.Namespace, env: TPUEnvConfig, extra_args: list[str]) -> int:
    """Submit a TPU job — saves config and spawns a background daemon that handles
    creation, setup, training launch, and preemption recovery."""
    from .watch import _map_v4_topology, spawn_watcher

    tpu_name = ns.name or env.tpu_name
    if not tpu_name:
        raise SystemExit("Error: no TPU name provided (use --name or set TPU_NAME)")

    command = " ".join(extra_args)  # empty string if no command provided
    topology = _map_v4_topology(ns.tpu_num) if ns.version == "v4" else None

    job = JobConfig(
        name=tpu_name,
        version=ns.version,
        tpu_num=ns.tpu_num,
        command=command,
        branch=ns.branch,
        setup_cmd=ns.setup_cmd,
        topology=topology,
    )

    job.save()

    # Spawn background daemon — it handles create, setup, training, and recovery
    pid = spawn_watcher(job, env)
    print(f"Submitted TPU job '{tpu_name}' ({ns.version}-{ns.tpu_num})")
    if command:
        print(f"  Command: {command}")
    print(f"  Watcher PID: {pid}")
    print(f"  Log file:    ~/.tpu-jobs/{tpu_name}/watch.log")
    print()
    print(f"  tpu status             Check job status")
    print(f"  tpu logs {tpu_name:<14s} View watcher log")
    print(f"  tpu logs {tpu_name:<14s} -f  Follow log in real time")
    print(f"  tpu delete {tpu_name:<12s} Stop and delete")
    return 0


# ---- logs ----


def _do_logs(ns: argparse.Namespace) -> int:
    name = ns.name
    if not name:
        raise SystemExit("Error: no TPU name provided")
    lp = log_path(name)
    if not lp.exists():
        print(f"No watcher log found for '{name}' (expected at {lp})")
        return 1
    import subprocess
    args = ["tail", f"-n{ns.lines}"]
    if ns.follow:
        args.append("-f")
    args.append(str(lp))
    try:
        return subprocess.run(args).returncode
    except KeyboardInterrupt:
        return 0


# ---- main ----


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    ns, unknown = ap.parse_known_args(argv)

    # --- --commands flag ---
    if getattr(ns, "commands", False):
        _print_commands()
        return 0

    if not ns.cmd:
        ap.print_help()
        return 0

    # --- create ---
    if ns.cmd == "create":
        extra = unknown
        if extra and extra[0] == "--":
            extra = extra[1:]
        env = TPUEnvConfig.from_env(require_tpu_name=not ns.name)
        return _do_create(ns, env, extra)

    # --- watch (legacy) ---
    if ns.cmd == "watch":
        from .watch import main as _watch_main

        return _watch_main([ns.version, *((ns.force and ["--force"]) or []), "-n", str(ns.tpu_num), *unknown])

    # --- list ---
    if ns.cmd == "list":
        env = TPUEnvConfig.from_env(require_tpu_name=False)
        return _do_list(env, ns.version)

    # --- status ---
    if ns.cmd == "status":
        env = TPUEnvConfig.from_env(require_tpu_name=False)
        return _do_status(env, ns.name)

    # --- logs ---
    if ns.cmd == "logs":
        return _do_logs(ns)

    # --- raw SSH shortcuts (v4/v5/v6 subcommands) ---
    if ns.cmd in {"v4", "v5", "v6"}:
        env = TPUEnvConfig.from_env()
        mgr = TPUManager(env)
        if getattr(ns, "rest", None) and len(ns.rest) >= 1 and ns.rest[0] == "setup":
            from .watch import run_setup

            worker = None if getattr(ns, "worker", None) is None else str(ns.worker)
            return run_setup(ns.cmd, env, worker=(worker or "all"))
        cmd = " ".join(ns.rest) if getattr(ns, "rest", None) else ""
        worker = None if getattr(ns, "worker", None) is None else str(ns.worker)
        return mgr.raw(ns.cmd, cmd=cmd, worker=(worker or "all"))

    # --- all other commands: resolve TPU by name ---
    env = TPUEnvConfig.from_env(require_tpu_name=not getattr(ns, "name", None))
    name = getattr(ns, "name", None)

    # delete also stops watcher
    if ns.cmd == "delete":
        tpu_name = name or env.tpu_name
        if not tpu_name:
            raise SystemExit("Error: no TPU name provided and TPU_NAME is not set")
        # Stop watcher if one is running
        if is_watcher_running(tpu_name):
            print(f"Stopping watcher for '{tpu_name}'...")
            stop_watcher(tpu_name)
        # Remove job state
        remove_job(tpu_name)
        # Try to resolve and delete the actual TPU (may already be gone)
        try:
            mgr = _resolve_mgr(env, name)
            ok = mgr.delete(mgr.version)
            return 0 if ok else 1
        except SystemExit:
            print(f"TPU '{tpu_name}' not found (already deleted or never created). Job state cleaned up.")
            return 0

    mgr = _resolve_mgr(env, name)
    v = mgr.version

    if ns.cmd == "stop":
        ok = mgr.stop(v)
        return 0 if ok else 1
    if ns.cmd == "start":
        ok = mgr.start(v)
        return 0 if ok else 1
    if ns.cmd == "tmux":
        cmd = " ".join(ns.rest) if getattr(ns, "rest", None) else ""
        ok = mgr.tmux(v, cmd=cmd, session=ns.session)
        return 0 if ok else 1
    if ns.cmd == "attach":
        return mgr.attach(v, session=ns.session, worker=ns.worker)
    if ns.cmd == "tmux-ls":
        ok = mgr.tmux_ls(v)
        return 0 if ok else 1
    if ns.cmd == "tail":
        return mgr.tail_log(v, worker=ns.worker)
    if ns.cmd == "tmux-kill-all":
        ok = mgr.tmux_kill_all(v)
        return 0 if ok else 1
    if ns.cmd == "kill-jax":
        ok = mgr.kill_jax(v)
        return 0 if ok else 1
    if ns.cmd == "clean-tmp":
        ok = mgr.clean_jax_tmp(v)
        return 0 if ok else 1
    if ns.cmd == "clean":
        ok = mgr.clean_logs(v)
        return 0 if ok else 1
    if ns.cmd == "nuke":
        ok = mgr.nuke_all(v)
        return 0 if ok else 1

    ap.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
