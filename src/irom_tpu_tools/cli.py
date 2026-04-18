from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

from .config import TPUEnvConfig
from .jobs import (
    JobConfig,
    is_watcher_running,
    last_preempted,
    log_path,
    preemption_count,
    remove_job,
    running_since,
    stop_watcher,
)
from .tpu import TPUManager

_ALLOWED_NAMES = {
    "tenny",
    "asher",
    "yanbo",
    "ola",
    "may",
    "apurva",
    "michael",
    "shresth",
    "lihan",
    "catherine",
}
_TPU_NAME_RE = re.compile(r"^v\d+-\d+-\d+-(.+)$")


def _creator_from_name(name: str) -> str:
    m = _TPU_NAME_RE.match(name)
    return m.group(1) if m else "-"


def _validate_tpu_name(name: str) -> None:
    m = _TPU_NAME_RE.match(name)
    if not m:
        raise SystemExit(
            f"Error: TPU name '{name}' does not match required format "
            "<tpu_type>-<num_tpus>-<index>-<your_name> (e.g. v6-64-01-lihan)"
        )
    your_name = m.group(1)
    if your_name not in _ALLOWED_NAMES:
        raise SystemExit(
            f"Error: TPU name suffix '{your_name}' is not allowed. "
            f"Must be one of: {', '.join(sorted(_ALLOWED_NAMES))}"
        )


def _add_name_arg(p: argparse.ArgumentParser) -> None:
    """Add optional TPU name positional; falls back to TPU_NAME env."""
    p.add_argument(
        "name", nargs="?", default=None, help="TPU name (default: $TPU_NAME env var)"
    )


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
                (
                    "tpu create v4 -n 8 --name my-tpu -- python train.py",
                    "Create TPU, setup, launch training, start background watcher",
                ),
                ("tpu delete my-tpu", "Delete TPU and stop its background watcher"),
                (
                    "tpu nuke my-tpu",
                    "Kill tmux + JAX processes + clean tmp (full reset) (preserve allocation, can restart later)",
                ),
            ],
        ),
        (
            "📋 Monitoring",
            [
                ("tpu list", "List all TPUs across all zones with watcher status"),
                ("tpu list v4", "List TPUs in a specific zone"),
                ("tpu status", "Show status of all managed jobs"),
                ("tpu status my-tpu", "Show status of a specific job"),
                (
                    "tpu info my-tpu",
                    "Show full job details (repo, setup, command, created, preemptions)",
                ),
                ("tpu logs my-tpu", "View background watcher logs"),
                ("tpu logs my-tpu -f", "Follow watcher logs in real time"),
            ],
        ),
        (
            "🔗 Connect",
            [
                ("tpu ssh my-tpu", "Open interactive SSH shell on worker 0 (no tmux)"),
                ("tpu ssh my-tpu --worker 1", "Open interactive SSH shell on a specific worker"),
                ("tpu attach my-tpu", "Attach to tmux session on worker 0"),
                ("tpu attach my-tpu --worker 1", "Attach to a specific worker"),
                ("tpu tail my-tpu", "Tail the training log on the TPU"),
                ("tpu tmux-ls my-tpu", "List tmux sessions on all workers"),
            ],
        ),
        (
            "🔧 Advanced",
            [
                ("tpu v4 -- ls -la", "Run raw SSH command on all v4 workers"),
                (
                    "tpu v4 --worker 0 -- nvidia-smi",
                    "Run raw command on a specific worker",
                ),
                ("tpu v4 setup", "Re-run the setup step on v4 workers"),
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
    c.print(
        Text(
            "Most commands auto-detect the TPU zone — just pass the name!",
            style="bright_magenta",
        )
    )
    c.print()


def build_parser() -> argparse.ArgumentParser:
    prog_name = (
        (sys.argv[0].rsplit("/", 1)[-1] or "tpu")
        if getattr(sys, "argv", None)
        else "tpu"
    )
    ap = argparse.ArgumentParser(
        prog=prog_name, description="Unified TPU utilities for v4/v5/v6"
    )
    ap.add_argument(
        "--commands",
        action="store_true",
        help="Show example commands with explanations",
    )
    sub = ap.add_subparsers(dest="cmd", required=False)

    # --- create: provision + setup + launch + background watcher ---
    p_create = sub.add_parser(
        "create",
        help="Create TPU, run setup, launch training, start background watcher",
    )
    p_create.add_argument("version", choices=["v4", "v5", "v6"], help="TPU version")
    p_create.add_argument(
        "--name", default=None, help="TPU name (default: $TPU_NAME env var)"
    )
    p_create.add_argument("--tpu-num", "-n", type=int, default=8, help="TPU chips")
    p_create.add_argument(
        "--repo",
        default=None,
        help="GitHub repo 'owner/name' to clone (omit for bare TPU)",
    )
    p_create.add_argument(
        "--branch",
        "-b",
        default="main",
        help="Git branch to checkout (ignored without --repo)",
    )
    p_create.add_argument(
        "--setup-cmd",
        "-s",
        default="uv sync",
        help="Setup command run inside the cloned repo (ignored without --repo)",
    )
    p_create.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="If TPU is already READY, re-run setup and command without prompting",
    )

    # --- list (optional version filter, shows watcher status) ---
    p_list = sub.add_parser("list", help="List TPUs with watcher status")
    p_list.add_argument(
        "version",
        nargs="?",
        choices=["v4", "v5", "v6"],
        default=None,
        help="Filter by version (omit for all)",
    )

    # --- status: show all managed jobs ---
    p_status = sub.add_parser("status", help="Show status of managed TPU jobs")
    p_status.add_argument(
        "name", nargs="?", default=None, help="Specific job name (omit for all)"
    )

    # --- info: show full details of a single managed job ---
    p_info = sub.add_parser(
        "info",
        help="Show full details of a managed TPU job (repo, setup, command, created, preemptions)",
    )
    p_info.add_argument(
        "name", nargs="?", default=None, help="Job name (default: $TPU_NAME env var)"
    )

    # --- logs: tail watcher log ---
    p_logs = sub.add_parser("logs", help="Tail the watcher log for a job")
    _add_name_arg(p_logs)
    p_logs.add_argument(
        "--lines", "-n", type=int, default=50, help="Number of lines to show"
    )
    p_logs.add_argument("--follow", "-f", action="store_true", help="Follow log output")

    # --- per-TPU commands: take optional name, auto-detect version/zone ---
    p_delete = sub.add_parser("delete", help="Delete a TPU (also stops watcher)")
    _add_name_arg(p_delete)

    p_tmux = sub.add_parser("tmux", help="Run a tmux command on all workers")
    _add_name_arg(p_tmux)
    p_tmux.add_argument("--session", default="tpu")
    p_tmux.add_argument(
        "rest", nargs=argparse.REMAINDER, help="Command to run in tmux session"
    )

    p_ssh = sub.add_parser("ssh", help="Open interactive SSH shell on a worker (no tmux)")
    _add_name_arg(p_ssh)
    p_ssh.add_argument("--worker", type=int, default=0)

    p_attach = sub.add_parser("attach", help="Attach to tmux on a worker")
    _add_name_arg(p_attach)
    p_attach.add_argument("--session", default="tpu")
    p_attach.add_argument("--worker", type=int, default=0)

    p_tail = sub.add_parser(
        "tail", help="Show last 50 lines of latest tmux log on a worker"
    )
    _add_name_arg(p_tail)
    p_tail.add_argument("--worker", type=int, default=0)

    p_clean_logs = sub.add_parser("clean", help="Truncate system logs on all workers")
    _add_name_arg(p_clean_logs)

    p_nuke = sub.add_parser("nuke", help="Kill tmux, JAX, and clean tmp on all workers")
    _add_name_arg(p_nuke)

    # --- raw SSH commands (version is the subcommand itself) ---
    p_v4 = sub.add_parser("v4", help="Run raw command on v4 workers (no tmux)")
    p_v4.add_argument(
        "--worker", type=int, default=None, help="Worker index (default: all)"
    )
    p_v4.add_argument("rest", nargs=argparse.REMAINDER, help="Command to run remotely")
    p_v5 = sub.add_parser("v5", help="Run raw command on v5 workers (no tmux)")
    p_v5.add_argument(
        "--worker", type=int, default=None, help="Worker index (default: all)"
    )
    p_v5.add_argument("rest", nargs=argparse.REMAINDER, help="Command to run remotely")
    p_v6 = sub.add_parser("v6", help="Run raw command on v6 workers (no tmux)")
    p_v6.add_argument(
        "--worker", type=int, default=None, help="Worker index (default: all)"
    )
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
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            "--zone",
            zone,
            "--project",
            project,
            "--format=json(name,state,acceleratorType)",
        ],
        capture_output=True,
        text=True,
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
    headers = [
        "NAME",
        "CREATOR",
        "STATE",
        "ACCELERATOR",
        "WATCHER",
        "RUNNING SINCE",
        "#PREEMPTIONS",
        "LAST PREEMPTED",
    ]
    keys = [
        "name",
        "creator",
        "state",
        "accel",
        "watcher",
        "running",
        "pcount",
        "preempted",
    ]
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
            rows.append(
                {
                    "name": name,
                    "creator": _creator_from_name(name),
                    "state": state,
                    "accel": accel,
                    "watcher": watcher,
                    "running": running,
                    "pcount": pcount,
                    "preempted": preempted,
                }
            )
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
        rows.append(
            {
                "name": n,
                "state": state,
                "accel": f"{job.version}-{job.tpu_num}",
                "watcher": watcher,
                "running": running,
                "pcount": pcount,
                "preempted": preempted,
            }
        )

    _print_tpu_table(rows)
    return 0


# ---- info ----


def _gcloud_describe_json(
    project: str, zone: str, name: str, timeout_s: int = 20
) -> dict:
    """Query full TPU details via gcloud describe --format=json. Returns {} on error."""
    try:
        proc = subprocess.run(
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
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError):
        return {}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}


def _do_info(env: TPUEnvConfig, name: str | None) -> int:
    """Show full details for a single managed job."""
    from rich.console import Console
    from rich.text import Text

    tpu_name = name or env.tpu_name
    if not tpu_name:
        raise SystemExit("Error: no TPU name provided and TPU_NAME is not set")

    try:
        job = JobConfig.load(tpu_name)
    except FileNotFoundError:
        print(f"No managed job named '{tpu_name}'.")
        return 1

    zone = env.zones[job.version]
    details = _gcloud_describe_json(env.tpu_project, zone, tpu_name)
    state = details.get("state", "UNKNOWN")
    health = details.get("health", "-")
    created = details.get("createTime", "-")
    accel = (details.get("acceleratorType") or f"{job.version}-{job.tpu_num}").rsplit(
        "/", 1
    )[-1]

    watcher = "running" if is_watcher_running(tpu_name) else "stopped"
    running = running_since(tpu_name) or "-"
    pcount = preemption_count(tpu_name)
    preempted = last_preempted(tpu_name) or "-"

    c = Console()
    c.print()
    c.print(Text(f"  TPU: {tpu_name}", style="bold bright_cyan"))
    c.print(Text("  ─" * 28, style="dim"))

    rows = [
        ("Accelerator", accel),
        ("Creator", _creator_from_name(tpu_name)),
        ("Zone", zone),
        ("State", state),
        ("Health", health),
        ("Created", created),
        ("Watcher", watcher),
        ("Running since", running),
        ("Preemptions", str(pcount)),
        ("Last preempted", preempted),
        ("Repo", job.repo or "(bare — no clone)"),
        ("Branch", job.branch if job.repo else "-"),
        ("Setup", job.setup_cmd if job.repo else "-"),
        ("Command", job.command or "-"),
    ]
    label_w = max(len(k) for k, _ in rows)
    for k, v in rows:
        line = Text("  ")
        line.append(f"{k:<{label_w}}  ", style="bold yellow")
        line.append(str(v))
        c.print(line)
    c.print()
    return 0


# ---- create ----


def _do_create(ns: argparse.Namespace, env: TPUEnvConfig, extra_args: list[str]) -> int:
    """Submit a TPU job — saves config and spawns a background daemon that handles
    creation, setup, training launch, and preemption recovery."""
    from .watch import _map_v4_topology, spawn_watcher

    tpu_name = ns.name or env.tpu_name
    if not tpu_name:
        raise SystemExit("Error: no TPU name provided (use --name or set TPU_NAME)")
    _validate_tpu_name(tpu_name)

    command = " ".join(extra_args)  # empty string if no command provided
    topology = _map_v4_topology(ns.tpu_num) if ns.version == "v4" else None

    # --repo resolution: explicit flag wins; else fall back to env vars; else bare.
    if ns.repo is not None:
        repo = ns.repo.strip()
    elif env.gh_owner and env.gh_repo_name:
        repo = f"{env.gh_owner}/{env.gh_repo_name}"
    else:
        repo = ""

    # Check if the TPU already exists before spawning the watcher.
    force_run = getattr(ns, "force", False)
    try:
        zone = env.zones[ns.version]
        pinned_mgr = TPUManager(env).for_tpu(tpu_name, ns.version, zone)
        current_state = pinned_mgr.describe(ns.version)
    except Exception:
        current_state = "UNKNOWN"

    if current_state == "READY":
        try:
            is_busy = pinned_mgr.check_activity(ns.version)
            activity = (
                "busy (JAX/Python processes detected)"
                if is_busy
                else "idle (no JAX processes detected)"
            )
        except Exception:
            activity = "activity unknown"

        print(f"TPU '{tpu_name}' is already READY — {activity}.")
        print(
            "  The TPU will NOT be re-created; setup and command will be re-run on the existing TPU."
        )
        if force_run:
            print("  --force specified: skipping confirmation.")
        else:
            ans = input("  Proceed? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                print("Aborted.")
                return 0
            force_run = True

    job = JobConfig(
        name=tpu_name,
        version=ns.version,
        tpu_num=ns.tpu_num,
        command=command,
        branch=ns.branch,
        setup_cmd=ns.setup_cmd,
        repo=repo,
        topology=topology,
    )

    job.save()

    # Spawn background daemon — it handles create, setup, training, and recovery
    pid = spawn_watcher(job, env, force_run=force_run)
    print(f"Submitted TPU job '{tpu_name}' ({ns.version}-{ns.tpu_num})")
    print(f"  Repo:    {repo or '(bare TPU — no clone)'}")
    if command:
        print(f"  Command: {command}")
    print(f"  Watcher PID: {pid}")
    print(f"  Log file:    ~/.tpu-jobs/{tpu_name}/watch.log")
    print()
    print("  tpu status             Check job status")
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

        return _watch_main(
            [
                ns.version,
                *((ns.force and ["--force"]) or []),
                "-n",
                str(ns.tpu_num),
                *unknown,
            ]
        )

    # --- list ---
    if ns.cmd == "list":
        env = TPUEnvConfig.from_env(require_tpu_name=False)
        return _do_list(env, ns.version)

    # --- status ---
    if ns.cmd == "status":
        env = TPUEnvConfig.from_env(require_tpu_name=False)
        return _do_status(env, ns.name)

    # --- info ---
    if ns.cmd == "info":
        env = TPUEnvConfig.from_env(require_tpu_name=not ns.name)
        return _do_info(env, ns.name)

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
            print(
                f"TPU '{tpu_name}' not found (already deleted or never created). Job state cleaned up."
            )
            return 0

    mgr = _resolve_mgr(env, name)
    v = mgr.version
    if ns.cmd == "ssh":
        return mgr.ssh(v, worker=ns.worker)
    if ns.cmd == "tmux":
        cmd = " ".join(ns.rest) if getattr(ns, "rest", None) else ""
        ok = mgr.tmux(v, cmd=cmd, session=ns.session)
        return 0 if ok else 1
    if ns.cmd == "attach":
        return mgr.attach(v, session=ns.session, worker=ns.worker)
    if ns.cmd == "tail":
        return mgr.tail_log(v, worker=ns.worker)
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
