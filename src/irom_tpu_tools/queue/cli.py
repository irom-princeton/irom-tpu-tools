from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time

from .backend import Backend, DryRunBackend, GCPBackend
from .config import (
    QueueConfig,
    bucket_for_resource,
    load_config,
    resource_for_request,
)
from . import interactive as interactive_tools
from .packaging import compute_checksum, create_code_tarball, generate_job_id
from .scheduler import Scheduler
from .types import (
    JobResources,
    JobSpec,
    JobState,
    JobStatus,
    ResourceConfig,
    TERMINAL_STATUSES,
    utc_now,
)


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        print("(none)")
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def _parse_kv(items: list[str] | None, label: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"{label} must use KEY=VALUE format: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"{label} has empty key: {item}")
        values[key] = value
    return values


def _backend(args: argparse.Namespace) -> Backend:
    if getattr(args, "dry_run", False):
        base_dir = Path(args.base_dir or "/tmp/irom_tpu_queue_dry_run")
        base_dir.mkdir(parents=True, exist_ok=True)
        return DryRunBackend(
            str(base_dir),
            provision_delay_seconds=float(getattr(args, "provision_delay", 0.0)),
        )
    return GCPBackend()


def _load_config(args: argparse.Namespace) -> QueueConfig:
    return load_config(getattr(args, "config", None))


def _state_url(config: QueueConfig) -> str:
    return f"{config.primary_bucket}/scheduler_state.json"


def _load_scheduler_state(backend: Backend, config: QueueConfig) -> list[dict]:
    state_json = backend.read_gcs(_state_url(config))
    if not state_json:
        return _scan_jobs(backend, config)
    try:
        data = json.loads(state_json)
    except json.JSONDecodeError:
        return _scan_jobs(backend, config)
    return list(data.get("jobs", []))


def _scan_jobs(backend: Backend, config: QueueConfig) -> list[dict]:
    jobs = []
    for bucket in config.buckets.values():
        for job_dir in backend.list_gcs(f"{bucket}/jobs/"):
            job_id = job_dir.rstrip("/").rsplit("/", 1)[-1]
            spec_json = backend.read_gcs(f"{bucket}/jobs/{job_id}/spec.json")
            status_json = backend.read_gcs(f"{bucket}/jobs/{job_id}/status.json")
            if not spec_json:
                continue
            try:
                spec = JobSpec.from_dict(json.loads(spec_json))
                state = (
                    JobState.from_dict(json.loads(status_json))
                    if status_json
                    else JobState.new()
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            jobs.append(
                {
                    "job_id": job_id,
                    "bucket": bucket,
                    "job_dir": f"{bucket}/jobs/{job_id}",
                    "spec": spec.to_dict(),
                    "state": state.to_dict(),
                }
            )
    return jobs


def _resolve_job(
    backend: Backend, config: QueueConfig, job_ref: str
) -> tuple[JobSpec, JobState, str]:
    matches = []
    for entry in _load_scheduler_state(backend, config):
        spec = JobSpec.from_dict(entry["spec"])
        state = JobState.from_dict(entry["state"])
        if job_ref in {entry["job_id"], spec.job_id, spec.display_name}:
            matches.append((spec, state, entry["job_dir"]))
        elif spec.job_id.endswith(job_ref):
            matches.append((spec, state, entry["job_dir"]))
    if not matches:
        raise SystemExit(f"Job not found: {job_ref}")
    if len(matches) > 1:
        names = ", ".join(m[0].job_id for m in matches)
        raise SystemExit(f"Job reference is ambiguous: {job_ref} ({names})")
    return matches[0]


def _resource_to_job_resources(resource: ResourceConfig) -> JobResources:
    return JobResources(
        resource_name=resource.name,
        accelerator_type=resource.accelerator_type,
        zone=resource.zone,
        project=resource.project,
        chips=resource.chips,
        workers=resource.workers,
        runtime_version=resource.runtime_version,
    )


def cmd_create(args: argparse.Namespace) -> int:
    config = _load_config(args)
    backend = _backend(args)
    resource = resource_for_request(
        config,
        version=args.version,
        chips=args.tpu_num,
        resource_name=args.resource,
    )
    if not resource.enabled:
        print(f"Resource {resource.name} is disabled by config.")
        return 1

    command_parts = list(getattr(args, "command", []) or [])
    if command_parts and command_parts[0] == "--":
        command_parts = command_parts[1:]
    command = " ".join(command_parts) or "true"
    display_name = args.name or resource.name
    job_id = generate_job_id(display_name)
    bucket = bucket_for_resource(config, resource)
    job_dir = f"{bucket}/jobs/{job_id}"

    code_dir = Path(args.code_dir).expanduser().resolve()
    if not code_dir.exists():
        print(f"Code directory does not exist: {code_dir}")
        return 1

    env_vars = _parse_kv(args.env, "--env")
    if os.environ.get("WANDB_USER_EMAIL") and "WANDB_USER_EMAIL" not in env_vars:
        env_vars["WANDB_USER_EMAIL"] = os.environ["WANDB_USER_EMAIL"]
    secret_refs = dict(config.secrets)
    secret_refs.update(_parse_kv(args.secret, "--secret"))

    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / "code.tar.gz"
        try:
            create_code_tarball(code_dir, archive)
        except RuntimeError as exc:
            print(f"Error: {exc}")
            return 1
        checksum = compute_checksum(archive)
        code_url = f"{job_dir}/code.tar.gz"
        if not backend.upload_file(str(archive), code_url):
            print(f"Failed to upload code archive to {code_url}")
            return 1

    submitted_by = args.user or os.environ.get("TPU_QUEUE_USER") or os.environ.get("USER") or "unknown"
    spec = JobSpec(
        job_id=job_id,
        display_name=display_name,
        code_tar_url=code_url,
        code_checksum=checksum,
        command=command,
        setup_cmd=args.setup_cmd,
        resources=_resource_to_job_resources(resource),
        max_attempts=args.max_attempts,
        submit_time=utc_now(),
        submitted_by=submitted_by,
        priority=args.priority,
        tags=args.tag or [],
        env_vars=env_vars,
        secret_refs=secret_refs,
        run_on_all_workers=not args.worker0_only,
    )
    state = JobState.new()
    backend.write_gcs(f"{job_dir}/spec.json", json.dumps(spec.to_dict(), indent=2))
    backend.write_gcs(f"{job_dir}/status.json", json.dumps(state.to_dict(), indent=2))

    print(f"Submitted job: {job_id}")
    print(f"  Name:     {display_name}")
    print(f"  Resource: {resource.name} ({resource.accelerator_type}, {resource.chips} chips)")
    print(f"  User:     {submitted_by}")
    print(f"  Command:  {command}")
    print()
    print(f"  tpu status {job_id}")
    print(f"  tpu logs {job_id}")
    print(f"  tpu delete {job_id}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    config = _load_config(args)
    backend = _backend(args)
    mode = (
        "resources"
        if getattr(args, "resources", False)
        else "jobs"
        if getattr(args, "jobs", False)
        else "auto"
    )
    job_only_filters = bool(args.user or args.active or args.status)
    if mode == "resources":
        if job_only_filters:
            raise SystemExit("--user, --active, and --status only apply to --jobs")
        return _print_resource_catalog(config, version=args.version)

    entries = _load_scheduler_state(backend, config)
    rows = []
    for entry in entries:
        spec = JobSpec.from_dict(entry["spec"])
        state = JobState.from_dict(entry["state"])
        if args.version and not spec.resources.resource_name.startswith(f"{args.version}-"):
            continue
        if args.user and spec.submitted_by != args.user:
            continue
        if args.active and state.status in TERMINAL_STATUSES:
            continue
        if args.status and state.status.value != args.status.upper():
            continue
        status = state.current_qr_state if state.current_qr_state else state.status.value
        rows.append(
            [
                spec.job_id,
                spec.display_name,
                status,
                f"{state.current_attempt}/{spec.max_attempts}",
                spec.resources.resource_name,
                str(spec.resources.chips),
                spec.submitted_by,
                spec.submit_time[:19],
            ]
        )
    rows.sort(key=lambda row: (row[2] in {"SUCCEEDED", "FAILED", "CANCELED"}, row[-1], row[0]))
    if mode == "auto" and not rows and not job_only_filters:
        if args.version:
            print(f"No queued {args.version} jobs found. Requestable resources:")
        else:
            print("No queued jobs found. Requestable resources:")
        print()
        return _print_resource_catalog(config, version=args.version)
    _print_table(
        ["JOB ID", "NAME", "STATUS", "ATT", "RESOURCE", "CHIPS", "USER", "SUBMITTED"],
        rows,
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config = _load_config(args)
    backend = _backend(args)
    spec, state, job_dir = _resolve_job(backend, config, args.job)
    print(f"Job: {spec.job_id}")
    print(f"  Name:       {spec.display_name}")
    print(f"  Status:     {state.status.value}")
    if state.current_qr_state:
        print(f"  QR state:   {state.current_qr_state}")
    if state.current_qr_name:
        print(f"  QR:         {state.current_qr_name}")
    print(f"  Attempt:    {state.current_attempt}/{spec.max_attempts}")
    print(f"  Resource:   {spec.resources.resource_name} ({spec.resources.accelerator_type})")
    print(f"  Chips:      {spec.resources.chips}")
    print(f"  Zone:       {spec.resources.zone}")
    print(f"  User:       {spec.submitted_by}")
    print(f"  Submitted:  {spec.submit_time}")
    print(f"  Job dir:    {job_dir}")
    print(f"  Command:    {spec.command}")
    if spec.setup_cmd:
        print(f"  Setup:      {spec.setup_cmd}")
    if state.provisioned_at:
        print(f"  Provisioned:{state.provisioned_at}")
    if state.attempts:
        print()
        print("Attempts:")
        for attempt in state.attempts:
            suffix = f" error={attempt.error}" if attempt.error else ""
            print(
                f"  {attempt.attempt}: {attempt.qr_name} "
                f"{attempt.started_at} -> {attempt.ended_at}{suffix}"
            )
    return 0


def _latest_attempt(state: JobState) -> int:
    return max(state.current_attempt + 1, len(state.attempts), 1)


def _read_log_content(
    backend: Backend, job_dir: str, attempt: int, worker: int | None
) -> str:
    prefix = f"{job_dir}/logs/attempt-{attempt}/"
    logs = backend.list_gcs(prefix)
    if worker is not None:
        logs = [url for url in logs if url.endswith(f"worker-{worker}.log")]
    content = []
    for url in sorted(logs):
        text = backend.read_gcs(url)
        if text is not None:
            content.append(f"=== {url} ===\n{text}")
    return "\n".join(content)


def cmd_logs(args: argparse.Namespace) -> int:
    config = _load_config(args)
    backend = _backend(args)
    _, state, job_dir = _resolve_job(backend, config, args.job)
    attempt = args.attempt or _latest_attempt(state)
    content = _read_log_content(backend, job_dir, attempt, args.worker)
    if not content:
        print(f"No logs found for attempt {attempt}.")
        return 0
    lines = content.splitlines()
    if args.lines and len(lines) > args.lines:
        lines = lines[-args.lines :]
    print("\n".join(lines))
    return 0


def cmd_tail(args: argparse.Namespace) -> int:
    config = _load_config(args)
    backend = _backend(args)
    _, state, job_dir = _resolve_job(backend, config, args.job)
    attempt = args.attempt or _latest_attempt(state)
    printed = 0
    try:
        while True:
            content = _read_log_content(backend, job_dir, attempt, args.worker)
            if content:
                chunk = content[printed:]
                if chunk:
                    print(chunk, end="" if chunk.endswith("\n") else "\n")
                    printed = len(content)
            if not args.follow:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def cmd_delete(args: argparse.Namespace) -> int:
    config = _load_config(args)
    backend = _backend(args)
    spec, state, job_dir = _resolve_job(backend, config, args.job)
    if state.status in TERMINAL_STATUSES:
        print(f"Job is already terminal: {state.status.value}")
        return 0
    backend.write_gcs(f"{job_dir}/canceled", f"Canceled at {utc_now()}\n")
    print(f"Cancellation requested for {spec.job_id}.")
    print("The scheduler will delete the queued resource and TPU VM.")
    return 0


def cmd_retry(args: argparse.Namespace) -> int:
    config = _load_config(args)
    backend = _backend(args)
    spec, state, job_dir = _resolve_job(backend, config, args.job)
    if state.status != JobStatus.FAILED:
        print(f"Job is not FAILED: {state.status.value}")
        return 1
    backend.write_gcs(f"{job_dir}/retry", f"Retry requested at {utc_now()}\n")
    print(f"Retry requested for {spec.job_id}.")
    return 0


def cmd_rerun(args: argparse.Namespace) -> int:
    config = _load_config(args)
    backend = _backend(args)
    old_spec, _, _ = _resolve_job(backend, config, args.job)
    resource = config.resources[old_spec.resources.resource_name]
    job_id = generate_job_id(args.name or old_spec.display_name)
    bucket = bucket_for_resource(config, resource)
    job_dir = f"{bucket}/jobs/{job_id}"
    spec = JobSpec(
        job_id=job_id,
        display_name=args.name or old_spec.display_name,
        code_tar_url=old_spec.code_tar_url,
        code_checksum=old_spec.code_checksum,
        command=old_spec.command,
        setup_cmd=old_spec.setup_cmd,
        resources=old_spec.resources,
        max_attempts=args.max_attempts or old_spec.max_attempts,
        submit_time=utc_now(),
        submitted_by=os.environ.get("TPU_QUEUE_USER") or os.environ.get("USER") or old_spec.submitted_by,
        priority=args.priority if args.priority is not None else old_spec.priority,
        tags=old_spec.tags,
        env_vars=old_spec.env_vars,
        secret_refs=old_spec.secret_refs,
        run_on_all_workers=old_spec.run_on_all_workers,
    )
    backend.write_gcs(f"{job_dir}/spec.json", json.dumps(spec.to_dict(), indent=2))
    backend.write_gcs(f"{job_dir}/status.json", json.dumps(JobState.new().to_dict(), indent=2))
    print(f"Submitted rerun: {job_id}")
    return 0


def _active_usage(
    entries: list[dict], config: QueueConfig
) -> tuple[dict[str, int], dict[str, int]]:
    by_quota: dict[str, int] = {}
    by_user: dict[str, int] = {}
    for entry in entries:
        spec = JobSpec.from_dict(entry["spec"])
        state = JobState.from_dict(entry["state"])
        if state.status not in {JobStatus.PROVISIONING, JobStatus.RUNNING}:
            continue
        resource = config.resources.get(spec.resources.resource_name)
        group = resource.quota_group if resource else spec.resources.resource_name
        by_quota[group] = by_quota.get(group, 0) + spec.resources.chips
        by_user[spec.submitted_by] = by_user.get(spec.submitted_by, 0) + spec.resources.chips
    return by_quota, by_user


def cmd_admin_jobs(args: argparse.Namespace) -> int:
    args.version = None
    args.user = getattr(args, "user", None)
    args.active = getattr(args, "active", False)
    args.status = getattr(args, "status", None)
    args.jobs = True
    args.resources = False
    return cmd_list(args)


def _print_resource_catalog(config: QueueConfig, *, version: str | None = None) -> int:
    resource_rows = [
        [
            r.name,
            r.accelerator_type,
            r.zone,
            str(r.chips),
            str(r.workers),
            r.quota_group,
            "yes" if r.enabled else "no",
        ]
        for r in sorted(config.resources.values(), key=lambda x: x.name)
        if version is None or r.version == version
    ]
    _print_table(
        ["RESOURCE", "ACCEL", "ZONE", "CHIPS", "WORKERS", "GROUP", "ENABLED"],
        resource_rows,
    )
    interactive_rows = [
        [
            t.name,
            ", ".join(t.aliases) if t.aliases else "-",
            t.zone,
            str(t.workers),
            t.description or "-",
        ]
        for t in sorted(config.interactive_tpus.values(), key=lambda x: x.name)
        if version is None or t.version == version
    ]
    if interactive_rows:
        print()
        print("Shared interactive TPUs:")
        _print_table(["NAME", "ALIASES", "ZONE", "WORKERS", "DESCRIPTION"], interactive_rows)
    return 0


def cmd_admin_resources(args: argparse.Namespace) -> int:
    config = _load_config(args)
    backend = _backend(args)
    entries = _load_scheduler_state(backend, config)
    by_quota, by_user = _active_usage(entries, config)
    rows = []
    for name, quota in sorted(config.quota_groups.items()):
        rows.append([name, str(by_quota.get(name, 0)), str(quota.total_chips)])
    print("Quota groups:")
    _print_table(["GROUP", "USED", "LIMIT"], rows)
    print()
    print("Enabled resources:")
    _print_resource_catalog(config)
    print()
    if by_user:
        print("Active chips by user:")
        _print_table(["USER", "CHIPS"], [[u, str(c)] for u, c in sorted(by_user.items())])
    return 0


def cmd_admin_qrs(args: argparse.Namespace) -> int:
    config = _load_config(args)
    backend = _backend(args)
    rows = []
    seen: set[tuple[str, str]] = set()
    for resource in config.resources.values():
        key = (resource.project, resource.zone)
        if key in seen:
            continue
        seen.add(key)
        for qr in backend.list_queued_resources(
            resource.project, resource.zone, f"{config.scheduler.qr_prefix}-"
        ):
            rows.append([qr, resource.project, resource.zone])
    _print_table(["QR", "PROJECT", "ZONE"], rows)
    return 0


def _heartbeat_age_seconds(
    backend: Backend, job_dir: str, state: JobState
) -> float | None:
    attempt = _latest_attempt(state)
    text = backend.read_gcs(f"{job_dir}/attempts/attempt-{attempt}/heartbeat")
    if not text:
        return None
    try:
        ts = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return (datetime.now(UTC) - ts).total_seconds()


def cmd_admin_cleanup(args: argparse.Namespace) -> int:
    config = _load_config(args)
    backend = _backend(args)
    entries = _load_scheduler_state(backend, config)
    tracked = {
        JobState.from_dict(e["state"]).current_qr_name: e
        for e in entries
        if JobState.from_dict(e["state"]).current_qr_name
    }
    actions: list[tuple[str, str, ResourceConfig, str | None]] = []
    seen: set[tuple[str, str]] = set()
    for resource in config.resources.values():
        key = (resource.project, resource.zone)
        if key in seen:
            continue
        seen.add(key)
        for qr in backend.list_queued_resources(
            resource.project, resource.zone, f"{config.scheduler.qr_prefix}-"
        ):
            entry = tracked.get(qr)
            if not entry:
                actions.append(("orphan", qr, resource, None))
                continue
            state = JobState.from_dict(entry["state"])
            if args.idle_minutes is not None:
                age = _heartbeat_age_seconds(backend, entry["job_dir"], state)
                if age is None or age >= args.idle_minutes * 60:
                    actions.append(("idle", qr, resource, entry["job_dir"]))
    if not actions:
        print("No queue-owned orphan or idle resources found.")
        return 0
    for reason, qr, resource, job_dir in actions:
        print(f"{reason}: {qr} ({resource.zone})")
        if args.yes:
            if job_dir:
                backend.write_gcs(f"{job_dir}/canceled", f"Admin cleanup at {utc_now()}\n")
            backend.delete_tpu_vm(qr, resource.project, resource.zone)
            backend.delete_queued_resource(qr, resource.project, resource.zone)
    if not args.yes:
        print()
        print("Dry run only. Re-run with --yes to delete these queue-owned resources.")
    return 0


def cmd_scheduler(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = _load_config(args)
    backend = _backend(args)
    scheduler = Scheduler(backend, config)
    if args.once:
        if isinstance(backend, DryRunBackend):
            backend.tick()
        scheduler.run_once()
        scheduler._maybe_write_scheduler_state(force=True)
        return 0
    while True:
        if isinstance(backend, DryRunBackend):
            backend.tick()
        scheduler.run_once()
        time.sleep(args.scan_interval or config.scheduler.scan_interval)


def _interactive_tpu(args: argparse.Namespace):
    return interactive_tools.resolve_interactive_tpu(_load_config(args), args.name)


def _worker_arg(value: str) -> int | str:
    return "all" if value == "all" else int(value)


def _validate_worker(tpu, worker: int | str) -> None:
    if worker == "all":
        return
    if worker < 0 or worker >= tpu.workers:
        raise SystemExit(
            f"Worker {worker} is outside configured range 0..{tpu.workers - 1} for {tpu.name}"
        )


def _command_from_args(parts: list[str]) -> str:
    command = list(parts or [])
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("No command provided after --")
    return " ".join(command)


def cmd_interactive_list(args: argparse.Namespace) -> int:
    config = _load_config(args)
    rows = interactive_tools.list_rows(config, live=args.live)
    _print_table(
        ["NAME", "ALIASES", "ZONE", "WORKERS", "ACCEL", "STATE", "HEALTH", "DESCRIPTION"],
        rows,
    )
    return 0


def cmd_interactive_info(args: argparse.Namespace) -> int:
    tpu = _interactive_tpu(args)
    print(f"Name:        {tpu.name}")
    print(f"Version:     {tpu.version}")
    print(f"Project:     {tpu.project}")
    print(f"Zone:        {tpu.zone}")
    print(f"Workers:     {tpu.workers}")
    print(f"Aliases:     {', '.join(tpu.aliases) if tpu.aliases else '-'}")
    print(f"Description: {tpu.description or '-'}")
    if args.live:
        data = interactive_tools.describe_interactive_tpu(tpu)
        print(f"State:       {data.get('state', 'UNKNOWN')}")
        print(f"Health:      {data.get('health', '-')}")
        if data.get("acceleratorType"):
            print(f"Accelerator: {str(data['acceleratorType']).rsplit('/', 1)[-1]}")
        if data.get("error"):
            print(f"Error:       {data['error']}")
    return 0


def cmd_interactive_ssh(args: argparse.Namespace) -> int:
    tpu = _interactive_tpu(args)
    _validate_worker(tpu, args.worker)
    return interactive_tools.ssh_shell(tpu, worker=args.worker)


def cmd_interactive_run(args: argparse.Namespace) -> int:
    tpu = _interactive_tpu(args)
    worker = _worker_arg(args.worker)
    _validate_worker(tpu, worker)
    return interactive_tools.run_command(
        tpu,
        command=_command_from_args(args.command),
        worker=worker,
    )


def cmd_interactive_tmux(args: argparse.Namespace) -> int:
    tpu = _interactive_tpu(args)
    worker = _worker_arg(args.worker)
    _validate_worker(tpu, worker)
    return interactive_tools.tmux_command(
        tpu,
        command=_command_from_args(args.command),
        session=args.session,
        worker=worker,
    )


def cmd_interactive_attach(args: argparse.Namespace) -> int:
    tpu = _interactive_tpu(args)
    _validate_worker(tpu, args.worker)
    return interactive_tools.attach_tmux(
        tpu, session=args.session, worker=args.worker
    )


def cmd_interactive_output(args: argparse.Namespace) -> int:
    tpu = _interactive_tpu(args)
    _validate_worker(tpu, args.worker)
    return interactive_tools.tail_output(
        tpu,
        session=args.session,
        worker=args.worker,
        lines=args.lines,
        follow=args.follow,
    )


def cmd_interactive_tmux_ls(args: argparse.Namespace) -> int:
    tpu = _interactive_tpu(args)
    _validate_worker(tpu, args.worker)
    return interactive_tools.tmux_ls(tpu, worker=args.worker)


def cmd_interactive_put(args: argparse.Namespace) -> int:
    tpu = _interactive_tpu(args)
    _validate_worker(tpu, args.worker)
    return interactive_tools.scp_to(
        tpu,
        local_path=args.local_path,
        remote_path=args.remote_path,
        worker=args.worker,
        recurse=args.recurse,
    )


def cmd_interactive_get(args: argparse.Namespace) -> int:
    tpu = _interactive_tpu(args)
    _validate_worker(tpu, args.worker)
    return interactive_tools.scp_from(
        tpu,
        remote_path=args.remote_path,
        local_path=args.local_path,
        worker=args.worker,
        recurse=args.recurse,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tpu", description="IROM TPU queue CLI")
    parser.add_argument("--config", help="Queue resources YAML (default: package config)")
    parser.add_argument("--dry-run", action="store_true", help="Use local dry-run backend")
    parser.add_argument("--base-dir", help="Dry-run state directory")
    parser.add_argument("--provision-delay", type=float, default=0.0)
    sub = parser.add_subparsers(dest="cmd")

    create = sub.add_parser(
        "create",
        help="Submit a queued TPU job",
        usage=(
            "tpu create {v4,v5,v6} [options] -- <training command>\n"
            "       tpu create v6 -n 32 --name run --code-dir . -- python train.py"
        ),
    )
    create.add_argument("version", choices=["v4", "v5", "v6"])
    create.add_argument("--name", "-N")
    create.add_argument("--tpu-num", "-n", type=int, default=8)
    create.add_argument("--resource", "-r")
    create.add_argument("--code-dir", "-c", default=".")
    create.add_argument("--setup-cmd", "-s", default="true")
    create.add_argument("--max-attempts", type=int, default=20)
    create.add_argument("--priority", "-p", type=int, choices=[0, 1, 2], default=1)
    create.add_argument("--tag", action="append")
    create.add_argument("--env", action="append")
    create.add_argument("--secret", action="append")
    create.add_argument("--user")
    create.add_argument("--worker0-only", action="store_true")
    create.set_defaults(func=cmd_create)

    list_p = sub.add_parser(
        "list",
        help="List queued jobs or requestable TPU resources",
        description=(
            "List queued jobs. If no matching jobs exist and no job-only filters "
            "are used, show requestable configured resources instead."
        ),
    )
    list_p.add_argument("version", nargs="?", choices=["v4", "v5", "v6"])
    mode = list_p.add_mutually_exclusive_group()
    mode.add_argument("--jobs", action="store_true", help="Only show queued jobs")
    mode.add_argument("--resources", action="store_true", help="Only show resources")
    list_p.add_argument("--user")
    list_p.add_argument("--active", "-a", action="store_true")
    list_p.add_argument("--status")
    list_p.set_defaults(func=cmd_list)

    for name in ("status", "info"):
        p = sub.add_parser(name, help="Show job status")
        p.add_argument("job")
        p.set_defaults(func=cmd_status)

    logs = sub.add_parser("logs", help="Show uploaded job logs")
    logs.add_argument("job")
    logs.add_argument("--attempt", "-a", type=int)
    logs.add_argument("--worker", "-w", type=int)
    logs.add_argument("--lines", "-n", type=int, default=200)
    logs.set_defaults(func=cmd_logs)

    output = sub.add_parser("output", help="Alias for logs")
    output.add_argument("job")
    output.add_argument("--attempt", "-a", type=int)
    output.add_argument("--worker", "-w", type=int)
    output.add_argument("--lines", "-n", type=int, default=200)
    output.set_defaults(func=cmd_logs)

    tail = sub.add_parser("tail", help="Poll uploaded logs")
    tail.add_argument("job")
    tail.add_argument("--attempt", "-a", type=int)
    tail.add_argument("--worker", "-w", type=int)
    tail.add_argument("--follow", "-f", action="store_true")
    tail.add_argument("--interval", type=float, default=5.0)
    tail.set_defaults(func=cmd_tail)

    for name in ("delete", "cancel"):
        p = sub.add_parser(name, help="Request job cancellation")
        p.add_argument("job")
        p.set_defaults(func=cmd_delete)

    retry = sub.add_parser("retry", help="Retry a failed job")
    retry.add_argument("job")
    retry.set_defaults(func=cmd_retry)

    rerun = sub.add_parser("rerun", help="Submit a new job from an old spec")
    rerun.add_argument("job")
    rerun.add_argument("--name")
    rerun.add_argument("--max-attempts", type=int)
    rerun.add_argument("--priority", type=int, choices=[0, 1, 2])
    rerun.set_defaults(func=cmd_rerun)

    scheduler = sub.add_parser("scheduler", help="Run central scheduler")
    scheduler.add_argument("--once", action="store_true")
    scheduler.add_argument("--scan-interval", type=int)
    scheduler.add_argument("--verbose", "-v", action="store_true")
    scheduler.set_defaults(func=cmd_scheduler)

    interactive = sub.add_parser(
        "interactive",
        help="Use allowlisted shared v4 interactive TPUs",
        description=(
            "Connect-only commands for configured shared v4 interactive TPUs. "
            "This group intentionally has no create/delete/stop/start commands."
        ),
    )
    interactive_sub = interactive.add_subparsers(dest="interactive_cmd")
    interactive.set_defaults(func=lambda args: (interactive.print_help() or 0))

    ilist = interactive_sub.add_parser("list", help="List allowlisted shared TPUs")
    ilist.add_argument("--live", action="store_true", help="Also query live TPU state")
    ilist.set_defaults(func=cmd_interactive_list)

    iinfo = interactive_sub.add_parser("info", help="Show shared TPU config")
    iinfo.add_argument("name")
    iinfo.add_argument("--live", action="store_true", help="Also query live TPU state")
    iinfo.set_defaults(func=cmd_interactive_info)

    issh = interactive_sub.add_parser("ssh", help="Open an interactive shell")
    issh.add_argument("name")
    issh.add_argument("--worker", type=int, default=0)
    issh.set_defaults(func=cmd_interactive_ssh)

    irun = interactive_sub.add_parser("run", help="Run a shell command")
    irun.add_argument("name")
    irun.add_argument("--worker", default="0", help="Worker index or 'all'")
    irun.add_argument("command", nargs=argparse.REMAINDER)
    irun.set_defaults(func=cmd_interactive_run)

    itmux = interactive_sub.add_parser("tmux", help="Run command in tmux")
    itmux.add_argument("name")
    itmux.add_argument("--session", default="tpu")
    itmux.add_argument("--worker", default="all", help="Worker index or 'all'")
    itmux.add_argument("command", nargs=argparse.REMAINDER)
    itmux.set_defaults(func=cmd_interactive_tmux)

    iattach = interactive_sub.add_parser("attach", help="Attach to tmux")
    iattach.add_argument("name")
    iattach.add_argument("--session", default="tpu")
    iattach.add_argument("--worker", type=int, default=0)
    iattach.set_defaults(func=cmd_interactive_attach)

    for command_name in ("output", "tail"):
        iout = interactive_sub.add_parser(command_name, help="Read latest tmux log")
        iout.add_argument("name")
        iout.add_argument("--session", default="tpu")
        iout.add_argument("--worker", type=int, default=0)
        iout.add_argument("--lines", "-n", type=int, default=200)
        iout.add_argument("--follow", "-f", action="store_true")
        iout.set_defaults(func=cmd_interactive_output)

    ils = interactive_sub.add_parser("tmux-ls", help="List tmux sessions")
    ils.add_argument("name")
    ils.add_argument("--worker", type=int, default=0)
    ils.set_defaults(func=cmd_interactive_tmux_ls)

    iput = interactive_sub.add_parser("put", help="Copy local file/dir to TPU")
    iput.add_argument("name")
    iput.add_argument("local_path")
    iput.add_argument("remote_path")
    iput.add_argument("--worker", type=int, default=0)
    iput.add_argument("--recurse", "-r", action="store_true")
    iput.set_defaults(func=cmd_interactive_put)

    iget = interactive_sub.add_parser("get", help="Copy TPU file/dir to local path")
    iget.add_argument("name")
    iget.add_argument("remote_path")
    iget.add_argument("local_path")
    iget.add_argument("--worker", type=int, default=0)
    iget.add_argument("--recurse", "-r", action="store_true")
    iget.set_defaults(func=cmd_interactive_get)

    admin = sub.add_parser("admin", help="Admin queue/resource commands")
    admin_sub = admin.add_subparsers(dest="admin_cmd")
    admin.set_defaults(func=lambda args: (admin.print_help() or 0))
    resources = admin_sub.add_parser("resources", help="Show quota and resource config")
    resources.set_defaults(func=cmd_admin_resources)
    jobs = admin_sub.add_parser("jobs", help="List queue jobs across users")
    jobs.add_argument("--user")
    jobs.add_argument("--active", "-a", action="store_true")
    jobs.add_argument("--status")
    jobs.set_defaults(func=cmd_admin_jobs)
    qrs = admin_sub.add_parser("qrs", help="List queue-owned queued resources")
    qrs.set_defaults(func=cmd_admin_qrs)
    cleanup = admin_sub.add_parser("cleanup", help="Delete queue-owned orphan/idle resources")
    cleanup.add_argument("--idle-minutes", type=int)
    cleanup.add_argument("--yes", action="store_true")
    cleanup.set_defaults(func=cmd_admin_cleanup)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args, unknown = parser.parse_known_args(raw_argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    if args.cmd == "create":
        if unknown and unknown[0] == "--":
            unknown = unknown[1:]
        args.command = unknown
    elif unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return args.func(args)
