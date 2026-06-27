from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
import io
import json
from pathlib import Path
import tempfile
import unittest

from irom_tpu_tools.queue.backend import DryRunBackend
from irom_tpu_tools.queue.cli import build_parser
from irom_tpu_tools.queue.config import QueueConfig, load_config
from irom_tpu_tools.queue.interactive import _permission_hint, resolve_interactive_tpu
from irom_tpu_tools.queue.scheduler import Scheduler
from irom_tpu_tools.queue.startup_script import build_startup_script
from irom_tpu_tools.queue.types import (
    JobResources,
    JobSpec,
    JobState,
    JobStatus,
    InteractiveTPUConfig,
    QuotaGroupConfig,
    ResourceConfig,
    SchedulerConfig,
    UserLimitConfig,
    utc_now,
)


def make_config(
    tmp: Path, *, user_limit: int | None = None, quota_total: int = 8
) -> QueueConfig:
    return QueueConfig(
        resources={
            "v6-8": ResourceConfig(
                name="v6-8",
                version="v6",
                accelerator_type="v6e-8",
                runtime_version="v2-alpha-tpuv6e",
                zone="us-east1-d",
                project="test-project",
                chips=8,
                workers=1,
                spot=True,
                enabled=True,
                quota_group="v6",
                service_account="worker@test-project.iam.gserviceaccount.com",
            )
        },
        quota_groups={"v6": QuotaGroupConfig(name="v6", total_chips=quota_total)},
        scheduler=SchedulerConfig(
            scan_interval=1,
            active_no_claim_timeout=60,
            heartbeat_timeout=60,
            status_write_interval=1,
            qr_prefix="iqtest",
        ),
        buckets={"us-east1": "gs://test-bucket/queue"},
        primary_bucket_region="us-east1",
        secrets={"WANDB_API_KEY": "wandb-api-key"},
        user_limits=UserLimitConfig(default_max_chips=user_limit),
        interactive_tpus={
            "v4-4-01-interactive": InteractiveTPUConfig(
                name="v4-4-01-interactive",
                version="v4",
                zone="us-central2-b",
                project="test-project",
                workers=1,
                aliases=("v4-interactive",),
            )
        },
    )


def make_spec(job_id: str, *, user: str = "alice") -> JobSpec:
    resources = JobResources(
        resource_name="v6-8",
        accelerator_type="v6e-8",
        zone="us-east1-d",
        project="test-project",
        chips=8,
        workers=1,
        runtime_version="v2-alpha-tpuv6e",
    )
    return JobSpec(
        job_id=job_id,
        display_name=job_id,
        code_tar_url=f"gs://test-bucket/queue/jobs/{job_id}/code.tar.gz",
        code_checksum="abc",
        command="python train.py",
        setup_cmd="uv sync",
        resources=resources,
        max_attempts=3,
        submit_time=utc_now(),
        submitted_by=user,
        secret_refs={"WANDB_API_KEY": "wandb-api-key"},
    )


def write_job(backend: DryRunBackend, bucket: str, spec: JobSpec) -> str:
    job_dir = f"{bucket}/jobs/{spec.job_id}"
    backend.write_gcs(f"{job_dir}/spec.json", json.dumps(spec.to_dict()))
    backend.write_gcs(f"{job_dir}/status.json", json.dumps(JobState.new().to_dict()))
    return job_dir


def write_config_file(tmp: Path) -> Path:
    path = tmp / "resources.yaml"
    path.write_text(
        """
quota_groups:
  v6:
    total_chips: 8
resources:
  v6-8:
    version: v6
    accelerator_type: v6e-8
    runtime_version: v2-alpha-tpuv6e
    zone: us-east1-d
    project: test-project
    chips: 8
    workers: 1
    spot: true
    enabled: true
    quota_group: v6
buckets:
  us-east1: gs://test-bucket/queue
primary_bucket_region: us-east1
scheduler:
  qr_prefix: iqtest
interactive_tpus: {}
""".lstrip()
    )
    return path


class SchedulerTests(unittest.TestCase):
    def test_schedules_and_requeues_after_preemption(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d))
            write_job(backend, config.primary_bucket, make_spec("job-a"))
            scheduler = Scheduler(backend, config)

            scheduler.run_once()
            self.assertEqual(len(backend.queued_resources), 1)
            qr_name = next(iter(backend.queued_resources))
            state = json.loads(
                backend.read_gcs(f"{config.primary_bucket}/jobs/job-a/status.json") or "{}"
            )
            self.assertEqual(state["status"], "PROVISIONING")

            backend.force_active(qr_name)
            scheduler.run_once()
            state = json.loads(
                backend.read_gcs(f"{config.primary_bucket}/jobs/job-a/status.json") or "{}"
            )
            self.assertEqual(state["status"], "RUNNING")

            backend.force_preempt(qr_name)
            scheduler.run_once()
            state = json.loads(
                backend.read_gcs(f"{config.primary_bucket}/jobs/job-a/status.json") or "{}"
            )
            self.assertEqual(state["status"], "PROVISIONING")
            self.assertEqual(state["current_attempt"], 1)
            self.assertEqual(len(backend.queued_resources), 1)

    def test_requeues_ready_unhealthy_maintenance_tpu(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d))
            write_job(backend, config.primary_bucket, make_spec("job-a"))
            scheduler = Scheduler(backend, config)

            scheduler.run_once()
            qr_name = next(iter(backend.queued_resources))
            backend.force_active(qr_name)
            scheduler.run_once()

            backend.force_unhealthy_maintenance(qr_name)
            scheduler.run_once()
            state = json.loads(
                backend.read_gcs(f"{config.primary_bucket}/jobs/job-a/status.json") or "{}"
            )
            self.assertEqual(state["status"], "PROVISIONING")
            self.assertEqual(state["current_attempt"], 1)
            self.assertEqual(
                state["attempts"][0]["error"],
                "TPU_VM_HEALTH_UNHEALTHY_MAINTENANCE",
            )
            self.assertEqual(len(backend.queued_resources), 1)

    def test_user_limit_keeps_second_job_pending(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d), user_limit=8)
            write_job(backend, config.primary_bucket, make_spec("job-a", user="alice"))
            write_job(backend, config.primary_bucket, make_spec("job-b", user="alice"))
            scheduler = Scheduler(backend, config)

            scheduler.run_once()
            self.assertEqual(len(backend.queued_resources), 1)
            status_b = json.loads(
                backend.read_gcs(f"{config.primary_bucket}/jobs/job-b/status.json") or "{}"
            )
            self.assertEqual(status_b["status"], JobStatus.PENDING.value)

    def test_user_limit_null_override_is_unlimited(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d), user_limit=8, quota_total=16)
            config.user_limits.users["admin"] = None
            write_job(backend, config.primary_bucket, make_spec("job-a", user="admin"))
            write_job(backend, config.primary_bucket, make_spec("job-b", user="admin"))
            scheduler = Scheduler(backend, config)

            scheduler.run_once()
            self.assertEqual(len(backend.queued_resources), 2)

    def test_default_config_has_admin_unlimited(self) -> None:
        config = load_config()
        self.assertIsNone(config.user_limits.max_chips_for("admin"))

    def test_startup_script_has_centralized_sentinels_and_no_local_watcher(self) -> None:
        script = build_startup_script(
            job_id="job-a",
            spec=make_spec("job-a"),
            qr_name="iqtest-123-v6-8-a1",
            job_dir="gs://test-bucket/queue/jobs/job-a",
            attempt=1,
            project="test-project",
        )
        self.assertIn("/attempts/attempt-$ATTEMPT/claimed", script)
        self.assertIn("/attempts/attempt-$ATTEMPT/heartbeat", script)
        self.assertIn("/logs/attempt-$ATTEMPT/worker-$WORKER_ID.log", script)
        self.assertNotIn(".tpu-jobs", script)
        self.assertNotIn("watch.pid", script)

    def test_interactive_tpus_are_allowlisted(self) -> None:
        config = make_config(Path("/tmp"))
        tpu = resolve_interactive_tpu(config, "v4-interactive")
        self.assertEqual(tpu.name, "v4-4-01-interactive")
        with self.assertRaises(SystemExit):
            resolve_interactive_tpu(config, "not-allowlisted")

    def test_interactive_permission_hint_mentions_read_only_tpu_access(self) -> None:
        config = make_config(Path("/tmp"))
        tpu = resolve_interactive_tpu(config, "v4-interactive")
        hint = _permission_hint(tpu)
        self.assertIn("roles/tpu.viewer", hint)
        self.assertIn("tpu.nodes.get", hint)
        self.assertIn("us-central2-b", hint)
        self.assertIn("No TPU Admin role is required", hint)

    def test_default_config_has_v4_interactive_entry(self) -> None:
        config = load_config()
        self.assertIn("v4-4-01-interactive", config.interactive_tpus)
        self.assertEqual(config.interactive_tpus["v4-4-01-interactive"].version, "v4")
        self.assertIn("v4-16-01-interactive", config.interactive_tpus)
        self.assertIn("v4-4-04-interactive", config.interactive_tpus)
        self.assertEqual(config.interactive_tpus["v4-16-01-interactive"].workers, 4)

    def test_interactive_parser_has_no_lifecycle_verbs(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("interactive", help_text)
        # Do not rely on argparse internals for full traversal; command parsing is
        # enough to prove lifecycle verbs are not accepted under interactive.
        for forbidden in ("create", "delete", "stop", "start"):
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                parser.parse_args(["interactive", forbidden, "v4-interactive"])

    def test_list_defaults_to_jobs_and_live_only_when_no_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            parser = build_parser()
            args = parser.parse_args(
                [
                    "--config",
                    str(write_config_file(root)),
                    "--dry-run",
                    "--base-dir",
                    str(root / "state"),
                    "list",
                    "v6",
                ]
            )
            out = io.StringIO()
            with redirect_stdout(out):
                args.func(args)
            text = out.getvalue()
            self.assertIn("Queued jobs:", text)
            self.assertIn("Live TPU VMs:", text)
            self.assertNotIn("Requestable resources:", text)
            self.assertNotIn("Shared interactive TPUs:", text)

    def test_list_hides_terminal_jobs_unless_requested(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config_path = write_config_file(root)
            config = load_config(config_path)
            state_dir = root / "state"
            backend = DryRunBackend(str(state_dir))
            write_job(backend, config.primary_bucket, make_spec("job-pending"))
            failed_dir = write_job(backend, config.primary_bucket, make_spec("job-failed"))
            failed_state = JobState.new()
            failed_state.status = JobStatus.FAILED
            backend.write_gcs(f"{failed_dir}/status.json", json.dumps(failed_state.to_dict()))

            parser = build_parser()
            args = parser.parse_args(
                [
                    "--config",
                    str(config_path),
                    "--dry-run",
                    "--base-dir",
                    str(state_dir),
                    "list",
                    "--jobs",
                    "v6",
                ]
            )
            out = io.StringIO()
            with redirect_stdout(out):
                args.func(args)
            text = out.getvalue()
            self.assertIn("job-pending", text)
            self.assertNotIn("job-failed", text)

            args_all = parser.parse_args(
                [
                    "--config",
                    str(config_path),
                    "--dry-run",
                    "--base-dir",
                    str(state_dir),
                    "list",
                    "--jobs",
                    "--all",
                    "v6",
                ]
            )
            out_all = io.StringIO()
            with redirect_stdout(out_all):
                args_all.func(args_all)
            self.assertIn("job-failed", out_all.getvalue())

    def test_list_jobs_can_still_show_empty_job_list(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            parser = build_parser()
            args = parser.parse_args(
                [
                    "--config",
                    str(write_config_file(root)),
                    "--dry-run",
                    "--base-dir",
                    str(root / "state"),
                    "list",
                    "--jobs",
                    "v6",
                ]
            )
            out = io.StringIO()
            with redirect_stdout(out):
                args.func(args)
            self.assertEqual(out.getvalue(), "(none)\n")

    def test_list_live_shows_active_tpu_vms(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config_path = write_config_file(root)
            state_dir = root / "state"
            backend = DryRunBackend(str(state_dir))
            backend.create_queued_resource(
                name="iqtest-live",
                node_id="iqtest-live",
                project="test-project",
                zone="us-east1-d",
                accelerator_type="v6e-8",
                runtime_version="v2-alpha-tpuv6e",
                spot=True,
                startup_script_path="",
                service_account=None,
            )
            backend.force_active("iqtest-live")

            parser = build_parser()
            args = parser.parse_args(
                [
                    "--config",
                    str(config_path),
                    "--dry-run",
                    "--base-dir",
                    str(state_dir),
                    "list",
                    "--live",
                    "v6",
                ]
            )
            out = io.StringIO()
            with redirect_stdout(out):
                args.func(args)
            text = out.getvalue()
            self.assertIn("iqtest-live", text)
            self.assertIn("v6e-8", text)
            self.assertIn("READY/HEALTHY", text)

    def test_admin_activity_reports_live_status_without_ssh(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config_path = write_config_file(root)
            state_dir = root / "state"
            backend = DryRunBackend(str(state_dir))
            backend.create_queued_resource(
                name="iqtest-live",
                node_id="iqtest-live",
                project="test-project",
                zone="us-east1-d",
                accelerator_type="v6e-8",
                runtime_version="v2-alpha-tpuv6e",
                spot=True,
                startup_script_path="",
                service_account=None,
            )
            backend.force_active("iqtest-live")

            parser = build_parser()
            args = parser.parse_args(
                [
                    "--config",
                    str(config_path),
                    "--dry-run",
                    "--base-dir",
                    str(state_dir),
                    "admin",
                    "activity",
                    "--no-ssh",
                    "iqtest-live",
                ]
            )
            out = io.StringIO()
            with redirect_stdout(out):
                args.func(args)
            text = out.getvalue()
            self.assertIn("## iqtest-live", text)
            self.assertIn("Status:  READY/HEALTHY", text)
            self.assertIn("Local watchers: (none)", text)


if __name__ == "__main__":
    unittest.main()
