from __future__ import annotations

import argparse
import base64
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from irom_tpu_tools.queue.backend import DryRunBackend, GCPBackend, GCSReadResult
from irom_tpu_tools.queue.cli import (
    _command_from_args,
    _scheduler_lock,
    _shell_join_command,
    build_parser,
)
from irom_tpu_tools.queue.config import QueueConfig, load_config
from irom_tpu_tools.queue.interactive import _permission_hint, resolve_interactive_tpu
from irom_tpu_tools.queue.scheduler import Scheduler
from irom_tpu_tools.queue.startup_script import build_startup_script
from irom_tpu_tools.queue.types import (
    AttemptRecord,
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
from irom_tpu_tools.ssh import SSHOptions, gcloud_tpu_ssh, gcloud_tpu_ssh_stream


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
    def test_single_worker_ssh_keeps_bash_script_in_one_remote_argument(self) -> None:
        command = "set -e; printf quoted-ok"
        expected_remote = "bash -lc 'set -e; printf quoted-ok'"

        with patch("irom_tpu_tools.ssh.run_streaming", return_value=0) as stream:
            rc = gcloud_tpu_ssh_stream(
                tpu_name="v4-interactive",
                project="test-project",
                zone="us-central2-b",
                worker="0",
                command=command,
                ssh=SSHOptions(forward_agent=False),
            )
        self.assertEqual(rc, 0)
        self.assertEqual(stream.call_args.args[0][-1], expected_remote)

        completed = subprocess.CompletedProcess([], 0, "", "")
        with patch("irom_tpu_tools.ssh.run_with_timeout", return_value=completed) as run:
            result = gcloud_tpu_ssh(
                tpu_name="v4-interactive",
                project="test-project",
                zone="us-central2-b",
                worker="0",
                command=command,
                ssh=SSHOptions(forward_agent=False),
            )
        self.assertIs(result, completed)
        self.assertEqual(run.call_args.args[2][-1], expected_remote)

        local = subprocess.run(
            ["bash", "-c", expected_remote],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(local.returncode, 0)
        self.assertEqual(local.stdout, "quoted-ok")

    def test_shell_command_join_preserves_nested_bash_command(self) -> None:
        command = [
            "bash",
            "-lc",
            'set -euo pipefail; printf "%s" "$RUN_NAME"',
        ]

        expected = "bash -lc 'set -euo pipefail; printf \"%s\" \"$RUN_NAME\"'"
        self.assertEqual(_shell_join_command(command, default="true"), expected)
        self.assertEqual(_command_from_args(["--", *command]), expected)
        self.assertEqual(_shell_join_command([], default="true"), "true")

        failure = _shell_join_command(
            ["bash", "-lc", "set -euo pipefail; false; echo unreachable"]
        )
        result = subprocess.run(
            ["bash", "-lc", failure],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_gcp_cleanup_uses_async_force_delete_for_active_resources(self) -> None:
        backend = GCPBackend()
        run = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
        backend._run = run

        self.assertTrue(backend.delete_queued_resource("qr-a", "project", "zone"))
        qr_command = run.call_args.args[0]
        self.assertIn("--force", qr_command)
        self.assertIn("--async", qr_command)

        run.reset_mock()
        self.assertTrue(backend.delete_tpu_vm("qr-a", "project", "zone"))
        vm_command = run.call_args.args[0]
        self.assertIn("--async", vm_command)

    def test_gcp_create_rejects_timeout_result(self) -> None:
        backend = GCPBackend()
        backend._run = Mock(
            return_value=subprocess.CompletedProcess([], 124, "", "timed out")
        )

        self.assertFalse(
            backend.create_queued_resource(
                name="qr-a",
                node_id="qr-a",
                project="project",
                zone="zone",
                accelerator_type="v6e-8",
                runtime_version="v2-alpha-tpuv6e",
                spot=True,
                startup_script_path="/tmp/startup.sh",
                service_account=None,
            )
        )

    def test_gcp_read_distinguishes_missing_object_from_probe_failure(self) -> None:
        backend = GCPBackend()
        backend._run = Mock(
            return_value=subprocess.CompletedProcess(
                [], 1, "", "The following URLs matched no objects or files"
            )
        )
        missing = backend.read_gcs_result("gs://bucket/missing")
        self.assertTrue(missing.succeeded)
        self.assertIsNone(missing.content)

        backend._run = Mock(
            return_value=subprocess.CompletedProcess([], 124, "", "timed out")
        )
        failed = backend.read_gcs_result("gs://bucket/heartbeat")
        self.assertFalse(failed.succeeded)
        self.assertIsNone(failed.content)

    def test_failed_create_is_backed_off_before_retry(self) -> None:
        class FailOnceBackend(DryRunBackend):
            def __init__(self, base_dir: str):
                super().__init__(base_dir)
                self.create_calls = 0

            def create_queued_resource(self, **kwargs) -> bool:
                self.create_calls += 1
                if self.create_calls == 1:
                    return False
                return super().create_queued_resource(**kwargs)

        with tempfile.TemporaryDirectory() as d:
            backend = FailOnceBackend(d)
            config = make_config(Path(d))
            config.scheduler.create_failure_backoff = 300
            write_job(backend, config.primary_bucket, make_spec("job-a"))
            scheduler = Scheduler(backend, config)
            scheduler.scan_jobs()

            scheduler.schedule_pending_jobs()
            self.assertEqual(backend.create_calls, 1)
            self.assertIn("job-a", scheduler._create_retry_not_before)

            scheduler.schedule_pending_jobs()
            self.assertEqual(backend.create_calls, 1)

            scheduler._create_retry_not_before["job-a"] = 0.0
            scheduler.schedule_pending_jobs()
            self.assertEqual(backend.create_calls, 2)
            self.assertNotIn("job-a", scheduler._create_retry_not_before)
            self.assertEqual(scheduler.jobs["job-a"].state.status, JobStatus.PROVISIONING)

    def test_step_exception_does_not_block_other_steps_or_state_write(self) -> None:
        # Regression test for the ENOSPC-during-schedule_pending_jobs
        # incident: write_startup_script raised OSError(28, "No space left
        # on device") from inside _create_queued_resource, and run_once()
        # had no per-step isolation, so the whole iteration aborted before
        # _maybe_write_scheduler_state() ever ran. That froze the aggregated
        # scheduler_state.json cache that `tpu list`/`tpu status`/
        # `tpu admin jobs` read from for *every* user, even though earlier
        # steps in that same iteration may have already written their own
        # per-job status.json fine.
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d))

            # job-a is pending; its queued-resource create will blow up.
            write_job(backend, config.primary_bucket, make_spec("job-a"))

            # job-b is terminal and past retention. reap_terminal_jobs runs
            # *after* schedule_pending_jobs in run_once(), so it only reaps
            # job-b if the earlier failure did not abort the whole pass.
            job_b_spec = make_spec("job-b")
            job_b_state = JobState.new()
            job_b_state.status = JobStatus.FAILED
            job_b_state.last_updated = (
                datetime.now(UTC)
                - timedelta(days=config.scheduler.job_retention_days + 1)
            ).isoformat()
            job_b_dir = f"{config.primary_bucket}/jobs/job-b"
            backend.write_gcs(f"{job_b_dir}/spec.json", json.dumps(job_b_spec.to_dict()))
            backend.write_gcs(
                f"{job_b_dir}/status.json", json.dumps(job_b_state.to_dict())
            )

            scheduler = Scheduler(backend, config)

            with patch(
                "irom_tpu_tools.queue.scheduler.write_startup_script",
                side_effect=OSError(28, "No space left on device"),
            ):
                with self.assertLogs(
                    "irom_tpu_tools.queue.scheduler", level="ERROR"
                ) as logs:
                    scheduler.run_once()

            # (a) the failure is logged with step and job context, not
            # swallowed silently.
            joined = "\n".join(logs.output)
            self.assertIn("job-a", joined)
            self.assertIn("No space left on device", joined)

            # The create-failure backoff still registers for job-a (the
            # exception is treated the same as a normal create failure, so
            # the scheduler will not hammer the API on every scan) and
            # job-a is left PENDING to retry rather than lost.
            self.assertIn("job-a", scheduler._create_retry_not_before)
            self.assertEqual(scheduler.jobs["job-a"].state.status, JobStatus.PENDING)

            # (b) later steps in the same run_once() pass still executed:
            # the unrelated, already-terminal job-b was reaped.
            self.assertNotIn("job-b", scheduler.jobs)

            # (c) _maybe_write_scheduler_state() still ran despite the
            # earlier step's exception, so the CLI-facing cache is current.
            cache = json.loads(
                backend.read_gcs(f"{config.primary_bucket}/scheduler_state.json") or "{}"
            )
            self.assertIn("updated_at", cache)
            self.assertEqual({j["job_id"] for j in cache["jobs"]}, {"job-a"})

    def test_scans_independent_job_records_concurrently(self) -> None:
        class TrackingBackend(DryRunBackend):
            def __init__(self, base_dir: str):
                super().__init__(base_dir)
                self.active_reads = 0
                self.max_active_reads = 0
                self.read_lock = threading.Lock()

            def read_gcs(self, url: str) -> str | None:
                with self.read_lock:
                    self.active_reads += 1
                    self.max_active_reads = max(self.max_active_reads, self.active_reads)
                try:
                    time.sleep(0.01)
                    return super().read_gcs(url)
                finally:
                    with self.read_lock:
                        self.active_reads -= 1

        with tempfile.TemporaryDirectory() as d:
            backend = TrackingBackend(d)
            config = make_config(Path(d))
            for index in range(4):
                write_job(backend, config.primary_bucket, make_spec(f"job-{index}"))

            scheduler = Scheduler(backend, config)
            scheduler.scan_jobs()

            self.assertEqual(len(scheduler.jobs), 4)
            self.assertGreater(backend.max_active_reads, 1)

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
            self.assertEqual(
                state["attempts"][0]["failure_type"],
                "INFRASTRUCTURE_PREEMPTION",
            )
            self.assertTrue(state["attempts"][0]["retryable"])
            self.assertEqual(len(backend.queued_resources), 1)

    def test_transient_heartbeat_read_failure_does_not_requeue_live_job(self) -> None:
        class FailingHeartbeatBackend(DryRunBackend):
            fail_heartbeat_read = False

            def read_gcs_result(self, url: str) -> GCSReadResult:
                if self.fail_heartbeat_read and url.endswith("/heartbeat"):
                    self.fail_heartbeat_read = False
                    return GCSReadResult(content=None, succeeded=False)
                return super().read_gcs_result(url)

        with tempfile.TemporaryDirectory() as d:
            backend = FailingHeartbeatBackend(d)
            config = make_config(Path(d))
            job_dir = write_job(backend, config.primary_bucket, make_spec("job-a"))
            scheduler = Scheduler(backend, config)

            scheduler.run_once()
            qr_name = next(iter(backend.queued_resources))
            backend.force_active(qr_name)
            scheduler.run_once()
            scheduler.jobs["job-a"].state.provisioned_at = (
                datetime.now(UTC) - timedelta(seconds=120)
            ).isoformat()
            backend.write_gcs(f"{job_dir}/attempts/attempt-1/claimed", "claimed")
            backend.write_gcs(
                f"{job_dir}/attempts/attempt-1/heartbeat", datetime.now(UTC).isoformat()
            )

            backend.fail_heartbeat_read = True
            scheduler.run_once()

            state = scheduler.jobs["job-a"].state
            self.assertEqual(state.status, JobStatus.RUNNING)
            self.assertEqual(state.current_attempt, 0)
            self.assertIn(qr_name, backend.queued_resources)

            scheduler.run_once()
            self.assertEqual(state.status, JobStatus.RUNNING)

    def test_missing_heartbeat_still_requeues_stale_job(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d))
            job_dir = write_job(backend, config.primary_bucket, make_spec("job-a"))
            scheduler = Scheduler(backend, config)

            scheduler.run_once()
            qr_name = next(iter(backend.queued_resources))
            backend.force_active(qr_name)
            scheduler.run_once()
            scheduler.jobs["job-a"].state.provisioned_at = (
                datetime.now(UTC) - timedelta(seconds=120)
            ).isoformat()
            backend.write_gcs(f"{job_dir}/attempts/attempt-1/claimed", "claimed")

            scheduler.run_once()

            state = scheduler.jobs["job-a"].state
            self.assertEqual(state.status, JobStatus.PROVISIONING)
            self.assertEqual(state.current_attempt, 1)
            self.assertEqual(state.attempts[0].error, "HEARTBEAT_TIMEOUT")

    def test_application_retry_advances_to_a_fresh_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d))
            job_dir = write_job(backend, config.primary_bucket, make_spec("job-a"))
            scheduler = Scheduler(backend, config)

            scheduler.run_once()
            first_qr = next(iter(backend.queued_resources))
            backend.force_active(first_qr)
            scheduler.run_once()

            backend.delete_gcs(f"{job_dir}/attempts/attempt-1/startup_version")
            backend.write_gcs(f"{job_dir}/failed", "FAILED with exit code 1")
            scheduler.run_once()
            failed_state = json.loads(backend.read_gcs(f"{job_dir}/status.json") or "{}")
            self.assertEqual(failed_state["status"], JobStatus.FAILED.value)
            self.assertEqual(failed_state["current_attempt"], 1)
            self.assertEqual(failed_state["attempts"][0]["attempt"], 1)

            backend.write_gcs(f"{job_dir}/retry", "retry")
            scheduler.run_once()
            retry_state = json.loads(backend.read_gcs(f"{job_dir}/status.json") or "{}")
            self.assertEqual(retry_state["status"], JobStatus.PROVISIONING.value)
            self.assertEqual(retry_state["current_attempt"], 1)
            self.assertNotEqual(retry_state["current_qr_name"], first_qr)
            self.assertTrue(retry_state["current_qr_name"].endswith("-a2"))

    def test_structured_worker_failure_is_terminal_and_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d))
            job_dir = write_job(backend, config.primary_bucket, make_spec("job-a"))
            scheduler = Scheduler(backend, config)

            scheduler.run_once()
            qr_name = next(iter(backend.queued_resources))
            backend.force_active(qr_name)
            scheduler.run_once()
            report = {
                "failure_type": "SETUP_ERROR",
                "phase": "setup",
                "worker_id": "3",
                "exit_code": 17,
                "message": "Worker 3 exited with code 17 during setup",
            }
            backend.write_gcs(
                f"{job_dir}/attempts/attempt-1/failed",
                json.dumps(report),
            )

            scheduler.run_once()
            state = json.loads(backend.read_gcs(f"{job_dir}/status.json") or "{}")
            self.assertEqual(state["status"], JobStatus.FAILED.value)
            attempt = state["attempts"][0]
            self.assertEqual(attempt["failure_type"], "SETUP_ERROR")
            self.assertFalse(attempt["retryable"])
            self.assertEqual(attempt["phase"], "setup")
            self.assertEqual(attempt["worker_id"], 3)
            self.assertEqual(attempt["exit_code"], 17)

    def test_new_attempt_ignores_stale_legacy_root_failure(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d))
            job_dir = write_job(backend, config.primary_bucket, make_spec("job-a"))
            scheduler = Scheduler(backend, config)

            scheduler.run_once()
            qr_name = next(iter(backend.queued_resources))
            backend.force_active(qr_name)
            scheduler.run_once()
            backend.write_gcs(f"{job_dir}/failed", "stale old-attempt failure")

            scheduler.run_once()
            state = json.loads(backend.read_gcs(f"{job_dir}/status.json") or "{}")
            self.assertEqual(state["status"], JobStatus.RUNNING.value)
            self.assertEqual(state["current_attempt"], 0)

    def test_focused_user_reconciliation_skips_other_users_scheduling(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d), quota_total=16)
            write_job(
                backend,
                config.primary_bucket,
                make_spec("job-alice", user="alice"),
            )
            bob_dir = write_job(
                backend,
                config.primary_bucket,
                make_spec("job-bob", user="bob"),
            )
            scheduler = Scheduler(backend, config)

            scheduler.run_once(focus_user="alice")

            alice_state = json.loads(
                backend.read_gcs(
                    f"{config.primary_bucket}/jobs/job-alice/status.json"
                )
                or "{}"
            )
            bob_state = json.loads(backend.read_gcs(f"{bob_dir}/status.json") or "{}")
            self.assertEqual(alice_state["status"], JobStatus.PROVISIONING.value)
            self.assertEqual(bob_state["status"], JobStatus.PENDING.value)
            self.assertEqual(set(scheduler.queued_resources.values()), {"job-alice"})

    def test_focused_user_still_honors_other_users_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d), quota_total=16)
            write_job(
                backend,
                config.primary_bucket,
                make_spec("job-alice", user="alice"),
            )
            bob_dir = write_job(
                backend,
                config.primary_bucket,
                make_spec("job-bob", user="bob"),
            )
            backend.write_gcs(f"{bob_dir}/canceled", "cancel")
            scheduler = Scheduler(backend, config)

            scheduler.run_once(focus_user="alice")

            bob_state = json.loads(backend.read_gcs(f"{bob_dir}/status.json") or "{}")
            self.assertEqual(bob_state["status"], JobStatus.CANCELED.value)
            self.assertEqual(set(scheduler.queued_resources.values()), {"job-alice"})

    def test_focused_user_cancels_other_users_active_tpu(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d), quota_total=16)
            write_job(
                backend,
                config.primary_bucket,
                make_spec("job-bob", user="bob"),
            )
            scheduler = Scheduler(backend, config)
            scheduler.run_once(focus_user="bob")
            qr_name = next(iter(scheduler.queued_resources))
            backend.force_active(qr_name)
            scheduler.run_once(focus_user="bob")

            bob_dir = f"{config.primary_bucket}/jobs/job-bob"
            backend.write_gcs(f"{bob_dir}/canceled", "cancel")
            focused = Scheduler(backend, config)
            focused.run_once(focus_user="alice")

            bob_state = json.loads(backend.read_gcs(f"{bob_dir}/status.json") or "{}")
            self.assertEqual(bob_state["status"], JobStatus.CANCELED.value)
            self.assertNotIn(qr_name, backend.queued_resources)

    def test_focused_user_counts_other_users_active_quota(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d), quota_total=8)
            write_job(
                backend,
                config.primary_bucket,
                make_spec("job-alice", user="alice"),
            )
            bob_dir = write_job(
                backend,
                config.primary_bucket,
                make_spec("job-bob", user="bob"),
            )
            bob_state = JobState.new()
            bob_state.status = JobStatus.RUNNING
            bob_state.current_qr_name = "iqtest-bob-a1"
            bob_state.current_qr_state = "ACTIVE"
            backend.write_gcs(f"{bob_dir}/status.json", json.dumps(bob_state.to_dict()))

            scheduler = Scheduler(backend, config)
            scheduler.run_once(focus_user="alice")

            alice_state = json.loads(
                backend.read_gcs(
                    f"{config.primary_bucket}/jobs/job-alice/status.json"
                )
                or "{}"
            )
            self.assertEqual(alice_state["status"], JobStatus.PENDING.value)
            self.assertEqual(len(scheduler.queued_resources), 1)
            self.assertEqual(
                scheduler.queued_resources["iqtest-bob-a1"],
                "job-bob",
            )

    def test_focused_user_does_not_refresh_other_terminal_history(self) -> None:
        class RecordingBackend(DryRunBackend):
            def __init__(self, base_dir: str):
                super().__init__(base_dir)
                self.read_urls: list[str] = []

            def read_gcs(self, url: str) -> str | None:
                self.read_urls.append(url)
                return super().read_gcs(url)

        with tempfile.TemporaryDirectory() as d:
            backend = RecordingBackend(d)
            config = make_config(Path(d), quota_total=24)
            write_job(
                backend,
                config.primary_bucket,
                make_spec("job-alice", user="alice"),
            )
            bob_dir = write_job(
                backend,
                config.primary_bucket,
                make_spec("job-bob", user="bob"),
            )
            charlie_dir = write_job(
                backend,
                config.primary_bucket,
                make_spec("job-charlie", user="charlie"),
            )
            bob_state = JobState.new()
            bob_state.status = JobStatus.SUCCEEDED
            backend.write_gcs(f"{bob_dir}/status.json", json.dumps(bob_state.to_dict()))
            charlie_state = JobState.new()
            charlie_state.status = JobStatus.RUNNING
            charlie_state.current_qr_name = "iqtest-charlie-a1"
            charlie_state.current_qr_state = "ACTIVE"
            backend.write_gcs(
                f"{charlie_dir}/status.json",
                json.dumps(charlie_state.to_dict()),
            )
            scheduler = Scheduler(backend, config)
            scheduler.run_once(focus_user="alice")
            backend.read_urls.clear()

            scheduler.run_once(focus_user="alice")

            self.assertNotIn(f"{bob_dir}/status.json", backend.read_urls)
            self.assertIn(f"{charlie_dir}/status.json", backend.read_urls)

    def test_focused_reconciliation_skips_unrelated_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d), quota_total=16)
            write_job(backend, config.primary_bucket, make_spec("job-a"))
            job_b_dir = write_job(backend, config.primary_bucket, make_spec("job-b"))
            scheduler = Scheduler(backend, config)

            scheduler.run_once()
            qr_by_job = {job_id: qr for qr, job_id in scheduler.queued_resources.items()}
            backend.force_active(qr_by_job["job-a"])
            backend.force_active(qr_by_job["job-b"])
            scheduler.run_once()

            backend.force_preempt(qr_by_job["job-a"])
            backend.write_gcs(f"{job_b_dir}/canceled", "cancel")
            focused = Scheduler(backend, config)
            focused.run_once(focus_job_ref="job-a")

            state_a = json.loads(
                backend.read_gcs(f"{config.primary_bucket}/jobs/job-a/status.json") or "{}"
            )
            state_b = json.loads(
                backend.read_gcs(f"{config.primary_bucket}/jobs/job-b/status.json") or "{}"
            )
            self.assertEqual(state_a["status"], JobStatus.PROVISIONING.value)
            self.assertEqual(state_a["current_attempt"], 1)
            self.assertEqual(state_b["status"], JobStatus.RUNNING.value)
            self.assertIn(qr_by_job["job-b"], backend.queued_resources)

    def test_focused_reconciliation_schedules_only_focused_pending_job(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d), quota_total=8)
            write_job(backend, config.primary_bucket, make_spec("job-older"))
            write_job(backend, config.primary_bucket, make_spec("job-focus"))

            scheduler = Scheduler(backend, config)
            scheduler.run_once(focus_job_ref="job-focus")

            state_older = json.loads(
                backend.read_gcs(f"{config.primary_bucket}/jobs/job-older/status.json") or "{}"
            )
            state_focus = json.loads(
                backend.read_gcs(f"{config.primary_bucket}/jobs/job-focus/status.json") or "{}"
            )
            self.assertEqual(state_older["status"], JobStatus.PENDING.value)
            self.assertEqual(state_focus["status"], JobStatus.PROVISIONING.value)

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

    def test_requeues_ready_unresponsive_tpu(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d))
            write_job(backend, config.primary_bucket, make_spec("job-a"))
            scheduler = Scheduler(backend, config)

            scheduler.run_once()
            qr_name = next(iter(backend.queued_resources))
            backend.force_active(qr_name)
            scheduler.run_once()

            backend.queued_resources[qr_name].health = "TIMEOUT"
            scheduler.run_once()
            state = json.loads(
                backend.read_gcs(f"{config.primary_bucket}/jobs/job-a/status.json") or "{}"
            )
            self.assertEqual(state["status"], "PROVISIONING")
            self.assertEqual(state["current_attempt"], 1)
            self.assertEqual(
                state["attempts"][0]["error"],
                "TPU_VM_HEALTH_TIMEOUT",
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
        self.assertEqual(config.scheduler.create_failure_backoff, 300)

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
        self.assertIn('"$ATTEMPT_DIR/setup-ready/worker-$WORKER_ID"', script)
        self.assertIn('"$ATTEMPT_DIR/setup-complete"', script)
        self.assertIn('"$ATTEMPT_DIR/failed"', script)
        self.assertIn('"$ATTEMPT_DIR/succeeded"', script)
        self.assertIn("/logs/attempt-$ATTEMPT/worker-$WORKER_ID.log", script)
        self.assertIn("sha256sum --check", script)
        self.assertIn('JOB_PHASE="setup"', script)
        self.assertIn('JOB_PHASE="setup_barrier"', script)
        self.assertIn('JOB_PHASE="command"', script)
        self.assertIn('"failure_type":"%s"', script)
        self.assertIn("log_upload_loop &", script)
        self.assertIn('gsutil -q cp "$LOG_DIR/worker-$WORKER_ID.log"', script)
        self.assertLess(script.index("log_upload_loop &"), script.index('echo "Running setup"'))
        self.assertLess(script.index("heartbeat_loop &"), script.index('echo "Running setup"'))
        self.assertLess(script.rindex("wait_for_setup_barrier"), script.index('echo "Running command"'))
        self.assertIn("EXPECTED_WORKERS=1", script)
        self.assertIn("SETUP_BARRIER_TIMEOUT=1800", script)
        self.assertIn("Preserving the first worker failure report", script)
        self.assertLess(script.index("heartbeat_loop &"), script.index('JOB_PHASE="setup"'))
        self.assertLess(script.index('JOB_PHASE="setup_barrier"'), script.index('JOB_PHASE="command"'))
        self.assertNotIn(".tpu-jobs", script)
        self.assertNotIn("watch.pid", script)
        syntax = subprocess.run(
            ["bash", "-n"],
            input=script,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_startup_script_uses_resource_worker_count_for_setup_barrier(self) -> None:
        spec = make_spec("job-a")
        spec = replace(
            spec,
            resources=replace(spec.resources, chips=64, workers=16),
        )
        script = build_startup_script(
            job_id="job-a",
            spec=spec,
            qr_name="iqtest-123-v6-64-a1",
            job_dir="gs://test-bucket/queue/jobs/job-a",
            attempt=1,
            project="test-project",
            setup_barrier_timeout=90,
        )
        self.assertIn("EXPECTED_WORKERS=16", script)
        self.assertIn("SETUP_BARRIER_TIMEOUT=90", script)

    def test_status_explains_terminal_error_and_worker_log_command(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config_path = write_config_file(root)
            config = load_config(config_path)
            state_dir = root / "state"
            backend = DryRunBackend(str(state_dir))
            job_dir = write_job(backend, config.primary_bucket, make_spec("job-a"))
            state = JobState.new()
            state.status = JobStatus.FAILED
            state.current_attempt = 1
            state.attempts.append(
                AttemptRecord(
                    attempt=1,
                    qr_name="iqtest-job-a-a1",
                    started_at=utc_now(),
                    ended_at=utc_now(),
                    error="Worker 3 exited with code 17 during setup",
                    failure_type="SETUP_ERROR",
                    retryable=False,
                    phase="setup",
                    worker_id=3,
                    exit_code=17,
                )
            )
            backend.write_gcs(f"{job_dir}/status.json", json.dumps(state.to_dict()))

            args = build_parser().parse_args(
                [
                    "--config",
                    str(config_path),
                    "--dry-run",
                    "--base-dir",
                    str(state_dir),
                    "status",
                    "job-a",
                ]
            )
            out = io.StringIO()
            with redirect_stdout(out):
                args.func(args)
            text = out.getvalue()
            self.assertIn("terminal error; agent diagnosis is required", text)
            self.assertIn(
                "tpu logs job-a --attempt 1 --worker 3 --lines 220",
                text,
            )

    def test_scheduler_lock_rejects_second_local_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            lock_path = Path(d) / "scheduler.lock"
            with _scheduler_lock(lock_path):
                with self.assertRaisesRegex(SystemExit, "Another local TPU scheduler"):
                    with _scheduler_lock(lock_path):
                        self.fail("second scheduler unexpectedly acquired the lock")

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
        self.assertIn("tpu.nodes.update", hint)
        self.assertIn("tpu interactive add-key", hint)
        self.assertIn("exact local SSH key", hint)
        self.assertIn("us-central2-b", hint)
        self.assertIn("No TPU Admin role is required", hint)

    def test_interactive_run_parses_options_after_name(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "interactive",
                "run",
                "v4-16-interactive",
                "--worker",
                "all",
                "--",
                "python",
                "scratch.py",
                "--worker",
                "7",
            ]
        )

        self.assertEqual(args.name, "v4-16-interactive")
        self.assertEqual(args.worker, "all")
        self.assertEqual(
            _command_from_args(args.command),
            "python scratch.py --worker 7",
        )

    def test_interactive_tmux_parses_options_after_name_and_preserves_quoting(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "interactive",
                "tmux",
                "v4-16-interactive",
                "--session",
                "alice-train",
                "--worker",
                "all",
                "--",
                "bash",
                "-lc",
                "cd ~/repo && uv run python scripts/train.py --fsdp-devices 4",
            ]
        )

        self.assertEqual(args.name, "v4-16-interactive")
        self.assertEqual(args.session, "alice-train")
        self.assertEqual(args.worker, "all")
        self.assertEqual(
            _command_from_args(args.command),
            "bash -lc 'cd ~/repo && uv run python scripts/train.py --fsdp-devices 4'",
        )

    def test_interactive_run_keeps_options_before_name_compatible(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "interactive",
                "run",
                "--worker",
                "all",
                "v4-16-interactive",
                "--",
                "hostname",
            ]
        )

        self.assertEqual(args.worker, "all")
        self.assertEqual(_command_from_args(args.command), "hostname")

    def test_default_config_has_v4_interactive_entry(self) -> None:
        config = load_config()
        self.assertIn("v4-4-01-interactive", config.interactive_tpus)
        self.assertEqual(config.interactive_tpus["v4-4-01-interactive"].version, "v4")
        self.assertIn("v4-16-01-interactive", config.interactive_tpus)
        self.assertIn("v4-4-04-interactive", config.interactive_tpus)
        self.assertEqual(config.interactive_tpus["v4-16-01-interactive"].workers, 4)

    def test_default_config_has_v6e_four_chips_per_worker(self) -> None:
        config = load_config()

        # v6e-8 is a single host regardless of chip count, matching the
        # v4-8/v5-8 single-host pattern; larger slices scale at 4 chips/worker.
        v6_8 = config.resources["v6-8"]
        self.assertEqual(v6_8.workers, 1)
        self.assertEqual(v6_8.chips, 8)

        expected_workers = {
            "v6-16": 4,
            "v6-32": 8,
            "v6-64": 16,
            "v6-128": 32,
        }

        for resource_name, workers in expected_workers.items():
            resource = config.resources[resource_name]
            self.assertEqual(resource.workers, workers)
            self.assertEqual(resource.chips, workers * 4)

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

    def _ssh_keys_config_and_backend(self, d: str) -> tuple[Path, DryRunBackend]:
        root = Path(d)
        config_path = root / "resources.yaml"
        config_path.write_text(
            """
quota_groups:
  v4:
    total_chips: 8
resources:
  v4-8:
    version: v4
    accelerator_type: v4-8
    runtime_version: tpu-ubuntu2204-base
    zone: us-central2-b
    project: test-project
    chips: 8
    workers: 1
    spot: true
    enabled: true
    quota_group: v4
buckets:
  us-east1: gs://test-bucket/queue
primary_bucket_region: us-east1
scheduler:
  qr_prefix: iqtest
interactive_tpus:
  v4-4-01-interactive:
    version: v4
    zone: us-central2-b
    project: test-project
    workers: 1
  v4-4-02-interactive:
    version: v4
    zone: us-central2-b
    project: test-project
    workers: 1
""".lstrip()
        )
        backend = DryRunBackend(str(root / "state"))
        return config_path, backend

    def _run_admin_ssh_keys(
        self, config_path: Path, backend: DryRunBackend, extra: list[str]
    ) -> tuple[int, str]:
        parser = build_parser()
        args = parser.parse_args(
            ["--config", str(config_path), "--dry-run", "admin", "ssh-keys", *extra]
        )
        out = io.StringIO()
        with patch("irom_tpu_tools.queue.cli._backend", return_value=backend):
            with redirect_stdout(out):
                rc = args.func(args)
        return rc, out.getvalue()

    def test_admin_ssh_keys_syncs_union_across_interactive_nodes(self) -> None:
        alice = "alice:ssh-rsa " + base64.b64encode(b"alice-key").decode() + " alice"
        bob = "bob:ssh-rsa " + base64.b64encode(b"bob-key").decode() + " bob"
        odd = "not a parseable entry"
        with tempfile.TemporaryDirectory() as d:
            config_path, backend = self._ssh_keys_config_and_backend(d)
            backend.set_tpu_vm_ssh_keys(
                "v4-4-01-interactive", "test-project", "us-central2-b", f"{alice}\n{bob}"
            )
            backend.set_tpu_vm_ssh_keys(
                "v4-4-02-interactive", "test-project", "us-central2-b", f"{alice}\n{odd}"
            )

            rc, text = self._run_admin_ssh_keys(config_path, backend, [])
            self.assertEqual(rc, 0)
            self.assertIn("MISSING: bob", text)
            self.assertIn("Dry run only", text)
            node2 = backend.get_tpu_vm_ssh_keys(
                "v4-4-02-interactive", "test-project", "us-central2-b"
            )
            self.assertNotIn("bob:", node2)

            rc, text = self._run_admin_ssh_keys(config_path, backend, ["--yes"])
            self.assertEqual(rc, 0)
            node2 = backend.get_tpu_vm_ssh_keys(
                "v4-4-02-interactive", "test-project", "us-central2-b"
            )
            self.assertEqual(node2.splitlines(), [alice, odd, bob])
            node1 = backend.get_tpu_vm_ssh_keys(
                "v4-4-01-interactive", "test-project", "us-central2-b"
            )
            self.assertEqual(node1.splitlines(), [alice, bob])

            rc, text = self._run_admin_ssh_keys(config_path, backend, [])
            self.assertEqual(rc, 0)
            self.assertIn("already has every known key", text)

    def test_admin_ssh_keys_add_provisions_new_user_everywhere(self) -> None:
        alice = "alice:ssh-rsa " + base64.b64encode(b"alice-key").decode() + " alice"
        carol_pub = "ssh-rsa " + base64.b64encode(b"carol-key").decode() + " carol@laptop"
        with tempfile.TemporaryDirectory() as d:
            config_path, backend = self._ssh_keys_config_and_backend(d)
            for node in ("v4-4-01-interactive", "v4-4-02-interactive"):
                backend.set_tpu_vm_ssh_keys(node, "test-project", "us-central2-b", alice)
            key_file = Path(d) / "carol.pub"
            key_file.write_text(carol_pub + "\n")

            rc, _ = self._run_admin_ssh_keys(
                config_path, backend, ["--add", f"carol={key_file}", "--yes"]
            )
            self.assertEqual(rc, 0)
            carol_blob = base64.b64encode(b"carol-key").decode()
            for node in ("v4-4-01-interactive", "v4-4-02-interactive"):
                keys = backend.get_tpu_vm_ssh_keys(node, "test-project", "us-central2-b")
                # The comment is normalized to the username by the guard.
                self.assertEqual(
                    keys.splitlines(), [alice, f"carol:ssh-rsa {carol_blob} carol"]
                )

    def test_admin_ssh_keys_skips_unreadable_node(self) -> None:
        alice = "alice:ssh-rsa " + base64.b64encode(b"alice-key").decode() + " alice"
        with tempfile.TemporaryDirectory() as d:
            config_path, backend = self._ssh_keys_config_and_backend(d)
            backend.set_tpu_vm_ssh_keys(
                "v4-4-01-interactive", "test-project", "us-central2-b", alice
            )
            # v4-4-02-interactive has no entry: describe fails -> None.

            rc, text = self._run_admin_ssh_keys(config_path, backend, ["--yes"])
            self.assertEqual(rc, 1)
            self.assertIn("UNREADABLE", text)
            self.assertIsNone(
                backend.get_tpu_vm_ssh_keys(
                    "v4-4-02-interactive", "test-project", "us-central2-b"
                )
            )

    def test_scheduler_provisions_requested_ssh_key(self) -> None:
        alice = "alice:ssh-rsa " + base64.b64encode(b"alice-key").decode() + " alice"
        bob_pub = "ssh-rsa " + base64.b64encode(b"bob-key").decode() + " bob@laptop"
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d))
            backend.set_tpu_vm_ssh_keys(
                "v4-4-01-interactive", "test-project", "us-central2-b", alice
            )
            request_url = f"{config.primary_bucket}/ssh-key-requests/bob.pub"
            backend.write_gcs(request_url, bob_pub + "\n")

            scheduler = Scheduler(backend, config)
            scheduler.run_once(focus_user="lzha")

            keys = backend.get_tpu_vm_ssh_keys(
                "v4-4-01-interactive", "test-project", "us-central2-b"
            )
            bob_blob = base64.b64encode(b"bob-key").decode()
            self.assertEqual(keys.splitlines(), [alice, f"bob:ssh-rsa {bob_blob} bob"])
            self.assertIsNone(backend.read_gcs(request_url))

            # A second pass after the interval is a no-op.
            scheduler._last_interactive_key_sync = 0.0
            scheduler.sync_interactive_ssh_key_requests()
            keys_again = backend.get_tpu_vm_ssh_keys(
                "v4-4-01-interactive", "test-project", "us-central2-b"
            )
            self.assertEqual(keys_again, keys)

    def test_scheduler_rejects_key_request_with_mismatched_username(self) -> None:
        alice = "alice:ssh-rsa " + base64.b64encode(b"alice-key").decode() + " alice"
        smuggled = "lzha:ssh-rsa " + base64.b64encode(b"evil-key").decode() + " lzha"
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d))
            backend.set_tpu_vm_ssh_keys(
                "v4-4-01-interactive", "test-project", "us-central2-b", alice
            )
            request_url = f"{config.primary_bucket}/ssh-key-requests/mallory.pub"
            backend.write_gcs(request_url, smuggled + "\n")

            scheduler = Scheduler(backend, config)
            scheduler.sync_interactive_ssh_key_requests()

            keys = backend.get_tpu_vm_ssh_keys(
                "v4-4-01-interactive", "test-project", "us-central2-b"
            )
            self.assertEqual(keys, alice)
            self.assertIsNone(backend.read_gcs(request_url))

    def test_scheduler_keeps_key_request_when_node_unreadable(self) -> None:
        bob_pub = "ssh-rsa " + base64.b64encode(b"bob-key").decode() + " bob"
        with tempfile.TemporaryDirectory() as d:
            backend = DryRunBackend(d)
            config = make_config(Path(d))
            # No ssh-keys seeded: describe fails -> None -> request must survive.
            request_url = f"{config.primary_bucket}/ssh-key-requests/bob.pub"
            backend.write_gcs(request_url, bob_pub + "\n")

            scheduler = Scheduler(backend, config)
            scheduler.sync_interactive_ssh_key_requests()

            self.assertIsNotNone(backend.read_gcs(request_url))
            self.assertIsNone(
                backend.get_tpu_vm_ssh_keys(
                    "v4-4-01-interactive", "test-project", "us-central2-b"
                )
            )

    def test_interactive_add_key_uploads_request(self) -> None:
        carol_pub = "ssh-rsa " + base64.b64encode(b"carol-key").decode() + " carol@laptop"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config_path = write_config_file(root)
            config_path.write_text(
                config_path.read_text().replace(
                    "interactive_tpus: {}",
                    "interactive_tpus:\n"
                    "  v4-4-01-interactive:\n"
                    "    version: v4\n"
                    "    zone: us-central2-b\n"
                    "    project: test-project\n"
                    "    workers: 1\n",
                )
            )
            backend = DryRunBackend(str(root / "state"))
            key_file = root / "carol.pub"
            key_file.write_text(carol_pub + "\n")

            parser = build_parser()
            args = parser.parse_args(
                [
                    "--config",
                    str(config_path),
                    "--dry-run",
                    "interactive",
                    "add-key",
                    "--key-file",
                    str(key_file),
                    "--user",
                    "carol",
                ]
            )
            out = io.StringIO()
            with patch("irom_tpu_tools.queue.cli._backend", return_value=backend):
                with redirect_stdout(out):
                    rc = args.func(args)
            self.assertEqual(rc, 0)
            self.assertIn("Requested SSH key provisioning for carol", out.getvalue())
            request = backend.read_gcs(
                "gs://test-bucket/queue/ssh-key-requests/carol.pub"
            )
            self.assertEqual(request, carol_pub + "\n")

    def test_delete_reports_sentinel_write_failure(self) -> None:
        class FailingWriteBackend(DryRunBackend):
            def write_gcs(self, url: str, content: str) -> bool:
                if url.endswith("/canceled"):
                    return False
                return super().write_gcs(url, content)

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config_path = write_config_file(root)
            state_dir = root / "state"
            backend = FailingWriteBackend(str(state_dir))
            config = load_config(str(config_path))
            job_dir = write_job(backend, config.primary_bucket, make_spec("job-del"))

            parser = build_parser()
            args = parser.parse_args(
                [
                    "--config",
                    str(config_path),
                    "--dry-run",
                    "--base-dir",
                    str(state_dir),
                    "delete",
                    "job-del",
                ]
            )
            out = io.StringIO()
            with patch("irom_tpu_tools.queue.cli._backend", return_value=backend):
                with redirect_stdout(out):
                    rc = args.func(args)
            self.assertEqual(rc, 1)
            self.assertIn("Failed to write cancellation sentinel", out.getvalue())

            DryRunBackend.write_gcs(backend, f"{job_dir}/canceled", "cancel")
            out = io.StringIO()
            with patch("irom_tpu_tools.queue.cli._backend", return_value=backend):
                with redirect_stdout(out):
                    rc = args.func(args)
            self.assertEqual(rc, 0)
            self.assertIn("Cancellation already requested", out.getvalue())

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


class GCPBackendAccountPinningTests(unittest.TestCase):
    """Control-plane gcloud calls are pinned to a configured service account so
    queue operations never ride the ambient (possibly personal, RAPT-expiring)
    gcloud identity -- while interactive SSH stays on the invoking user."""

    SA = "irom-service-account@mae-irom-lab-guided-data.iam.gserviceaccount.com"

    @staticmethod
    def _mock_run() -> Mock:
        # returncode 0 with parseable "{}" stdout keeps every JSON-parsing
        # method on its success path so each records exactly one subprocess call.
        return Mock(return_value=subprocess.CompletedProcess([], 0, "{}", ""))

    def _exercise_control_plane(self, backend: GCPBackend) -> None:
        backend.read_gcs("gs://b/x")
        backend.write_gcs("gs://b/x", "data")
        backend.exists_gcs("gs://b/x")
        backend.list_gcs("gs://b/")
        backend.delete_gcs("gs://b/x", recursive=True)
        backend.upload_file("/tmp/x", "gs://b/x")
        backend.create_queued_resource(
            name="q",
            node_id="q",
            project="p",
            zone="z",
            accelerator_type="v6e-8",
            runtime_version="rt",
            spot=True,
            startup_script_path="/tmp/s.sh",
            service_account=None,
        )
        backend.delete_queued_resource("q", "p", "z")
        backend.delete_tpu_vm("q", "p", "z")
        backend.get_queued_resource_state("q", "p", "z")
        backend.get_tpu_vm_state("q", "p", "z")
        backend.list_queued_resources("p", "z")
        backend.list_tpu_vms("p", "z")
        backend.get_tpu_vm_ssh_keys("q", "p", "z")
        backend.set_tpu_vm_ssh_keys("q", "p", "z", "user:key")

    def test_control_plane_calls_pin_account_when_configured(self) -> None:
        backend = GCPBackend(control_plane_account=self.SA)
        run = self._mock_run()
        with patch("irom_tpu_tools.queue.backend.subprocess.run", run):
            self._exercise_control_plane(backend)

        self.assertGreaterEqual(run.call_count, 15)
        for call in run.call_args_list:
            cmd = call.args[0]
            env = call.kwargs.get("env")
            self.assertIsNotNone(env, f"expected a pinned env for: {cmd}")
            self.assertEqual(
                env.get("CLOUDSDK_CORE_ACCOUNT"),
                self.SA,
                f"account not pinned for: {cmd}",
            )
            # Parent environment must be inherited, not replaced wholesale.
            self.assertIn("PATH", env)

    def test_interactive_ssh_does_not_pin_account(self) -> None:
        # The critical scoping guarantee: SSH sessions keep the invoking user's
        # identity even when a control-plane account is configured.
        backend = GCPBackend(control_plane_account=self.SA)
        run = self._mock_run()
        with patch("irom_tpu_tools.queue.backend.subprocess.run", run):
            backend.ssh_tpu_vm("node", "p", "z", 0, "hostname")

        self.assertEqual(run.call_count, 1)
        env = run.call_args.kwargs.get("env")
        if env is not None:
            self.assertNotIn("CLOUDSDK_CORE_ACCOUNT", env)

    def test_no_account_configured_preserves_ambient_env(self) -> None:
        # Default construction must be byte-for-byte the pre-change behavior:
        # subprocess.run receives no env override anywhere.
        backend = GCPBackend()
        run = self._mock_run()
        with patch("irom_tpu_tools.queue.backend.subprocess.run", run):
            self._exercise_control_plane(backend)
            backend.ssh_tpu_vm("node", "p", "z", 0, "hostname")

        self.assertGreater(run.call_count, 0)
        for call in run.call_args_list:
            self.assertIsNone(
                call.kwargs.get("env"),
                f"no env override expected when account unset: {call.args[0]}",
            )

    def test_empty_account_string_is_treated_as_unset(self) -> None:
        backend = GCPBackend(control_plane_account="")
        self.assertIsNone(backend.control_plane_account)
        run = self._mock_run()
        with patch("irom_tpu_tools.queue.backend.subprocess.run", run):
            backend.read_gcs("gs://b/x")
        self.assertIsNone(run.call_args.kwargs.get("env"))

    def test_dry_run_never_invokes_subprocess(self) -> None:
        backend = GCPBackend(dry_run_commands=True, control_plane_account=self.SA)
        run = self._mock_run()
        with patch("irom_tpu_tools.queue.backend.subprocess.run", run):
            backend.write_gcs("gs://b/x", "data")
            backend.delete_queued_resource("q", "p", "z")
        run.assert_not_called()

    def test_backend_factory_reads_account_from_env(self) -> None:
        from irom_tpu_tools.queue.cli import _backend

        args = argparse.Namespace(dry_run=False)
        with patch.dict(os.environ, {"TPU_CONTROL_PLANE_ACCOUNT": self.SA}):
            backend = _backend(args)
        self.assertIsInstance(backend, GCPBackend)
        self.assertEqual(backend.control_plane_account, self.SA)

        env_without = {
            k: v for k, v in os.environ.items() if k != "TPU_CONTROL_PLANE_ACCOUNT"
        }
        with patch.dict(os.environ, env_without, clear=True):
            backend = _backend(args)
        self.assertIsNone(backend.control_plane_account)


if __name__ == "__main__":
    unittest.main()
