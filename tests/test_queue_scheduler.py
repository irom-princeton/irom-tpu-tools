from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile
import unittest

from irom_tpu_tools.queue.backend import DryRunBackend
from irom_tpu_tools.queue.config import QueueConfig
from irom_tpu_tools.queue.scheduler import Scheduler
from irom_tpu_tools.queue.startup_script import build_startup_script
from irom_tpu_tools.queue.types import (
    JobResources,
    JobSpec,
    JobState,
    JobStatus,
    QuotaGroupConfig,
    ResourceConfig,
    SchedulerConfig,
    UserLimitConfig,
    utc_now,
)


def make_config(tmp: Path, *, user_limit: int | None = None) -> QueueConfig:
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
        quota_groups={"v6": QuotaGroupConfig(name="v6", total_chips=8)},
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


if __name__ == "__main__":
    unittest.main()
