from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import logging
import tempfile
import time
import uuid

from .backend import Backend
from .config import QueueConfig
from .startup_script import write_startup_script
from .types import (
    AttemptRecord,
    Job,
    JobSpec,
    JobState,
    JobStatus,
    QRState,
    ResourceConfig,
    TERMINAL_STATUSES,
    utc_now,
)

logger = logging.getLogger(__name__)

TERMINAL_TPU_VM_STATES = {"PREEMPTED", "TERMINATED", "STOPPED", "DELETED", "FAILED"}


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class Scheduler:
    def __init__(self, backend: Backend, config: QueueConfig):
        self.backend = backend
        self.config = config
        self.jobs: dict[str, Job] = {}
        self.queued_resources: dict[str, str] = {}
        self._last_state_write = 0.0
        self._last_orphan_check = 0.0

    def _job_dir(self, job: Job) -> str:
        return job.job_dir

    def scan_jobs(self) -> None:
        for bucket in self.config.buckets.values():
            self._scan_bucket(bucket)

    def _scan_bucket(self, bucket: str) -> None:
        for job_dir in self.backend.list_gcs(f"{bucket.rstrip('/')}/jobs/"):
            job_id = job_dir.rstrip("/").rsplit("/", 1)[-1]
            if job_id in self.jobs:
                continue
            spec_json = self.backend.read_gcs(f"{bucket.rstrip('/')}/jobs/{job_id}/spec.json")
            if not spec_json:
                continue
            try:
                spec = JobSpec.from_dict(json.loads(spec_json))
                status_json = self.backend.read_gcs(
                    f"{bucket.rstrip('/')}/jobs/{job_id}/status.json"
                )
                state = (
                    JobState.from_dict(json.loads(status_json))
                    if status_json
                    else JobState.new()
                )
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("Skipping invalid job %s: %s", job_id, exc)
                continue
            job = Job(spec=spec, state=state, bucket=bucket.rstrip("/"))
            self.jobs[job_id] = job
            if (
                state.status in {JobStatus.PROVISIONING, JobStatus.RUNNING}
                and state.current_qr_name
            ):
                self.queued_resources[state.current_qr_name] = job_id

    def _write_status(self, job_id: str) -> None:
        job = self.jobs[job_id]
        job.state.last_updated = utc_now()
        self.backend.write_gcs(
            f"{job.job_dir}/status.json", json.dumps(job.state.to_dict(), indent=2)
        )

    def _cancel_requested(self, job: Job) -> bool:
        return self.backend.exists_gcs(f"{job.job_dir}/canceled")

    def check_canceled_jobs(self) -> None:
        for job_id, job in list(self.jobs.items()):
            if job.state.status in TERMINAL_STATUSES:
                continue
            if self._cancel_requested(job):
                self._cancel_job(job_id, "CANCELED")

    def _cancel_job(self, job_id: str, reason: str) -> None:
        job = self.jobs[job_id]
        current_qr = job.state.current_qr_name
        resource = self.config.resources.get(job.spec.resources.resource_name)
        if current_qr and resource:
            self._cleanup_job_resources(current_qr, resource)
        job.state.status = JobStatus.CANCELED
        job.state.current_qr_name = None
        job.state.current_qr_state = None
        logger.info("Job %s canceled: %s", job_id, reason)
        self._write_status(job_id)

    def check_completed_jobs(self) -> None:
        for job_id, job in list(self.jobs.items()):
            if job.state.status != JobStatus.RUNNING:
                continue
            if self.backend.exists_gcs(f"{job.job_dir}/succeeded"):
                self._finish_job(job_id, None)
            elif self.backend.exists_gcs(f"{job.job_dir}/failed"):
                error = self.backend.read_gcs(f"{job.job_dir}/failed") or "FAILED"
                self._finish_job(job_id, error.strip())

    def _finish_job(self, job_id: str, error: str | None) -> None:
        job = self.jobs[job_id]
        attempt = job.state.current_attempt + 1
        job.state.attempts.append(
            AttemptRecord(
                attempt=attempt,
                qr_name=job.state.current_qr_name or "",
                started_at=job.state.provisioned_at or "",
                ended_at=utc_now(),
                error=error,
            )
        )
        job.state.status = JobStatus.FAILED if error else JobStatus.SUCCEEDED
        current_qr = job.state.current_qr_name
        resource = self.config.resources.get(job.spec.resources.resource_name)
        job.state.current_qr_name = None
        job.state.current_qr_state = None
        self._write_status(job_id)
        if current_qr and resource:
            self._cleanup_job_resources(current_qr, resource)

    def check_retry_requests(self) -> None:
        for job_id, job in list(self.jobs.items()):
            if job.state.status != JobStatus.FAILED:
                continue
            marker = f"{job.job_dir}/retry"
            if not self.backend.exists_gcs(marker):
                continue
            self.backend.delete_gcs(marker)
            self.backend.delete_gcs(f"{job.job_dir}/failed")
            if job.state.current_attempt >= job.spec.max_attempts:
                logger.warning("Retry requested for %s but max attempts reached", job_id)
                self._write_status(job_id)
                continue
            job.state.status = JobStatus.PENDING
            job.state.current_qr_name = None
            job.state.current_qr_state = None
            self._write_status(job_id)

    def poll_queued_resources(self) -> None:
        for qr_name, job_id in list(self.queued_resources.items()):
            job = self.jobs.get(job_id)
            if not job:
                self.queued_resources.pop(qr_name, None)
                continue
            resource = self.config.resources.get(job.spec.resources.resource_name)
            if not resource:
                continue
            if job.state.current_qr_name != qr_name:
                self._cleanup_job_resources(qr_name, resource)
                continue
            state = self.backend.get_queued_resource_state(
                qr_name, resource.project, resource.zone
            )
            if state is None:
                self._handle_preemption(job_id, qr_name, resource, "QR disappeared")
                continue
            if job.state.current_qr_state != state.value:
                job.state.current_qr_state = state.value
                self._write_status(job_id)
            self._handle_qr_state(job_id, qr_name, state, resource)

    def _handle_qr_state(
        self, job_id: str, qr_name: str, state: QRState, resource: ResourceConfig
    ) -> None:
        job = self.jobs[job_id]
        if job.state.status in TERMINAL_STATUSES:
            self._cleanup_job_resources(qr_name, resource)
            return
        if state == QRState.ACTIVE:
            vm_state = self.backend.get_tpu_vm_state(qr_name, resource.project, resource.zone)
            if vm_state in TERMINAL_TPU_VM_STATES:
                self._handle_preemption(job_id, qr_name, resource, f"TPU_VM_{vm_state}")
                return
            if job.state.status == JobStatus.PROVISIONING:
                job.state.status = JobStatus.RUNNING
                job.state.provisioned_at = utc_now()
                self.backend.write_gcs(f"{job.job_dir}/running", job.state.provisioned_at)
                self._write_status(job_id)
            elif job.state.status == JobStatus.RUNNING:
                self._check_active_timeouts(job_id, qr_name, resource)
        elif state in {QRState.SUSPENDING, QRState.SUSPENDED}:
            self._handle_preemption(job_id, qr_name, resource, state.value)
        elif state == QRState.FAILED:
            self._handle_preemption(job_id, qr_name, resource, "QR_FAILED")

    def _check_active_timeouts(
        self, job_id: str, qr_name: str, resource: ResourceConfig
    ) -> None:
        job = self.jobs[job_id]
        provisioned_at = _parse_time(job.state.provisioned_at)
        if not provisioned_at:
            return
        now = datetime.now(UTC)
        attempt = job.state.current_attempt + 1
        claimed = self.backend.exists_gcs(f"{job.job_dir}/attempts/attempt-{attempt}/claimed")
        if not claimed:
            elapsed = (now - provisioned_at).total_seconds()
            if elapsed > self.config.scheduler.active_no_claim_timeout:
                self._handle_preemption(job_id, qr_name, resource, "ACTIVE_NO_CLAIM_TIMEOUT")
            return
        heartbeat_text = self.backend.read_gcs(
            f"{job.job_dir}/attempts/attempt-{attempt}/heartbeat"
        )
        heartbeat_at = _parse_time((heartbeat_text or "").strip())
        if not heartbeat_at:
            heartbeat_at = provisioned_at
        elapsed = (now - heartbeat_at).total_seconds()
        if elapsed > self.config.scheduler.heartbeat_timeout:
            self._handle_preemption(job_id, qr_name, resource, "HEARTBEAT_TIMEOUT")

    def _handle_preemption(
        self, job_id: str, qr_name: str, resource: ResourceConfig, error: str
    ) -> None:
        job = self.jobs[job_id]
        job.state.attempts.append(
            AttemptRecord(
                attempt=job.state.current_attempt + 1,
                qr_name=qr_name,
                started_at=job.state.provisioned_at or "",
                ended_at=utc_now(),
                error=error,
            )
        )
        job.state.current_attempt += 1
        job.state.provisioned_at = None
        job.state.current_qr_name = None
        job.state.current_qr_state = None
        if job.state.current_attempt >= job.spec.max_attempts:
            job.state.status = JobStatus.FAILED
        else:
            job.state.status = JobStatus.PENDING
        self._cleanup_job_resources(qr_name, resource)
        self._write_status(job_id)

    def _cleanup_job_resources(self, qr_name: str, resource: ResourceConfig) -> None:
        self.backend.delete_tpu_vm(qr_name, resource.project, resource.zone)
        if self.backend.delete_queued_resource(qr_name, resource.project, resource.zone):
            self.queued_resources.pop(qr_name, None)

    def schedule_pending_jobs(self) -> None:
        chips_by_quota: dict[str, int] = {}
        chips_by_user: dict[str, int] = {}
        for job in self.jobs.values():
            if job.state.status not in {JobStatus.PROVISIONING, JobStatus.RUNNING}:
                continue
            resource = self.config.resources.get(job.spec.resources.resource_name)
            if not resource:
                continue
            chips_by_quota[resource.quota_group] = (
                chips_by_quota.get(resource.quota_group, 0) + resource.chips
            )
            chips_by_user[job.spec.submitted_by] = (
                chips_by_user.get(job.spec.submitted_by, 0) + resource.chips
            )

        pending = [
            (job.spec.priority, job.spec.submit_time, job_id)
            for job_id, job in self.jobs.items()
            if job.state.status == JobStatus.PENDING
        ]
        pending.sort()
        for _, _, job_id in pending:
            job = self.jobs[job_id]
            resource = self.config.resources.get(job.spec.resources.resource_name)
            if not resource or not resource.enabled:
                continue
            quota = self.config.quota_groups[resource.quota_group]
            if chips_by_quota.get(resource.quota_group, 0) + resource.chips > quota.total_chips:
                continue
            max_user_chips = self.config.user_limits.max_chips_for(job.spec.submitted_by)
            if (
                max_user_chips is not None
                and chips_by_user.get(job.spec.submitted_by, 0) + resource.chips
                > max_user_chips
            ):
                continue
            if self._create_queued_resource(job_id, resource):
                chips_by_quota[resource.quota_group] = (
                    chips_by_quota.get(resource.quota_group, 0) + resource.chips
                )
                chips_by_user[job.spec.submitted_by] = (
                    chips_by_user.get(job.spec.submitted_by, 0) + resource.chips
                )

    def _create_queued_resource(self, job_id: str, resource: ResourceConfig) -> bool:
        job = self.jobs[job_id]
        attempt = job.state.current_attempt + 1
        qr_name = self._generate_qr_name(job, attempt)
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as tmp:
            script_path = tmp.name
        write_startup_script(
            job_id=job_id,
            spec=job.spec,
            qr_name=qr_name,
            job_dir=job.job_dir,
            attempt=attempt,
            project=resource.project,
            output_path=script_path,
            heartbeat_interval=max(30, min(self.config.scheduler.heartbeat_timeout // 3, 120)),
        )
        ok = self.backend.create_queued_resource(
            name=qr_name,
            node_id=qr_name,
            project=resource.project,
            zone=resource.zone,
            accelerator_type=resource.accelerator_type,
            runtime_version=resource.runtime_version,
            spot=resource.spot,
            startup_script_path=script_path,
            service_account=resource.service_account,
            label_workaround=self.config.scheduler.qr_label_workaround,
        )
        if not ok:
            return False
        self.queued_resources[qr_name] = job_id
        job.state.status = JobStatus.PROVISIONING
        job.state.current_qr_name = qr_name
        job.state.current_qr_state = QRState.WAITING_FOR_RESOURCES.value
        self._write_status(job_id)
        return True

    def _generate_qr_name(self, job: Job, attempt: int) -> str:
        prefix = self.config.scheduler.qr_prefix
        resource = job.spec.resources.resource_name.replace("_", "-")
        return f"{prefix}-{uuid.uuid4().hex[:8]}-{resource}-a{attempt}"

    def reconcile_orphaned_qrs(self) -> None:
        tracked = set(self.queued_resources)
        seen_zones: set[tuple[str, str]] = set()
        for resource in self.config.resources.values():
            key = (resource.project, resource.zone)
            if key in seen_zones:
                continue
            seen_zones.add(key)
            for qr_name in self.backend.list_queued_resources(
                resource.project, resource.zone, f"{self.config.scheduler.qr_prefix}-"
            ):
                if qr_name in tracked:
                    continue
                logger.warning("Deleting orphaned queue QR %s", qr_name)
                self._cleanup_job_resources(qr_name, resource)

    def _maybe_reconcile_orphans(self) -> None:
        now = time.time()
        if now - self._last_orphan_check < 300:
            return
        self._last_orphan_check = now
        self.reconcile_orphaned_qrs()

    def _maybe_write_scheduler_state(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_state_write < self.config.scheduler.status_write_interval:
            return
        self._last_state_write = now
        jobs = []
        for job_id, job in sorted(self.jobs.items()):
            jobs.append(
                {
                    "job_id": job_id,
                    "bucket": job.bucket,
                    "job_dir": job.job_dir,
                    "spec": job.spec.to_dict(),
                    "state": job.state.to_dict(),
                }
            )
        self.backend.write_gcs(
            f"{self.config.primary_bucket}/scheduler_state.json",
            json.dumps({"updated_at": utc_now(), "jobs": jobs}, indent=2),
        )

    def reap_terminal_jobs(self) -> None:
        cutoff = datetime.now(UTC) - timedelta(days=self.config.scheduler.job_retention_days)
        for job_id, job in list(self.jobs.items()):
            if job.state.status not in TERMINAL_STATUSES:
                continue
            updated = _parse_time(job.state.last_updated)
            if updated and updated < cutoff:
                self.backend.delete_gcs(job.job_dir, recursive=True)
                del self.jobs[job_id]

    def run_once(self) -> None:
        self.scan_jobs()
        self.check_canceled_jobs()
        self.check_completed_jobs()
        self.check_retry_requests()
        self.poll_queued_resources()
        self.schedule_pending_jobs()
        self.reap_terminal_jobs()
        self._maybe_reconcile_orphans()
        self._maybe_write_scheduler_state()

    def run_forever(self) -> None:
        while True:
            try:
                self.run_once()
            except Exception:
                logger.exception("Scheduler iteration failed")
            time.sleep(self.config.scheduler.scan_interval)
