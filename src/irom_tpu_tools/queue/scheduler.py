from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import json
import logging
import tempfile
import time
import uuid

from .backend import Backend
from .config import QueueConfig
from . import interactive
from .startup_script import write_startup_script
from .types import (
    AttemptFailureType,
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
RETRY_TPU_VM_HEALTH = {"TIMEOUT", "UNHEALTHY_MAINTENANCE"}
JOB_SCAN_WORKERS = 8
INTERACTIVE_KEY_SYNC_INTERVAL = 300


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
        self._last_interactive_key_sync = 0.0
        self._create_retry_not_before: dict[str, float] = {}

    def _job_dir(self, job: Job) -> str:
        return job.job_dir

    def scan_jobs(self) -> None:
        for bucket in self.config.buckets.values():
            self._scan_bucket(bucket)

    def _scan_bucket(self, bucket: str) -> None:
        bucket = bucket.rstrip("/")
        pending: list[tuple[str, str]] = []
        for job_dir in self.backend.list_gcs(f"{bucket}/jobs/"):
            job_id = job_dir.rstrip("/").rsplit("/", 1)[-1]
            if job_id in self.jobs:
                continue
            pending.append((job_id, bucket))

        with ThreadPoolExecutor(max_workers=JOB_SCAN_WORKERS) as executor:
            loaded = executor.map(lambda item: self._load_job(*item), pending)
            for job_id, job in loaded:
                if job is None:
                    continue
                self._add_job(job_id, job)

    def _load_job(self, job_id: str, bucket: str) -> tuple[str, Job | None]:
        spec_json = self.backend.read_gcs(f"{bucket}/jobs/{job_id}/spec.json")
        if not spec_json:
            return job_id, None
        try:
            spec = JobSpec.from_dict(json.loads(spec_json))
            status_json = self.backend.read_gcs(f"{bucket}/jobs/{job_id}/status.json")
            state = (
                JobState.from_dict(json.loads(status_json))
                if status_json
                else JobState.new()
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Skipping invalid job %s: %s", job_id, exc)
            return job_id, None
        return job_id, Job(spec=spec, state=state, bucket=bucket)

    def _add_job(self, job_id: str, job: Job) -> None:
        state = job.state
        self.jobs[job_id] = job
        if (
            state.status in {JobStatus.PROVISIONING, JobStatus.RUNNING}
            and state.current_qr_name
        ):
            self.queued_resources[state.current_qr_name] = job_id

    def _refresh_job_states(self, job_ids: set[str]) -> None:
        def load_state(job_id: str) -> tuple[str, JobState | None]:
            job = self.jobs[job_id]
            status_json = self.backend.read_gcs(f"{job.job_dir}/status.json")
            if not status_json:
                return job_id, None
            try:
                return job_id, JobState.from_dict(json.loads(status_json))
            except (json.JSONDecodeError, KeyError, ValueError):
                return job_id, None

        with ThreadPoolExecutor(max_workers=JOB_SCAN_WORKERS) as executor:
            refreshed = executor.map(load_state, sorted(job_ids))
            for job_id, state in refreshed:
                if state is None:
                    continue
                for qr_name, mapped_job_id in list(self.queued_resources.items()):
                    if mapped_job_id == job_id:
                        self.queued_resources.pop(qr_name, None)
                self.jobs[job_id].state = state
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

    def check_canceled_jobs(self, job_ids: set[str] | None = None) -> None:
        for job_id, job in list(self.jobs.items()):
            if job_ids is not None and job_id not in job_ids:
                continue
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

    def check_completed_jobs(self, job_ids: set[str] | None = None) -> None:
        for job_id, job in list(self.jobs.items()):
            if job_ids is not None and job_id not in job_ids:
                continue
            if job.state.status != JobStatus.RUNNING:
                continue
            attempt = job.state.current_attempt + 1
            succeeded = self.backend.read_gcs(
                f"{job.job_dir}/attempts/attempt-{attempt}/succeeded"
            )
            failed = self.backend.read_gcs(
                f"{job.job_dir}/attempts/attempt-{attempt}/failed"
            )
            # Existing attempts created by older startup scripts use root markers.
            startup_version = self.backend.read_gcs(
                f"{job.job_dir}/attempts/attempt-{attempt}/startup_version"
            )
            if startup_version is None:
                if succeeded is None:
                    succeeded = self.backend.read_gcs(f"{job.job_dir}/succeeded")
                if failed is None:
                    failed = self.backend.read_gcs(f"{job.job_dir}/failed")
            if failed is not None:
                report = self._parse_failure_report(failed)
                self._finish_job(job_id, report["message"], report)
            elif succeeded is not None:
                self._finish_job(job_id, None)

    @staticmethod
    def _parse_failure_report(text: str) -> dict[str, object]:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        failure_type = str(
            raw.get("failure_type") or AttemptFailureType.APPLICATION_ERROR.value
        )
        message = str(raw.get("message") or text.strip() or "FAILED")

        def optional_int(value: object) -> int | None:
            try:
                return None if value is None else int(value)
            except (TypeError, ValueError):
                return None

        return {
            "failure_type": failure_type,
            "message": message,
            "phase": raw.get("phase"),
            "worker_id": optional_int(raw.get("worker_id")),
            "exit_code": optional_int(raw.get("exit_code")),
        }

    def _finish_job(
        self,
        job_id: str,
        error: str | None,
        failure_report: dict[str, object] | None = None,
    ) -> None:
        job = self.jobs[job_id]
        attempt = job.state.current_attempt + 1
        report = failure_report or {}
        job.state.attempts.append(
            AttemptRecord(
                attempt=attempt,
                qr_name=job.state.current_qr_name or "",
                started_at=job.state.provisioned_at or "",
                ended_at=utc_now(),
                error=error,
                failure_type=(
                    str(report["failure_type"])
                    if report.get("failure_type") is not None
                    else None
                ),
                retryable=False if error else None,
                phase=(str(report["phase"]) if report.get("phase") is not None else None),
                worker_id=(
                    int(report["worker_id"])
                    if report.get("worker_id") is not None
                    else None
                ),
                exit_code=(
                    int(report["exit_code"])
                    if report.get("exit_code") is not None
                    else None
                ),
            )
        )
        job.state.current_attempt = attempt
        job.state.status = JobStatus.FAILED if error else JobStatus.SUCCEEDED
        current_qr = job.state.current_qr_name
        resource = self.config.resources.get(job.spec.resources.resource_name)
        job.state.current_qr_name = None
        job.state.current_qr_state = None
        self._write_status(job_id)
        if error:
            logger.error(
                "Job %s attempt %s failed (%s); inspect worker logs before retrying: %s",
                job_id,
                attempt,
                report.get("failure_type") or AttemptFailureType.APPLICATION_ERROR.value,
                error,
            )
        else:
            logger.info("Job %s attempt %s succeeded", job_id, attempt)
        if current_qr and resource:
            self._cleanup_job_resources(current_qr, resource)

    def check_retry_requests(self, job_ids: set[str] | None = None) -> None:
        for job_id, job in list(self.jobs.items()):
            if job_ids is not None and job_id not in job_ids:
                continue
            if job.state.status != JobStatus.FAILED:
                continue
            marker = f"{job.job_dir}/retry"
            if not self.backend.exists_gcs(marker):
                continue
            self.backend.delete_gcs(marker)
            self.backend.delete_gcs(f"{job.job_dir}/failed")
            self.backend.delete_gcs(f"{job.job_dir}/succeeded")
            if job.state.current_attempt >= job.spec.max_attempts:
                logger.warning("Retry requested for %s but max attempts reached", job_id)
                self._write_status(job_id)
                continue
            job.state.status = JobStatus.PENDING
            job.state.current_qr_name = None
            job.state.current_qr_state = None
            self._write_status(job_id)

    def poll_queued_resources(self, job_ids: set[str] | None = None) -> None:
        for qr_name, job_id in list(self.queued_resources.items()):
            if job_ids is not None and job_id not in job_ids:
                continue
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
            vm_status = self.backend.get_tpu_vm_status(
                qr_name, resource.project, resource.zone
            )
            vm_state = vm_status.state
            if vm_state in TERMINAL_TPU_VM_STATES:
                self._handle_preemption(job_id, qr_name, resource, f"TPU_VM_{vm_state}")
                return
            vm_health = vm_status.health
            if vm_health in RETRY_TPU_VM_HEALTH:
                self._handle_preemption(
                    job_id,
                    qr_name,
                    resource,
                    f"TPU_VM_HEALTH_{vm_health}",
                )
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
        heartbeat_result = self.backend.read_gcs_result(
            f"{job.job_dir}/attempts/attempt-{attempt}/heartbeat"
        )
        if not heartbeat_result.succeeded:
            logger.warning(
                "Skipping heartbeat timeout check for job %s attempt %s after GCS read failure",
                job_id,
                attempt,
            )
            return
        heartbeat_text = heartbeat_result.content
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
        attempt = job.state.current_attempt + 1
        retryable = attempt < job.spec.max_attempts
        job.state.attempts.append(
            AttemptRecord(
                attempt=attempt,
                qr_name=qr_name,
                started_at=job.state.provisioned_at or "",
                ended_at=utc_now(),
                error=error,
                failure_type=AttemptFailureType.INFRASTRUCTURE_PREEMPTION.value,
                retryable=retryable,
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
        self.backend.delete_gcs(f"{job.job_dir}/failed")
        self.backend.delete_gcs(f"{job.job_dir}/succeeded")
        self._cleanup_job_resources(qr_name, resource)
        self._write_status(job_id)
        action = "automatic retry" if retryable else "max attempts reached"
        logger.warning(
            "Job %s attempt %s had infrastructure failure %s; %s",
            job_id,
            attempt,
            error,
            action,
        )

    def _cleanup_job_resources(self, qr_name: str, resource: ResourceConfig) -> None:
        self.backend.delete_tpu_vm(qr_name, resource.project, resource.zone)
        if self.backend.delete_queued_resource(qr_name, resource.project, resource.zone):
            self.queued_resources.pop(qr_name, None)

    def schedule_pending_jobs(self, *, only_job_ids: set[str] | None = None) -> None:
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
            and (only_job_ids is None or job_id in only_job_ids)
        ]
        pending.sort()
        for _, _, job_id in pending:
            if self._create_retry_not_before.get(job_id, 0.0) > time.monotonic():
                continue
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
            try:
                created = self._create_queued_resource(job_id, resource)
            except Exception:
                # A raised exception here (e.g. the ENOSPC OSError from
                # write_startup_script during a disk-full incident) must be
                # treated the same as a normal create failure: it still needs
                # to register the create-failure backoff for this one job so
                # the scheduler does not hammer the API on retry, and it must
                # not abort scheduling of the *other* pending jobs in this
                # same pass. Log with job/resource context so operators can
                # still diagnose it in scheduler.log.
                logger.exception(
                    "Queued-resource create raised for job %s (resource=%s)",
                    job_id,
                    resource.name,
                )
                created = False
            if created:
                self._create_retry_not_before.pop(job_id, None)
                chips_by_quota[resource.quota_group] = (
                    chips_by_quota.get(resource.quota_group, 0) + resource.chips
                )
                chips_by_user[job.spec.submitted_by] = (
                    chips_by_user.get(job.spec.submitted_by, 0) + resource.chips
                )
            else:
                delay = max(0, self.config.scheduler.create_failure_backoff)
                self._create_retry_not_before[job_id] = time.monotonic() + delay
                logger.warning(
                    "Deferring queued-resource create for job %s by %s seconds",
                    job_id,
                    delay,
                )

    def _create_queued_resource(self, job_id: str, resource: ResourceConfig) -> bool:
        job = self.jobs[job_id]
        attempt = job.state.current_attempt + 1
        qr_name = self._generate_qr_name(job, attempt)
        # Clear only legacy root markers. New startup scripts write attempt-scoped markers.
        self.backend.delete_gcs(f"{job.job_dir}/failed")
        self.backend.delete_gcs(f"{job.job_dir}/succeeded")
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
        self.backend.write_gcs(
            f"{job.job_dir}/attempts/attempt-{attempt}/startup_version",
            "3\n",
        )
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

    def sync_interactive_ssh_key_requests(self) -> None:
        """Provision user-requested SSH keys onto all interactive TPUs.

        Users upload their public key to
        `{primary_bucket}/ssh-key-requests/<user>.pub` via
        `tpu interactive add-key`; only the scheduler identity has the
        tpu.nodes.update permission needed to write node metadata. The
        username embedded in the key entry must match the request filename so
        a request cannot provision a key under someone else's login. Keys are
        only appended, never removed. A request is deleted once the key is
        present on every configured interactive TPU; unreadable or failed
        nodes keep the request alive for the next sync pass.
        """
        if not self.config.interactive_tpus:
            return
        prefix = f"{self.config.primary_bucket}/{interactive.SSH_KEY_REQUEST_PREFIX}/"
        for url in self.backend.list_gcs(prefix):
            name = url.rstrip("/").rsplit("/", 1)[-1]
            if not name.endswith(".pub"):
                continue
            user = name[: -len(".pub")]
            content = self.backend.read_gcs(url)
            if content is None:
                continue
            try:
                entry = interactive.normalized_key_entry(user, content)
            except ValueError as exc:
                logger.warning("Rejecting SSH key request %s: %s", name, exc)
                self.backend.delete_gcs(url)
                continue
            identity = interactive.ssh_key_identity(entry)
            provisioned_everywhere = True
            for tpu in sorted(
                self.config.interactive_tpus.values(), key=lambda t: t.name
            ):
                raw = self.backend.get_tpu_vm_ssh_keys(tpu.name, tpu.project, tpu.zone)
                if raw is None:
                    logger.warning(
                        "Cannot read ssh-keys on %s; keeping key request for %s",
                        tpu.name,
                        user,
                    )
                    provisioned_everywhere = False
                    continue
                lines = [line.strip() for line in raw.splitlines() if line.strip()]
                if any(interactive.ssh_key_identity(line) == identity for line in lines):
                    continue
                if self.backend.set_tpu_vm_ssh_keys(
                    tpu.name, tpu.project, tpu.zone, "\n".join([*lines, entry])
                ):
                    logger.info("Provisioned SSH key for %s on %s", user, tpu.name)
                else:
                    provisioned_everywhere = False
            if provisioned_everywhere:
                self.backend.delete_gcs(url)
                logger.info("Completed SSH key request for %s", user)

    def _maybe_sync_interactive_ssh_keys(self) -> None:
        now = time.time()
        if now - self._last_interactive_key_sync < INTERACTIVE_KEY_SYNC_INTERVAL:
            return
        self._last_interactive_key_sync = now
        self.sync_interactive_ssh_key_requests()

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

    def _resolve_job_id(self, job_ref: str) -> str:
        matches = [
            job_id
            for job_id, job in self.jobs.items()
            if job_ref in {job_id, job.spec.job_id, job.spec.display_name}
            or job.spec.job_id.endswith(job_ref)
        ]
        if not matches:
            raise ValueError(f"Job not found: {job_ref}")
        if len(matches) > 1:
            raise ValueError(f"Job reference is ambiguous: {job_ref} ({', '.join(matches)})")
        return matches[0]

    def _run_step(self, step_name: str, func, *args, **kwargs):
        """Run one run_once step, isolating its exceptions from the rest.

        A step failing (for example the ENOSPC OSError from
        write_startup_script during a disk-full incident, raised out of
        schedule_pending_jobs) must not prevent the other steps in the same
        run_once() pass from still running, and must not prevent
        _maybe_write_scheduler_state() from still executing afterwards. The
        exception is always logged with full context via logger.exception so
        operators can still see it in scheduler.log; it is never swallowed
        silently.
        """
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.exception(
                "Scheduler step %r failed; continuing with remaining steps",
                step_name,
            )
            return None

    def run_once(
        self,
        *,
        focus_job_ref: str | None = None,
        focus_user: str | None = None,
    ) -> None:
        if focus_job_ref and focus_user:
            raise ValueError("Cannot combine focus_job_ref and focus_user")
        try:
            self._run_once_steps(focus_job_ref=focus_job_ref, focus_user=focus_user)
        finally:
            # Always attempt to flush the aggregated scheduler_state.json
            # cache that `tpu list`/`tpu status`/`tpu admin jobs` read from,
            # even if one or more steps above raised. This is the specific
            # behavior that was missing: previously a single step exception
            # aborted the whole run_once() iteration before this call ever
            # ran, freezing that cache -- even though the underlying per-job
            # status.json writes from earlier steps in the same broken
            # iteration may have already succeeded. Route through _run_step
            # too so a failure while writing scheduler_state.json itself
            # (e.g. the same disk-full condition) is logged, not raised, and
            # does not mask whatever exception the try block already saw.
            self._run_step("write_scheduler_state", self._maybe_write_scheduler_state)

    def _run_once_steps(
        self,
        *,
        focus_job_ref: str | None,
        focus_user: str | None,
    ) -> None:
        self._run_step("scan_jobs", self.scan_jobs)

        focus_job_id = (
            self._resolve_job_id(focus_job_ref) if focus_job_ref else None
        )
        if focus_job_id:
            job_ids = {focus_job_id}
            cancel_job_ids = job_ids
        elif focus_user:
            job_ids = {
                job_id
                for job_id, job in self.jobs.items()
                if job.spec.submitted_by == focus_user
            }
            quota_job_ids = {
                job_id
                for job_id, job in self.jobs.items()
                if job_id not in job_ids and job.state.status not in TERMINAL_STATUSES
            }
            self._run_step("refresh_job_states", self._refresh_job_states, quota_job_ids)
            # Cancellation only releases resources, so honor every user's
            # sentinel even when lifecycle work is otherwise focus-restricted.
            # Otherwise non-focus users can never delete pending/active jobs.
            cancel_job_ids = job_ids | quota_job_ids
        else:
            job_ids = None
            cancel_job_ids = None
        self._run_step("check_canceled_jobs", self.check_canceled_jobs, cancel_job_ids)
        if focus_job_id is None:
            # Like cancellations, key provisioning is a safe user request the
            # scheduler identity must serve even in focus-user mode.
            self._run_step("sync_interactive_ssh_keys", self._maybe_sync_interactive_ssh_keys)
        self._run_step("poll_queued_resources", self.poll_queued_resources, job_ids)
        self._run_step("check_completed_jobs", self.check_completed_jobs, job_ids)
        self._run_step("check_retry_requests", self.check_retry_requests, job_ids)
        if job_ids is not None:
            self._run_step(
                "schedule_pending_jobs", self.schedule_pending_jobs, only_job_ids=job_ids
            )
        else:
            self._run_step("schedule_pending_jobs", self.schedule_pending_jobs)
            self._run_step("reap_terminal_jobs", self.reap_terminal_jobs)
            self._run_step("reconcile_orphans", self._maybe_reconcile_orphans)

    def run_forever(self) -> None:
        while True:
            try:
                self.run_once()
            except Exception:
                logger.exception("Scheduler iteration failed")
            time.sleep(self.config.scheduler.scan_interval)
