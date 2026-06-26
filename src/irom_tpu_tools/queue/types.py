from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class QRState(Enum):
    WAITING_FOR_RESOURCES = "WAITING_FOR_RESOURCES"
    PROVISIONING = "PROVISIONING"
    ACTIVE = "ACTIVE"
    SUSPENDING = "SUSPENDING"
    SUSPENDED = "SUSPENDED"
    FAILED = "FAILED"


class JobStatus(Enum):
    PENDING = "PENDING"
    PROVISIONING = "PROVISIONING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


TERMINAL_STATUSES = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELED}


@dataclass(frozen=True)
class ResourceConfig:
    name: str
    version: str
    accelerator_type: str
    runtime_version: str
    zone: str
    project: str
    chips: int
    workers: int
    spot: bool
    enabled: bool
    quota_group: str
    service_account: str | None = None


@dataclass(frozen=True)
class QuotaGroupConfig:
    name: str
    total_chips: int


@dataclass(frozen=True)
class UserLimitConfig:
    default_max_chips: int | None = None
    users: dict[str, int] = field(default_factory=dict)

    def max_chips_for(self, user: str) -> int | None:
        return self.users.get(user, self.default_max_chips)


@dataclass(frozen=True)
class InteractiveTPUConfig:
    name: str
    version: str
    zone: str
    project: str
    workers: int = 1
    description: str = ""
    aliases: tuple[str, ...] = ()


@dataclass
class SchedulerConfig:
    scan_interval: int = 30
    active_no_claim_timeout: int = 1800
    heartbeat_timeout: int = 600
    status_write_interval: int = 60
    job_retention_days: int = 30
    qr_prefix: str = "iq"
    qr_label_workaround: bool = False


@dataclass(frozen=True)
class JobResources:
    resource_name: str
    accelerator_type: str
    zone: str
    project: str
    chips: int
    workers: int
    runtime_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_name": self.resource_name,
            "accelerator_type": self.accelerator_type,
            "zone": self.zone,
            "project": self.project,
            "chips": self.chips,
            "workers": self.workers,
            "runtime_version": self.runtime_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobResources":
        return cls(
            resource_name=str(data["resource_name"]),
            accelerator_type=str(data["accelerator_type"]),
            zone=str(data["zone"]),
            project=str(data["project"]),
            chips=int(data["chips"]),
            workers=int(data["workers"]),
            runtime_version=str(data["runtime_version"]),
        )


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    display_name: str
    code_tar_url: str
    code_checksum: str
    command: str
    setup_cmd: str
    resources: JobResources
    max_attempts: int
    submit_time: str
    submitted_by: str
    priority: int = 1
    tags: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    secret_refs: dict[str, str] = field(default_factory=dict)
    run_on_all_workers: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "display_name": self.display_name,
            "code_tar_url": self.code_tar_url,
            "code_checksum": self.code_checksum,
            "command": self.command,
            "setup_cmd": self.setup_cmd,
            "resources": self.resources.to_dict(),
            "max_attempts": self.max_attempts,
            "submit_time": self.submit_time,
            "submitted_by": self.submitted_by,
            "priority": self.priority,
            "tags": self.tags,
            "env_vars": self.env_vars,
            "secret_refs": self.secret_refs,
            "run_on_all_workers": self.run_on_all_workers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobSpec":
        return cls(
            job_id=str(data["job_id"]),
            display_name=str(data.get("display_name") or data["job_id"]),
            code_tar_url=str(data["code_tar_url"]),
            code_checksum=str(data["code_checksum"]),
            command=str(data.get("command") or ""),
            setup_cmd=str(data.get("setup_cmd") or ""),
            resources=JobResources.from_dict(data["resources"]),
            max_attempts=int(data.get("max_attempts", 20)),
            submit_time=str(data["submit_time"]),
            submitted_by=str(data.get("submitted_by") or "unknown"),
            priority=int(data.get("priority", 1)),
            tags=[str(x) for x in data.get("tags", [])],
            env_vars={str(k): str(v) for k, v in data.get("env_vars", {}).items()},
            secret_refs={
                str(k): str(v) for k, v in data.get("secret_refs", {}).items()
            },
            run_on_all_workers=bool(data.get("run_on_all_workers", True)),
        )


@dataclass(frozen=True)
class AttemptRecord:
    attempt: int
    qr_name: str
    started_at: str
    ended_at: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "qr_name": self.qr_name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttemptRecord":
        return cls(
            attempt=int(data["attempt"]),
            qr_name=str(data.get("qr_name") or ""),
            started_at=str(data.get("started_at") or ""),
            ended_at=str(data.get("ended_at") or ""),
            error=(None if data.get("error") is None else str(data.get("error"))),
        )


@dataclass
class JobState:
    status: JobStatus
    current_attempt: int
    attempts: list[AttemptRecord]
    created_at: str
    last_updated: str
    current_qr_name: str | None = None
    current_qr_state: str | None = None
    provisioned_at: str | None = None

    @classmethod
    def new(cls) -> "JobState":
        now = utc_now()
        return cls(
            status=JobStatus.PENDING,
            current_attempt=0,
            attempts=[],
            created_at=now,
            last_updated=now,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "current_attempt": self.current_attempt,
            "attempts": [a.to_dict() for a in self.attempts],
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "current_qr_name": self.current_qr_name,
            "current_qr_state": self.current_qr_state,
            "provisioned_at": self.provisioned_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobState":
        return cls(
            status=JobStatus(str(data["status"])),
            current_attempt=int(data.get("current_attempt", 0)),
            attempts=[
                AttemptRecord.from_dict(a) for a in data.get("attempts", [])
            ],
            created_at=str(data["created_at"]),
            last_updated=str(data.get("last_updated") or data["created_at"]),
            current_qr_name=data.get("current_qr_name"),
            current_qr_state=data.get("current_qr_state"),
            provisioned_at=data.get("provisioned_at"),
        )


@dataclass
class Job:
    spec: JobSpec
    state: JobState
    bucket: str

    @property
    def job_dir(self) -> str:
        return f"{self.bucket.rstrip('/')}/jobs/{self.spec.job_id}"
