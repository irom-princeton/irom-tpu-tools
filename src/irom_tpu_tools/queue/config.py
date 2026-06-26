from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml

from .types import (
    QuotaGroupConfig,
    ResourceConfig,
    SchedulerConfig,
    UserLimitConfig,
)


@dataclass(frozen=True)
class QueueConfig:
    resources: dict[str, ResourceConfig]
    quota_groups: dict[str, QuotaGroupConfig]
    scheduler: SchedulerConfig
    buckets: dict[str, str]
    primary_bucket_region: str
    secrets: dict[str, str]
    user_limits: UserLimitConfig

    @property
    def primary_bucket(self) -> str:
        return self.buckets[self.primary_bucket_region]


def _default_config_path() -> Path:
    return Path(__file__).with_name("resources.yaml")


def load_config(path: str | Path | None = None) -> QueueConfig:
    if path is None:
        path = os.environ.get("TPU_QUEUE_CONFIG") or _default_config_path()
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Queue config not found: {config_path}")
    raw_text = os.path.expandvars(config_path.read_text())
    raw = yaml.safe_load(raw_text) or {}
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> QueueConfig:
    quota_groups = {
        name: QuotaGroupConfig(name=name, total_chips=int(cfg["total_chips"]))
        for name, cfg in raw.get("quota_groups", {}).items()
    }

    resources: dict[str, ResourceConfig] = {}
    for name, cfg in raw.get("resources", {}).items():
        quota_group = str(cfg["quota_group"])
        if quota_group not in quota_groups:
            raise ValueError(f"Resource {name} references unknown quota group {quota_group}")
        resources[name] = ResourceConfig(
            name=name,
            version=str(cfg["version"]),
            accelerator_type=str(cfg["accelerator_type"]),
            runtime_version=str(cfg["runtime_version"]),
            zone=str(cfg["zone"]),
            project=str(cfg["project"]),
            chips=int(cfg["chips"]),
            workers=int(cfg["workers"]),
            spot=bool(cfg.get("spot", True)),
            enabled=bool(cfg.get("enabled", True)),
            quota_group=quota_group,
            service_account=cfg.get("service_account"),
        )

    if not resources:
        raise ValueError("Queue config must define at least one resource")

    buckets = {str(k): str(v).rstrip("/") for k, v in raw.get("buckets", {}).items()}
    if not buckets:
        raise ValueError("Queue config must define buckets")
    primary_region = str(raw.get("primary_bucket_region") or next(iter(buckets)))
    if primary_region not in buckets:
        raise ValueError(f"primary_bucket_region {primary_region!r} is not in buckets")

    sched_raw = raw.get("scheduler", {})
    scheduler = SchedulerConfig(
        scan_interval=int(sched_raw.get("scan_interval_seconds", 30)),
        active_no_claim_timeout=int(
            sched_raw.get("active_no_claim_timeout_seconds", 1800)
        ),
        heartbeat_timeout=int(sched_raw.get("heartbeat_timeout_seconds", 600)),
        status_write_interval=int(sched_raw.get("status_write_interval_seconds", 60)),
        job_retention_days=int(sched_raw.get("job_retention_days", 30)),
        qr_prefix=str(sched_raw.get("qr_prefix", "iq")),
        qr_label_workaround=bool(sched_raw.get("qr_label_workaround", False)),
    )

    limits_raw = raw.get("user_limits", {})
    user_limits = UserLimitConfig(
        default_max_chips=limits_raw.get("default_max_chips"),
        users={str(k): int(v) for k, v in limits_raw.get("users", {}).items()},
    )

    return QueueConfig(
        resources=resources,
        quota_groups=quota_groups,
        scheduler=scheduler,
        buckets=buckets,
        primary_bucket_region=primary_region,
        secrets={str(k): str(v) for k, v in raw.get("secrets", {}).items()},
        user_limits=user_limits,
    )


def zone_to_region(zone: str) -> str:
    parts = zone.split("-")
    if len(parts) < 3:
        return zone
    return "-".join(parts[:-1])


def bucket_for_resource(config: QueueConfig, resource: ResourceConfig) -> str:
    region = zone_to_region(resource.zone)
    try:
        return config.buckets[region]
    except KeyError as exc:
        raise ValueError(
            f"No queue bucket configured for region {region} (resource {resource.name})"
        ) from exc


def resource_for_request(
    config: QueueConfig,
    *,
    version: str | None = None,
    chips: int | None = None,
    resource_name: str | None = None,
) -> ResourceConfig:
    if resource_name:
        try:
            return config.resources[resource_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown resource {resource_name!r}; available: {', '.join(config.resources)}"
            ) from exc
    if not version or chips is None:
        raise ValueError("Either --resource or both version and --tpu-num are required")
    inferred = f"{version}-{chips}"
    try:
        return config.resources[inferred]
    except KeyError as exc:
        matches = [
            r.name
            for r in config.resources.values()
            if r.version == version and r.chips == chips
        ]
        if len(matches) == 1:
            return config.resources[matches[0]]
        raise ValueError(
            f"No resource configured for {version} with {chips} chips; "
            f"available: {', '.join(config.resources)}"
        ) from exc
