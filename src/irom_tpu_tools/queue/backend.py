from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
import json
import logging
import shutil
import subprocess
from typing import Any

from .types import QRState

logger = logging.getLogger(__name__)


class Backend(ABC):
    @abstractmethod
    def create_queued_resource(
        self,
        *,
        name: str,
        node_id: str,
        project: str,
        zone: str,
        accelerator_type: str,
        runtime_version: str,
        spot: bool,
        startup_script_path: str,
        service_account: str | None,
        label_workaround: bool = False,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_queued_resource_state(
        self, name: str, project: str, zone: str
    ) -> QRState | None:
        raise NotImplementedError

    @abstractmethod
    def get_tpu_vm_state(self, name: str, project: str, zone: str) -> str | None:
        raise NotImplementedError

    def get_tpu_vm_status(self, name: str, project: str, zone: str) -> TpuVmStatus:
        return TpuVmStatus(state=self.get_tpu_vm_state(name, project, zone))

    @abstractmethod
    def delete_queued_resource(self, name: str, project: str, zone: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def delete_tpu_vm(self, name: str, project: str, zone: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list_queued_resources(
        self, project: str, zone: str, name_prefix: str = ""
    ) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def list_tpu_vms(
        self, project: str, zone: str, name_prefix: str = ""
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def ssh_tpu_vm(
        self, name: str, project: str, zone: str, worker: int, command: str
    ) -> subprocess.CompletedProcess[str]:
        raise NotImplementedError

    @abstractmethod
    def read_gcs(self, url: str) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def write_gcs(self, url: str, content: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def exists_gcs(self, url: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list_gcs(self, prefix: str) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def delete_gcs(self, url: str, recursive: bool = False) -> bool:
        raise NotImplementedError

    @abstractmethod
    def upload_file(self, local_path: str, gcs_url: str) -> bool:
        raise NotImplementedError


class GCPBackend(Backend):
    def __init__(self, *, dry_run_commands: bool = False):
        self.dry_run_commands = dry_run_commands

    def _run(
        self,
        cmd: list[str],
        *,
        check: bool = True,
        input_data: str | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        logger.debug("Running: %s", " ".join(cmd))
        if self.dry_run_commands:
            logger.info("[dry-run] %s", " ".join(cmd))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        try:
            return subprocess.run(
                cmd,
                check=check,
                capture_output=True,
                text=True,
                input=input_data,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            logger.warning("Command timed out after %.1fs: %s", timeout or 0, " ".join(cmd))
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            return subprocess.CompletedProcess(cmd, 124, stdout, stderr)

    def create_queued_resource(
        self,
        *,
        name: str,
        node_id: str,
        project: str,
        zone: str,
        accelerator_type: str,
        runtime_version: str,
        spot: bool,
        startup_script_path: str,
        service_account: str | None,
        label_workaround: bool = False,
    ) -> bool:
        cmd = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "queued-resources",
            "create",
            name,
            "--project",
            project,
            "--zone",
            zone,
            "--node-id",
            node_id,
            "--accelerator-type",
            accelerator_type,
            "--runtime-version",
            runtime_version,
            f"--metadata-from-file=startup-script={startup_script_path}",
            "--async",
        ]
        if spot:
            cmd.extend(["--provisioning-model", "SPOT", "--spot"])
        if service_account:
            cmd.extend(["--service-account", service_account])
        if label_workaround:
            cmd.append("--labels=env=prod")
        try:
            self._run(cmd, timeout=120)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("Failed to create queued resource %s: %s", name, exc.stderr)
            return False

    def get_queued_resource_state(
        self, name: str, project: str, zone: str
    ) -> QRState | None:
        cmd = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "queued-resources",
            "describe",
            name,
            "--project",
            project,
            "--zone",
            zone,
            "--format",
            "json",
        ]
        result = self._run(cmd, check=False, timeout=30)
        if result.returncode != 0 or not result.stdout.strip():
            return None
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None
        state = data.get("state")
        if isinstance(state, dict):
            state = state.get("state") or state.get("stateName") or state.get("name")
        if not state:
            return None
        try:
            return QRState(str(state))
        except ValueError:
            logger.warning("Unknown queued resource state for %s: %s", name, state)
            return None

    def get_tpu_vm_state(self, name: str, project: str, zone: str) -> str | None:
        return self.get_tpu_vm_status(name, project, zone).state

    def get_tpu_vm_status(self, name: str, project: str, zone: str) -> TpuVmStatus:
        cmd = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            name,
            "--project",
            project,
            "--zone",
            zone,
            "--format",
            "json(state,health,healthDescription)",
        ]
        result = self._run(cmd, check=False, timeout=30)
        if result.returncode != 0 or not result.stdout.strip():
            return TpuVmStatus()
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return TpuVmStatus()
        state = data.get("state")
        health = data.get("health")
        health_description = data.get("healthDescription")
        return TpuVmStatus(
            state=str(state) if state else None,
            health=str(health) if health else None,
            health_description=str(health_description) if health_description else None,
        )

    def delete_queued_resource(self, name: str, project: str, zone: str) -> bool:
        cmd = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "queued-resources",
            "delete",
            name,
            "--project",
            project,
            "--zone",
            zone,
            "--quiet",
        ]
        result = self._run(cmd, check=False, timeout=120)
        if result.returncode == 0:
            return True
        if "not found" in (result.stderr or result.stdout).lower():
            return True
        logger.error("Failed to delete queued resource %s: %s", name, result.stderr)
        return False

    def delete_tpu_vm(self, name: str, project: str, zone: str) -> bool:
        cmd = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            name,
            "--project",
            project,
            "--zone",
            zone,
            "--quiet",
        ]
        result = self._run(cmd, check=False, timeout=120)
        if result.returncode == 0:
            return True
        if "not found" in (result.stderr or result.stdout).lower():
            return True
        logger.error("Failed to delete TPU VM %s: %s", name, result.stderr)
        return False

    def list_queued_resources(
        self, project: str, zone: str, name_prefix: str = ""
    ) -> list[str]:
        cmd = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "queued-resources",
            "list",
            "--project",
            project,
            "--zone",
            zone,
            "--format",
            "value(name)",
        ]
        if name_prefix:
            cmd.append(f"--filter=name:{name_prefix}")
        result = self._run(cmd, check=False, timeout=30)
        if result.returncode != 0:
            return []
        return [line.rsplit("/", 1)[-1] for line in result.stdout.splitlines() if line]

    def list_tpu_vms(
        self, project: str, zone: str, name_prefix: str = ""
    ) -> list[dict[str, Any]]:
        cmd = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            "--project",
            project,
            "--zone",
            zone,
            "--format",
            "json(name,state,health,acceleratorType,createTime)",
        ]
        if name_prefix:
            cmd.append(f"--filter=name:{name_prefix}")
        result = self._run(cmd, check=False, timeout=30)
        if result.returncode != 0 or not result.stdout.strip():
            return []
        try:
            rows = json.loads(result.stdout)
        except json.JSONDecodeError:
            return []
        for row in rows:
            if "name" in row:
                row["name"] = str(row["name"]).rsplit("/", 1)[-1]
        return rows

    def ssh_tpu_vm(
        self, name: str, project: str, zone: str, worker: int, command: str
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            name,
            "--project",
            project,
            "--zone",
            zone,
            "--worker",
            str(worker),
            "--command",
            command,
        ]
        return self._run(cmd, check=False, timeout=60)

    def read_gcs(self, url: str) -> str | None:
        result = self._run(["gsutil", "cat", url], check=False, timeout=30)
        if result.returncode != 0:
            return None
        return result.stdout

    def write_gcs(self, url: str, content: str) -> bool:
        result = self._run(["gsutil", "cp", "-", url], check=False, input_data=content, timeout=30)
        return result.returncode == 0

    def exists_gcs(self, url: str) -> bool:
        return self._run(["gsutil", "stat", url], check=False, timeout=20).returncode == 0

    def list_gcs(self, prefix: str) -> list[str]:
        result = self._run(["gsutil", "ls", prefix], check=False, timeout=30)
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def delete_gcs(self, url: str, recursive: bool = False) -> bool:
        cmd = ["gsutil", "rm"]
        if recursive:
            cmd.append("-r")
        cmd.append(url)
        return self._run(cmd, check=False, timeout=60).returncode == 0

    def upload_file(self, local_path: str, gcs_url: str) -> bool:
        return self._run(["gsutil", "cp", local_path, gcs_url], check=False, timeout=120).returncode == 0


@dataclass
class TpuVmStatus:
    state: str | None = None
    health: str | None = None
    health_description: str | None = None


@dataclass
class SimulatedQR:
    name: str
    node_id: str
    project: str
    zone: str
    accelerator_type: str
    runtime_version: str
    spot: bool
    state: QRState = QRState.WAITING_FOR_RESOURCES
    health: str = "HEALTHY"
    health_description: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    active_at: datetime | None = None


class DryRunBackend(Backend):
    def __init__(self, base_dir: str, *, provision_delay_seconds: float = 0.0):
        self.base_dir = Path(base_dir)
        self.gcs_dir = self.base_dir / "gcs"
        self.gcs_dir.mkdir(parents=True, exist_ok=True)
        self.provision_delay_seconds = provision_delay_seconds
        self.qr_state_path = self.base_dir / "qrs.json"
        self.queued_resources: dict[str, SimulatedQR] = {}
        self._now: datetime | None = None
        self._load_qrs()

    def now(self) -> datetime:
        return self._now or datetime.now(UTC)

    def set_time(self, value: datetime) -> None:
        self._now = value

    def advance_time(self, seconds: float) -> None:
        self._now = self.now() + timedelta(seconds=seconds)
        self.tick()

    def _load_qrs(self) -> None:
        if not self.qr_state_path.exists():
            return
        try:
            data = json.loads(self.qr_state_path.read_text())
        except json.JSONDecodeError:
            return
        for name, raw in data.items():
            self.queued_resources[name] = SimulatedQR(
                name=name,
                node_id=raw["node_id"],
                project=raw["project"],
                zone=raw["zone"],
                accelerator_type=raw["accelerator_type"],
                runtime_version=raw["runtime_version"],
                spot=bool(raw["spot"]),
                state=QRState(raw["state"]),
                health=str(raw.get("health") or "HEALTHY"),
                health_description=raw.get("health_description"),
                created_at=datetime.fromisoformat(raw["created_at"]),
                active_at=(
                    datetime.fromisoformat(raw["active_at"])
                    if raw.get("active_at")
                    else None
                ),
            )

    def _save_qrs(self) -> None:
        data = {}
        for name, qr in self.queued_resources.items():
            data[name] = {
                "node_id": qr.node_id,
                "project": qr.project,
                "zone": qr.zone,
                "accelerator_type": qr.accelerator_type,
                "runtime_version": qr.runtime_version,
                "spot": qr.spot,
                "state": qr.state.value,
                "health": qr.health,
                "health_description": qr.health_description,
                "created_at": qr.created_at.isoformat(),
                "active_at": qr.active_at.isoformat() if qr.active_at else None,
            }
        self.qr_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.qr_state_path.write_text(json.dumps(data, indent=2))

    def tick(self) -> None:
        for qr in list(self.queued_resources.values()):
            if qr.state == QRState.WAITING_FOR_RESOURCES:
                elapsed = (self.now() - qr.created_at).total_seconds()
                if elapsed >= self.provision_delay_seconds:
                    qr.state = QRState.ACTIVE
                    qr.active_at = self.now()
                    self._save_qrs()
            elif qr.state == QRState.SUSPENDING:
                qr.state = QRState.SUSPENDED
                self._save_qrs()

    def _gcs_path(self, url: str) -> Path:
        if not url.startswith("gs://"):
            raise ValueError(f"DryRunBackend only supports gs:// URLs: {url}")
        return self.gcs_dir / url[5:]

    def create_queued_resource(
        self,
        *,
        name: str,
        node_id: str,
        project: str,
        zone: str,
        accelerator_type: str,
        runtime_version: str,
        spot: bool,
        startup_script_path: str,
        service_account: str | None,
        label_workaround: bool = False,
    ) -> bool:
        if name in self.queued_resources:
            return False
        self.queued_resources[name] = SimulatedQR(
            name=name,
            node_id=node_id,
            project=project,
            zone=zone,
            accelerator_type=accelerator_type,
            runtime_version=runtime_version,
            spot=spot,
            created_at=self.now(),
        )
        self._save_qrs()
        return True

    def get_queued_resource_state(
        self, name: str, project: str, zone: str
    ) -> QRState | None:
        qr = self.queued_resources.get(name)
        if not qr or qr.project != project or qr.zone != zone:
            return None
        return qr.state

    def get_tpu_vm_state(self, name: str, project: str, zone: str) -> str | None:
        return self.get_tpu_vm_status(name, project, zone).state

    def get_tpu_vm_status(self, name: str, project: str, zone: str) -> TpuVmStatus:
        qr = self.queued_resources.get(name)
        if not qr or qr.project != project or qr.zone != zone:
            return TpuVmStatus()
        if qr.state == QRState.ACTIVE:
            return TpuVmStatus(
                state="READY",
                health=qr.health,
                health_description=qr.health_description,
            )
        if qr.state in {QRState.SUSPENDING, QRState.SUSPENDED}:
            return TpuVmStatus(
                state="PREEMPTED",
                health=qr.health,
                health_description=qr.health_description,
            )
        return TpuVmStatus()

    def force_active(self, name: str) -> None:
        self.queued_resources[name].state = QRState.ACTIVE
        self.queued_resources[name].active_at = self.now()
        self._save_qrs()

    def force_preempt(self, name: str) -> None:
        self.queued_resources[name].state = QRState.SUSPENDING
        self._save_qrs()

    def force_unhealthy_maintenance(self, name: str) -> None:
        qr = self.queued_resources[name]
        qr.state = QRState.ACTIVE
        qr.active_at = qr.active_at or self.now()
        qr.health = "UNHEALTHY_MAINTENANCE"
        qr.health_description = "The TPU had a maintenance event"
        self._save_qrs()

    def delete_queued_resource(self, name: str, project: str, zone: str) -> bool:
        self.queued_resources.pop(name, None)
        self._save_qrs()
        return True

    def delete_tpu_vm(self, name: str, project: str, zone: str) -> bool:
        return True

    def list_queued_resources(
        self, project: str, zone: str, name_prefix: str = ""
    ) -> list[str]:
        return sorted(
            name
            for name, qr in self.queued_resources.items()
            if qr.project == project
            and qr.zone == zone
            and (not name_prefix or name.startswith(name_prefix))
        )

    def list_tpu_vms(
        self, project: str, zone: str, name_prefix: str = ""
    ) -> list[dict[str, Any]]:
        rows = []
        for name, qr in self.queued_resources.items():
            if qr.project != project or qr.zone != zone:
                continue
            if name_prefix and not name.startswith(name_prefix):
                continue
            if qr.state == QRState.ACTIVE:
                rows.append(
                    {
                        "name": name,
                        "state": "READY",
                        "health": qr.health,
                        "acceleratorType": qr.accelerator_type,
                        "createTime": qr.created_at.isoformat(),
                    }
                )
        return rows

    def ssh_tpu_vm(
        self, name: str, project: str, zone: str, worker: int, command: str
    ) -> subprocess.CompletedProcess[str]:
        if not any(
            qr.name == name and qr.project == project and qr.zone == zone
            for qr in self.queued_resources.values()
        ):
            return subprocess.CompletedProcess(
                ["dry-run-ssh", name], 1, "", f"TPU VM not found: {name}"
            )
        return subprocess.CompletedProcess(
            ["dry-run-ssh", name],
            0,
            "tmux:\n(none)\nprocesses:\n(none)\nlogs:\n(none)\n",
            "",
        )

    def read_gcs(self, url: str) -> str | None:
        path = self._gcs_path(url)
        return path.read_text() if path.exists() and path.is_file() else None

    def write_gcs(self, url: str, content: str) -> bool:
        path = self._gcs_path(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return True

    def exists_gcs(self, url: str) -> bool:
        return self._gcs_path(url).exists()

    def list_gcs(self, prefix: str) -> list[str]:
        path = self._gcs_path(prefix)
        if path.is_dir():
            return [
                f"gs://{p.relative_to(self.gcs_dir)}" + ("/" if p.is_dir() else "")
                for p in sorted(path.iterdir())
            ]
        parent = path.parent
        if not parent.exists():
            return []
        return [
            f"gs://{p.relative_to(self.gcs_dir)}" + ("/" if p.is_dir() else "")
            for p in sorted(parent.iterdir())
            if p.name.startswith(path.name)
        ]

    def delete_gcs(self, url: str, recursive: bool = False) -> bool:
        path = self._gcs_path(url)
        if path.is_dir() and recursive:
            shutil.rmtree(path)
            return True
        if path.is_file():
            path.unlink()
            return True
        return False

    def upload_file(self, local_path: str, gcs_url: str) -> bool:
        dst = self._gcs_path(gcs_url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dst)
        return True
