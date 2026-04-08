from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class TPUEnvConfig:
    """Configuration derived from environment variables for TPU zones/buckets."""

    tpu_name: str
    tpu_project: str
    tpu_zone_v4: str
    tpu_zone_v5: str
    tpu_zone_v6: str
    tpu_bucket_v4: str
    tpu_bucket_v5: str
    tpu_bucket_v6: str
    tpu_service_account: str
    gh_repo_name: str
    wandb_api_key: str
    gh_token: str
    gh_owner: str

    @staticmethod
    def from_env() -> TPUEnvConfig:
        def must_get(name: str) -> str:
            val = os.environ.get(name, "").strip()
            if not val:
                raise RuntimeError(f"Missing required environment variable: {name}")
            return val

        return TPUEnvConfig(
            tpu_name=must_get("TPU_NAME"),
            tpu_project=must_get("TPU_PROJECT"),
            tpu_zone_v4=must_get("TPU_ZONE_v4"),
            tpu_zone_v5=must_get("TPU_ZONE_v5"),
            tpu_zone_v6=must_get("TPU_ZONE_v6"),
            tpu_bucket_v4=must_get("TPU_BUCKET_v4"),
            tpu_bucket_v5=must_get("TPU_BUCKET_v5"),
            tpu_bucket_v6=must_get("TPU_BUCKET_v6"),
            tpu_service_account=must_get("TPU_SERVICE_ACCOUNT"),
            gh_repo_name=must_get("GH_REPO_NAME"),
            wandb_api_key=must_get("WANDB_API_KEY"),
            gh_token=must_get("GH_TOKEN"),
            gh_owner=must_get("GH_OWNER"),
        )
