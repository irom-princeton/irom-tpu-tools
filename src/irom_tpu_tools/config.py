from __future__ import annotations

from dataclasses import dataclass
import os
import re


# Region-specific service accounts and their co-located buckets.
# Disabled by default; set TPU_REGION_SA_ENABLED=1 to opt in.
# When enabled, TPU VMs are created with the SA matching their zone's region,
# avoiding cross-region GCS transfer costs.
REGION_SERVICE_ACCOUNTS: dict[str, str] = {
    "us-central1": "tpu-sa-us-central1@mae-irom-lab-guided-data.iam.gserviceaccount.com",
    "us-central2": "tpu-sa-us-central2@mae-irom-lab-guided-data.iam.gserviceaccount.com",
    "us-east1": "tpu-sa-us-east1@mae-irom-lab-guided-data.iam.gserviceaccount.com",
}

REGION_STORAGE_BUCKETS: dict[str, str] = {
    "us-central1": "gs://v5_central1_a",
    "us-central2": "gs://pi0-cot",
    "us-east1": "gs://v6_east1d",
}


def zone_to_region(zone: str) -> str:
    """Strip the trailing zone letter from a GCP zone, e.g. us-east1-d → us-east1."""
    return re.sub(r"-[a-z]$", "", zone)


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
    def service_account_for_zone(self, zone: str) -> str:
        region = zone_to_region(zone)
        sa = REGION_SERVICE_ACCOUNTS.get(region)
        if sa is None:
            raise RuntimeError(
                f"No region service account configured for zone '{zone}' (region '{region}'). "
                f"Known regions: {list(REGION_SERVICE_ACCOUNTS)}"
            )
        return sa

    @property
    def zones(self) -> dict[str, str]:
        """Return {version: zone} mapping for all configured TPU versions."""
        return {
            "v4": self.tpu_zone_v4,
            "v5": self.tpu_zone_v5,
            "v6": self.tpu_zone_v6,
        }

    @staticmethod
    def from_env(require_tpu_name: bool = True) -> TPUEnvConfig:
        def must_get(name: str) -> str:
            val = os.environ.get(name, "").strip()
            if not val:
                raise RuntimeError(f"Missing required environment variable: {name}")
            return val

        def maybe_get(name: str) -> str:
            return os.environ.get(name, "").strip()

        tpu_name = must_get("TPU_NAME") if require_tpu_name else maybe_get("TPU_NAME")

        return TPUEnvConfig(
            tpu_name=tpu_name,
            tpu_project=must_get("TPU_PROJECT"),
            tpu_zone_v4=must_get("TPU_ZONE_v4"),
            tpu_zone_v5=must_get("TPU_ZONE_v5"),
            tpu_zone_v6=must_get("TPU_ZONE_v6"),
            tpu_bucket_v4=must_get("TPU_BUCKET_v4"),
            tpu_bucket_v5=must_get("TPU_BUCKET_v5"),
            tpu_bucket_v6=must_get("TPU_BUCKET_v6"),
            tpu_service_account=must_get("TPU_SERVICE_ACCOUNT"),
            wandb_api_key=must_get("WANDB_API_KEY"),
            # Repo-related vars are optional fallback defaults; a job's --repo
            # flag takes precedence and bare-TPU mode needs none of these.
            gh_repo_name=maybe_get("GH_REPO_NAME"),
            gh_token=maybe_get("GH_TOKEN"),
            gh_owner=maybe_get("GH_OWNER"),
        )
