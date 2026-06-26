from __future__ import annotations

import shlex
from pathlib import Path

from .types import JobSpec


def _export_lines(values: dict[str, str]) -> str:
    if not values:
        return "# no plain environment variables"
    return "\n".join(f"export {k}={shlex.quote(v)}" for k, v in sorted(values.items()))


def _secret_lines(secrets: dict[str, str], project: str) -> str:
    if not secrets:
        return "# no secret manager variables"
    lines = []
    for env_name, secret_name in sorted(secrets.items()):
        lines.append(
            "export "
            + env_name
            + "=$(gcloud secrets versions access latest "
            + f"--secret={shlex.quote(secret_name)} --project={shlex.quote(project)})"
        )
    return "\n".join(lines)


def build_startup_script(
    *,
    job_id: str,
    spec: JobSpec,
    qr_name: str,
    job_dir: str,
    attempt: int,
    project: str,
    heartbeat_interval: int = 60,
) -> str:
    env_vars = {
        "TPU_QUEUE_JOB_ID": job_id,
        "TPU_QUEUE_ATTEMPT": str(attempt),
        "TPU_QUEUE_QR_NAME": qr_name,
    }
    env_vars.update(spec.env_vars)

    worker_gate_start = ""
    worker_gate_end = ""
    if not spec.run_on_all_workers:
        worker_gate_start = 'if [[ "$WORKER_ID" == "0" ]]; then\n'
        worker_gate_end = "\nelse\n  echo \"Worker $WORKER_ID idle because run_on_all_workers=false\"\nfi"

    setup_cmd = shlex.quote(spec.setup_cmd or "true")
    run_cmd = shlex.quote(spec.command or "true")

    return f"""#!/usr/bin/env bash
set -euo pipefail

export HOME="${{HOME:-/root}}"
JOB_ID={shlex.quote(job_id)}
JOB_DIR={shlex.quote(job_dir)}
ATTEMPT={attempt}
QR_NAME={shlex.quote(qr_name)}
DEPLOY_DIR="$HOME/deployed_code/$JOB_ID/attempt-$ATTEMPT"
LOG_DIR=/job_logs
HEARTBEAT_INTERVAL={int(heartbeat_interval)}

get_worker_id() {{
  if [[ -n "${{TPU_WORKER_ID:-}}" ]]; then
    echo "$TPU_WORKER_ID"
    return 0
  fi
  local host
  host="$(hostname)"
  if [[ "$host" =~ -w-([0-9]+)$ ]]; then
    echo "${{BASH_REMATCH[1]}}"
    return 0
  fi
  echo "0"
}}

WORKER_ID="$(get_worker_id)"
mkdir -p "$LOG_DIR"
touch "$LOG_DIR/worker-$WORKER_ID.log"
chmod 755 "$LOG_DIR"
chmod 644 "$LOG_DIR/worker-$WORKER_ID.log"
exec > >(tee -a "$LOG_DIR/worker-$WORKER_ID.log") 2>&1

echo "JOB_START=$(date -Iseconds)"
echo "JOB_ID=$JOB_ID"
echo "QR_NAME=$QR_NAME"
echo "ATTEMPT=$ATTEMPT"
echo "WORKER_ID=$WORKER_ID"

RECEIVED_SIGTERM=0
HEARTBEAT_PID=""

heartbeat_loop() {{
  while true; do
    date -Iseconds | gsutil cp - "$JOB_DIR/attempts/attempt-$ATTEMPT/heartbeat" || true
    sleep "$HEARTBEAT_INTERVAL"
  done
}}

handle_sigterm() {{
  echo "Received SIGTERM"
  RECEIVED_SIGTERM=1
  exit 143
}}
trap handle_sigterm SIGTERM

cleanup() {{
  local rc=$?
  if [[ "$RECEIVED_SIGTERM" == "1" ]]; then
    rc=143
  fi
  if [[ -n "$HEARTBEAT_PID" ]]; then
    kill "$HEARTBEAT_PID" 2>/dev/null || true
  fi
  echo "JOB_EXIT=$rc"
  echo "JOB_END=$(date -Iseconds)"
  gsutil cp "$LOG_DIR/worker-$WORKER_ID.log" "$JOB_DIR/logs/attempt-$ATTEMPT/worker-$WORKER_ID.log" || true
  if [[ "$WORKER_ID" == "0" ]]; then
    if [[ "$rc" == "0" ]]; then
      echo "SUCCESS $(date -Iseconds)" | gsutil cp - "$JOB_DIR/succeeded"
    elif [[ "$rc" == "42" || "$rc" == "143" ]]; then
      echo "Preemption-like exit $rc; scheduler will retry from QR state"
    else
      echo "FAILED with exit code $rc" | gsutil cp - "$JOB_DIR/failed"
    fi
  fi
}}
trap cleanup EXIT

{_secret_lines(spec.secret_refs, project)}
{_export_lines(env_vars)}

mkdir -p "$DEPLOY_DIR"
cd "$DEPLOY_DIR"
echo "Downloading code archive"
gsutil cp {shlex.quote(spec.code_tar_url)} code.tar.gz
tar -xzf code.tar.gz
rm -f code.tar.gz

echo "Running setup"
SETUP_CMD={setup_cmd}
bash -lc "$SETUP_CMD"

if [[ "$WORKER_ID" == "0" ]]; then
  date -Iseconds | gsutil cp - "$JOB_DIR/attempts/attempt-$ATTEMPT/claimed"
  heartbeat_loop &
  HEARTBEAT_PID=$!
fi

echo "Running command"
RUN_CMD={run_cmd}
{worker_gate_start}bash -lc "$RUN_CMD"{worker_gate_end}
"""


def write_startup_script(
    *,
    job_id: str,
    spec: JobSpec,
    qr_name: str,
    job_dir: str,
    attempt: int,
    project: str,
    output_path: str,
    heartbeat_interval: int = 60,
) -> str:
    path = Path(output_path)
    path.write_text(
        build_startup_script(
            job_id=job_id,
            spec=spec,
            qr_name=qr_name,
            job_dir=job_dir,
            attempt=attempt,
            project=project,
            heartbeat_interval=heartbeat_interval,
        )
    )
    path.chmod(0o700)
    return str(path)
