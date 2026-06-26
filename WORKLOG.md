# Worklog

## 2026-06-26 - Queued TPU Toolbox Branch

Goal: migrate `irom-tpu-tools` to a queue-backed TPU workflow on branch
`codex/queued-tpu-toolbox`, removing local watcher state and moving TPU
Admin operations behind central scheduler/admin commands.

Plan:
- Add GCS-backed queue job specs, status, logs, sentinels, and scheduler state.
- Keep normal users on queue submission/cancel/log commands that only need queue
  bucket access.
- Group direct TPU Admin operations under scheduler/admin paths.
- Validate with local dry-run backend and unit tests; no real TPU jobs launched.

Result:
- Added `src/irom_tpu_tools/queue/` with config, typed job state, GCP and
  dry-run backends, code packaging, startup script generation, scheduler, and
  queue CLI.
- Replaced the top-level `tpu` CLI path with queue submission/status/log/admin
  commands.
- Added default IROM resource/quota config and tests for scheduling,
  preemption requeue, user chip limits, and no local watcher state in startup
  scripts.

Validation:
- `python3 -m compileall src tests`
- `PYTHONPATH=src python3 -m unittest discover -s tests`
- Dry-run submit/scheduler/list/admin QR smoke with
  `/tmp/irom-tpu-queue-smoke`.

No real TPU or GCP queued resource was launched.
