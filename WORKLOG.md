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

Implementation commit: `1b646295e94352a98526a5151d8ef0b25cd7775f`.

## 2026-06-26 - Restricted Shared Interactive TPU Commands

Goal: allow users to use pre-existing shared v4 interactive TPUs without
restoring direct TPU lifecycle commands.

Plan:
- Add an `interactive_tpus` allowlist to queue config.
- Add `tpu interactive` commands for list/info/ssh/run/tmux/attach/output and
  file copy.
- Do not add create/delete/stop/start under `tpu interactive`.
- Validate parsing and allowlist behavior locally; do not launch or mutate TPU
  resources.

Result:
- Added `InteractiveTPUConfig` and `interactive_tpus` parsing with v4-only
  validation.
- Added connect-only `tpu interactive` subcommands:
  `list`, `info`, `ssh`, `run`, `tmux`, `attach`, `output`, `tail`,
  `tmux-ls`, `put`, and `get`.
- Added default allowlist entry `v4-4-01-interactive` with alias
  `v4-interactive`.
- Added tests for allowlist resolution, default config, and rejection of
  lifecycle verbs under `tpu interactive`.

Validation:
- `python3 -m compileall src tests`
- `PYTHONPATH=src python3 -m unittest discover -s tests`
- `PYTHONPATH=src python3 -m irom_tpu_tools.cli interactive --help`
- `PYTHONPATH=src python3 -m irom_tpu_tools.cli interactive list`
- Parser smoke for `tpu interactive run v4-interactive -- hostname`

No TPU command was executed against GCP.

## 2026-06-27 - Health-Aware TPU Requeue

Goal: make the queue scheduler requeue jobs when a TPU VM is still reported as
`READY` but health has moved to `UNHEALTHY_MAINTENANCE`, because worker SSH is
unavailable in that state and queue status can otherwise remain stale.

Result:
- Added `TpuVmStatus` with state, health, and health description.
- Extended the GCP and dry-run backends to expose TPU VM health from
  `gcloud alpha compute tpus tpu-vm describe`.
- Changed scheduler active-QR polling to treat
  `TPU_VM_HEALTH_UNHEALTHY_MAINTENANCE` as a retry/preemption signal.
- Added a dry-run scheduler test for `READY` plus `UNHEALTHY_MAINTENANCE`.

Validation:
- `python3 -m py_compile src/irom_tpu_tools/queue/backend.py src/irom_tpu_tools/queue/scheduler.py tests/test_queue_scheduler.py`
- `.venv/bin/python -m unittest tests.test_queue_scheduler`
- `PYTHONPATH=src python3 -m unittest tests.test_queue_scheduler`
- `git diff --check`
