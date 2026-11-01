# Worklog

## 2026-10-31 - Retry GCS/TPU Reads; Warn Instead of Showing "(none)"

Goal: `tpu list` printed "Queued jobs: (none)" and "Live TPU VMs: (none)"
during a brief gcloud failure, with only `Failed to read GCS object
.../scheduler_state.json (exit 1)` and no reason, so a live queue looked empty.

Root causes:
- The scheduler rewrites `scheduler_state.json` every ~60-70 s; a read that
  lands mid-rewrite gets `not found: 404` (1 in 74 reads in a loop), which
  `read_gcs_result` did not recognise as missing and reported as a failure.
- `list_gcs`, `list_queued_resources` and `list_tpu_vms` returned `[]` on any
  gcloud failure without logging, so a network/auth blip printed "(none)".
- The read warning dropped gcloud's stderr, hiding the actual error.

Changes (`queue/backend.py`):
- New `GCPBackend._run_read` retries read-only gcloud calls twice (1 s, 2 s)
  on a nonzero exit. "Matched no objects" (object/prefix really absent) and
  timeouts (exit 124, already waited the full timeout) return at once, so the
  scheduler's common missing-sentinel reads cost nothing extra.
- `read_gcs_result`, `list_gcs`, `list_queued_resources`, `list_tpu_vms` use
  it, and log a warning with gcloud's ERROR line when they still fail; an
  empty GCS prefix stays a silent empty listing.
- Tests: mid-rewrite 404 retried to success, persistent failure gives up after
  3 tries and logs the gcloud error, missing object/timeout not retried, list
  failures warn while an empty prefix does not.

Goal: schedule every user's jobs, not only lzha's. zheng reported that jobs
submitted today ran before priority-0 jobs from yesterday; the queue showed
his three v6-64 priority-0 jobs PENDING with 256 v6 chips free while lzha's
default-priority jobs from today were ACTIVE. Root cause: the deployed
service ran `--focus-user=lzha`, which never schedules other users' pending
jobs.

Changes:
- `contrib/systemd/irom-tpu-scheduler.service` now runs `tpu scheduler`
  unfocused; the unit is the central scheduler for the whole queue.
- README reframes `--focus-user` as an optional narrow/debugging mode and
  warns against running a focused service as the only scheduler, and
  documents queue ordering: priority ascending (0 highest, default 1), then
  submit time, with quota/user-cap/backoff skipping and GCP capacity deciding
  activation order.
- Updated the irom-tpu-vm-workflow skill (`.claude` and `.codex` copies):
  SKILL.md now describes the central deployment, current command names, and
  ordering semantics; `personal_scheduler_preflight.sh` now fails if the
  service command contains `--focus-user` instead of requiring it.

Deployment and live verification (2026-07-02):
- Pushed `597fdb0`, ran `pipx reinstall irom-tpu-tools`, installed the updated
  unit, `daemon-reload`, restarted `irom-tpu-scheduler.service`. The updated
  preflight reported `mode=central` with one service PID at `597fdb0`.
- The first central scan drained the cancellation backlog: eight jobs whose
  `canceled` sentinels had been written by zheng, yixunhu, and ah4775 while
  the focused scheduler ignored them (verified `Canceled at ...` sentinel
  contents and timestamps, all matching deliberate user cancel-and-resubmit
  patterns) transitioned to CANCELED and released ~192 v6 chips.
- zheng's priority-0 `train_ac_wan` (20260701-185937) left PENDING and reached
  PROVISIONING on QR `iq-39b4a5ca-v6-64-a1` within minutes of the restart;
  remaining pending jobs are being scheduled in priority/submit order.

## 2026-07-02 - Self-Service Interactive SSH Key Requests

Goal: let users provision their own interactive TPU SSH keys without any TPU
write permission, using the same sentinel pattern as job cancellation.

Changes:
- Added `tpu interactive add-key [--key-file PATH] [--user NAME]`. It
  validates the local gcloud public key and uploads it to
  `{primary_bucket}/ssh-key-requests/<user>.pub`; users only need existing
  queue-bucket write access.
- Added a scheduler sync pass (every 300 seconds, cold start included, active
  in focus-user mode but not focus-job one-shots) that appends requested keys
  to every configured interactive TPU and deletes the request once the key is
  present everywhere. Requests survive unreadable or failed nodes for retry.
- Username-match guard in shared `normalized_key_entry`: the provisioned
  metadata entry always uses the request filename's username, an embedded
  `user:` prefix must match it, and the key comment is replaced with that
  username, so a request cannot install a key under another login. Malformed
  or mismatched requests are logged and deleted.
- Moved the ssh-key parsing/fingerprint helpers into `queue/interactive.py`,
  shared by the scheduler, `tpu admin ssh-keys` (whose `--add` now applies the
  same normalization), and the new command. The interactive permission hint
  now points at `tpu interactive add-key`.

Validation:
- `PYTHONPATH=src python3 -m unittest discover -s tests`: 48 passed, with new
  tests for request provisioning under focus-user mode, username-mismatch
  rejection, unreadable-node retention, and the add-key upload.
- `python3 -m compileall -q src tests` and `uvx ruff check src tests`: passed.
- Dry-run smoke: `interactive add-key` uploaded the request and a scheduler
  one-shot kept it alive while node metadata was unreadable, as designed.

## 2026-07-02 - Interactive TPU SSH Key Sync Command

Goal: make interactive TPU SSH key provisioning automatic and robust after
diagnosing ah4775's v4 SSH failure. Audit logs showed gcloud attempting
`tpu.nodes.update` on `v4-4-02/03/04-interactive` on 2026-06-30 because his
key was only present on the two nodes his own gcloud had updated in mid-June
while he still had TPU write permissions; viewer-only users cannot self-add
keys, so first connections to newer nodes failed. The key was manually copied
to the remaining nodes on 2026-07-01.

Changes:
- Added `tpu admin ssh-keys [--version vX] [--add USER=PUBKEY_FILE] [--yes]`.
  It inventories `ssh-keys` metadata on every configured interactive TPU
  (user + SHA256 fingerprint), computes the cross-node union plus any `--add`
  public key files, and reports missing entries per node. `--yes` appends the
  missing keys via `gcloud alpha compute tpus tpu-vm update
  --metadata-from-file=ssh-keys=...`. It never removes keys, preserves
  unrecognized metadata lines without propagating them, and skips (and flags)
  nodes whose describe fails.
- Added `get_tpu_vm_ssh_keys`/`set_tpu_vm_ssh_keys` to the GCP and dry-run
  backends. Reads distinguish "no ssh-keys metadata" from "describe failed" so
  an unreachable node is never treated as empty and overwritten.
- Documented the sync command and the onboarding flow in the README admin and
  IAM sections.

Validation:
- `PYTHONPATH=src python3 -m unittest discover -s tests`: 44 passed, including
  new tests for union sync, `--add` onboarding, unrecognized-entry
  preservation, and the unreadable-node guard.
- `python3 -m compileall -q src tests` and `uvx ruff check src tests`: passed.
- Live read-only run against the five real interactive nodes printed matching
  ah4775/lzha fingerprints everywhere and reported no missing keys.

## 2026-07-02 - CLI Dedup, Non-Admin Delete Fix, Current-Version README

Goal: remove duplicated CLI surface, fix the bug where non-admin users could
not delete pending/active TPU jobs under the deployed focused scheduler, and
rewrite the README to describe only the current queue-backed tool.

Changes:
- Removed alias subcommands so each action has one name: `info` (use
  `status`), `output` (use `logs`), `cancel` (use `delete`), and
  `tpu interactive tail` (use `tpu interactive output`, which already supports
  `--follow`).
- Deleted the dead legacy watcher modules `tpu.py`, `watch.py`, `jobs.py`, and
  top-level `config.py`. They were unreachable from the `tpu` entry point and
  duplicated queue functionality. `ssh.py` remains; `queue/interactive.py`
  uses it.
- Fixed focused (`--focus-user`) scheduling to honor every user's cancellation
  sentinel. Previously `run_once` restricted `check_canceled_jobs` to the
  focus user's jobs, so a non-focus user's `tpu delete` was written but never
  processed and their pending/active job and TPU resources were never
  released. Scheduling, completion, retry, and QR polling remain
  focus-restricted; `--focus-job` one-shots keep their narrow scope.
- Made `tpu delete` report an existing sentinel and fail loudly when the
  sentinel write fails instead of always claiming success.
- Rewrote the README to describe the current version only, without the
  branch-migration framing or removed alias commands.

Validation:
- `python3 -m compileall -q src tests`: passed.
- `PYTHONPATH=src python3 -m unittest discover -s tests`: 41 passed.
- `uvx ruff check src tests`: passed.
- Dry-run repro: a `--focus-user=lzha` pass now cancels another user's
  pending job and removes its queued resource; before the fix the job stayed
  `PENDING` with the sentinel unprocessed.

## 2026-06-30 - Exclusive Focused Scheduling

- Changed `tpu scheduler --once --focus-job JOB` to schedule only `JOB` when it is pending. Previously, the focused pass walked and attempted every older pending job before the target, which could create unrelated TPU requests while an operator was monitoring a quota-blocked job.
- Global unfocused scheduling retains priority and submission-time ordering.
- Added a regression test that leaves the older job pending and provisions only the focused job.

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

## 2026-06-30 - Interactive TPU Launch Parsing And Access Guidance

Agent: `interactive-access-20260630`

Goal: repair the connect-only interactive TPU CLI so documented `run` and
`tmux` options work after the TPU name, preserve nested shell argument
boundaries, and accurately explain the SSH-key permission path without granting
users TPU Admin.

Base revision: `bafaf331bbc610fd599fe23866e2ee4056a48deb` on isolated branch
`codex/interactive-access-20260630`.

Findings and changes:
- `argparse.REMAINDER` consumed `--worker` and `--session` when they followed
  the TPU name. Replaced it with one-or-more command arguments, retaining `--`
  as the launcher/remote-command separator and preserving both option orders.
- Plain string joining discarded nested `bash -lc` quoting. Added a shared
  `shlex.join` helper for queued and interactive commands.
- Single-worker SSH also passed `bash`, `-lc`, and the raw script as separate
  trailing SSH arguments. Because OpenSSH flattens them, shell operators could
  escape into the outer shell. Both streaming and bounded SSH paths now pass
  one safely quoted remote shell command.
- Current gcloud TPU-VM SSH checks project/node `ssh-keys` metadata and calls
  `tpu.nodes.update` when the exact local key is missing. Updated the runtime
  hint and README to prefer admin key pre-provisioning plus viewer/OS Login/IAP
  access, with a narrow update permission only when pre-provisioning is not
  possible. TPU Admin remains unnecessary for interactive users.
- Added focused quoting, parser compatibility, command-flag preservation, and
  permission-hint tests, plus an explicit four-worker tmux example.

Validation:
- Six focused unit tests for command quoting, single-worker SSH wrapping,
  post-name parsing, pre-name compatibility, and permission guidance passed.
- `PYTHONPATH=src python3 -m unittest discover -s tests` passed all 25 tests.
- `PYTHONPATH=src python3 -m compileall -q src tests` passed.
- `ruff check src/irom_tpu_tools/queue/cli.py src/irom_tpu_tools/queue/interactive.py src/irom_tpu_tools/ssh.py tests/test_queue_scheduler.py`
  passed.
- `git diff --check` passed.

No TPU, GCP, scheduler, SSH, tmux, or remote command was launched. No local
long-running process remains.

Implementation commits:
- `a1fb0367ab10762eabfe5d29f61acc34f8d7b02c`
- `4df0f8cf06ebfaf82a6933fc29b12629a678624b`

## 2026-06-30 - Correct v6e Worker Topology

Agent: `gemma3-oxe-b2048-20260630`

Goal: make queue resource metadata match the current four-chip-per-worker v6e
topology observed by multi-host JAX and the TPU VM SSH API.

Findings and changes:
- A live `v6e-64` allocation exposed workers 0 through 15, each with four local
  TPU devices. JAX reported 16 processes and 64 global devices. Existing queue
  metadata incorrectly advertised eight workers.
- Updated all v6e resource entries consistently: `v6-8` has 2 workers,
  `v6-16` has 4, `v6-32` has 8, `v6-64` has 16, and disabled `v6-128` has 32.
  Chip counts, accelerator types, quotas, and launch behavior are unchanged.
- Added a default-config regression test that checks every v6e worker count and
  enforces four chips per worker.

Validation:
- `uv run --with pytest pytest -q`: 26 passed.
- `uvx ruff check tests/test_queue_scheduler.py`: passed.
- `git diff --check`: passed.

## 2026-07-01 - Personal Scheduler Auto-Resume And Failure Classification

Agent: `personal-tpu-autoresume-20260701`

Goal: replace agent-owned one-shot reconciliation loops with one local
single-user scheduler that automatically recovers infrastructure preemptions,
while keeping setup and application errors terminal and actionable.

Base revision: `96170c83631cb247478041c7e5359b65db1d96fc`, containing the
13 unmerged queue/preemption fixes currently installed on this workstation.
The separate command-quoting branch was patch-equivalent to later commits in
this base and required no additional integration commit.

Changes:
- Added continuous `--focus-user` scheduling. Other users' status is refreshed
  for quota accounting, but lifecycle operations, scheduling, cancellation,
  completion, and retries are restricted to the selected user.
- Added a local `flock` singleton guard and a systemd user-service template so
  per-job scheduler loops cannot race the personal scheduler. The service
  uses the CLI's bounded rotating log under `~/.local/state/irom-tpu-tools/`
  because this workstation does not expose the user journal to the account.
- Added structured attempt outcomes for infrastructure preemption, setup
  errors, and application errors, including retryability, phase, worker, and
  exit code. `tpu status` now prints the recovery policy and exact log command.
- Changed new workers to use attempt-scoped success/failure markers. Every
  worker can report a terminal error; infrastructure state is reconciled before
  worker markers so preemption wins over secondary distributed-process exits.
- Versioned new startup attempts so stale legacy root markers cannot terminate
  a replacement attempt, while preserving old-attempt compatibility.
- Verified each downloaded code archive against the immutable SHA-256 checksum
  stored in the submitted job spec before setup or training begins.
- Documented installation, scope, retry policy, workstation availability, and
  log-polling constraints.

Validation:
- `PYTHONPATH=src python3 -m unittest discover -s tests`: 32 passed.
- `PYTHONPATH=src python3 -m compileall -q src tests`: passed.
- `uvx ruff check src tests`: passed.
- `systemd-analyze --user verify contrib/systemd/irom-tpu-scheduler.service`:
  passed.
- Focused-user dry-run scheduler smoke under
  `/tmp/irom-personal-scheduler-smoke-20260701`: passed.
- `git diff --check`: passed.

Deployment and live verification:

- Fast-forwarded the complete installed queue/preemption history and this
  change onto `main`, then pushed `main` through `046d423`.
- Reinstalled `irom-tpu-tools` from the merged checkout with `pipx` and enabled
  `/home/lzha/.config/systemd/user/irom-tpu-scheduler.service` with
  `Linger=yes`. The active command is the installed CLI with
  `--focus-user=lzha --scan-interval=30` and the rotating local log at
  `~/.local/state/irom-tpu-tools/scheduler.log`.
- Stopped legacy per-job scheduler loops without stopping or resubmitting any
  training process. A live singleton smoke returned the expected
  `Another local TPU scheduler is active` error, and process inspection showed
  exactly one remaining scheduler.
- The service classified existing maintenance/suspension exits as
  `INFRASTRUCTURE_PREEMPTION`. It queued a replacement attempt for the existing
  reasoning evaluation; that resource reached `WAITING_FOR_RESOURCES`. A
  replacement request for the existing G3 training job received GCP HTTP 429
  because the 512-chip v6e preemptible project/zone quota was exhausted, so the
  job correctly remained `PENDING` for a later scan.
- `tpu status` exposed retryability and recovery guidance for the affected
  attempts. No new TPU job spec was submitted during implementation or
  deployment.

### Live Follow-Up: Create Backoff

The first deployed scan exposed a repeated HTTP 429 path while the project was
at its 512-chip v6e preemptible quota. The job correctly remained `PENDING`, but
the scheduler attempted another queued-resource create after every full scan.

Changes:
- Added `scheduler.create_failure_backoff_seconds`, defaulting to 300 seconds,
  and an in-memory per-job delay after a failed queued-resource create.
- Fixed `GCPBackend.create_queued_resource` so the synthetic exit code 124
  returned by a bounded command timeout is treated as failure instead of a
  successfully provisioned request.
- Limited focused-user quota refreshes to other users' nonterminal jobs.
  Terminal history is still discovered at cold start but no longer causes a
  status-object read on every scheduler pass.
- Updated personal-service and one-shot guidance so validation uses a separate
  lock and focused one-shots are never run beside the service.

Validation:
- `PYTHONPATH=src python3 -m unittest discover -s tests`: 35 passed.
- `PYTHONPATH=src python3 -m compileall -q src tests`: passed.
- `uvx ruff check src tests`: passed.

Deployment verification:
- Fast-forwarded and pushed `main` through `284d99e`, reinstalled the pipx
  package from the canonical `/home/lzha/code/irom-tpu-tools` checkout, and
  restarted the enabled user service. The installed scheduler source hash
  matched `main`.
- The live service received HTTP 429 quota responses for the existing reasoning
  evaluation and G3 training jobs, left both jobs pending, and logged a
  300-second create deferral for each job.
- Process inspection showed exactly one scheduler process and no legacy
  per-job scheduler loop. No new job spec was submitted.
