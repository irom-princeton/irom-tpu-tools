# irom-tpu-tools

Queue-backed TPU scheduling for IROM. This branch intentionally removes the
old local watcher workflow from the default CLI. Normal users submit jobs to a
central GCS queue; a central scheduler service account creates/deletes queued
resources and TPU VMs.

## Why This Branch Exists

The old workflow let every user run TPU Admin operations from their own shell.
That made IAM broad and cleanup decentralized. The new split is:

- Users: package code, submit job specs, read status/logs, request cancel/retry.
- Scheduler: creates queued resources, handles preemption, retries attempts,
  cleans up TPU VMs and QRs.
- Admins: inspect quotas, all jobs, queue-owned QRs, and delete orphan/idle
  queue resources.

Normal job lifecycle state no longer lives in `~/.tpu-jobs`. It lives in GCS:

```text
gs://.../tpu-job-queue/
  scheduler_state.json
  jobs/<job_id>/
    spec.json
    status.json
    code.tar.gz
    canceled
    retry
    running
    succeeded
    failed
    attempts/attempt-1/claimed
    attempts/attempt-1/heartbeat
    logs/attempt-1/worker-0.log
```

## Installation

```bash
pipx install --force /home/lzha/code/irom-tpu-tools
export PATH="$HOME/.local/bin:$PATH"
tpu --help
```

Optional custom config:

```bash
export TPU_QUEUE_CONFIG=/path/to/resources.yaml
```

The packaged default config is
`src/irom_tpu_tools/queue/resources.yaml`.

## User Commands

Submit a job:

```bash
tpu create v6 -n 32 --name train-openpi \
  --code-dir /home/lzha/code/ego-lap \
  --setup-cmd "uv sync --group tpu && uv pip install -e ." \
  --env WANDB_PROJECT=openpi \
  -- python scripts/train.py --config-name=my_config
```

List and inspect:

```bash
tpu list
tpu list v6
tpu list --jobs v6 --active
tpu list --resources v4
tpu status <job_id_or_name>
tpu logs <job_id_or_name> --lines 200
tpu tail <job_id_or_name> --follow
```

By default, `tpu list [v4|v5|v6]` lists queued jobs. If no matching jobs exist,
it shows the requestable resource catalog instead of a bare empty table. Use
`--jobs` for a strict queued-job view, or `--resources` for a strict resource
catalog view.

Cancel or retry:

```bash
tpu delete <job_id_or_name>
tpu retry <job_id_or_name>
tpu rerun <job_id_or_name>
```

`tpu delete` writes a cancellation sentinel. The scheduler performs the actual
TPU VM and queued-resource cleanup.

## Shared Interactive TPUs

Shared v4 interactive TPUs are configured under `interactive_tpus` in
`resources.yaml`. Users can connect to and move files on these allowlisted TPUs,
but this command group intentionally has no create/delete/stop/start actions.
Admins create, stop, start, and delete these shared TPUs outside the user
command path.

```bash
tpu interactive list
tpu interactive list --live
tpu interactive info v4-interactive
tpu interactive ssh v4-interactive --worker 0
tpu interactive run v4-interactive -- hostname
tpu interactive tmux v4-interactive --session debug -- python scratch.py
tpu interactive attach v4-interactive --session debug
tpu interactive output v4-interactive --session debug --lines 200
tpu interactive put v4-interactive ./local.txt ~/local.txt
tpu interactive get v4-interactive ~/remote.txt ./remote.txt
```

Interactive commands resolve only configured TPU names or aliases. The packaged
default includes `v4-4-01-interactive` with alias `v4-interactive`; admins should
edit that allowlist to match the actual shared v4 TPU inventory.

Interactive users need only the project/IAP/OS Login permissions required to SSH
to the existing TPU VM and perform normal file I/O on it. They do not need TPU
Admin because `tpu interactive` never creates, deletes, stops, or starts TPU
resources.

## Scheduler

The scheduler must run under an identity with TPU Admin permissions:

```bash
tpu scheduler --scan-interval 30
```

For local validation without GCP:

```bash
tpu --dry-run --base-dir /tmp/irom-tpu-queue scheduler --once
```

The scheduler loop:

1. Scans regional queue buckets for jobs.
2. Handles cancel/retry/completion sentinels.
3. Polls queue-owned QRs.
4. Requeues preempted/suspended/failed attempts until `max_attempts`.
5. Enforces quota groups and per-user chip limits.
6. Deletes terminal or orphaned queue-owned resources.
7. Writes `scheduler_state.json` for fast CLI listing.

## Admin Commands

These commands are intended for the central admin account:

```bash
tpu admin resources
tpu admin qrs
tpu admin cleanup --idle-minutes 30
tpu admin cleanup --idle-minutes 30 --yes
```

Cleanup is dry-run by default. It only targets queue-owned resources whose names
start with the configured `qr_prefix`.

## IAM Model

Normal users need:

- `storage.objects.create/get/list` on queue prefixes.
- Read access to their logs/status.
- SSH/IAP/OS Login access to allowlisted shared interactive TPUs, if they use
  `tpu interactive`.
- No TPU Admin role.

Scheduler/admin identity needs:

- TPU Admin on the TPU project.
- Service Account User for TPU worker service accounts.
- Secret Manager access for configured worker secrets.
- Read/write/delete access to queue buckets.

TPU worker service accounts need:

- Read access to queued code archives.
- Write access to job logs/sentinels.
- Dataset/checkpoint bucket permissions required by the training code.
- Secret Manager access for configured secrets, such as `WANDB_API_KEY`.

## Development Validation

```bash
python3 -m compileall src tests
PYTHONPATH=src python3 -m unittest discover -s tests
```
