# irom-tpu-tools

Unified TPU utilities and background watcher for any repo across **v4 / v5 / v6**.

## Installation

```bash
git clone https://github.com/irom-princeton/irom-tpu-tools.git
pipx install ./irom-tpu-tools

# Ensure ~/.local/bin is on PATH
export PATH="$HOME/.local/bin:$PATH"

# Verify
tpu --help       # 
tpu --commands   # command cheat sheet
```

When you make local changes to any file in the package, run `pipx install --force ./irom-tpu-tools` for it to take effect.

---

## Environment Setup

Export the following variables (e.g. in your `~/.bashrc` or `~/.zshrc`):

```bash
export TPU_NAME=<default_tpu_name>                   # optional; used when --name is omitted
export TPU_PROJECT=<gcp_project_id>
export TPU_ZONE_v4=us-central2-b
export TPU_ZONE_v5=us-central1-a
export TPU_ZONE_v6=us-east1-d
export TPU_BUCKET_v4=gs://my-bucket-v4
export TPU_BUCKET_v5=gs://my-bucket-v5
export TPU_BUCKET_v6=gs://my-bucket-v6
export TPU_SERVICE_ACCOUNT=<service_account_email>   # ask your project admin
export GH_REPO_NAME=<github_repo_name>               # repo to clone on the TPU
export GH_OWNER=<your_github_username>               # owner of the repo/fork
export GH_TOKEN=<your_github_personal_access_token>  # needs repo read access
export WANDB_API_KEY=<your_wandb_api_key>
```

`GH_OWNER` / `GH_TOKEN` are used to clone via HTTPS (`https://<token>@github.com/<owner>/<repo>`), so this works with your own fork — you do **not** need to be the upstream repo owner.

---

## Quickstart

`tpu create` is the main entrypoint. It provisions the TPU, runs setup, launches training in a tmux session, and **spawns a background daemon** that monitors the TPU and automatically recreates it + relaunches training on preemption.

```bash
# Submit a job — returns immediately; watcher runs in the background.
tpu create v6 --name my-tpu -n 8 \
  -- XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --group tpu scripts/train.py --config my_config

# Check what's going on.
tpu status                    # all managed jobs
tpu logs tpu-name -f            # follow the watcher log

# Attach to the training tmux session on worker 0.
tpu attach tpu-name

# When you're done.
tpu delete tpu-name             # stops watcher + deletes TPU
```

Everything after `--` is the training command. It will be run from inside the cloned repo, after `git fetch && git checkout <branch> && git pull`.

---

## Commands

Most per-TPU commands take a TPU **name** and auto-detect the version/zone by querying gcloud. If the name is omitted, `$TPU_NAME` is used.

### 🚀 Lifecycle

| Command | Description |
|---|---|
| `tpu create <version> --name NAME -n N [-b BRANCH] [-s SETUP] -- <cmd...>` | Create TPU, setup, launch training, start background watcher |
| `tpu delete [NAME]` | Delete the TPU and stop its background watcher |
| `tpu stop [NAME]` | Stop the TPU (preserve allocation; can restart later) |
| `tpu start [NAME]` | Restart a stopped TPU |

`create` flags:

| Flag | Description |
|---|---|
| `version` | `v4`, `v5`, or `v6` (positional, required) |
| `--name` | TPU name (default: `$TPU_NAME`) |
| `-n / --tpu-num` | Number of chips (v4: 4/8/16/32 · v5: 16/32/64 · v6: 8/16/32/64/128) |
| `-b / --branch` | Git branch to check out (default: `main`) |
| `-s / --setup-cmd` | Shell command(s) to run after cloning the repo (default: `uv sync`) |
| `-- <cmd...>` | Training command — everything after `--` |

### 📋 Monitoring

| Command | Description |
|---|---|
| `tpu list` | List all TPUs across all zones (includes watcher status) |
| `tpu list v6` | List TPUs in a specific zone |
| `tpu status` | Show status of all managed jobs (watcher, preemptions, running since) |
| `tpu status NAME` | Show status of a single job |
| `tpu logs NAME` | Show last 50 lines of the watcher log |
| `tpu logs NAME -f` | Follow the watcher log in real time |
| `tpu logs NAME -n 500` | Show last N lines |

The status/list tables include:
- `STATE` — current TPU state (READY / PREEMPTED / STOPPED / ...)
- `WATCHER` — whether the background daemon is alive
- `RUNNING SINCE` — when the TPU most recently became READY
- `#PREEMPTIONS` — total preemptions observed by the watcher
- `LAST PREEMPTED` — timestamp of the most recent preemption

### 🔗 Connect

| Command | Description |
|---|---|
| `tpu attach NAME` | Attach to tmux session on worker 0 |
| `tpu attach NAME --worker 1` | Attach on a specific worker |
| `tpu tail NAME` | Tail last 50 lines of the latest tmux log on worker 0 |
| `tpu tail NAME --worker 1` | Tail on a specific worker |
| `tpu tmux-ls NAME` | List tmux sessions on all workers |
| `tpu tmux NAME --session s -- <cmd>` | Run a command in a new/named tmux session on all workers |

### 🧹 Cleanup

| Command | Description |
|---|---|
| `tpu nuke NAME` | Kill tmux + JAX processes + clean tmp (full reset) |
| `tpu kill-jax NAME` | Kill only JAX/XLA processes |
| `tpu tmux-kill-all NAME` | Kill tmux server on all workers |
| `tpu clean-tmp NAME` | Clean JAX/XLA temp files |
| `tpu clean NAME` | Truncate system logs to free disk |

### 🔧 Advanced

| Command | Description |
|---|---|
| `tpu v4 -- <cmd>` | Run a raw SSH command on all v4 workers (no tmux) |
| `tpu v4 --worker 0 -- <cmd>` | Run a raw command on a specific worker |
| `tpu v4 setup` | Re-run the setup step (clone + install) on v4 workers |
| `tpu watch v4 -n 8 -f main -- <cmd>` | **[Legacy]** Foreground watch loop (no daemon) |

Replace `v4` with `v5` or `v6` as needed.

You can use `tpu-tools` instead of `tpu` for any command.

---

## Package Structure

```
irom-tpu-tools/
  pyproject.toml      # Console scripts: tpu, tpu-tools
  src/irom_tpu_tools/
    cli.py            # CLI dispatcher
    config.py         # Env var loader
    jobs.py           # Persistent job state in ~/.tpu-jobs/
    tpu.py            # TPU lifecycle helpers (create/list/delete/tmux/kill/nuke)
    watch.py          # Watcher daemon + legacy foreground watch
    __init__.py
  README.md
  LICENSE
```

---

## Help

```bash
tpu --help
tpu --commands        # cheat-sheet of all commands
tpu create --help
```
