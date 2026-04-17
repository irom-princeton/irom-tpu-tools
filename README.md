# irom-tpu-tools

Unified TPU utilities and background watcher for any repo across **v4 / v5 / v6**.

## Installation

First, set up the Google Cloud CLI by following the [official installation guide](https://docs.cloud.google.com/sdk/docs/install-sdk).

```bash
git clone https://github.com/irom-princeton/irom-tpu-tools.git
pipx install ./irom-tpu-tools

# Ensure ~/.local/bin is on PATH
export PATH="$HOME/.local/bin:$PATH"

# set up tab-completion for tpu name (optional)
tpu install-completion
source ~/.bashrc

# Verify
tpu --help 
tpu --commands # command cheat sheet
```

When you make local changes to any file in the package, run `pipx install --force ./irom-tpu-tools` for it to take effect.

---

## Environment Setup

Export the following variables (e.g. in your `~/.bashrc` or `~/.zshrc`):

```bash
export TPU_NAME=<default_tpu_name>                   # optional fallback when --name is omitted. Format: <tpu_type>-<num_tpus>-<index>-<your_name> (e.g. v6-64-01-lihan)
export TPU_PROJECT=<gcp_project_id>                  # ask your project admin
export TPU_ZONE_v4=us-central2-b
export TPU_ZONE_v5=us-central1-a
export TPU_ZONE_v6=us-east1-d
export TPU_BUCKET_v4=gs://my-bucket-v4               # ask your project admin
export TPU_BUCKET_v5=gs://my-bucket-v5               # ask your project admin
export TPU_BUCKET_v6=gs://my-bucket-v6               # ask your project admin
export TPU_SERVICE_ACCOUNT=<service_account_email>   # ask your project admin
export WANDB_API_KEY=<your_wandb_api_key>

# Optional — used as defaults when `tpu create --repo` is omitted.
export GH_REPO_NAME=<github_repo_name>               # repo to clone on the TPU
export GH_OWNER=<your_github_username>               # owner of the repo/fork
export GH_TOKEN=<your_github_personal_access_token>  # needs repo read access
```

`GH_OWNER` / `GH_TOKEN` are used to clone via HTTPS (`https://<token>@github.com/<owner>/<repo>`), so this works with your own fork — you do **not** need to be the upstream repo owner.

After this, you can run `export TPU_NAME=pi0 && tpu v4` to initialize. You will be prompted with some questions from google CLI, and answer yes. After the set up, type and run `exit` in your terminal to terminate the setup process.

---

## Quickstart

### Monitoring Status

```bash
tpu list                    # check all running tpu jobs
tpu status                  # check all tpu jobs managed by you
```

### Creating a TPU instance

Different options for creating a tpu instance:
```bash
tpu create v6 --name my-tpu -n 8 # create bare tpu instance
tpu create v6 --name my-tpu -n 8 --repo usrname/reponame  --branch main --setup-cmd "..."  # create tpu instance with cloned repo and custom setup command
tpu create v6 --name my-tpu -n 8 --repo usrname/reponame  --branch main --setup-cmd "..." \
  -- XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --group tpu scripts/train.py --config my_config # create tpu instance with repo, setup, and command that will relaunch after preemption
```

To access your instance after creation
```bash
tpu attach my-tpu                   # attach to the training tmux session on worker 0 (Ctrl-B+D to exit)
tpu tail my-tpu                     # read final lines of output
tpu info my-tpu                     # get information about the tpu
```

### Stopping / Restarting / Deleting a TPU instance

`tpu stop` and `tpu start` both **preserve the TPU's resource allocation** — the VM and its attached disk stay provisioned, so your reservation/quota slot is held across the stop. Only `tpu delete` releases allocation.

```bash
tpu stop my-tpu                      # stop watcher + stop the TPU (allocation preserved)
tpu start my-tpu                     # restart and respawn watcher with the saved config
tpu start my-tpu --repo D/E --branch main --setup-cmd "uv sync" \
  -- python scripts/other_train.py   # restart with a NEW repo / setup / command
tpu delete my-tpu                    # stop watcher + delete TPU (releases allocation)
```

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
tpu start --help
```
