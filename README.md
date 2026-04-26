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
export TPU_PROJECT=mae-irom-lab-guided-data          # ask your project admin
export TPU_ZONE_v4=us-central2-b
export TPU_ZONE_v5=us-central1-a
export TPU_ZONE_v6=us-east1-d
export TPU_BUCKET_v4=gs://pi0-cot
export TPU_BUCKET_v5=gs://v5_central1_a
export TPU_BUCKET_v6=gs://v6_east1d
export WANDB_API_KEY=<your_wandb_api_key>
export GH_TOKEN=<your_github_personal_access_token>  # needs repo read access

# Optional — used as defaults when `tpu create --repo` is omitted.
export TPU_NAME=<default_tpu_name>                   # optional fallback when --name is omitted. Format: <tpu_type>-<num_tpus>-<index>-<your_name> (e.g. v6-64-01-lihan)
export GH_REPO_NAME=<github_repo_name>               # repo to clone on the TPU
export GH_OWNER=<your_github_username>               # owner of the repo/fork
```

`GH_OWNER` / `GH_TOKEN` are used to clone via HTTPS (`https://<token>@github.com/<owner>/<repo>`), so this works with your own fork — you do **not** need to be the upstream repo owner.

After this, you need to run `tpu attach pi0` to initialize. You will be prompted with some questions from google CLI, and answer yes. After the set up, type and run `exit` in your terminal to terminate the setup process.

---

## Quickstart

### Monitoring Status

```bash
tpu list                 # check all running tpu jobs
tpu status               # check all tpu jobs managed by you
```

### Creating a TPU instance

Different options for creating a tpu instance. Note, `my-tpu` should follow the format of <tpu_type>-<num_tpus>-<index>-<your_name> (e.g. v6-64-01-lihan).
```bash
# create bare tpu instance
tpu create v6 --name my-tpu -n 8

# create tpu instance with cloned repo and custom setup command
tpu create v6 --name my-tpu -n 8 --repo usrname/reponame  --branch main --setup-cmd "..."  

# create tpu instance with repo, setup, and command that will relaunch after preemption
tpu create v6 --name my-tpu -n 8 --repo usrname/reponame  --branch main --setup-cmd "..." \
  -- XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --group tpu scripts/train.py --config my_config 
```

To access your instance after creation
```bash
tpu ssh my-tpu                      # interactive SSH shell on worker 0 (no tmux)
tpu attach my-tpu                   # attach to the training tmux session on worker 0 (Ctrl-B+D to exit)
tpu tail my-tpu                     # read final lines of output
tpu info my-tpu                     # get information about the tpu
tpu logs my-tpu                     # show last 50 lines of the watcher log
```

### Re-running an existing job

If you want to relaunch the same setup + command on a managed TPU (e.g. after manually killing training, or to restart on a still-allocated TPU), use `rerun`. It loads the saved config and reuses the `tpu create` flow:

```bash
tpu rerun my-tpu        # prompts before relaunching on a READY TPU
tpu rerun my-tpu -f     # skip prompt
```

`tpu info my-tpu` shows what will be re-run (repo, branch, setup, command).

### Deleting a TPU instance
```
# stop watcher + delete TPU (releases allocation)
tpu delete my-tpu
```

---

## Full List of Commands

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
| `tpu ssh NAME` | Open interactive SSH shell on worker 0 (no tmux) |
| `tpu ssh NAME --worker 1` | Open interactive SSH shell on a specific worker |
| `tpu attach NAME` | Attach to tmux session on worker 0 |
| `tpu attach NAME --worker 1` | Attach on a specific worker |
| `tpu tail NAME` | Tail last 50 lines of the latest tmux log on worker 0 |
| `tpu tail NAME --worker 1` | Tail on a specific worker |
| `tpu tmux NAME --session s -- <cmd>` | Run a command in a new/named tmux session on all workers |

### 🧹 Cleanup

| Command | Description |
|---|---|
| `tpu nuke NAME` | Kill tmux + JAX processes + clean tmp (full reset) |
| `tpu clean NAME` | Truncate system logs to free disk |

### 🔧 Advanced

| Command | Description |
|---|---|
| `tpu v4 -- <cmd>` | Run a raw SSH command on all v4 workers (no tmux) |
| `tpu v4 --worker 0 -- <cmd>` | Run a raw command on a specific worker |
| `tpu v4 setup` | Re-run the setup step (clone + install) on v4 workers |

Replace `v4` with `v5` or `v6` as needed.

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
    watch.py          # Watcher daemon
    __init__.py
  README.md
  LICENSE
```
