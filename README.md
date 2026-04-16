# irom-tpu-tools

Unified TPU utilities and watcher for any repo across **v4 / v5 / v6**.

## Installation

```bash
git clone https://github.com/irom-princeton/irom-tpu-tools.git
pipx install ./irom-tpu-tools

# Ensure ~/.local/bin is on PATH
export PATH="$HOME/.local/bin:$PATH"

# Verify
tpu --help
```

When you make local changes to any file in the package, run `pipx install --force ./irom-tpu-tools` for it to take effect.

---

## Environment Setup

Export the following variables (e.g. in your `~/.bashrc` or `~/.zshrc`):

```bash
export TPU_NAME=<tpu_name>  # Unique identifier for the TPU VM. Must be set before each creation. Use format: <tpu_type>-<num_tpus>-<index>-<your_name> (e.g., v6-64-01-lihan)
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

## Watch & Run

`tpu watch` is the main command for submitting and monitoring training jobs. It:

1. Checks TPU state in a loop.
2. Creates the TPU if it doesn't exist (or was preempted/stopped).
3. SSHes into all workers, clones your repo (if not already present), and runs the setup command.
4. Launches your training command inside a `tmux` session.
5. Keeps watching and re-launches automatically if the TPU is preempted.

### Command structure

```
tpu watch <version> [flags] <branch> <run_command...>
```

| Argument / Flag | Description |
|---|---|
| `version` | TPU version: `v4`, `v5`, or `v6` |
| `-n / --tpu-num` | Number of chips (v4: 4/8/16/32 · v5: 16/32/64 · v6: 8/16/32/64/128) |
| `-f / --force` | Force re-setup and re-launch even if the TPU is already READY |
| `-s / --setup-cmd` | Shell command(s) to run after cloning the repo (default: `uv sync`) |
| `branch` | Git branch to check out before running (first positional arg after flags) |
| `run_command...` | The full command to execute on the TPU (everything after the branch) |

### Examples

**Basic training job on 8 v6 chips:**
```bash
tpu watch v6 -n 8 main \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --group tpu scripts/train.py --config my_config
```

**Custom dependency setup (e.g. editable install on top of uv sync):**
```bash
tpu watch v6 -n 8 --setup-cmd "uv sync && uv pip install -e ." main \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --group tpu scripts/train.py --config my_config
```

**Force re-run on an already-running TPU (e.g. after a failed job):**
```bash
tpu nuke v6                      # kill existing processes first
tpu watch v6 -f -n 8 my-branch \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --group tpu scripts/train.py --config my_config
```

> **Note:** `-f` launches the job once and exits. To resume continuous watching (auto-relaunch on preemption), re-run the same command **without** `-f`. To monitor the logs, run `tpu tail v6`.

### How setup works

On first launch (or after a preemption), the tool SSHes into all TPU workers and:
1. Writes environment variables to `~/.zshrc`.
2. Installs `uv`.
3. Clones your repo via HTTPS using `GH_TOKEN` (skipped if already cloned).
4. Runs `--setup-cmd` inside the repo directory (default: `uv sync`).

Setup is **idempotent** — re-running on an already-set-up worker is safe.

---

## Utility Commands

Replace `v6` with `v4` or `v5` as needed.

| Command | Description |
|---|---|
| `tpu nuke v6` | Kill all running processes on the TPU |
| `tpu list v6` | List TPUs in the v6 zone |
| `tpu tail v6` | Tail the most recent log. Default for worker 0 |
| `tpu delete v6` | Delete the current TPU |
| `tpu delete-name v6 NAME` | Delete a TPU by name |
| `tpu v6 setup` | Run the setup step manually (clone + install) |
| `tpu tmux v6 --session s <cmd>` | Run a command in a tmux session on the TPU |
| `tpu attach v6 --session s --worker 0` | Attach to a tmux session on worker 0 |
| `tpu tmux-ls v6` | List tmux sessions |
| `tpu tmux-kill-all v6` | Kill all tmux sessions |
| `tpu kill-jax v6` | Kill all JAX processes |
| `tpu clean-tmp v6` | Clean `/tmp` on all workers |

You can use `tpu-tools` instead of `tpu` for any command.

---

## Package Structure

```
irom-tpu-tools/
  pyproject.toml      # Console scripts: tpu, tpu-tools
  src/irom_tpu_tools/
    config.py         # Env var loader
    ssh.py            # gcloud SSH wrapper with timeouts
    tpu.py            # TPU lifecycle helpers (list/delete/tmux/kill/nuke)
    watch.py          # Watch-and-run logic
    cli.py            # CLI dispatcher
    __init__.py
  README.md
  LICENSE
```

---

## Help

```bash
tpu --help
tpu watch --help
```
