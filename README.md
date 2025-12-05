# openpi-tpu-tools

Unified TPU utilities and watcher for OpenPI-CoT across **v4 / v5 / v6**.

## Installation

```bash

# Install from GitHub
pipx install git+https://github.com/lihzha/openpi-tpu-tools.git

# or install from local path after cloning
git clone https://github.com/lihzha/openpi-tpu-tools.git
pipx install ./openpi-tpu-tools

# Ensure ~/.local/bin is on PATH (for --user installs)
export PATH="$HOME/.local/bin:$PATH"

# Verify (both names work)
tpu --help
# or
tpu-tools --help
```

When you make local changes to any file in the package, run `pipx install --force <PACKAGE_DIR>` for it to take effect.

---


## Environment Setup

Set up your environment variables. Below is an example:

```bash
export TPU_NAME=<tpu_name>
export TPU_PROJECT=mae-irom-lab-guided-data
export TPU_ZONE_v4=us-central2-b
export TPU_ZONE_v5=us-central1-a
export TPU_ZONE_v6=us-east1-d
export TPU_BUCKET_v4=gs://pi0-cot
export TPU_BUCKET_v5=gs://v5_central1_a
export TPU_BUCKET_v6=gs://v6_east1d
export GH_REPO_NAME="openpi-cot"  # repo name. Only need to change if you want to extend this package to other repos
export GH_OWNER="lihzha"  # Your github name
export TPU_SERVICE_ACCOUNT="<YOUR_SERVICE_ACCOUNT_HERE>"  # Ask your project admin for service account
export GH_TOKEN="<YOUR_GITHUB_TOKEN_HERE>"  # Your github personal access tokens. Required for accessing private repos
export WANDB_API_KEY="<YOUR_API_KEY_HERE>"
```
<!-- 
Optional SSH settings:

```
GCLOUD_SSH_KEY_FILE
SSH_CONNECT_TIMEOUT
SSH_ALIVE_INTERVAL
SSH_ALIVE_COUNT_MAX
SSH_TOTAL_TIMEOUT
SSH_KILL_AFTER
DESCRIBE_TIMEOUT
SLEEP_SECS
``` -->

If you want to use single node to debug code, run `tpu v4 setup` after you finish above steps. Then, first clone the repo, and then you can interactive running commands using `tpu v4 "source ~/.zshrc && cd openpi-cot && git checkout -b <your_branch_name> && git pull origin <your_branch_name> && uv run <your_scripts_and_args>`.

---


## Watch & Run

If you want to submit a training job, use command in the section.

```bash
# v6 example (8 workers)
tpu watch v6 -f -n 8 -- <extra args>

# v5 example (16 workers)
tpu watch v5 -f -n 16 -- <extra args>

# v4 example (8 workers, maps 8→2x2x2 topology)
tpu watch v4 -f -n 8 -- <extra args>
```

Use `--` to separate TPU arguments from training script arguments:

```bash
tpu watch v6 -f -n 8 -- --config.some_flag=value
```

---



## Utility Commands

Replaces functions from `.tpu_funcs.sh`:

| Command                                        | Description                     |
| ---------------------------------------------- | ------------------------------- |
| `tpu list v6`                                  | List TPUs in v6                 |
| `tpu delete v6`                                | Delete current TPU              |
| `tpu delete-name v6 NAME`                      | Delete TPU by name              |
| `tpu tmux v6 --session s <cmd>`                | Run command in tmux on TPU      |
| `tpu attach v6 --session s --worker 0`         | Attach to tmux session worker 0 |
| `tpu tmux-ls v6`                               | List tmux sessions              |
| `tpu tail v6 --worker 0`                       | Tail last log for worker 0      |
| `tpu tmux-kill-all v6`                         | Kill all tmux sessions          |
| `tpu kill-jax v6`                              | Kill all JAX processes          |
| `tpu clean-tmp v6`                             | Clean `/tmp` on TPU             |
| `tpu nuke v6`                                  | Kill tmux, JAX, and clean tmp   |

(You can use `tpu-tools` instead of `tpu` if you prefer.)

---

## Development

### Extending for Different Training Frameworks

This tool is currently configured for OpenPI/OpenPI-CoT training, but can be easily adapted for other frameworks. **The only file you need to modify is `watch.py`** - it contains all the training-specific logic.

#### Key areas to customize in `watch.py`:

1. **Environment setup template** (lines ~120-182):
   - Modify environment variables (WANDB_API_KEY, data paths, etc.)
   - Change repository URL and clone commands
   - Update dependency installation (uv/conda/pip)

2. **Training command** (lines ~199-207):
   - Replace the `uv run scripts/train.py` command
   - Update target names (currently `pi_droid_cot_v4/v5/v6`)
   - Modify environment variables like `XLA_PYTHON_CLIENT_MEM_FRACTION`

3. **Configuration variables**:
   - Repository name, owner, and structure
   - Training script paths and arguments
   - Data storage locations

#### Example customization for a different framework:

```python
# In watch.py, replace the training command section:
train_cmd = (
    "source ~/.zshrc && cd my-training-repo && "
    "git pull origin main && "
    "python train.py --config my_config.yaml " + extra
)
```

All other components (TPU management, SSH handling, tmux operations) work generically across frameworks.

---

## Package Structure

```
openpi-tpu-tools/
  pyproject.toml      # Console scripts: tpu, tpu-tools
  src/openpi_tpu_tools/
    config.py         # Env loader for v4/v5/v6
    ssh.py            # gcloud SSH wrapper w/ timeouts
    tpu.py            # List/delete/tmux/kill/nuke helpers
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
# or
tpu-tools --help
```