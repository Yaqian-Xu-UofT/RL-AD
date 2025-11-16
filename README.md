# RL-AD

RL experiments on highway-env with multiple agent implementations.

## Directory Structure

```
RL-AD/
├── agents/
│   ├── rule_based/    # Rule-based agent implementations
│   ├── dqn/           # DQN agent
│   ├── ppo/           # PPO agent
│   └── sac/           # SAC agent
├── utils/             # Shared utilities (env wrappers, metrics, logging)
├── configs/           # Experiment configurations
└── results/           # Put your training outputs here (gitignored)
```

Each agent directory is independent to avoid merge conflicts when multiple people work on different agents.

## Setup with uv

We use `uv` for dependency management (like `pip` but faster and with better dependency resolution).

**Install dependencies** (like `pip install -r requirements.txt`):
```bash
uv sync
```
This creates a `.venv/` and installs all packages from `pyproject.toml`.

**Add new library** (like `pip install package-name`):
```bash
uv add package-name
```
This installs the package AND automatically updates `pyproject.toml` and `uv.lock`.

**Run scripts:**
```bash
uv run python agents/rule_based/train.py    # Run directly with uv
# or activate the virtual environment first
source .venv/bin/activate
python agents/rule_based/train.py
```

**Resolve conflicts:**
- If `uv.lock` conflicts: `uv lock --upgrade` to regenerate
- If dependency errors: `uv sync --refresh` to clear cache and re-sync

