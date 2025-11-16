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

**Sync dependencies:**
```bash
uv sync
```

**Add new library:**
```bash
uv add package-name
```

**Run scripts:**
```bash
uv run python agents/rule_based/train.py
# or activate venv
source .venv/bin/activate
python agents/rule_based/train.py
```

**Resolve conflicts:**
- If `uv.lock` conflicts: `uv lock --upgrade` to regenerate
- If dependency errors: `uv sync --refresh` to clear cache and re-sync

