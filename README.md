# RL-AD

RL experiments on highway-env with multiple agent implementations.

## Directory Structure

```
RL-AD/
├── agents/
│   ├── rule_based/    # Rule-based agent implementation
│   ├── dqn/           # DQN agent
│   ├── ppo/           # PPO agent
│   └── sac/           # SAC agent
├── demo/              # Quick demo run for each type of agent
├── eval_results/      # Output diagrams from evaluatiom runs
└── results/           # Training outputs (e.g. DRL models, tensorboard logs)
```

## Setup with uv

We use `uv` for dependency management (like `pip` but faster and with better dependency resolution).

**Install dependencies** (like `pip install -r requirements.txt`):
```bash
uv sync
```
This creates a `.venv/` and installs all packages from `pyproject.toml`.

## Agent Documentation

Each agent implementation has its own detailed README with training and configuration information:

- [Rule-Based Agent](agents/rule_based/readme.md)
- [DQN Agent](agents/dqn/README.md)
- [PPO Agent](agents/ppo/README.md)
- [SAC Agent](agents/sac/readme.md)

## Running Demos

To run quick demonstrations of trained agents, use `demo/demo_script.py`:

```bash
source .venv/bin/activate
python demo/demo_script.py
```

Edit the `__main__` block to select which agent to demo (uncomment the desired function).

Each demo function:
- Loads a trained agent model
- Records 3 episodes as videos
- Saves videos to `demo/{agent_type}_demo_videos/`

## Running Evaluations

To evaluate and compare agent performance, use `test_evaluation_framework.py`:

```bash
source .venv/bin/activate
python test_evaluation_framework.py
```

Edit the `__main__` block to select which evaluation to run:

```python
if __name__ == "__main__":
    eval_rule_based()                    # Evaluate rule-based agent
    # eval_rule_based_and_dqn()          # Compare rule-based vs DQN
    # eval_rule_based_and_ppo()          # Compare rule-based vs PPO
    # eval_rule_based_and_sac()          # Compare rule-based vs SAC
```

Evaluation results:
- Metrics are printed to console (collision rate, avg speed, etc.)
- Plots are saved to `eval_results/`

