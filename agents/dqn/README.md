# DQN Agent for RL-AD

This directory contains the DQN (Deep Q-Network) agent implementation used in the RL-AD project.  
It includes training scripts with different configurations, environment wrappers, and evaluation tools.

---

## Folder Structure Overview

```
agents/dqn/
│
├── sb3_dqn_train.py  # Basic DQN training script
├── sb3_dqn_config2_train.py  # DQN training with noisy observations
├── sb3_dqn_config3_train.py  # DQN training with custom reward function
├── sb3_dqn_run.py  # Script to run and test trained models
├── custom_env_test.py  # Environment wrappers and testing utilities
│
├── results/models/  # Trained DQN models (output directory)
└── results/logs/  # Training logs (output directory)
```

---

## Training a DQN Model

### Basic DQN Training (Config 1)

Train a standard DQN agent with default configuration:

```bash
python sb3_dqn_train.py
```

This trains a model with:
- 2-layer MLP policy (256x256)
- 200,000 training steps
- Buffer size: 15,000
- Learning rate: 5e-4

### DQN with Noisy Observations (Config 2)

Train a DQN agent with noisy observations to simulate sensor uncertainty:

```bash
python sb3_dqn_config2_train.py
```

This adds Gaussian noise to position and velocity observations using the `NoisyObservationWrapper`.

### DQN with Custom Reward (Config 3)

Train a DQN agent with a custom reward function that penalizes unsafe following distances:

```bash
python sb3_dqn_config3_train.py
```

This configuration applies a `CustomRewardWrapper` that modifies rewards based on:
- Distance to front vehicles
- Relative speed differences
- Lane-changing safety margins

---

## Training Output

During training, these outputs will be generated:

- **`results/models/`**  
  Stores trained DQN models (e.g., `dqn_default_2e5/`, `dqn_config2_2e5/`, `dqn_config3_2e5/`)

- **`results/logs/`**  
  Contains TensorBoard logs for monitoring training progress

---

## Environment Wrappers

The `custom_env_test.py` file provides experiment with DQN on different environment wrappers that maybe used:

- **`NoisyObservationWrapper`**  
  Adds Gaussian noise to observations to simulate sensor uncertainty

- **`CustomRewardWrapper`** (commented out)  
  Custom reward shaping for safer driving behavior

---

## Running a Trained Model

```bash
python sb3_dqn_run.py
```

The model path in the script needs to point to the trained model location.

---

## Model Architecture

All DQN models use:
- **Policy:** MLP (Multi-Layer Perceptron)
- **Network architecture:** [256, 256]
- **Optimizer:** Adam
- **Gamma (discount factor):** 0.8
- **Target network update interval:** 50 steps

---

