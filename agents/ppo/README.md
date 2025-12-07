# PPO Agent for RL-AD

This directory contains the PPO (Proximal Policy Optimization) agent implementation used in the RL-AD project.  
It includes training scripts, pretrained models, evaluation tools, and rollout videos.

---

## Folder Structure Overview

agents/ppo/
│
├── ppo.py # Training script
├── test_evaluation_framework.py # Evaluation utilities
│
├── save_models/ # Pretrained PPO models
│ └── 11.29_penalty_2 # Best PPO model
│
├── training_model/ # Training logs
└── videos/ # Training rollout videos


---

## Training a PPO Model

To train a new PPO agent, run:

```bash
python ppo.py
```

---

## Training Output

During training, the following outputs are generated:

- **`training_model/`**  
  Stores all training logs.

- **Training videos**  
  Saved automatically in the **`videos/`** folder for visual inspection of agent behavior.

---

## Pretrained Models

The folder **`save_models/`** contains several pretrained PPO models.

- **Best model:**  
    `11.29_penalty_2` — This is currently the best-performing model among all PPO variants tested.

You can load these models directly in your scripts.

---

## Evaluation

To evaluate a model using the framework, run the evaluation function:

```bash
python test_evaluation_framework.py
```

