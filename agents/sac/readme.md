# 🚗 SAC for Highway-Env (Autonomous Driving)

This folder contains the final implementation of Soft Actor-Critic (SAC) agents for the `highway-env` autonomous driving simulation. It includes training scripts, evaluation frameworks, and job submission scripts for server environments.

## 📂 File Structure

### Training Scripts
* `train_sb3_ori.py`: Baseline SAC implementation (Original).
* `train_sb3_penalty.py`: SAC implementation with penalty-based reward shaping.
* `train_sb3_noise_penalty.py`: Combined implementation using both noise injection and penalties.
* `custom.py`: Custom environment wrappers and feature extractors.

### Evaluation Frameworks
> **Important Note:** The following test scripts were copied from the project root (`../../`). Ensure you have the correct directory context when running them.

* `test_evaluation_framework_ori.py`: Evaluates the Original model.
* `test_evaluation_framework_noise.py`: Evaluates the Noise-based model.
* `test_evaluation_framework_noise_penalty.py`: Evaluates the Noise + Penalty model.

### Server & Job Scripts
* `job_train_sb3_ori.sh`: Slurm/Bash script for training the Original agent.
* `job_train_sb3_noise.sh`: Slurm/Bash script for training the Noise agent.
* `job_train_sb3_noise_penalty.sh`: Slurm/Bash script for training the Noise + Penalty agent.

### Directories
* `ckpt/`: Stores model checkpoints (`.zip`) and PyTorch weights (`.pth`).
* `eval_results/`: Stores output plots (e.g., `agent_comparison_*.png`) and metrics.

---

## 🚀 Usage

### 1. Training
To train an agent locally, run the corresponding python script:

```bash
# Train the baseline original agent
python train_sb3_ori.py 
# or use uv
uv run train_sb3_ori.py


# Train the penalty-based agent
python train_sb3_penalty.py

# Train under noisy environment
python train_sb3_noise_penalty.py
```

To submit a training job on the server (using Slurm/Bash):
```bash
sh job_train_sb3_ori.sh
# or if using Slurm
sbatch job_train_sb3_ori.sh 
```

### 2. Evaluation
To test a trained agent and generate evaluation metrics:
```bash
python test_evaluation_framework_ori.py
```
_Ensure that the corresponding checkpoints exist in the `ckpt/` folder before running evaluation_

## 📊 Results
Training and evaluation results, including comparison plots between the different agent variations (Original vs. Noise vs. Penalty), are saved in the `eval_results/` directory.

## ⚙️ Dependencies 
> **Important Note:** Consistent with pyproject.toml of this repository.
- Python 3.x
- Stable-Baselines3
- Highway-Env
- PyTorch
- Gymnasium / Gym

## ⚠️ Important Notes
- Path Dependencies: The testing scripts (`test_xxx`) were copied from a parent directory (`../../`). If you encounter `ImportError` regarding modules not found, check that your `PYTHONPATH` includes the project root or that relative imports inside `custom.py` and the test scripts align with the current directory structure.

 - Checkpoints: The `ckpt/` folder contains zipped Stable-Baselines3 models. Do not rename the internal files if you plan to load them manually using `torch.load`; allow SB3 to handle the loading via `.load()`.

