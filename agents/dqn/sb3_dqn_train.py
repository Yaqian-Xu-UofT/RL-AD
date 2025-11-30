import gymnasium
import highway_env
import torch
from stable_baselines3 import DQN

# # Detect device
# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# print(f"Using device: {device}")


config = {
  "lanes_count": 4,
  "vehicles_density": 1,
  "duration": 60,
  "vehicles_count": 25,
  "policy_frequency": 2,
  "simulation_frequency": 15,
}

env = gymnasium.make("highway-v0", config=config)

model = DQN(
  'MlpPolicy', 
  env,
  policy_kwargs=dict(net_arch=[256, 256]),
  learning_rate=5e-4,
  buffer_size=15000,
  learning_starts=200,
  batch_size=32,
  gamma=0.8,
  train_freq=1,
  gradient_steps=1,
  target_update_interval=50,
  verbose=1,
  tensorboard_log="results/logs/dqn_final/",
  # device=device
)
model.learn(int(1e5))
model.save("results/models/dqn_default/")
