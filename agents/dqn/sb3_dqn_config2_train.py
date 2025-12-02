import gymnasium
import highway_env
import torch
from stable_baselines3 import DQN
from custom_env_test import NoisyObservationWrapper

# # Detect device
# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# print(f"Using device: {device}")


config = {
    "observation": {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx", "vy"], # Explicitly listing features for noise wrapper
    },
    # "other_vehicles_type": "__main__.CustomVehicle", # Register custom vehicle type as string
    "lanes_count": 4,
    "vehicles_density": 1,
    "duration": 60,
    "vehicles_count": 25,
    "policy_frequency": 2,
    "simulation_frequency": 15,
}

# Create Environment
env = gymnasium.make('highway-v0', config=config)

# Apply Wrappers
env = NoisyObservationWrapper(env, speed_std=5, dist_std=5)
# env = CustomRewardWrapper(env)

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
  tensorboard_log="results/logs/dqn_config2_2e5/",
  # device=device
)
model.learn(int(2e5))
model.save("results/models/dqn_config2_2e5/")
