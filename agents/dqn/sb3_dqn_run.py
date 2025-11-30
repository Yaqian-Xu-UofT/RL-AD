import gymnasium
import highway_env
import torch
from stable_baselines3 import DQN

# # Detect device
# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# print(f"Using device: {device}")


config = {"duration": 100,}
env = gymnasium.make("highway-v0", render_mode="human", config=config)

# Load and test saved model
model = DQN.load(
  "results/models/dqn_default/",
  # device=device,
)
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
