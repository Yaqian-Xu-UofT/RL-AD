import gymnasium
import highway_env
import torch
from stable_baselines3 import DQN

# Detect device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


train = False


if train:
  env = gymnasium.make("highway-fast-v0")

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
    tensorboard_log="results/logs/dqn/",
    device=device
  )
  model.learn(int(1e5))
  model.save("results/models/dqn/")
else:
  config = {"duration": 100,}
  env = gymnasium.make("highway-v0", render_mode="human", config=config)

  # Load and test saved model
  model = DQN.load("results/models/dqn/", device=device)
  while True:
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, truncated, info = env.step(action)
      env.render()
