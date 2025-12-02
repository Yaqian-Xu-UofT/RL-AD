import gymnasium
import highway_env
from stable_baselines3 import DQN
from gymnasium import ObservationWrapper, Wrapper



### 需要在config里加 observation："order": "sorted"
config = {
 "observation": {
    "type": "Kinematics",
    "features": ["presence", "x", "y", "vx", "vy"],
    # "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    "absolute": False,
    "order": "sorted",
    "vehicles_count": 10,
  },
  "vehicles_count": 25,
  "lanes_count": 4,
  "vehicles_density": 1,
  "duration": 60,
  "policy_frequency": 2,
  "simulation_frequency": 15,
}


class CustomRewardWrapper(Wrapper):
   def step(
       self, action
   ):
       obs, reward, done, truncated, info = self.env.step(action)

       if info['crashed']:
           return obs, 0, done, truncated, info

       for i in range(1, 6):
           other_x = obs[i][1]
           other_y = obs[i][2]
           other_vx = obs[i][3]
           # one vehicle length from the front vehicle in the same lane
           if abs(other_y) < 0.03 and abs(other_x) < 0.05:
               reward = 0
               break
           # 1.5 vehicle length from the front vehicle during lane changing
           elif abs(other_y) < 0.125 and abs(other_x) < 0.0625:
               reward *= 0.5
               break
           # two vehicle length from the front vehicle in the same lane with 10 m/s speed difference
           elif abs(other_y) < 0.03 and abs(other_x) < 0.075 and other_vx < -0.12:
               reward *= 0.5
               break

       return obs, reward, done, truncated, info


# Create Environment
env = gymnasium.make('highway-v0', config=config)

# Apply Wrappers
env = CustomRewardWrapper(env)

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
  tensorboard_log="results/logs/dqn_config3_2e5/",
  # device=device
)
model.learn(int(2e5))
model.save("results/models/dqn_config3_2e5/")
