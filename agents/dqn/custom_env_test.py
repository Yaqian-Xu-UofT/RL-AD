import gymnasium as gym
import numpy as np
import torch
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle
from gymnasium import ObservationWrapper, Wrapper
from stable_baselines3 import DQN


# # 1. Modify Vehicle Speed Distribution
# # Reference: https://highway-env.farama.org/dynamics/vehicle/behavior/
# class CustomVehicle(IDMVehicle):
#     @classmethod
#     def create_random(cls, road, speed=None, lane_from=None, lane_to=None, lane_id=None, spacing=1):
#         # Override to use Gaussian distribution for speed instead of Uniform
#         if speed is None:
#             # Example: Mean 25, Std 5, clipped between 15 and 35
#             speed = road.np_random.normal(loc=25.0, scale=10.0)
#             speed = np.clip(speed, 10.0, 45.0)
            
#         return super().create_random(road, speed, lane_from, lane_to, lane_id, spacing)

# 2. Add Gaussian Noise to Observation
# Reference: https://gymnasium.farama.org/api/wrappers/observation_wrappers/
class NoisyObservationWrapper(ObservationWrapper):
    def __init__(self, env, speed_std=0.1, dist_std=0.1):
        super().__init__(env)
        self.speed_std = speed_std
        self.dist_std = dist_std

    def observation(self, observation):
        # Add noise to positions (distance) and velocities (speed) of each vehicle
        if observation.shape[1] == 5:
            # Assuming features: [presence, x, y, vx, vy]
            scale = [0, 0.01, 0.05, 0.03, 0]
        else:
            # Assuming features: [x, y, vx, vy]
            scale = [0.01, 0.05, 0.03, 0]

        noise = self.env.unwrapped.np_random.normal(
            loc=0.0, 
            scale=scale, 
            size=observation.shape
        )
        return (observation + noise).astype(np.float32)


# # 3. Modify Reward Calculation
# class CustomRewardWrapper(Wrapper):
#     def step(self, action):
#         obs, reward, done, truncated, info = self.env.step(action)
        
#         # Access the vehicle via unwrapped env
#         vehicle = self.env.unwrapped.vehicle
#         current_speed = vehicle.speed
#         is_crashed = vehicle.crashed
        
#         new_reward = 0
        
#         # 1. Modified speed reward
#         if current_speed > 28:
#             new_reward += 1.0
#         elif current_speed > 20:
#             new_reward += 0.5
            
#         # 2. Collision penalty (heavier)
#         if is_crashed:
#             new_reward -= 5.0
            
#         # 3. Lane change penalty
#         # 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT
#         if action == 0 or action == 2:
#             new_reward -= 0.2
            
#         return obs, new_reward, done, truncated, info

def main():
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
    env = gym.make('highway-v0', render_mode='human', config=config)
    
    # Apply Wrappers
    env = NoisyObservationWrapper(env, speed_std=5, dist_std=5)
    # env = CustomRewardWrapper(env)
    
    print("Environment created with CustomVehicle, NoisyObservation, and CustomReward.")

    # # Detect device
    # device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # print(f"Using device: {device}")

    model = DQN.load("results/models/dqn/")
    
    # Test Run
    obs, info = env.reset()
    done = truncated = False
    total_reward = 0
    steps = 0
    
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        # action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
    print(f"Test run finished. Steps: {steps}, Total Reward: {total_reward}")
    print(f"Observation shape: {obs.shape}")
    print(f"Sample observation (row 0 - Ego): {obs[0]}")

if __name__ == "__main__":
    main()

