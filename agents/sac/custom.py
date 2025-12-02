import gymnasium as gym
from gymnasium import ObservationWrapper, Wrapper
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle
import numpy as np
import torch
import os

class CustomVehicle(IDMVehicle):
    @classmethod
    def create_random(cls, road, speed=None, lane_from=None, lane_to=None, lane_id=None, spacing=1):
        # Override to use Gaussian distribution for speed instead of Uniform
        if speed is None:
            # Example: Mean 25, Std 5, clipped between 15 and 35
            speed = road.np_random.normal(loc=25.0, scale=10.0)
            speed = np.clip(speed, 10.0, 45.0)
            
        return super().create_random(road, speed, lane_from, lane_to, lane_id, spacing)

class NoisyObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)


    def observation(self, observation):
        # Add noise to positions (distance) and velocities (speed) of each vehicle
        scale = [0, 0.01, 0.05, 0.03, 0, 0, 0]

        noise = self.env.unwrapped.np_random.normal(
            loc=0.0, 
            scale=scale, 
            size=observation.shape
        )
        return (observation + noise).astype(np.float32)


# 3. Modify Reward function based on speed 
class SpeedRewardWrapper(Wrapper):
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Access the vehicle via unwrapped env
        vehicle = self.env.unwrapped.vehicle
        current_speed = vehicle.speed
        is_crashed = vehicle.crashed
        
        new_reward = 0
        
        # 1. Modified speed reward
        if current_speed > 28:
            new_reward += 1.0
        elif current_speed > 20:
            new_reward += 0.5
            
        # 2. Collision penalty (heavier)
        if is_crashed:
            new_reward -= 5.0
            
        # 3. Lane change penalty
        # 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT
        if action == 0 or action == 2:
            new_reward -= 0.2
            
        return obs, new_reward, done, truncated, info

# from Benjamin's implementation
class SafetyRewardWrapper(Wrapper):
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

        ### Yaqian's 居中奖励
        ### 其他人comment掉这部分
        ego_y = obs[0][2]
        ego_cos_h = obs[0][5]
        # driving straight
        if abs(ego_cos_h - 1) < 0.05:
            # four lanes
            delta_0 = abs(ego_y - 0)
            delta_1 = abs(ego_y - 0.25)
            delta_2 = abs(ego_y - 0.5)
            delta_3 = abs(ego_y - 0.75)

            min_delta = min(delta_0, delta_1, delta_2, delta_3)
            if min_delta > 0.03:
                reward *= 0.5
            elif min_delta > 0.15:
                reward *= 0.75
        ### 其他人comment掉这部分


        return obs, reward, done, truncated, info