import gymnasium as gym
from gymnasium import ObservationWrapper, Wrapper
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle
import numpy as np
import torch
import os

class CustomIDMVehicle(IDMVehicle):
    @classmethod
    def create_random(cls, road, speed=None, lane_from=None, lane_to=None, lane_id=None, spacing=1):
        # Override to use Gaussian distribution for speed instead of Uniform
        if speed is None:
            speed = road.np_random.normal(loc=25.0, scale=10.0)
            speed = np.clip(speed, 10.0, 45.0)
        return super().create_random(road, speed, lane_from, lane_to, lane_id, spacing)

class NoisyObservationWrapper(ObservationWrapper):
    def __init__(self, env, speed_std=0.1, dist_std=0.1):
        super().__init__(env)
        self.speed_std = speed_std
        self.dist_std = dist_std

    def observation(self, observation):
        # Add noise to positions (distance) and velocities (speed) of each vehicle
        if observation.shape[1] == 5:
            # Assuming features: [presence, x, y, vx, vy]
            scale = [0, self.dist_std, self.dist_std, self.speed_std, self.speed_std]
        else:
            # Assuming features: [x, y, vx, vy]
            scale = [self.dist_std, self.dist_std, self.speed_std, self.speed_std]

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


class SafetyRewardWrapper(Wrapper):
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        if info['crashed']:
            return obs, 0, done, truncated, info    # reward as 0 on crash TODO
        
        ego_y = obs[0][2]       # lane
        for i in range(1, 6):
            other_x = obs[i][1]  # position of other vehicle
            other_y = obs[i][2]  # lane of other vehicle
            other_vx = obs[i][3] # speed of other vehicle
            # one vehicle length from the front vehicle in the same lane
            if abs(ego_y - other_y) < 0.03 and abs(other_x) < 0.05:
                reward = 0
                break
            # one vehicle length from the front vehicle during lane changing
            elif abs(ego_y - other_y) < 0.125 and abs(other_x) < 0.05 and other_vx < -0.125:
                reward = 0
                break
            # two vehicle length from the front vehicle during lane changing
            elif abs(ego_y - other_y) < 0.125 and abs(other_x) < 0.075 and other_vx < -0.13:
                reward *= 0.5
                break

        return obs, reward, done, truncated, info
