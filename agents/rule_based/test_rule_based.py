import sys
import os
import numpy as np

# Test run in RL-AD folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from highway_env.vehicle.kinematics import Vehicle
from agents.rule_based.agent import RuleBasedAgent
from gymnasium import ObservationWrapper, Wrapper



class NoisyObservationWrapper(ObservationWrapper):
    def __init__(self, env, speed_std=0.1, dist_std=0.1):
        super().__init__(env)
        self.speed_std = speed_std
        self.dist_std = dist_std

    def observation(self, observation):
        # Add noise to positions (distance) and velocities (speed) of each vehicle
        scale = [0, 0.01, 0.05, 0.03, 0, 0, 0]

        noise = self.env.unwrapped.np_random.normal(
            loc=0.0, 
            scale=scale, 
            size=observation.shape
        )
        return (observation + noise).astype(np.float32)

def make_configure_env(**kwargs):
    env = gym.make(kwargs["id"], config=kwargs["config"])
    # env = CustomRewardWrapper(env)
    env = NoisyObservationWrapper(env, speed_std=0, dist_std=0)
    env.reset()
    return env


env_name = "highway-v0"

config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "normalize": False,
        "absolute": True,
        "order": "sorted"
    },
    "duration": 60,
    "lane_count": 4,
    "simulation_frequency": 15,
    "policy_frequency": 2,
    "vehicle_density": 1,
    "reward_speed_range": [20, 30],
    "vehicles_count": 25
}


if __name__ == "__main__":
    env = gym.make("highway-v0", render_mode="rgb_array", config=config)

    # Record videos 
    env = RecordVideo(
        env,
        video_folder="agents/rule_based/videos",
        name_prefix="rule_based",
        episode_trigger=lambda x: True 
    )

    # Track statistics for every episode
    env = RecordEpisodeStatistics(env)

    agent = RuleBasedAgent(env, target_speed=30)

    obs, info = env.reset()
    done = truncated = False
    total_reward = 0

    # Run a single episode
    while not (done or truncated):
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

    print(f"Episode finished. Total reward: {total_reward}")
    env.close()