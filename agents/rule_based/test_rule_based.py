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



env_name = "highway-v0"

# # Set up logging for episode statistics
# logging.basicConfig(level=logging.INFO, format='%(message)s')

# Create environment with periodic video recording

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
    "policy_frequency": 3,
    "vehicle_density": 1.5, # denser traffic for overtaking
    "reward_speed_range": [20, 30], # Speed range for maximum reward (in m/s). Default is [20, 30].
    "speed_limit": 30, # Speed limit for the road (m/s). Other vehicles will drive around this speed.
    "action": {
        "type": "DiscreteMetaAction",
        # "target_speeds": np.linspace(15, 40, num=26), # The agent will choose from these target speeds (m/s).
    }
}


if __name__ == "__main__":
    # # Set max speed of vehicles in the environment (default 40 m/s)
    # Vehicle.MAX_SPEED = 40  
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

    agent = RuleBasedAgent(env, target_speed=40)

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