import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from agents.rule_based import RuleBasedAgent


env_name = "highway-v0"

# # Set up logging for episode statistics
# logging.basicConfig(level=logging.INFO, format='%(message)s')

# Create environment with periodic video recording

config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "normalize": False,
        "absolute": False,
        "order": "sorted"
    }
}

env = gym.make("highway-v0", render_mode="rgb_array", config=config)


# Record videos 
env = RecordVideo(
    env,
    video_folder="initial_demo/highway-v0",
    name_prefix="rule_based",
    episode_trigger=lambda x: True 
)

# Track statistics for every episode (lightweight)
env = RecordEpisodeStatistics(env)


agent = RuleBasedAgent(env)

obs, info = env.reset()
done = truncated = False
total_reward = 0
while not (done or truncated):
    action = agent.act(obs)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

print(f"Episode finished. Total reward: {total_reward}")
env.close()