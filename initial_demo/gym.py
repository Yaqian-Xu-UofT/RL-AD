import logging
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import highway_env


env_name = "highway-v0"

# # Set up logging for episode statistics
# logging.basicConfig(level=logging.INFO, format='%(message)s')

# Create environment with periodic video recording

env = gym.make("highway-v0", render_mode="rgb_array")


# Record videos 
env = RecordVideo(
    env,
    video_folder="highway-v0",
    name_prefix="training",
    episode_trigger=lambda x: True 
)

# Track statistics for every episode (lightweight)
env = RecordEpisodeStatistics(env)



obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

env.close()