import logging
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import highway_env


def demo_ppo():
    from stable_baselines3 import PPO
    config = {
            "lanes_count": 4,
            "duration": 60,  # longer episode for overtaking
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 10,   # observe 15 vehicles around ego
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": False,
                "order": "sorted"
            },
            "policy_frequency": 2,
            "vehicles_count": 25,
        }

    # Create environment with periodic video recording
    env = gym.make("highway-v0", render_mode="rgb_array", config=config)

    # Record videos 
    env = RecordVideo(
        env,
        video_folder="demo/ppo_demo_videos",
        name_prefix="demo",
        episode_trigger=lambda x: True 
    )

    # Track statistics for every episode (lightweight)
    env = RecordEpisodeStatistics(env)

    # Load model
    model_path = "agents/ppo/save_models/11.29_penalty_2/model"
    agent = PPO.load(model_path, env=env)
    print("load succes")

    for _ in range(3):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)

    env.close()

if __name__ == "__main__":
    demo_ppo()