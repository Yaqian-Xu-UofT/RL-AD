import gymnasium as gym
import highway_env
from stable_baselines3 import SAC
from gymnasium.wrappers import RecordVideo
import os

def test():
    # 1. Same Config as Training
    config = {
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True
        },
        "duration": 100,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "normalize": True,
            "absolute": False
        }
    }

    # 2. Setup Environment with Video
    env = gym.make("highway-v0", render_mode="human", config=config)

    # Save video to 'video_sb3' folder
    # env = RecordVideo(env, video_folder="video_sb3/", 
    # name_prefix="sac_sb3_highway_test", 
    # episode_trigger=lambda x: True,
    # disable_logger=True)

    # 3. Load the trained SB3 SAC model
    model_path = "/home/yqxu/links/scratch/RL-AD/101477/checkpoints/sac_sb3_highway_final.zip"
    model = SAC.load(model_path, env=env)

    # 4. Run Test Loop
    episodes = 1
    for ep in range(episodes):
        obs, info = env.reset()
        done = truncated = False
        total_reward = 0

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()  
        
        print(f"Episode {ep+1} Reward: {total_reward:.2f}")
        env.close()

    print("Test videos saved in video_sb3/ folder.")


if __name__ == "__main__":
    test()