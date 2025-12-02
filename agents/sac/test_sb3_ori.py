# 3-lane, collision_reward -0.1
import gymnasium as gym
import highway_env
from stable_baselines3 import SAC
from gymnasium.wrappers import RecordVideo
import os
import numpy as np
from stable_baselines3.common.env_util import make_vec_env


model_name = "sac_sb3_ori"
tensorboard_log_dir = os.environ.get("LOGDIR", ".")

def test():
    # job-108132
    config = {
        # common configurations
        "lanes_count": 4,
        "vehicles_density": 1,
        "vehicles_count": 25,
        "duration": 60,
        "policy_frequency": 2,
        "simulation_frequency": 15,

        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
            "steering_range": [-np.deg2rad(15), np.deg2rad(15)],
            "dynamical": True,
            "clip": True
        },
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,   # observe 15 vehicles around ego
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "normalize": True,
            "absolute": False,
            "order": "sorted"
        },
        "lane_change_reward": 1,  # reward for successful lane changes
        "collision_reward": -0.1,   # TODO tune back to -1.0 after testing
        "right_lane_reward": 0.0,   # encourage lane changing
        "high_speed_reward": 2.0,
        "reward_speed_range": [25, 30], # default [20, 30], ie [72, 108] km/h
        "offroad_terminal": True,
        "normalize_reward": True,

    }

    # 2. Setup Environment with Video
    env = make_vec_env(
        env_id="highway-v0",
        # n_envs=1,
        seed=0,
        # vec_env_cls=SubprocVecEnv, 

        env_kwargs={
            "render_mode": None, 
            "config": config     
        }
    )

    # Save video to 'video_sb3' folder
    env = RecordVideo(env, video_folder="video_sb3/", 
    name_prefix="CustomIDMVehicle", 
    episode_trigger=lambda x: True,
    disable_logger=True)

    # 3. Load the trained SB3 SAC model
    model_path = f"./ckpt/{model_name}"    # TODO change env vehicle speed
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
            print(f"truncated: {truncated}")
            print(f"done: {done}")
            print(f"info: {info}")
            print('\n')
            total_reward += reward
            env.render()  
        
        print(f"Episode {ep+1} Reward: {total_reward:.2f}")
        env.close()

    print("Test videos saved in video_sb3/ folder.")


if __name__ == "__main__":
    test()