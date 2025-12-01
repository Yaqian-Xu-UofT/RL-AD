# Updated Nov-30
# Base config-1: train_sb3_multicore.py
import gymnasium as gym
import highway_env
import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

model_name = "sac_sb3_ori"
tensorboard_log_dir = os.environ.get("LOGDIR", ".")
def train():
    # 1. Configuration and Environment Setup
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
            "steering_range": [-np.deg2rad(10), np.deg2rad(10)],
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
    

        "lane_change_reward": 1,
        "collision_reward": -0.1,
        "right_lane_reward": 0.0,
        "high_speed_reward": 2.5,
        "reward_speed_range": [25, 30],
        "offroad_terminal": True,
        "normalize_reward": False    # TODO
    }
    print("Env Config", config)
    # 2. Create Vectorized Environment (Multi-core Setup)
    n_cpu = 20
    
    print(f"Creating {n_cpu} parallel environments...")
    
    # make_vec_env replacing gym.make
    env = make_vec_env(
        env_id="highway-v0",
        n_envs=n_cpu,
        seed=0,
        vec_env_cls=SubprocVecEnv, 
        env_kwargs={
            "render_mode": None, 
            "config": config     
        }
    )

    # 3. Initialize SB3 SAC Agent
    model = SAC(
        "MlpPolicy",
        env,
        device="cuda",
        verbose=1,      # Training information
        batch_size=1024,
        ent_coef="auto",    # Automatically tune entropy (exploration)
        buffer_size=1_000_000,    # buffer more transitions for better learning
        learning_starts=200_000,  # collect more transitions with random policy before learning
        train_freq=64,   # every steps we do a training step
        gradient_steps=64, # how many gradient steps to do after each rollout 
        tau=0.005,          # target smoothing coefficient
        gamma=0.99,         # discount factor
        learning_rate=3e-4,
        tensorboard_log=tensorboard_log_dir
    )
    # model.load("/home/yqxu/links/scratch/RL-AD/107283/checkpoints/sac_sb3_600000_steps.zip")
    print("Starting Training with SB3 SAC Agent (Multi-core)...")


    # Checkpoint callback to save the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=200_000 // n_cpu,  # Save every 200,000 steps divided by number of CPUs
        save_path=os.environ.get("CKPTDIR", "."),
        name_prefix=model_name
    )

    # 4. Training Loop
    # total_timesteps is the total number of steps to train across all environments
    model.learn(total_timesteps=1_600_000, progress_bar=True, callback=checkpoint_callback)   

    # 5. Save the model
    final_save_path = os.path.join(os.environ.get("CKPTDIR", "."), model_name + ".zip")
    model.save(final_save_path)
    print(f"Model saved to {final_save_path}")
    
    env.close()


if __name__ == "__main__":
    train()