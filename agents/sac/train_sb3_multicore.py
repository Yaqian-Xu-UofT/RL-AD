import gymnasium as gym
import highway_env
import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

def train():
    # 1. Configuration and Environment Setup
    config = {
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
            "steering_range": [-np.deg2rad(15), np.deg2rad(15)],
            "dynamical": True,
            "clip": True
        },
        "lanes_count": 3,
        "duration": 60,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,   # observe 15 vehicles around ego
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "normalize": True,
            "absolute": False
        },
        "lane_change_reward": 1,
        "collision_reward": -0.1,
        "right_lane_reward": 0.0,
        "high_speed_reward": 2.5,
        "reward_speed_range": [30, 35],
        "vehicle_density": 1.5,
        "offroad_terminal": True,
        "normalize_reward": False
    }
    print("Env Config", config)
    # 2. Create Vectorized Environment (Multi-core Setup)
    # 设置并行运行的进程数 (CPU核心数)
    # 如果您的电脑有8核，通常设置为 4-8 之间
    n_cpu = 20
    
    print(f"Creating {n_cpu} parallel environments...")
    
    # 使用 make_vec_env 替代 gym.make
    env = make_vec_env(
        env_id="highway-v0",
        n_envs=n_cpu,
        seed=0,
        vec_env_cls=SubprocVecEnv, # 使用子进程进行真正的并行计算
        env_kwargs={
            "render_mode": None, # 训练时通常不需要渲染，节省资源
            "config": config     # 将您的配置传递给每个环境
        }
    )
    tensorboard_log_dir = os.path.join(os.environ.get("LOGDIR", "."), "sac_sb3_multicore_tensorboard")

    # 3. Initialize SB3 SAC Agent
    model = SAC(
        "MlpPolicy",
        env,
        device="cuda",
        verbose=1,      # Training information
        batch_size=1024,
        ent_coef="auto",    # Automatically tune entropy (exploration)
        buffer_size=1000000,    # buffer more transitions for better learning
        learning_starts=200000,  # collect more transitions with random policy before learning
        train_freq=64,   # every steps we do a training step
        gradient_steps=64, # how many gradient steps to do after each rollout 
        tau=0.005,          # target smoothing coefficient
        gamma=0.99,         # discount factor
        learning_rate=3e-4,
        tensorboard_log=tensorboard_log_dir
    )
    model.load("/home/yqxu/links/scratch/RL-AD/107283/checkpoints/sac_sb3_600000_steps.zip")
    print("Starting Training with SB3 SAC Agent (Multi-core)...")


    # Checkpoint callback to save the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=200000 // n_cpu,  # Save every 200,000 steps divided by number of CPUs
        save_path=os.environ.get("CKPTDIR", "."),
        name_prefix="sac_sb3"
    )

    # 4. Training Loop
    # 注意：在多核环境下，total_timesteps 是所有环境步数的总和
    model.learn(total_timesteps=1000000, progress_bar=True, callback=checkpoint_callback)   

    # 5. Save the model
    final_save_path = os.path.join(os.environ.get("CKPTDIR", "."), "sac_sb3.zip")
    model.save(final_save_path)
    print(f"Model saved to {final_save_path}")
    
    # 关闭环境进程
    env.close()


if __name__ == "__main__":
    train()