import numpy as np
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

def demo_sac():
    from stable_baselines3 import SAC
    
    sac_config = {
        "lanes_count": 4,
        "vehicles_density": 1.0,
        "vehicles_count": 25,
        "duration": 60,
        "policy_frequency": 2,
        "simulation_frequency": 15,
        "speed_limit": 30,

        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
            "steering_range": [-np.deg2rad(10), np.deg2rad(10)],
            "speed_range": [0, 30],
            "dynamical": True,
            "clip": True
        },
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "normalize": True,
            "absolute": False,
            "order": "sorted"
        },
        "lane_change_reward": 0.8,
        "collision_reward": -2,
        "right_lane_reward": 0.0,
        "high_speed_reward": 1.2,
        "reward_speed_range": [25, 30],
        "offroad_terminal": True,
        "normalize_reward": True,
    }
    
    env_sac = gym.make("highway-v0", render_mode="rgb_array", config=sac_config)

    # Record videos 
    env_sac = RecordVideo(
        env_sac,
        video_folder="demo/sac_demo_videos",
        name_prefix="demo",
        episode_trigger=lambda x: True 
    )

    # Track statistics for every episode (lightweight)
    env_sac = RecordEpisodeStatistics(env_sac)
    
    # Load model
    model_path = "agents/sac/ckpt/sac_sb3_ori"

    agent_sac = SAC.load(model_path, env=env_sac)

    print("load succes")

    for _ in range(3):
        obs, info = env_sac.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = agent_sac.predict(obs)
            obs, reward, done, truncated, info = env_sac.step(action)

    env_sac.close()
    

def demo_rule_based(lccd=3, trt=1.2):
    import sys
    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_root)
    from agents.rule_based.agent import RuleBasedAgent

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
        "vehicles_count": 25,
    }

    env = gym.make("highway-v0", render_mode="rgb_array", config=config)
    
    # Record videos 
    env = RecordVideo(
        env,
        video_folder="demo/rulebased_demo_videos",
        name_prefix="demo",
        episode_trigger=lambda x: True 
    )

    # Track statistics for every episode (lightweight)
    env = RecordEpisodeStatistics(env)

    agent = RuleBasedAgent(env, lccd=lccd, trt=trt)

    for _ in range(3):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)

    env.close()


def demo_dqn():
    from stable_baselines3 import DQN
    config = {
            "lanes_count": 4,
            "duration": 60,  # longer episode for overtaking
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
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
        video_folder="demo/dqn_demo_videos",
        name_prefix="demo",
        episode_trigger=lambda x: True 
    )

    # Track statistics for every episode (lightweight)
    env = RecordEpisodeStatistics(env)

    # Load model
    model_path = "results/models/dqn_default"
    agent = DQN.load(model_path, env=env)
    print("load succes")

    for _ in range(3):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)

    env.close()


if __name__ == "__main__":
    # demo_ppo()
    # demo_sac()
    # demo_rule_based()
    demo_dqn()

