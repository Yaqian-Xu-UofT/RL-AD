import gymnasium as gym
import numpy as np
import highway_env
from highway_env.vehicle.kinematics import Vehicle

from agents.rule_based.agent import RuleBasedAgent
from agents.evaluation import EvaluationManager

from agents.custom.custom import NoisyObservationWrapper

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

def eval_rule_based_and_ppo():
    from stable_baselines3 import PPO
    import os

    eval_manager = EvaluationManager(save_dir="eval_results")
    num_episodes = 2

    # Setup PPO Agent
    print("\n--- Setting up PPO Agent ---")
    ppo_config = {
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
    
    env_ppo = gym.make("highway-v0", render_mode="rgb_array", config=ppo_config)
    
    # Load model
    model_path = "agents/ppo/save_models/11.29_penalty_2/model"
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        return

    agent_ppo = PPO.load(model_path, env=env_ppo)
    
    # Setup Rule Based Agent
    print("\n--- Setting up Rule Based Agent ---")
    rb_config = config.copy()
    
    env_rb = gym.make("highway-v0", render_mode="rgb_array", config=rb_config)
    agent_rb = RuleBasedAgent(env_rb, target_speed=29.99999) 
    
    # Compare using compare_agents with (agent, env) tuples
    print("\n--- Comparing Agents ---")
    
    agents_dict = {
        "PPO Agent": (agent_ppo, env_ppo),
        "Rule Based Agent": (agent_rb, env_rb)
    }
    
    # Pass None for env since each agent has its own env in the tuple
    results = eval_manager.compare_agents(agents_dict=agents_dict, env=None, num_episodes=num_episodes)
    
    for name, metrics in results.items():
        print(f"\n--- Results for {name} ---")
        eval_manager.print_results(metrics)
        
    env_ppo.close()
    env_rb.close()

def eval_ppo():
    from stable_baselines3 import PPO
    import os

    ppo_config = {
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
    
    env_ppo = gym.make("highway-v0", render_mode="rgb_array", config=ppo_config)

    # Load model
    model_path = "agents/ppo/save_models/11.29_penalty_2/model"
    # model_path = "agents/ppo/save_models/12.1_penalty_3/model"
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        return

    agent = PPO.load(model_path, env=env_ppo)
    print("load succes")

    eval_manager = EvaluationManager(save_dir="eval_results")
    num_episodes = 1 #100

    print("Starting evaluation...")
    metrics = eval_manager.evaluate_agent(agent=agent, env=env_ppo, num_episodes=num_episodes)
    eval_manager.print_results(metrics)
            
    # Plot metrics
    print("\nPlotting metrics...")
    eval_manager.plot_metrics(metrics, title="ppo Agent Evaluation")
    
    env_ppo.close()

if __name__ == "__main__":
    # eval_rule_based_and_ppo()
    eval_ppo()
