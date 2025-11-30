import gymnasium as gym
import numpy as np
import highway_env
from highway_env.vehicle.kinematics import Vehicle

from agents.rule_based.agent import RuleBasedAgent
from agents.evaluation import EvaluationManager

config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 25,
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
    "reward_speed_range": [20, 30]
}

def eval_rule_based():
    env = gym.make("highway-v0", render_mode="human", config=config)
    
    agent = RuleBasedAgent(env)
    eval_manager = EvaluationManager(save_dir="eval_results")
    num_episodes = 2

    print("Starting evaluation...")
    metrics = eval_manager.evaluate_agent(agent=agent, env=env, num_episodes=num_episodes)
    eval_manager.print_results(metrics)
            
    # Plot metrics
    print("\nPlotting metrics...")
    eval_manager.plot_metrics(metrics, title="Rule Based Agent Evaluation")
    
    env.close()


def eval_two_rule_based_agents():
    env = gym.make("highway-v0", render_mode="human", config=config)
    
    agent1 = RuleBasedAgent(env, target_speed=28)
    agent2 = RuleBasedAgent(env, target_speed=30)
    
    eval_manager = EvaluationManager(save_dir="eval_results")
    
    num_episodes = 2

    # Create dictionary of agents
    agents = {
        "Rule Based (28m/s)": agent1,
        "Rule Based (30m/s)": agent2
    }
    
    # Run comparison
    print("Starting comparison of agents...")
    results = eval_manager.compare_agents(agents, env, num_episodes=num_episodes)
    
    for name, metrics in results.items():
        print(f"\n--- Results for {name} ---")
        eval_manager.print_results(metrics)
    
    env.close()

def eval_rule_based_and_sac():
    from stable_baselines3 import SAC
    import os
    
    # # Common scenario parameters
    # common_config = {
    #     "lanes_count": 4,
    #     "duration": 60,
    #     "vehicle_density": 1,
    #     "offroad_terminal": True
    # }

    eval_manager = EvaluationManager(save_dir="eval_results")
    num_episodes = 2

    # Setup SAC Agent
    print("\n--- Setting up SAC Agent ---")
    sac_config = {
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
            "steering_range": [-np.deg2rad(15), np.deg2rad(15)],
            "dynamical": True,
            "clip": True
        },
        "lanes_count": 4,
        "duration": 60,  # longer episode for overtaking
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,   # observe 15 vehicles around ego
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "normalize": True,
            "absolute": False
        },
        "lane_change_reward": 1,  # reward for successful lane changes
        "collision_reward": -0.1,   # TODO tune back to -1.0 after testing
        "right_lane_reward": 0.0,   # encourage lane changing
        "high_speed_reward": 2.5,
        "reward_speed_range": [30, 35], # default [20, 30], ie [72, 108] km/h
        "vehicle_density": 1, # denser traffic for overtaking
        "offroad_terminal": True,
        "normalize_reward": False
    }
    
    env_sac = gym.make("highway-v0", render_mode="human", config=sac_config)
    
    # Load model
    model_path = "agents/sac/ckpt/sac_sb3_108132"
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        return

    agent_sac = SAC.load(model_path, env=env_sac)
    
    # Setup Rule Based Agent
    print("\n--- Setting up Rule Based Agent ---")
    rb_config = config.copy()
    
    env_rb = gym.make("highway-v0", render_mode="human", config=rb_config)
    agent_rb = RuleBasedAgent(env_rb, target_speed=30) 
    
    # Compare using compare_agents with (agent, env) tuples
    print("\n--- Comparing Agents ---")
    
    agents_dict = {
        "SAC Agent": (agent_sac, env_sac),
        "Rule Based Agent": (agent_rb, env_rb)
    }
    
    # Pass None for env since each agent has its own env in the tuple
    results = eval_manager.compare_agents(agents_dict=agents_dict, env=None, num_episodes=num_episodes)
    
    for name, metrics in results.items():
        print(f"\n--- Results for {name} ---")
        eval_manager.print_results(metrics)
        
    env_sac.close()
    env_rb.close()

if __name__ == "__main__":
    # eval_rule_based()
    # eval_two_rule_based_agents()
    eval_rule_based_and_sac()
