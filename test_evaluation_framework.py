import gymnasium as gym
import numpy as np
import highway_env
from highway_env.vehicle.kinematics import Vehicle

from agents.rule_based.agent import RuleBasedAgent
from agents.evaluation import EvaluationManager

from agents.custom.custom import NoisyObservationWrapper, SafetyRewardWrapper

# Rule-Based Agent Evaluation Configuration
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

def eval_rule_based(lccd=3, trt=1.2):
    env = gym.make("highway-v0", render_mode="human", config=config)
    
    agent = RuleBasedAgent(env, lccd=lccd, trt=trt)
    eval_manager = EvaluationManager(save_dir="eval_results")
    num_episodes = 100

    print("Starting evaluation...")
    print(f"\n=== Eval RB w. lccd={lccd} and trt={trt} ===")
    metrics = eval_manager.evaluate_agent(agent=agent, env=env, num_episodes=num_episodes)
    eval_manager.print_results(metrics)
            
    # Plot metrics
    print("\nPlotting metrics...")
    eval_manager.plot_metrics(metrics, title="Rule Based Agent Evaluation")
    
    env.close()

def eval_rule_based_w_noise(lccd=3, trt=1.2):
    env = gym.make("highway-v0", render_mode="human", config=config)
    env = NoisyObservationWrapper(env, normalize=False)
    
    agent = RuleBasedAgent(env, lccd=lccd, trt=trt)
    eval_manager = EvaluationManager(save_dir="eval_results")
    num_episodes = 100

    print("Starting evaluation with noisy observations...")
    print(f"\n=== Eval RB w. lccd={lccd} and trt={trt} ===")
    metrics = eval_manager.evaluate_agent(agent=agent, env=env, num_episodes=num_episodes)
    eval_manager.print_results(metrics)
            
    # Plot metrics
    print("\nPlotting metrics...")
    eval_manager.plot_metrics(metrics, title="Rule Based Agent Evaluation with Noise")
    
    env.close()

def eval_rule_based_w_custom_reward(lccd=3, trt=1.2):
    env = gym.make("highway-v0", render_mode="human", config=config)
    env = SafetyRewardWrapper(env, mid_lane_reward=False)
    
    agent = RuleBasedAgent(env, lccd=lccd, trt=trt)
    eval_manager = EvaluationManager(save_dir="eval_results")
    num_episodes = 20

    print("Starting evaluation with custom reward...")
    print(f"\n=== Eval RB w. lccd={lccd} and trt={trt} ===")
    metrics = eval_manager.evaluate_agent(agent=agent, env=env, num_episodes=num_episodes)
    eval_manager.print_results(metrics)
            
    # Plot metrics
    print("\nPlotting metrics...")
    eval_manager.plot_metrics(metrics, title="Rule Based Agent Evaluation with Noise")
    
    env.close()

def eval_rule_based_w_noise_custom_reward(lccd=3, trt=1.2):
    env = gym.make("highway-v0", render_mode="human", config=config)
    env = NoisyObservationWrapper(env, normalize=False)
    env = SafetyRewardWrapper(env, mid_lane_reward=False)
    
    agent = RuleBasedAgent(env, lccd=lccd, trt=trt)
    eval_manager = EvaluationManager(save_dir="eval_results")
    num_episodes = 20

    print("Starting evaluation with noisy observations and custom reward...")
    print(f"\n=== Eval RB w. lccd={lccd} and trt={trt} ===")
    metrics = eval_manager.evaluate_agent(agent=agent, env=env, num_episodes=num_episodes)
    eval_manager.print_results(metrics)
            
    # Plot metrics
    print("\nPlotting metrics...")
    eval_manager.plot_metrics(metrics, title="Rule Based Agent Evaluation with Noise")
    
    env.close()

def eval_two_rule_based_agents():
    # Example configurations for two different rule-based agents
    # Target speed set to 28 m/s and 30 m/s respectively
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

    eval_manager = EvaluationManager(save_dir="eval_results")
    num_episodes = 2

    # Setup SAC Agent
    print("\n--- Setting up SAC Agent ---")
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
        # "other_vehicles_type": "custom.CustomIDMVehicle",
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

def eval_rule_based_and_dqn():
    from stable_baselines3 import DQN
    import os

    eval_manager = EvaluationManager(save_dir="eval_results/dqn")
    num_episodes = 100

    # Setup PPO Agent
    print("\n--- Setting up DQN Agent ---")
    dqn_config = {
        "lanes_count": 4,
        "duration": 60,  # longer episode for overtaking
        "observation": {
            "type": "Kinematics",
            # "vehicles_count": 10,   # observe 15 vehicles around ego
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "order": "sorted"
        },
        "policy_frequency": 2,
        "vehicles_count": 25,
    }
    
    env_dqn = gym.make("highway-v0", render_mode="rgb_array", config=dqn_config)
    
    # Load model
    model_path = "results/models/dqn_default"
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        return

    agent_dqn = DQN.load(model_path, env=env_dqn)
    
    # Setup Rule Based Agent
    print("\n--- Setting up Rule Based Agent ---")
    rb_config = config.copy()
    
    env_rb = gym.make("highway-v0", render_mode="rgb_array", config=rb_config)
    agent_rb = RuleBasedAgent(env_rb, target_speed=29.99999) 
    
    # Compare using compare_agents with (agent, env) tuples
    print("\n--- Comparing Agents ---")
    
    agents_dict = {
        "DQN Agent": (agent_dqn, env_dqn),
        "Rule Based Agent": (agent_rb, env_rb)
    }
    
    # Pass None for env since each agent has its own env in the tuple
    results = eval_manager.compare_agents(agents_dict=agents_dict, env=None, num_episodes=num_episodes)
    
    for name, metrics in results.items():
        print(f"\n--- Results for {name} ---")
        eval_manager.print_results(metrics)
        
    env_dqn.close()
    env_rb.close()


if __name__ == "__main__":
    ## Evaluating Rule Based Agent Variants, DQN, PPO
    ## SAC and PPO evaluation scripts are under agents/sac and agents/ppo directories respectively.

    eval_rule_based()
    # eval_rule_based_w_noise()
    # eval_rule_based_w_custom_reward()
    # eval_rule_based_w_noise_custom_reward()
    
    # eval_two_rule_based_agents()

    # eval_rule_based_and_sac()
    # eval_rule_based_and_ppo()
    # eval_rule_based_and_dqn()

    # eval_ppo()
