import gymnasium as gym
import numpy as np
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
    # Configuration from initial_demo/test_rule_based.py

    
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


def eval_two_agents():
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

if __name__ == "__main__":
    # eval_rule_based()
    eval_two_agents()
