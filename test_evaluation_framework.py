import gymnasium as gym
import numpy as np
from highway_env.vehicle.kinematics import Vehicle

from agents.rule_based.agent import RuleBasedAgent
from agents.evaluation import EvaluationManager

def eval_rule_based():
    # Configuration from initial_demo/test_rule_based.py
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
        "reward_speed_range": [20, 30],
        # "speed_limit": 30,
        # "action": {
        #     "type": "DiscreteMetaAction",
        #     "target_speeds": np.linspace(15, 40, num=26),
        # }
    }
    
    env = gym.make("highway-v0", render_mode="human", config=config)
    
    agent = RuleBasedAgent(env)
    
    eval_manager = EvaluationManager(save_dir="eval_results")
    

    print("Starting evaluation...")
    metrics = eval_manager.evaluate_agent(agent, env, num_episodes=2)
    
    # Print results
    print("\nEvaluation Results:")
    for key, value in metrics.items():
        if key == "rewards":
            print(f"{key}: Mean={np.mean(value):.2f}, Std={np.std(value):.2f}")
        elif key == "episode_lengths":
            print(f"{key}: Mean={np.mean(value):.2f}, Std={np.std(value):.2f}")
        elif key == "average_speeds":
            print(f"{key}: Mean={np.mean(value):.2f}, Std={np.std(value):.2f}")
        elif key in ["collisions", "successes", 
                     "overtakes", "collision_rate", 
                     "success_rate", "overall_avg_reward", 
                     "overall_avg_speed"]:
            print(f"{key}: {value}")
            
    # Plot metrics
    print("\nPlotting metrics...")
    eval_manager.plot_metrics(metrics, title="Rule Based Agent Evaluation")
    
    env.close()

if __name__ == "__main__":
    eval_rule_based()
