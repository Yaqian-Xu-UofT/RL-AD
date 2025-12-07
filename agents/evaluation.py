import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, Callable, List
from matplotlib.ticker import MaxNLocator
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class EvaluationMetrics:
    rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    collisions: int = 0
    average_speeds: List[float] = field(default_factory=list)
    successes: int = 0
    collision_rate: float = 0.0
    success_rate: float = 0.0
    overall_avg_reward: float = 0.0
    overall_avg_speed: float = 0.0

    def __getitem__(self, key):
        return getattr(self, key)
    
    def items(self):
        return self.__dict__.items()

class EvaluationManager:
    def __init__(self, save_dir="results"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.num_episodes = 1

    def evaluate_agent(self, agent, env, num_episodes=10) -> EvaluationMetrics:
        metrics = EvaluationMetrics()

        self.num_episodes = num_episodes

        print(f"Starting evaluation for {num_episodes} episodes...")

        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            episode_speeds = []
            
            print(f"Episode {episode + 1} starts...")
            while not (done or truncated):
                # Handle both SB3 agents and simple rule-based agents
                if hasattr(agent, "predict"):
                    action, _ = agent.predict(obs, deterministic=True) # SB3 agent
                elif hasattr(agent, "act"):
                    action = agent.act(obs, episode=episode)  # RB agent
                else:
                    return None
                
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Check speed info
                episode_speeds.append(info.get("speed", 0))
                
                # Check for collision
                if info.get("crashed", False):
                    metrics.collisions += 1
                    print(f"Collision detected in episode {episode + 1}.")

            print(f"Episode {episode + 1} ends with reward {episode_reward} and length {episode_length}.")
                
                
            metrics.successes = num_episodes - metrics.collisions
            metrics.rewards.append(episode_reward)
            metrics.episode_lengths.append(episode_length)
            if episode_speeds:
                metrics.average_speeds.append(np.mean(episode_speeds))
            
        # Calculate rates
        metrics.collision_rate = metrics.collisions / num_episodes
        metrics.success_rate = metrics.successes / num_episodes
        metrics.overall_avg_reward = np.mean(metrics.rewards)
        metrics.overall_avg_speed = np.mean(metrics.average_speeds) if metrics.average_speeds else 0

        return metrics
    
    def print_results(self, metrics: EvaluationMetrics):
        print("\nEvaluation Results:")
        print(f"rewards: Mean={np.mean(metrics.rewards):.2f}, Std={np.std(metrics.rewards):.2f}")
        print(f"episode_lengths: Mean={np.mean(metrics.episode_lengths):.2f}, Std={np.std(metrics.episode_lengths):.2f}")
        print(f"average_speeds: Mean={np.mean(metrics.average_speeds):.2f}, Std={np.std(metrics.average_speeds):.2f}")
        print(f"collisions: {metrics.collisions}")
        print(f"successes: {metrics.successes}")
        print(f"collision_rate: {metrics.collision_rate:.2f}")
        print(f"success_rate: {metrics.success_rate:.2f}")
        print(f"overall_avg_reward: {metrics.overall_avg_reward:.2f}")
        print(f"overall_avg_speed: {metrics.overall_avg_speed:.2f}")

    def plot_metrics(self, metrics: EvaluationMetrics, title="Agent Performance"):
        # Set font to Arial and adjust size
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 15
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Rewards - Red
        axes[0].plot(range(1, self.num_episodes + 1), metrics.rewards, color='red', linewidth=2, marker='o', markersize=8)
        axes[0].set_title("Episode Rewards", fontsize=18, fontweight='bold', color='black')
        axes[0].set_xlabel("Episode", fontsize=15, fontweight='bold', color='black')
        axes[0].set_ylabel("Reward", fontsize=15, fontweight='bold', color='black')
        axes[0].set_xlim(1, self.num_episodes)
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].grid(True, linestyle='--', alpha=0.7)
        # put average reward value on the plot, with background being half transparent white box
        avg_reward = np.mean(metrics.rewards)
        axes[0].text(0.9, 0.1, 
                    f'Avg. Reward: {avg_reward:.2f}', 
                    transform=axes[0].transAxes, 
                    ha='right', va='top', 
                    fontsize=15, fontweight='bold', color='black', 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        
        # Speeds - Green
        axes[1].plot(range(1, self.num_episodes + 1), metrics.average_speeds, color='green', linewidth=2, marker='s', markersize=8)
        axes[1].set_title("Average Speed per Episode", fontsize=18, fontweight='bold', color='black')
        axes[1].set_xlabel("Episode", fontsize=15, fontweight='bold', color='black')
        axes[1].set_ylabel("Speed (m/s)", fontsize=15, fontweight='bold', color='black')
        axes[1].set_xlim(1, self.num_episodes)
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].grid(True, linestyle='--', alpha=0.7)
        # put average speed value on the plot, with background being half transparent white box
        avg_speed = np.mean(metrics.average_speeds)
        axes[1].text(0.9, 0.1, 
                    f'Avg. Speed: {avg_speed:.2f}', 
                    transform=axes[1].transAxes, 
                    ha='right', va='top', 
                    fontsize=15, fontweight='bold', color='black', 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        
        # Rates - Blue/Red
        rates = ["Collision Rate", "Success Rate"]
        values = [metrics.collision_rate, metrics.success_rate]
        # Red for collision (negative) and Blue for success (positive)
        bars = axes[2].bar(rates, values, color=['red', 'blue'], alpha=0.8, width=0.5)
        for bar in bars:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=15, fontweight='bold', color='black')
        axes[2].set_title("Performance Rates", fontsize=18, fontweight='bold', color='black')
        plt.setp(axes[2].get_xticklabels(), fontweight='bold')
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.suptitle(title, fontsize=22, fontweight='bold', color='black')
        plt.tight_layout()
        timestamp = datetime.now().strftime("%m%d_%H%M")
        save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{self.num_episodes}_runs_{timestamp}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()

    def compare_agents(self, agents_dict: Dict[str, Any], env=None, num_episodes=10):
        compare_results = {}
        
        for name, item in agents_dict.items():
            # Support (agent, env) tuple in agents_dict
            if isinstance(item, (tuple, list)) and len(item) == 2:
                agent, current_env = item
            else:
                agent = item
                current_env = env
            
            if current_env is None:
                raise ValueError(f"No environment provided for agent: {name}")

            print(f"Evaluating agent: {name}")
            metrics = self.evaluate_agent(agent, current_env, num_episodes)
            compare_results[name] = metrics
            
        self.plot_comparison(compare_results)
        return compare_results

    def plot_comparison(self, results: Dict[str, EvaluationMetrics]):
        # Set font to Arial
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 12

        agents = list(results.keys())
        metrics_map = {
            "Overall Average Reward": "overall_avg_reward",
            "Collision Rate": "collision_rate",
            "Overall Average Speed": "overall_avg_speed",
            "Completion Rate": "success_rate"
        }
        
        # Colors for agents
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        agent_colors = [colors[i % len(colors)] for i in range(len(agents))]
        
        fig, axes = plt.subplots(1, len(metrics_map), figsize=(6 * len(metrics_map), 6))
        
        for i, (title, metric_attr) in enumerate(metrics_map.items()):
            values = [getattr(results[agent], metric_attr) for agent in agents]
            bars = axes[i].bar(agents, values, color=agent_colors, alpha=0.8, width=0.6)
            axes[i].set_title(title, fontsize=18, fontweight='bold', color='black')
            if 'Reward' in title:
                avg_reward = np.mean(values)
                axes[0].text(0.9, 0.1, 
                            f'Avg. Reward: {avg_reward:.2f}', 
                            transform=axes[0].transAxes, 
                            ha='right', va='top', 
                            fontsize=15, fontweight='bold', color='black', 
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
            if 'Speed' in title:
                axes[i].set_ylabel("Speed (m/s)", fontsize=15, fontweight='bold', color='black')
                avg_speed = np.mean(values)
                axes[1].text(0.9, 0.1, 
                    f'Avg. Speed: {avg_speed:.2f}',
                    transform=axes[1].transAxes, 
                    ha='right', va='top', fontsize=15, fontweight='bold', color='black', 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
            axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)
                    

            plt.setp(axes[i].get_xticklabels(), fontweight='bold')
            
            # Value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=15, fontweight='bold', color='black')
            
        plt.tight_layout()

        timestamp = datetime.now().strftime("%m%d_%H%M")
        save_path = os.path.join(self.save_dir, f"agent_comparison_{self.num_episodes}_runs_{timestamp}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Comparison plot saved to {save_path}")
        plt.close()
