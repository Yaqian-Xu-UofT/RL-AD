import gymnasium as gym
import highway_env
import os
from stable_baselines3 import SAC

def train():
    # 1. Configuration and Environment Setup
    config = {
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True
        },
        "duration": 40,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "normalize": True,
            "absolute": False
        },
        "offroad_terminal": True
    }

    # 2. Create Environment
    env = gym.make("highway-v0", render_mode=None, config=config)   # TODO

    # 3. Initialize SB3 SAC Agent
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,      # Training information
        batch_size=256,
        ent_coef="auto",    # Automatically tune entropy (exploration)
        buffer_size=50000,
        # learning_starts=1000,  # number of steps before learning starts
        train_freq=1,   # every steps we do a training step
        gradient_steps=1, # how many gradient steps to do after each rollout 
        tau=0.005,          # target smoothing coefficient
        gamma=0.99,         # discount factor TODO 0.8?
        learning_rate=3e-4,
    )
    print("Starting Training with SB3 SAC Agent...")

    # 4. Training Loop
    model.learn(total_timesteps=50000, progress_bar=True)   # SAC typically needs 50k-100k steps to converge 

    # 5. Save the model
    save_path = os.path.join(os.environ.get("CKPTDIR", "."), "sac_sb3_highway_final.zip")
    model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train()