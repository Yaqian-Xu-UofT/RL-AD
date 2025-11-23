import gymnasium as gym
import numpy as np
import highway_env
from tqdm import tqdm
import torch
# from agents.sac.sac_agent import SACAgent
from agent import SAC, ReplayBuffer


def train():
    # 1. Setup environment with Continuous Actions
    
    config = {
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,   # enable throttle control
            "lateral": True          # enable steering control
            # acceleration_range – the range of acceleration values [m/s²]
            # steering_range – the range of steering values [rad]
            # speed_range – the range of reachable speeds [m/s]
        },
        "duration": 40,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],   # cos_h and sin_h for heading of vehicle in radians
            "normalize": True,
            "absolute": False   # Relative coordinates are easier to generalize
        },
        "offroad_terminal": True,  # terminate episode if ego vehicle goes offroad
        # "collision_terminal": True  # terminate episode on collision
    }
    # env = gym.make("highway-v0", render_mode="rgb_array", config=config)
    # env = gym.make("highway-v0", render_mode="human", config=config)
    env = gym.make("highway-v0", render_mode=None, config=config)
    env.reset()

    # 2. Initialize SAC Agent
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    # print("env.action_space:", env.action_space)  # env.action_space: Box(-1.0, 1.0, (2,), float32)
    max_action = float(env.action_space.high[0])

    agent = SAC(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(capacity=100000, state_dim=state_dim, action_dim=action_dim)

    # 3. Training Loop
    episodes = 3000     # SAC typically needs 50k-100k steps to converge (40 steps/episode*3k=120k steps)
    batch_size = 256

    for episode in tqdm(range(episodes)):
        state, _ = env.reset()
        state = state.flatten() # Flatten kinematics observation
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Select action
            action = agent.select_action(state)

            # Step
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = next_state.flatten()

            # Store in buffer
            real_done = done and not truncated
            replay_buffer.add(state, action, reward, next_state, float(real_done)) # TODO: float(done)?

            state = next_state
            episode_reward += reward

            # Train agent if buffer is sufficient
            if replay_buffer.size > batch_size:
                agent.train(replay_buffer, batch_size)

        # logging
        if episode % 100 == 0:
            print(f"Episode: {episode}, Reward: {episode_reward:.2f}")
    
    # Save the model
    torch.save(agent.actor.state_dict(), "sac_actor_highway.pth")
    env.close()
    print("Training finished")


def main():
    print("SAC Agent Main Function")
    train()


if __name__ == "__main__":
    main()