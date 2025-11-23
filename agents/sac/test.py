# TODO: load sac_actor_highway.pth and run test episodes and save video
import gymnasium as gym
import numpy as np
import torch
import highway_env
from gymnasium.wrappers import RecordVideo
from agent import Actor

def get_deterministic_action(actor, state):
    """
    For Testing, we want the best possible action (deterministic),
    not a random exploration action.
    """
    state_tensor = torch.FloatTensor(state.reshape(1, -1))

    with torch.no_grad():
        # Get mean action from actor
        mean, _ = actor.forward(state_tensor)

        # SAC applies tanh to squash output to [-1, 1], then scales by max_action
        action = torch.tanh(mean) * actor.max_action
    
    return action.cpu().numpy()[0]

def test():
    # 1. Exact same config as training
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
            "absolute": False   # Relative coordinates are easier to generalize
        }
    }
    # 2. Create Env with Video Recording
    env = gym.make("highway-v0", render_mode="rgb_array", config=config)

    # Wrap the env to save video.
    env = RecordVideo(env, video_folder="videos/", name_prefix="sac_highway_test", episode_trigger=lambda x: True)

    # 3. Recreate the Actor Architecture
    env.reset()
    action_dim = env.action_space.shape[0]
    state_dim = np.prod(env.observation_space.shape)
    max_action = float(env.action_space.high[0])

    actor = Actor(state_dim, action_dim, max_action)

    # 4. Load the trained weights
    print("Loading actor weights from sac_actor_highway.pth")
    actor.load_state_dict(torch.load("sac_actor_highway.pth"))

    obs, info = env.reset()
    state = obs.flatten()
    done = truncated = False
    total_reward = 0

    print("Starting simulation...")
    while not (done or truncated):
        # Use deterministic action for testing
        action = get_deterministic_action(actor, obs)
        obs, reward, done, truncated, info = env.step(action)
        obs = obs.flatten()
        total_reward += reward
        env.render()    # Required for the video recorder to capture frames

    print(f"Test Episode Reward: {total_reward:.2f}")
    print("Video saved in videos/ folder.")
    env.close()


if __name__ == "__main__":
    test()    