import os
import time
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import highway_env
from gymnasium import ObservationWrapper, Wrapper
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle



def make_env(env_id: str, seed: int = 0, config: dict = None):
    def _init():
        if config is None:
            env = gym.make(env_id)
        else:
            env = gym.make(env_id, config=config)
        obs, info = env.reset()
        return env

    return _init


def flatten_obs(obs):
    # Accepts obs being np.ndarray, list, or dict of arrays
    if isinstance(obs, dict):
        parts = []
        for k in sorted(obs.keys()):
            v = obs[k]
            parts.append(np.array(v).ravel())
        return np.concatenate(parts, axis=0)
    else:
        return np.array(obs).ravel()


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.shared = nn.Sequential(*layers)

        self.actor = nn.Sequential(nn.Linear(last, 128), nn.ReLU(), nn.Linear(128, action_dim))
        self.critic = nn.Sequential(nn.Linear(last, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(x)
        logits = self.actor(shared)
        value = self.critic(shared).squeeze(-1)
        return logits, value


class PPOBuffer:
    """On-policy rollout buffer for PPO with GAE-Lambda."""

    def __init__(self, obs_dim, size, n_envs, gamma=0.99, lam=0.95, device='cpu'):
        self.obs_buf = np.zeros((size * n_envs, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size * n_envs,), dtype=np.int64)
        self.adv_buf = np.zeros((size * n_envs,), dtype=np.float32)
        self.rew_buf = np.zeros((size * n_envs,), dtype=np.float32)
        self.ret_buf = np.zeros((size * n_envs,), dtype=np.float32)
        self.val_buf = np.zeros((size * n_envs,), dtype=np.float32)
        self.logp_buf = np.zeros((size * n_envs,), dtype=np.float32)
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size * n_envs
        self.gamma = gamma
        self.lam = lam
        self.device = device

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        # compute GAE-Lambda advantage and rewards-to-go
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE-Lambda
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = discount_cumsum(deltas, self.gamma * self.lam)
        self.adv_buf[path_slice] = adv

        # compute rewards-to-go
        ret = discount_cumsum(rews, self.gamma)[:-1]
        self.ret_buf[path_slice] = ret

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size, "Buffer has to be full before you can get()"
        self.ptr = 0
        self.path_start_idx = 0
        # normalize advantages
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf) + 1e-8
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs=self.obs_buf.copy(), act=self.act_buf.copy(), ret=self.ret_buf.copy(), adv=self.adv_buf.copy(), logp=self.logp_buf.copy())
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in data.items()}


def discount_cumsum(x, discount):
    """Compute discounted cumulative sums of vectors."""
    out = np.zeros_like(x)
    running = 0
    for i in reversed(range(len(x))):
        running = x[i] + discount * running
        out[i] = running
    return out


def compute_logp_and_action(pi_logits: torch.Tensor, actions: torch.Tensor):
    # discrete actions
    dist = torch.distributions.Categorical(logits=pi_logits)
    logp = dist.log_prob(actions)
    return logp, dist




def train(
    env_id='highway-v0',
    config=None,
    seed=0,
    n_envs=8,
    steps_per_env=128,
    total_timesteps=200_000,
    lr=5e-4,
    clip_ratio=0.2,
    train_iters=80,
    minibatch_size=64,
    gamma=0.8,
    lam=0.95,
    vf_coef=0.5,
    ent_coef=0,
    max_grad_norm=0.5,
    device='cpu',
    save_path='ppo_highway.pth',
    model = None
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create vectorized envs
    env_fns = [make_env(env_id, seed + i, config) for i in range(n_envs)]
    envs = gym.vector.AsyncVectorEnv(env_fns)

    obs0, _ = envs.reset()
    obs_flat0 = flatten_obs(obs0[0])
    obs_dim = flatten_obs(obs0[0]).shape[0]
    action_dim = envs.single_action_space.n

    if not model:
        model = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    rollout_len = steps_per_env
    buf = PPOBuffer(obs_dim, rollout_len, n_envs, gamma=gamma, lam=lam, device=device)

    obs = obs0
    ep_return = np.zeros(n_envs)
    ep_len = np.zeros(n_envs, dtype=int)

    num_updates = total_timesteps // (n_envs * rollout_len)
    print(f"Training for {total_timesteps} timesteps, {num_updates} updates")

    global_step = 0
    start_time = time.time()
    for update in range(num_updates):
        for step in range(rollout_len):
            obs_flat = np.stack([flatten_obs(o) for o in obs], axis=0)
            obs_t = torch.from_numpy(obs_flat.astype(np.float32)).to(device)

            with torch.no_grad():
                logits, values = model(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                logp = dist.log_prob(actions)

            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, truncs, infos = envs.step(actions_np)

            # store per-env
            for i in range(n_envs):
                buf.store(obs_flat[i], actions_np[i], rewards[i], float(values[i].cpu().numpy()), float(logp[i].cpu().numpy()))

            ep_return += rewards
            ep_len += 1
            for i, d in enumerate(dones):
                if d:
                    # terminal
                    last_val = 0.0
                    buf.finish_path(last_val)
                    ep_return[i] = 0
                    ep_len[i] = 0
            for i, t in enumerate(truncs):
                if t:
                    # truncated episode: bootstrap value
                    obs_i_flat = flatten_obs(next_obs[i])
                    obs_i_t = torch.from_numpy(obs_i_flat.astype(np.float32)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        _, last_val_t = model(obs_i_t)
                        last_val = last_val_t.item()
                    buf.finish_path(last_val)
                    ep_return[i] = 0
                    ep_len[i] = 0

            obs = next_obs
            global_step += n_envs

        # If some trajectories didn't finish, bootstrap their values
        # compute last values for each parallel env
        obs_flat = np.stack([flatten_obs(o) for o in obs], axis=0)
        obs_t = torch.from_numpy(obs_flat.astype(np.float32)).to(device)
        with torch.no_grad():
            _, last_vals = model(obs_t)
        for v in last_vals.cpu().numpy():
            buf.finish_path(float(v))

        data = buf.get()

        policy_losses = []
        value_losses = []
        entropies = []

        # PPO update
        for it in range(train_iters):
            # create minibatches
            idxs = np.arange(len(data['obs']))
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), minibatch_size):
                mb_idx = idxs[start:start + minibatch_size]
                mb = {k: v[mb_idx] for k, v in data.items()}

                logits, values = model(mb['obs'])
                logp_new, dist = compute_logp_and_action(logits, mb['act'].long())

                ratio = torch.exp(logp_new - mb['logp'])
                surr1 = ratio * mb['adv']
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb['adv']
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((mb['ret'] - values) ** 2).mean()

                entropy = dist.entropy().mean()

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        mean_policy_loss = float(np.mean(policy_losses)) if policy_losses else 0.0
        mean_value_loss = float(np.mean(value_losses)) if value_losses else 0.0
        mean_entropy = float(np.mean(entropies)) if entropies else 0.0


        elapsed = time.time() - start_time
        print(f"Update {update+1}/{num_updates} | steps {global_step}/{total_timesteps} | time {elapsed:.1f}s | "
              f"policy_loss {mean_policy_loss:.4f} | value_loss {mean_value_loss:.4f} | entropy {mean_entropy:.4f} ")



        if (update + 1) % 50 == 0:
            # save
            torch.save(model.state_dict(), save_path)

            avg_reward = evaluate_policy(update+1, model, episodes=1, video_folder="Save_gpt/highway-v0")
            print("Average evaluation reward:", avg_reward)

    print("Training finished")
    torch.save(model.state_dict(), save_path)

def evaluate_policy(update, model, episodes=3, video_folder="eval_videos"):
    os.makedirs(video_folder, exist_ok=True)
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
            "order": "sorted",
        },
        "action": {"type": "DiscreteMetaAction"},
        "lanes_count": 3,
        "vehicles_count": 15,
        "policy_frequency": 2,
        "duration": 100,
 
    }
    folder = f"{video_folder}_{update}"

    # create env with video recording
    env = gym.make("highway-v0", render_mode="rgb_array", config=config)
    env = RecordVideo(
        env,
        video_folder=folder,
        name_prefix="eval",
        episode_trigger=lambda ep: True,
    )
    env = RecordEpisodeStatistics(env)


    total_rew = 0.0


    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_rew = 0.0
        while not (done or truncated):
            obs_flat = flatten_obs(obs)
            obs_t = torch.from_numpy(obs_flat.astype(np.float32)).unsqueeze(0).to(next(model.parameters()).device)
            with torch.no_grad():
                logits, _ = model(obs_t)
                action = torch.argmax(logits, dim=-1).item()
            obs, reward, done, truncated, info = env.step(action)
            ep_rew += float(reward)
        total_rew += ep_rew


    env.close()
    return total_rew / episodes



class CustomRewardWrapper(Wrapper):
    def step(
        self, action
    ):
        obs, reward, done, truncated, info = self.env.step(action)

        if info['crashed']:
            return obs, 0, done, truncated, info

        # ego_y = obs[0][2]
        for i in range(1, 6):
            other_x = obs[i][1]
            other_y = obs[i][2]
            other_vx = obs[i][3]
            # one vehicle length from the front vehicle in the same lane
            if abs(other_y) < 0.03 and abs(other_x) < 0.05:
                reward = 0
                break
            # 1.5 vehicle length from the front vehicle during lane changing
            elif abs(other_y) < 0.125 and abs(other_x) < 0.0625:
                reward *= 0.5
                break
            # two vehicle length from the front vehicle in the same lane with 10 m/s speed difference
            elif abs(other_y) < 0.03 and abs(other_x) < 0.075 and other_vx < -0.12:
                reward *= 0.5
                break    

        return obs, reward, done, truncated, info

class NoisyObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        # Add noise to positions (distance) and velocities (speed) of each vehicle
        scale = [0, 0.00125, 0.015625, 0.005, 0, 0, 0]
        #  [0.       0.00125  0.015625 0.005    0.       0.       0.      ]

        noise = self.env.unwrapped.np_random.normal(
            loc=0.0, 
            scale=scale, 
            size=observation.shape
        )
        return (observation + noise).astype(np.float32)
    

# below is the code of using Stable Baseline 3 
# reference: https://github.com/Farama-Foundation/HighwayEnv/blob/master/scripts/sb3_highway_ppo.py


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.distributions import Categorical


def make_configure_env(**kwargs):
    env = gym.make(kwargs["id"], config=kwargs["config"])
    env = CustomRewardWrapper(env)
    # env = NoisyObservationWrapper(env)
    env.reset()
    return env



env_kwargs = {
    "id": "highway-v0",
    "config": {
        "lanes_count": 4,
        "vehicles_count": 25,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
            "order": "sorted"
        },
        "policy_frequency": 2,
        # "vehicles_density": 1,
        "duration": 60,
    },
}


if __name__ == '__main__':

    # config = {
    #     "observation": {
    #         "type": "Kinematics",
    #         "vehicles_count": 10,
    #         "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    #         "absolute": False,
    #         "order": "sorted",
    #     },
    #     "action": {"type": "DiscreteMetaAction"},
    #     "lanes_count": 4,
    #     "vehicles_count": 25,
    #     "policy_frequency": 2,
    #     "duration": 60,
 
    # }

    # env = gym.make("highway-v0", config=config)
    # model = ActorCritic(flatten_obs(env.reset()[0]).shape[0], env.action_space.n)
    # model.load_state_dict(torch.load('Save/ppo_highway_11.19.23.pth', map_location='cpu'))
    # env.close()
    # train(env_id="highway-v0", config=config, seed=0, n_envs=8, steps_per_env=96, total_timesteps=800000, device="cpu", 
    #       model = model
    #       )

    # # Evaluate after training
    # env = gym.make("highway-v0", render_mode="rgb_array", config=config)
    # model = ActorCritic(flatten_obs(env.reset()[0]).shape[0], env.action_space.n).to('cpu')
    # model.load_state_dict(torch.load('Save/ppo_highway_11.19.23.pth', map_location='cpu'))


    # avg_reward = evaluate_policy(0, model, episodes=3, video_folder="eval_videos")
    # print("Average evaluation reward:", avg_reward)

    # train = True
    train = False

    if train:
        n_cpu = 8
        env = make_vec_env(
            make_configure_env,
            n_envs=n_cpu,
            seed=0,
            vec_env_cls=SubprocVecEnv,
            env_kwargs=env_kwargs,
        )
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=512 // n_cpu,
            batch_size=64,
            learning_rate=5e-4,
            verbose=2,
            tensorboard_log="agents/ppo/training_model/",
            gamma=0.8,
        )
        model = PPO.load("agents/ppo/save_models/11.29_penalty_2/model", env=env, tensorboard_log="agents/ppo/training_model/")
        # Train the agent
        model.learn(total_timesteps=200000)
        # Save the agent
        model.save("agents/ppo/training_model/model")

    model = PPO.load("agents/ppo/save_models/11.29_penalty_2/model")
    env = gym.make("highway-v0", render_mode="rgb_array", config=env_kwargs["config"])
    # env = NoisyObservationWrapper(env, speed_std=0, dist_std=0)

        # Record videos 
    env = RecordVideo(
        env,
        video_folder="videos",
        name_prefix="training",
        episode_trigger=lambda x: True 
    )

    # Track statistics for every episode (lightweight)
    env = RecordEpisodeStatistics(env)

    for _ in range(5):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            # env.render()

    env.close()