import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((capacity, state_dim))
        self.action = np.zeros((capacity, action_dim))
        self.reward = np.zeros((capacity, 1))
        self.next_state = np.zeros((capacity, state_dim))
        self.done = np.zeros((capacity, 1))

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (torch.FloatTensor(self.state[ind]),
                torch.FloatTensor(self.action[ind]),
                torch.FloatTensor(self.reward[ind]),
                torch.FloatTensor(self.next_state[ind]),
                torch.FloatTensor(self.done[ind]))

# Actor Network (Gaussian Policy)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action
    
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean = self.mean(a)
        log_std = self.log_std(a)
        log_std = torch.clamp(log_std, -20, 2)  # TODO: why clamp?
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() # reparameterization trick TODO: what is this?
        action = torch.tanh(x_t) * self.max_action
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.max_action * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob
    
# Critic Network (Twin Q-Networks)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

# SAC Agent
class SAC:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.gamma = 0.99   # discount factor
        self.tau = 0.005
        self.alpha = 0.2  # Entropy coefficient TODO

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def train(self, replay_buffer, batch_size=256):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
            target_Q = reward + (1 - done) * self.gamma * target_Q
        
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action_new, log_prob_new = self.actor.sample(state)
        Q1_new, Q2_new = self.critic(state, action_new)
        Q_new = torch.min(Q1_new, Q2_new)
        actor_loss = (self.alpha * log_prob_new - Q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Target Networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data 
                                    + (1 - self.tau) * target_param.data)