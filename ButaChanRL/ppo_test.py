import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

# Define the Actor and Critic networks
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.actor(x)), self.critic(x)

# Define the PPO algorithm
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=2e-4, gamma=0.99, clip_ratio=0.2):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer_actor = optim.Adam(self.actor_critic.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.actor_critic.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.clip_ratio = clip_ratio

    def compute_advantage(self, rewards, values, next_value):
        returns = []
        advantages = []
        advantage = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            delta = r + self.gamma * next_value - v
            advantage = delta + self.gamma * advantage
            advantages.insert(0, advantage)
            returns.insert(0, advantage + v)
        return torch.tensor(advantages), torch.tensor(returns)

    def update_policy(self, states, actions, old_probs, rewards, dones):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        old_probs = torch.tensor(old_probs, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        _, values = self.actor_critic(states)
        advantages, returns = self.compute_advantage(rewards, values.detach().numpy(), 0)

        for _ in range(10):  # Number of PPO epochs
            _, values = self.actor_critic(states)
            new_probs = self.compute_probs(states, actions)

            ratio = new_probs / old_probs
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            loss_actor = -torch.min(ratio * advantages, clip_adv).mean()

            loss_critic = F.mse_loss(returns, values)

            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss_actor.backward()
            loss_critic.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()

    def compute_probs(self, states, actions):
        mu, _ = self.actor_critic(states)
        dist = torch.distributions.Normal(mu, 1.0)
        return dist.log_prob(actions).sum(axis=1)

# Training loop
def train(env_name='Pendulum-v0', num_episodes=1000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppo = PPO(state_dim, action_dim)

    for episode in range(num_episodes):
        states = []
        actions = []
        old_probs = []
        rewards = []

        state,_ = env.reset()

        while True:
            mu, _ = ppo.actor_critic(torch.tensor(state, dtype=torch.float32))
            dist = torch.distributions.Normal(mu, 1.0)
            action = dist.sample().numpy()

            next_state, reward, done,done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            old_probs.append(dist.log_prob(torch.tensor(action, dtype=torch.float32)).sum())
            rewards.append(reward)

            state = next_state

            if done:
                next_value = 0 if done else ppo.actor_critic(torch.tensor(next_state, dtype=torch.float32))[1].detach().numpy()
                ppo.update_policy(states, actions, old_probs, rewards, done, next_value)
                break

if __name__ == "__main__":
    train()