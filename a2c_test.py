import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        policy = self.actor(state)
        value = self.critic(state)
        return policy, value


# Actor-Critic Agent
class ACAgent:
    def __init__(self, state_size, action_size, lr_actor=0.001, lr_critic=0.001, gamma=0.99):
        self.actor_critic = ActorCritic(state_size, action_size)
        self.optimizer_actor = optim.Adam(self.actor_critic.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.actor_critic.critic.parameters(), lr=lr_critic)
        self.gamma = gamma

    def select_action(self, state):
        
        state = torch.FloatTensor(state).unsqueeze(0)
        
        policy, _ = self.actor_critic(state)
        action = torch.multinomial(policy, 1).item()
        return action

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.tensor([action])
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)

        # Compute the TD error
        _, value = self.actor_critic(state)
        _, next_value = self.actor_critic(next_state)
        td_error = reward + (1 - done) * self.gamma * next_value - value

        # Update Critic
        critic_loss = td_error.pow(2)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Update Actor
        policy, _ = self.actor_critic(state)
        selected_prob = policy.gather(1, action.unsqueeze(1))
        actor_loss = -torch.log(selected_prob) * td_error.detach()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()


# Training loop
def train(env_name='CartPole-v1', num_episodes=1000):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = ACAgent(state_size, action_size)

    for episode in range(1, num_episodes + 1):
        state,_ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done,done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        print(f"Episode: {episode}, Total Reward: {total_reward}")

    env.close()


# Run the training
train()