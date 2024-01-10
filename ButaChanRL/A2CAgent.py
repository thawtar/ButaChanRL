import torch
import random
from copy import deepcopy
import numpy as np
from butachanrl.ExperienceReplay import ReplayBuffer
from butachanrl.Network import ActorCritic
from butachanrl.Network import optimize_network



class A2CAgent:
    def __init__(self):
        self.name = "A2C"
        torch.autograd.set_detect_anomaly(True)
        self.device = None
        self.seed = 1 # random seed. Later can be changed by using set_seed method

    def set_seed(self,seed=1):
        self.seed = seed
        #random.seed(self.seed)
    

    def set_device(self,device):
        self.device = device
    
    def agent_init(self, agent_config):
        if(self.device==None):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = agent_config["network_config"].get("state_dim")
        self.num_hidden_layers = agent_config["network_config"].get("num_hidden_units")
        self.num_actions = agent_config["network_config"].get("num_actions")
        self.dueling = agent_config["network_config"].get("dueling")
        
        self.actor_critic_network = ActorCritic(agent_config['network_config']).to(self.device)
        
        self.actor_step_size = 1e-4 #agent_config['actor_step_size']
        self.critic_step_size =  1e-3 #agent_config['critic_step_size']
        self.avg_reward_step_size = 1e-3
        self.num_actions = agent_config['network_config']['num_actions']
        self.discount = agent_config['gamma']
        self.time_step = 0
        self.loss = []
        self.episode_rewards = []
        self.loss_capacity = 5_000
        self.last_state = None
        self.last_action = None
        self.sum_rewards = 0
        self.episode_steps = 0
        self.actor_optimizer = torch.optim.Adam(self.actor_critic_network.actor.parameters(),lr=self.actor_step_size)
        self.critic_optimizer = torch.optim.Adam(self.actor_critic_network.critic.parameters(),lr=self.critic_step_size)
        self.avg_reward = 0
        self.I = 1
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.terminals = []

    def select_action(self,state):
        state = torch.tensor(np.array(state),dtype=torch.float32,device=self.device)
        policy,_ = self.actor_critic_network(state)
        action_probabilities = policy.squeeze().detach().numpy()
        action = np.random.choice(len(action_probabilities), p=action_probabilities)
        return action

    def policy(self,state):
        #state = torch.tensor(state,dtype=torch.float32)
        
        policy,_ = self.actor_critic_network(state)
        m = torch.distributions.Categorical(policy)
        action = m.sample()
        lp = m.log_prob(action)
        #action = torch.multinomial(policy, 1).item()
        return action.item(),lp
    
    def value(self,state):
        #state = torch.tensor(state,dtype=torch.float32)
        _,value = self.actor_critic_network(state)
        return value

    # Work Required: No.
    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.sum_rewards = 0
        self.episode_steps = 0
        self.I = 1
        self.last_state = state #torch.tensor(np.array([state]),dtype=torch.float32,device=self.device)
        self.last_action = self.select_action(state)
        self.time_step += 1
        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        self.sum_rewards += reward
        self.episode_steps += 1
        action = self.select_action(state)
        self.states.append(self.last_state)
        self.actions.append(self.last_action)
        self.rewards.append(reward)
        self.terminals.append(False)
        self.next_states.append(state)
        self.last_action = action
        self.last_state = state
        return action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1
        
        self.states.append(self.last_state)
        self.actions.append(self.last_action)
        self.rewards.append(reward)
        state = np.zeros_like(self.last_state)
        self.terminals.append(True)
        self.next_states.append(state)
        loss=self.agent_update()
        return loss

    def discount_rewards(self, rewards):
        # Compute the gamma-discounted rewards over an episode
        
        running_add = 0
        discounted_r = np.zeros_like(rewards)
        for i in reversed(range(0,len(rewards))):
            
            running_add = running_add * self.discount + rewards[i]
            discounted_r[i] = running_add
        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) # divide by standard deviation
        return discounted_r

    def agent_update(self):
        #discount_r = self.discount_rewards(self.rewards)
        #policy,values = self.actor_critic_network(self.states)

        #advantages = discount_r - values
        R = self.discount_rewards(self.rewards)
        R = torch.FloatTensor(R)
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        next_states = torch.FloatTensor(self.next_states)
        dones = torch.FloatTensor(self.terminals)
        ones = torch.ones_like(dones)
        

        # Compute advantages
        _, next_values = self.actor_critic_network(next_states)
        advantages = R - self.actor_critic_network.critic(states).detach().squeeze()

        # Compute critic loss
        loss_c = torch.nn.MSELoss()
        critic_loss = loss_c(R,self.actor_critic_network.critic(states).squeeze())#, rewards + self.discount * next_values.detach().squeeze() * (ones-dones))

        # Compute actor loss
        policy, _ = self.actor_critic_network(states)
        selected_probs = policy.gather(1, actions.unsqueeze(1))
        actor_loss = -(torch.log(selected_probs) * advantages.detach()).mean()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        #print("Actor loss",actor_loss.detach().numpy())
        #+print("Critic loss",critic_loss.detach().numpy())
        # Total loss
        total_loss = actor_loss + critic_loss

        return total_loss.detach().numpy()



    def agent_message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")

    def get_loss(self):
        return np.average(np.array(self.loss))