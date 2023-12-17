import torch
import random
from copy import deepcopy
import numpy as np
from ExperienceReplay import ReplayBuffer
from Network import ActorCritic
from Network import optimize_network



class PPOAgent:
    def __init__(self):
        self.name = "PPO"
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
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'],
                                          agent_config['minibatch_sz'],self.seed)
        self.state_dim = agent_config["network_config"].get("state_dim")
        self.num_hidden_layers = agent_config["network_config"].get("num_hidden_units")
        self.num_actions = agent_config["network_config"].get("num_actions")
        self.dueling = agent_config["network_config"].get("dueling")
        
        self.actor_critic_network = ActorCritic(agent_config['network_config']).to(self.device)
        
        self.actor_step_size = 1e-4 #agent_config['actor_step_size']
        self.critic_step_size =  1e-3 #agent_config['critic_step_size']
        self.avg_reward_step_size = 1e-3
        self.num_actions = agent_config['network_config']['num_actions']
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.discount = agent_config['gamma']
        self.time_step = 0
        self.update_freq = agent_config['update_freq']
        self.loss = []
        self.episode_rewards = []
        self.loss_capacity = 5_000
        self.warmup_steps = agent_config['warmup_steps']
        self.last_state = None
        self.last_action = None
        self.sum_rewards = 0
        self.episode_steps = 0
        self.actor_optimizer = torch.optim.Adam(self.actor_critic_network.actor.parameters(),lr=self.actor_step_size)
        self.critic_optimizer = torch.optim.Adam(self.actor_critic_network.critic.parameters(),lr=self.critic_step_size)
        self.avg_reward = 0
        self.I = 1

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
        self.last_state = torch.tensor(np.array([state]),dtype=torch.float32,device=self.device)
        self.last_action,_ = self.policy(self.last_state)
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
        #print(state,reward)
        state = torch.tensor(np.array([state]),dtype=torch.float32,device=self.device)
        action,lp = self.policy(state)
        reward = torch.tensor([reward], dtype=torch.float32)
        last_value = self.value(self.last_state)
        current_value = self.value(state)
        delta = reward  - self.discount* current_value - last_value
        delta *= self.I
        #self.avg_reward += self.avg_reward_step_size*delta
        
        # update critic
        critic_loss = delta.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        actor_loss = -lp * delta.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        ### END CODE HERE
        # your code here
        self.last_state = state
        self.last_action = action
        self.time_step += 1
        self.I *= self.discount
        return action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1
        action,lp = self.policy(self.last_state)
        reward = torch.tensor([reward], dtype=torch.float32)
        last_value = self.value(self.last_state)
        
        delta = reward  - last_value
        delta *= self.I
        #self.avg_reward += self.avg_reward_step_size*delta
        
        # update critic
        critic_loss = delta.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        actor_loss = -lp * delta.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        ### END CODE HERE
        # your code here
        self.time_step += 1
        #self.I *= self.discount
    
        return critic_loss.detach().numpy()

    def agent_message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")

    def get_loss(self):
        return np.average(np.array(self.loss))