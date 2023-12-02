import torch
import random
from copy import deepcopy
import numpy as np
from ExperienceReplay import ReplayBuffer
from Network import DuelinDQN, DQN
from Network import optimize_network


class Agent:
    def __init__(self):
        self.name = "DQNAgent"
        self.device = None
        self.seed = 1 # random seed. Later can be changed by using set_seed method

    def set_seed(self,seed=1):
        self.seed = seed
        #random.seed(self.seed)
    
    def set_epsilon_decay(self,n_steps=10000):
        self.eps_decay = 1. - 1./n_steps

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
        if(self.dueling):
            self.q_network = DuelinDQN(agent_config['network_config']).to(self.device)
            self.target_network = DuelinDQN(agent_config['network_config']).to(self.device)
        else:
            self.q_network = DQN(agent_config['network_config']).to(self.device)
            self.target_network = DQN(agent_config['network_config']).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.step_size = agent_config['step_size']
        self.num_actions = agent_config['network_config']['num_actions']
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.discount = agent_config['gamma']
        self.epsilon = agent_config['epsilon']
        self.time_step = 0
        self.update_freq = agent_config['update_freq']
        self.loss = []
        self.episode_rewards = []
        self.loss_capacity = 5_000
        self.warmup_steps = agent_config['warmup_steps']
        self.eps_decay = 0.9999
        self.last_state = None
        self.last_action = None
        self.sum_rewards = 0
        self.episode_steps = 0
        self.optimizer = torch.optim.AdamW(self.q_network.parameters(),lr=self.step_size)

    def greedy_policy(self,state,epsilon=0.001):
        state = torch.tensor([state],dtype=torch.float32)
        a = random.random()
        if(a>=epsilon):
            with torch.no_grad():
                action_values = self.q_network(state)
            action = torch.argmax(action_values).item()
        else:
            action = random.choice(list(range(self.num_actions)))
        return action

    def epsilon_greedy_policy(self,state):
        epsilon = np.max([self.epsilon,0.05]) 
        self.epsilon *= self.eps_decay
        a = random.random()
        if(a>=epsilon):
            with torch.no_grad():
                action_values = self.q_network(state)
            action = torch.argmax(action_values).item()
        else:
            action = random.choice(list(range(self.num_actions)))
        return action

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
        self.last_state = torch.tensor(np.array([state]),dtype=torch.float32,device=self.device)
        self.last_action = self.epsilon_greedy_policy(self.last_state)
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
        state = torch.tensor(np.array([state]),dtype=torch.float32,device=self.device)
        action = self.epsilon_greedy_policy(state)
        terminal = False
        self.replay_buffer.append(self.last_state, self.last_action, reward, terminal, state)

        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size and self.time_step>self.warmup_steps: # and self.episode_steps%self.replay_buffer.minibatch_size==0:
            for _ in range(self.num_replay):
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()
                loss = optimize_network(experiences, self.discount, self.optimizer, self.target_network, self.q_network,self.device)
                if(len(self.loss)>=self.loss_capacity):
                    del self.loss[0]
                self.loss.append(loss)

        if(self.time_step%self.update_freq==0):
            #print("Updating network")
            self.update_target_network()
       
        self.last_state = None
        self.last_action = None

        ### END CODE HERE
        # your code here
        self.last_state = state
        self.last_action = action
        self.time_step += 1
        return action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1
        self.episode_rewards.append(self.sum_rewards)
        # Set terminal state to an array of zeros
        state = torch.zeros_like(self.last_state,device=self.device)

        # Append new experience to replay buffer
        # Note: look at the replay_buffer append function for the order of arguments
        end_loss = 0
        # your code here
        terminal = True
        self.replay_buffer.append(self.last_state, self.last_action, reward, terminal, state)
        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            for _ in range(self.num_replay):
                experiences = self.replay_buffer.sample()
                loss = optimize_network(experiences, self.discount, self.optimizer, self.target_network, self.q_network,self.device)
                end_loss = loss
                if(len(self.loss)>=self.loss_capacity):
                    del self.loss[0]
                self.loss.append(loss)
        
        if(self.time_step%self.update_freq==0):
            self.update_target_network()
        self.time_step += 1
    
        return end_loss

    def agent_message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_loss(self):
        return np.average(np.array(self.loss))