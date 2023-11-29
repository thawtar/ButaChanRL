import torch
import random
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy
from tqdm import tqdm
import os 
import gymnasium as gym
import json
import numpy as np
from ExperienceReplay import ReplayBuffer
from Network import DuelinDQN, DQN
from Network import optimize_network


class Agent:
    def __init__(self):
        self.name = "ButaChanRL"

    # Work Required: No.
    def agent_init(self, agent_config):
        """Setup for the agent called when the experiment first starts.
        Set parameters needed to setup the agent.
        Assume agent_config dict contains:
        {
            network_config: dictionary,
            optimizer_config: dictionary,
            replay_buffer_size: integer,
            minibatch_sz: integer,
            num_replay_updates_per_step: float
            discount_factor: float,
        }
        """
        global device
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'],
                                          agent_config['minibatch_sz'], agent_config.get("seed"))
        self.state_dim = agent_config["network_config"].get("state_dim")
        self.num_hidden_layers = agent_config["network_config"].get("num_hidden_units")
        self.num_actions = agent_config["network_config"].get("num_actions")
        self.dueling = agent_config["network_config"].get("dueling")
        if(self.dueling):
            self.q_network = DuelinDQN(agent_config['network_config']).to(device)
            self.target_network = DuelinDQN(agent_config['network_config']).to(device)
        else:
            self.q_network = DQN(agent_config['network_config']).to(device)
            self.target_network = DQN(agent_config['network_config']).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        #self.optimizer = torch.optim.RMSprop(self.q_network.parameters(),lr=1e-4)
        self.optimizer = torch.optim.AdamW(self.q_network.parameters(),lr=3e-4)
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
        self.visualize = agent_config['visualize']
        

    # Work Required: No.
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
        global device
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = torch.tensor(np.array([state]),dtype=torch.float32,device=device)
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
        global device
        self.sum_rewards += reward
        self.episode_steps += 1
        state = torch.tensor(np.array([state]),dtype=torch.float32,device=device)
        action = self.epsilon_greedy_policy(state)
        terminal = False
        self.replay_buffer.append(self.last_state, self.last_action, reward, terminal, state)

        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size and self.time_step>self.warmup_steps: # and self.episode_steps%self.replay_buffer.minibatch_size==0:
            for _ in range(self.num_replay):
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()
                loss = optimize_network(experiences, self.discount, self.optimizer, self.target_network, self.q_network)
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
        state = torch.zeros_like(self.last_state,device=device)

        # Append new experience to replay buffer
        # Note: look at the replay_buffer append function for the order of arguments

        # your code here
        terminal = True
        self.replay_buffer.append(self.last_state, self.last_action, reward, terminal, state)
        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            for _ in range(self.num_replay):
                experiences = self.replay_buffer.sample()
                loss = optimize_network(experiences, self.discount, self.optimizer, self.target_network, self.q_network)
                if(len(self.loss)>=self.loss_capacity):
                    del self.loss[0]
                self.loss.append(loss)
        
        if(self.time_step%self.update_freq==0):
            self.update_target_network()
        self.time_step += 1


    def agent_message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())