import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
import numpy as np
from Agent import Agent

class RL:
    def __init__(self) -> None:
        self.name = "ButaChanRL"
        pass

    def learn(self,agent,env,agent_parameters,NSTEPS=10000):
        epsiode = 1
        epsiodes = []
        loss = []
        # prepare agent
        agent.agent_init(agent_parameters)
        state,info= env.reset() 
        # choose initial action based on agent's results
        action = agent.agent_start(state)
        done = False
        epsiode_reward = 0
        epsiode_rewards = []
        for i in range(NSTEPS):
            state,reward,terminated,truncated,info=env.step(action)
            epsiode_reward += reward
            done = terminated or truncated
            if(done):
                agent.agent_end(reward)
                epsiode_rewards.append(epsiode_reward)
                epsiode += 1
            else:
                action = agent.agent_step(reward,state)
            


    def run(self):
        env = gym.make("CartPole-v1")
        experiment_parameters = {
            "num_runs" : 1,
            "num_episodes" : 1000,
            # OpenAI Gym environments allow for a timestep limit timeout, causing episodes to end after 
            # some number of timesteps. Here we use the default of 500.
            "timeout" : 500
        }
        n_state = env.observation_space.shape[0]
        n_actions = env.action_space.n

        agent_parameters = {
        'network_config': {
            'state_dim': n_state,
            'num_hidden_units': 128,
            'num_actions': n_actions
        },
        'replay_buffer_size': 1_000_0000,
        'minibatch_sz': 32,
        'num_replay_updates_per_step': 1,
        'gamma': 0.99,
        'epsilon': 1,
        'update_freq':1000,
        'warmup_steps':5000,
        'visualize':True
        }
        agent = Agent()
