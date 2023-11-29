import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os 
import gymnasium as gym
import numpy as np

class Utilities:
    def load_model(self,model,filename="model.weights"):
        model.target_network.load_state_dict(torch.load(filename))
        model.q_network.load_state_dict(torch.load(filename))

    def save_model(self,model,filename="model.weights"):
        torch.save(model.target_network.state_dict(),filename)


    def test_agent(self,agent,test_env,start_tick_max=10000,n_episode=100,episode_length=500):
        actions = ["BUY","SELL","HOLD"]
        test_episode_rewards = []
        test_episode_length = episode_length
        for i in range(n_episode):
            start_tick = np.random.choice(start_tick_max)
            end_tick = start_tick+episode_length
            
            test_env = TradingEnv(data_file="data.csv",start_tick=start_tick,end_tick=end_tick,window_length=9,debug=True)
            print(f"Episode: {i+1}")
            print(f"Ticks: {start_tick},{end_tick}")
            test_epi_rew = 0.
            state,_=test_env.reset()
            done = False
            n_step = 0
            while(done!=True):
                action_values = agent.target_network.get_action_values(state)
                action = np.argmax(action_values)
                state,reward,terminated,truncated,info=test_env.step(action)
                
                done = terminated or truncated
                n_step += 1
                test_epi_rew += reward
            print("Total Reward: ",test_epi_rew)
            test_episode_rewards.append(test_epi_rew)
        test_episode_rewards = np.array(test_episode_rewards)
        plt.plot(test_episode_rewards)
        test_avg_rew = np.average(test_episode_rewards)
        print("average random reward: ",test_avg_rew)

    def visualize_values(self,agent,env):
        state,_=env.reset()
        done = False
        n_step = 0
        values = []
        while(done!=True):
            action_values = agent.target_network.get_action_values(state)
            values.append(action_values)
            action = np.argmax(action_values)
            state,reward,terminated,truncated,info=env.step(action)
            
            done = terminated or truncated
        values = np.array(values)
        values = values.reshape(values.shape[0],agent.num_actions)
        print(values.shape)
        plt.plot(values)
        plt.show()