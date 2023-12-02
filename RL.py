import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
import numpy as np
from Agent import Agent
import matplotlib.pyplot as plt
from Utilities import Utilities
import random

class RL:
    def __init__(self) -> None:
        self.name = "ButaChanRL"
        self.mean_episode_length = 0
        self.mean_episode_rew = 0
        self.mean_loss= 0
        self.step = 0
        self.output_step = 1000
        self.epsiode_rewards = []
        self.episode_lens = []
        self.loss = []
        self.utils = Utilities()

    def set_seed(self,seed=1):
        random.seed(seed)
        np.random.seed(seed)


    def visualize_values(self,agent,env,runs=1):
        self.utils.visualize_values(agent,env,runs)
     
    def plot_live(self,data,n_mean=20,plot_start=20):
        plt.ion()
        plt.figure(1)
        plot_data = torch.tensor(data, dtype=torch.float)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.plot(plot_data.numpy(),"o")
        # Take 100 episode averages and plot them too
        if len(plot_data ) >= plot_start:
            means = plot_data .unfold(0, n_mean, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(n_mean), means))
            plt.plot(means.numpy())
        plt.pause(0.1)  # pause a bit so that plots are updated

    def plot_validate(self,data,test_data):
        plt.ion()
        plt.figure(1)
        plot_data = torch.tensor(data, dtype=torch.float)
        test_data = torch.tensor(test_data,dtype=torch.float32)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.plot(plot_data.numpy(),"o")
        plt.plot(test_data.numpy(),"rx")
        plt.pause(0.1)  # pause a bit so that plots are updated

    def episode_summarize(self,episode,episode_reward):
        print(f"Episode: {episode}, Reward: {episode_reward}")

    def summarize(self):
        self.mean_episode_length = 0
        self.mean_episode_rew = 0
        if(len(self.episode_lens)>0):
            self.mean_episode_length = np.average(self.episode_lens)
            self.mean_episode_rew = np.average(self.epsiode_rewards)
        num_episodes = len(self.epsiode_rewards)
        self.mean_loss = 0
        if(len(self.loss)>0):
            self.mean_loss = np.average(self.loss)
        print(f"Step:{self.step}, Episode:{num_episodes} Mean_Epi_Len: {self.mean_episode_length:5.2f},Mean_Epi_Rew {self.mean_episode_rew:5.2f}, Loss: {self.mean_loss:5.2f}")

    def learn(self,agent,env,agent_parameters,NSTEPS=10000,visualize=False):
        epsiode = 1
        epsiodes = []
        
        # prepare agent
        agent.agent_init(agent_parameters)
        agent.set_epsilon_decay(NSTEPS//2)
        state,info= env.reset() 
        # choose initial action based on agent's results
        action = agent.agent_start(state)
        done = False
        epsiode_reward = 0
        episode_len = 0
        
        for i in tqdm(range(1,NSTEPS+1)):
            self.step = i
            state,reward,terminated,truncated,info=env.step(action)
            #print(i,state,reward,action,done)
            epsiode_reward += reward
            done = terminated or truncated
            if(self.step%self.output_step==0):
                self.summarize()
                if(visualize):
                    if(len(self.epsiode_rewards)>0):
                        self.plot_live(self.epsiode_rewards)
            if(done):
                loss = agent.agent_end(reward)
                #print("Loss length",len(agent.loss))
                self.loss.append(loss)
                self.epsiode_rewards.append(epsiode_reward)
                self.episode_lens.append(episode_len)
                if(len(self.epsiode_rewards)>=1000):
                    del self.epsiode_rewards[0]
                    del self.episode_lens[0]
                epsiode += 1
                
                # restart next episode
                state,_= env.reset() 
                action = agent.agent_start(state)
                done = False
                epsiode_reward = 0
                episode_len = 0
            else:
                action = agent.agent_step(reward,state)
                episode_len+=1
        return agent

    def evaluate(self,agent,env,n_episodes=10,seed=1,visualize=False,eval_espilon=0.001):
        epsiode_rewards = []
        for episode in range(1,n_episodes+1):
            state,info = env.reset()
            action = agent.greedy_policy(state,eval_espilon)
            done = False
            epsiode_reward = 0
            episode_len = 0
            while not done:
                state,reward,terminated,truncated,info=env.step(action)
                epsiode_reward += reward
                done = terminated or truncated
                action = agent.greedy_policy(state,eval_espilon)
                episode_len += 1
            epsiode_rewards.append(epsiode_reward)
            self.episode_summarize(episode,epsiode_reward)
        mean_rew = np.average(epsiode_rewards)
        std_rew = np.std(epsiode_rewards)
        return (mean_rew,std_rew)   
            

def main():
    print("---------------------")
    print("Welcome to ButaChanRL")
    print("---------------------")
    print("Please use this framework from a run.py file")
    

if __name__=="__main__":
    main()
