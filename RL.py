import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
import numpy as np
from Agent import Agent

class RL:
    def __init__(self) -> None:
        self.name = "ButaChanRL"
        self.mean_episode_length = 0
        self.mean_episode_rew = 0
        self.mean_episode_loss= 0
        self.step = 0
        self.output_step = 1000
        self.epsiode_rewards = []
        self.episode_lens = []
        self.loss = []
        

    def summarize(self):
        self.mean_episode_length = np.average(self.episode_lens)
        self.mean_episode_rew = np.average(self.epsiode_rewards)
        num_episodes = len(self.epsiode_rewards)
        self.mean_episode_loss = np.average(self.loss)
        print(f"Step:{self.step}, Episode:{num_episodes} Mean_Epi_Len: {self.mean_episode_length:5.2f},Mean_Epi_Rew {self.mean_episode_rew:5.2f}, Loss: {self.mean_episode_loss:5.2f}")


    def learn(self,agent,env,agent_parameters,NSTEPS=10000):
        epsiode = 1
        epsiodes = []
        
        # prepare agent
        agent.agent_init(agent_parameters)
        state,info= env.reset() 
        # choose initial action based on agent's results
        action = agent.agent_start(state)
        done = False
        epsiode_reward = 0
        episode_len = 0
        
        for i in range(1,NSTEPS+1):
            self.step = i
            #print("Step:",i)
            state,reward,terminated,truncated,info=env.step(action)
            #print(i,state,reward,action,done)
            epsiode_reward += reward
            done = terminated or truncated
            if(self.step%self.output_step==0):
                self.summarize()
            if(done):
                agent.agent_end(reward)
                self.loss.append(agent.get_loss())
                print("Loss:",agent.get_loss())
                if(len(self.epsiode_rewards)<1000):
                    self.epsiode_rewards.append(epsiode_reward)
                    self.episode_lens.append(episode_len)
                epsiode += 1
                # restart next episode
                agent.agent_init(agent_parameters)
                state,info= env.reset() 
                action = agent.agent_start(state)
                done = False
                epsiode_reward = 0
                episode_len = 0
            else:
                action = agent.agent_step(reward,state)
                episode_len+=1
            
            


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
            'num_actions': n_actions,
            "dueling": False
        },
        'replay_buffer_size': 1_000_0000,
        'minibatch_sz': 32,
        'num_replay_updates_per_step': 4,
        'gamma': 0.99,
        'epsilon': 0.1,
        'update_freq':10,
        'warmup_steps':500,
        'visualize':True
        }
        agent = Agent()
        self.learn(agent,env,agent_parameters)



def main():
    rl = RL()
    rl.run()

if __name__=="__main__":
    main()
