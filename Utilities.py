import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
import numpy as np

class Utilities:
    def load_model(self,model,filename="model.weights"):
        model.target_network.load_state_dict(torch.load(filename))
        model.q_network.load_state_dict(torch.load(filename))

    def save_model(self,model,filename="model.weights"):
        torch.save(model.target_network.state_dict(),filename)


    def test_agent(self,agent,test_env):
        pass

    def visualize_values(self,agent,env):
        state,_=env.reset()
        done = False
        n_step = 0
        values = []
        while(done!=True):
            action_values = agent.q_network.get_action_values(state)
            values.append(action_values)
            action = np.argmax(action_values)
            state,reward,terminated,truncated,info=env.step(action)
            done = terminated or truncated
        values = np.array(values)
        values = values.reshape(values.shape[0],agent.num_actions)
        print(values.shape)
        plt.plot(values)
        plt.show()