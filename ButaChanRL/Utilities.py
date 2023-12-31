import torch
import matplotlib.pyplot as plt
import numpy as np


class Utilities:
    def load_model(self,model,filename="model.weights"):
        if(model.name == "DQN"):
            model.target_network.load_state_dict(torch.load(filename))
            model.q_network.load_state_dict(torch.load(filename))
        elif(model.name=="ActorCritic"):
            model.actor_critic_network.load_state_dict(torch.load(filename))
        else:
            NotImplementedError()


    def save_model(self,model,filename="model.weights"):
        if(model.name == "DQN"):
            torch.save(model.q_network.state_dict(),filename)
        elif(model.name=="ActorCritic"):
            torch.save(model.actor_critic_network.state_dict(),filename)


    def test_agent(self,agent,test_env):
        pass

    def visualize_values(self,agent,env,runs=1):
        for i in range(runs):
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