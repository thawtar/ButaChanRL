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

class Agent:
    def __init__(self):
        self.name = "torch_dqn"


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
        self.q_network = ActionValueNetwork(agent_config['network_config']).to(device)
        #self.q_network = DQN(agent_config['network_config']).to(device)
        self.target_network = ActionValueNetwork(agent_config['network_config']).to(device)
        #self.target_network = DQN(agent_config['network_config']).to(device)
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
        #random.seed(agent_config.get("seed"))

        self.last_state = None
        self.last_action = None

        self.sum_rewards = 0
        self.episode_steps = 0
        self.visualize = agent_config['visualize']
        

    # Work Required: No.
    def epsilon_greedy_policy(self,state):
        epsilon = 0.05 #np.max([self.epsilon,0.05]) 
        self.epsilon *= self.eps_decay
        #print("Epsilon: ",epsilon)
        a = random.random()
        if(a>=epsilon):
            with torch.no_grad():
                action_values = self.q_network(state)
                #action_values = self.target_network(state)
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

        # Make state an array of shape (1, state_dim) to add a batch dimension and
        # to later match the get_action_values() and get_TD_update() functions
        state = torch.tensor(np.array([state]),dtype=torch.float32,device=device)

        # Select action
        # your code here
        action = self.epsilon_greedy_policy(state)

        # Append new experience to replay buffer
        # Note: look at the replay_buffer append function for the order of arguments

        # your code here
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

                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()

                # Call optimize_network to update the weights of the network
                # your code here
                loss = optimize_network(experiences, self.discount, self.optimizer, self.target_network, self.q_network)
                if(len(self.loss)>=self.loss_capacity):
                    del self.loss[0]
                self.loss.append(loss)
        
        if(self.time_step%self.update_freq==0):
            self.update_target_network()
        show_loss(self.loss)
        if(self.visualize):
            plot_live(self.episode_rewards)
        self.time_step += 1


    def agent_message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

class Utilities:
    def load_model(self,model,filename="model.weights"):
        model.target_network.load_state_dict(torch.load(filename))
        model.q_network.load_state_dict(torch.load(filename))

    def save_model(self,model,filename="model.weights"):
        torch.save(model.target_network.state_dict(),filename)


    def test_agent(self,agent,start_tick_max=10000,n_episode=100,episode_length=500):
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

def run_rand_experiment(agent,agent_parameters, experiment_parameters):
    
    # save sum of reward at the end of each episode
    agent_sum_reward = np.zeros((experiment_parameters["num_runs"], 
                                 experiment_parameters["num_episodes"]))

    env_info = {}
    episode_length = 500
    agent_info = agent_parameters
    agents = [] # to store multiple models from multiple runs
    # one agent setting
    uti = Utilities()
    models_dir = "./models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    for run in range(1, experiment_parameters["num_runs"]+1):
        
        #initialize agent every run
        #agent = Agent()
        agent.agent_init(agent_parameters)
 
        file_name = "./ga_models_v4/model_30"
        #uti.load_model(agent,file_name)
        agent_info["seed"] = run
        agent_info["network_config"]["seed"] = run
        env_info["seed"] = run
        max_episode_reward = 0
        ep = 1
        for episode in tqdm(range(1, experiment_parameters["num_episodes"]+1)):
            """
            This part of the code was modified by Thaw Tar in order to replace RLGlue with 
            hand coded RL stepping.
            """
            # create a new environment
            start_tick = 50 #np.random.choice(10000)+1
            end_tick = start_tick+episode_length
            print(f"\nStart Tick: {start_tick}, End Tick: {end_tick}")
            environment = TradingEnv(data_file="usdjpy.csv",start_tick=start_tick,end_tick=end_tick,window_length=9,debug=False,pos=True)
            # prepare the environment
            state,info= environment.reset()
            
            # choose initial action based on agent's results
            action = agent.agent_start(state)
            done = False
            episode_reward = 0.0
            # run episode
            episode_step = 0
            last_period_reward = 0
            current_period_reward = 0
            while(done!=True):
                state,reward,terminated,truncated,info=environment.step(action)
                
                done = terminated or truncated
                if(done):
                    agent.agent_end(reward)
                else:
                    action = agent.agent_step(reward,state)
                episode_reward += reward
                current_period_reward += reward
            #episode_reward = rl_glue.rl_agent_message("get_sum_reward")
                if(episode_step>0 and episode_step%1000==0):
                    
                    #print(f"Episode: {ep}, Step: {episode_step} Reward: {current_period_reward*10_0000}, Balance:{environment.balance}")
                    current_period_reward = 0
                episode_step+=1
            episode_reward = environment.cumulative_reward
            agent_sum_reward[run - 1, episode - 1] = episode_reward
            environment.summarize()

            #print(f"Episode: {ep}, Reward: {episode_reward*10_0000}")
            #environment.summarize()
            ep += 1
            if(episode_reward>max_episode_reward):
                max_episode_reward = episode_reward
                model_file_name = f"{models_dir}/optimal_model_run_{run}_episode_{episode}"
                uti.save_model(agent,model_file_name)
            else:
                model_file_name = f"{models_dir}/model_run_{run}_episode_{episode}"
                uti.save_model(agent,model_file_name)
        agents.append(agent)
        
    save_name = "{}".format(agent.name)
    if not os.path.exists('results'):
        os.makedirs('results')
    np.save("results/sum_reward_{}".format(save_name), agent_sum_reward)
    
    return agents,agent_sum_reward

def run():
    # Experiment parameters
    experiment_parameters = {
        "num_runs" : 1,
        "num_episodes" : 1000,
        # OpenAI Gym environments allow for a timestep limit timeout, causing episodes to end after 
        # some number of timesteps. Here we use the default of 500.
        "timeout" : 500
    }

    # Environment parameters
    environment_parameters = {}

    #current_env = gym.make("LunarLander-v2",render_mode="rgb_array")

    #current_env = TradingEnv(data_file="usdjpy.csv",start_tick=1,end_tick=500,window_length=9,debug=False,pos=True)
    n_state = 57 #current_env.observation_space.shape[0]
    n_actions = 3 #current_env.action_space.n
    #print(n_state,n_actions)
    # Agent parameters
    agent_parameters = {
        'network_config': {
            'state_dim': n_state,
            'num_hidden_units': 128,
            'num_actions': n_actions
        },
        'optimizer_config': {
            'step_size': 3e-4,
            'beta_m': 0.9, 
            'beta_v': 0.999,
            'epsilon': 1e-8
        },
        'replay_buffer_size': 1_000_0000,
        'minibatch_sz': 32,
        'num_replay_updates_per_step': 1,
        'gamma': 0.99,
        'epsilon': 1,
        'update_freq':1000,
        'warmup_steps':5000,
        "dueling":True,
        'visualize':True
    }
    current_agent = Agent()
    #current_agent.agent_init(agent_parameters)

    # run experiment
    agents,sum_reward=run_rand_experiment(current_agent,agent_parameters, experiment_parameters)
    for i in range(sum_reward.shape[0]):
        plt.plot(sum_reward[i,:])
    plt.show()
    return agents, sum_reward

agents, sum_reward = run()
avg_reward = np.average(sum_reward,axis=0)
plt.plot(avg_reward)
plt.show()