import gymnasium as gym
from RL import RL
from DQNAgent import DQNAgent
from ActorCriticAgent import ActorCriticAgent
from A2CAgent import A2CAgent
from SARSAAgent import SARSAAgent
import torch

def run():
    env = gym.make("CartPole-v1") #TradingEnv(data_file="usdjpy.csv",start_tick=100,end_tick=500,window_length=9,debug=False,pos=False,tech=False)
    #test_env = TradingEnv(data_file="usdjpy.csv",start_tick=10000,end_tick=11000,window_length=9,debug=False,pos=False,tech=False)
    
    
    n_state = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent_parameters = {
    'network_config': {
        'state_dim': n_state,
        'num_hidden_units': 128,
        'num_actions': n_actions,
        "network_type":"dqn"
    },
    'replay_buffer_size': 1_000_000,
    'minibatch_sz': 32,
    'observation_size':n_state,
    'num_replay_updates_per_step': 1,
    "step_size": 3e-4,
    'gamma': 0.99,
    'epsilon': 1,
    'update_freq':500,
    'warmup_steps':500,
    'double_dqn':True
    }
    #agent = ActorCriticAgent() #DQNAgent()
    agent = A2CAgent()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    #agent.set_device(device=device)
    #agent = SARSAAgent()
    #agent = ActorCriticAgent()
    rl = RL()
    #rl.set_seed(1)
    trained_agent = rl.learn(agent,env,agent_parameters,NSTEPS=50_000,visualize=True,save_best_weights=False)
    mean,std=rl.evaluate(trained_agent,env,n_episodes=5,visualize=True)
    print(f"Mean reward: {mean}, Standard deviation: {std}")
    


def main():
    run()

if __name__=="__main__":
    main()