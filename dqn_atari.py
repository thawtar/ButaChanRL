import gymnasium as gym
from RL import RL
from DQNAgent import DQNAgent
import matplotlib.pyplot as plt
from preprocess import resize_frame

def run():
    #env = gym.make("BreakoutNoFrameskip-v4")
    env = gym.make("BreakoutDeterministic-v4")
    n_state = (84,84,4)#env.observation_space
    n_actions = env.action_space.n
    
    agent_parameters = {
    'network_config': {
        'state_dim': n_state,
        'num_hidden_units': 128,
        'num_actions': n_actions,
        "network_type":"cnn"
    },
    'replay_buffer_size': 1_000_000,
    'minibatch_sz': 32,
    'num_replay_updates_per_step': 2,
    "step_size": 3e-4,
    'gamma': 0.99,
    'epsilon': 1,
    'update_freq':100,
    'warmup_steps':500,
    'double_dqn':False
    }
    agent = DQNAgent()
    rl = RL()
    trained_agent = rl.learn(agent,env,agent_parameters,NSTEPS=50_000,visualize=True,save_best_weights=False)
    mean,std=rl.evaluate(trained_agent,env,n_episodes=5,visualize=True)
    print(f"Mean reward: {mean}, Standard deviation: {std}")
    


def main():
    run()

if __name__=="__main__":
    main()