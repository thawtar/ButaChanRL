import gymnasium as gym
from RL import RL
from Agent import Agent


def run():
    env = gym.make("CartPole-v1")
    
    n_state = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent_parameters = {
    'network_config': {
        'state_dim': n_state,
        'num_hidden_units': 128,
        'num_actions': n_actions,
        "dueling": False
    },
    'replay_buffer_size': 1_000_000,
    'minibatch_sz': 32,
    'num_replay_updates_per_step': 1,
    'gamma': 0.99,
    'epsilon': 1,
    'update_freq':100,
    'warmup_steps':500,
    }
    agent = Agent()
    rl = RL()
    rl.learn(agent,env,agent_parameters,visualize=False)



def main():
    run()

if __name__=="__main__":
    main()