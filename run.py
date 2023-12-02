import gymnasium as gym
from RL import RL
from Agent import Agent


def run():
    env = gym.make("CartPole-v1")
    #env.reset(seed=1)
    n_state = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent_parameters = {
    'network_config': {
        'state_dim': n_state,
        'num_hidden_units': 128,
        'num_actions': n_actions,
        "dueling": True
    },
    'replay_buffer_size': 1_000_000,
    'minibatch_sz': 32,
    'num_replay_updates_per_step': 1,
    'gamma': 0.99,
    'epsilon': 1,
    'update_freq':1000,
    'warmup_steps':500,
    }
    agent = Agent()
    rl = RL()
    #rl.set_seed(1)
    trained_agent = rl.learn(agent,env,agent_parameters,NSTEPS=10000,visualize=True)
    mean,std=rl.evaluate(trained_agent,env,n_episodes=5)
    print(f"Mean reward: {mean}, Standard deviation: {std}")
    


def main():
    run()

if __name__=="__main__":
    main()