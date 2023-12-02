import gymnasium as gym
from RL import RL
from Agent import Agent
from TradingEnv7 import TradingEnv


def run():
    #env = gym.make("CartPole-v1")
    #env = gym.make("LunarLander-v2")
    env = TradingEnv(data_file="usdjpy.csv",start_tick=100,end_tick=500,window_length=9,debug=False,pos=False)
    test_env = TradingEnv(data_file="usdjpy.csv",start_tick=1000,end_tick=1500,window_length=9,debug=False,pos=False)
    #env.reset(seed=1)
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
    "step_size": 1e-3,
    'gamma': 0.99,
    'epsilon': 1,
    'update_freq':1000,
    'warmup_steps':500,
    }
    agent = Agent()
    rl = RL()
    #rl.set_seed(1)
    trained_agent = rl.learn(agent,env,agent_parameters,NSTEPS=100_000,visualize=True,save_best_weights=True)
    mean,std=rl.evaluate(trained_agent,test_env,n_episodes=5)
    test_env.summarize()
    print(f"Mean reward: {mean}, Standard deviation: {std}")
    


def main():
    run()

if __name__=="__main__":
    main()