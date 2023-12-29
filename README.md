# ButaChanRL🐷
Welcome to ButaChanRL! ButaChanRL is a Deep Reinforcement Learning (DRL) framework created for easy testing and implementation of Reinforcement Learning (RL) agents.

Built on the PyTorch backend, ButaChanRL provides a collection of state-of-the-art deep reinforcement learning algorithms, including DQN, Double DQN, Dueling DQN, and Prioritized Experience Replay (PER is on the way!).

For Continuous action space problems, Continuous and Discrete Actor-Critic algorithms (Vanilla actor-critic, A2C, SAC) are at the development state.

Installation:

ButaChanRL can be installed by downloading the package and using setup.py as follows

```
pip install -e .
```

It can also be download directly from PyPI as follows
```
pip install butachanrl
```

Documentation:

A documentation will be created later for detailed information on ButaChanRL's modules, classes, and methods, refer to the documentation.

Examples:

Explore the examples directory for sample scripts demonstrating the use of ButaChanRL with different environments and algorithms.
ButaChanRL can easily be used with Gymnasium environments like other RL packages.

For the example, you can easily create the training code as follows. Just change the "CartPole-v1" to other Discrete Gym environment and you are good to go!

```python
import gymnasium as gym
import torch
# you will need RL and Agent classes
from butachanrl.RL import RL
from butachanrl.DQNAgent import DQNAgent
from butachanrl.ActorCriticAgent import ActorCriticAgent


torch.set_num_threads(1) # to keep the CPU load low. 
env = gym.make("CartPole-v1") # just create gym environment natively

n_state = env.observation_space.shape[0]
n_actions = env.action_space.n

agent_parameters = { # this is where you can choose neural networks adjust hyperparameters 
'network_config': {
    'state_dim': n_state,
    'num_hidden_units': 128,
    'num_actions': n_actions,
    "network_type":"dqn" # may choose between dqn, dueling, lstm and cnn networks
},
'replay_buffer_size': 1_000_000, # Replay buffer to store transitions
'minibatch_sz': 32, # minibatch size to choose from replay buffer and batch train the neural network
'num_replay_updates_per_step': 2, # number of gradient updates per time steps
"step_size": 3e-4, # learning rate for Adam optimizer
'gamma': 0.99, # discount factor
'epsilon': 1, # epsilon greedy is used with annealing. This is initial epsilon value.
'update_freq':1000, # the target network is updated every update_freq time steps
'warmup_steps':500, # steps without gradient descent
'double_dqn':True # whether to use Double DQN
}
agent = DQNAgent() # You can change to other agents such as SARSA, ActorCritic

# default device is CPU but if you want to use GPU, you can easily turn on as follows
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#agent.set_device(device) # to set the device for neural network operations

rl = RL() # You need this to control the overall process of training

trained_agent = rl.learn(agent,env,agent_parameters,NSTEPS=50_000,visualize=True,save_best_weights=False) # training loop
mean,std=rl.evaluate(trained_agent,env,n_episodes=5,visualize=True) # evaluation
print(f"Mean reward: {mean}, Standard deviation: {std}")

```


![alt text](https://github.com/thawtar/ButaChanRL/blob/master/images/training.png)


Requirements:

Requirements for ButaChanRL can be installed via pip as follows:

```
pip install numpy torch matplotlib tqdm gymnasium
```
(OR)
Just install via requirement.txt

```
pip install -r requirements.txt
```
Contributing:
Contributions to ButaChanRL are welcome! Please refer to CONTRIBUTING.md for guidelines on how to contribute.

License:
This project is licensed under the MIT License - see the LICENSE file for details.

Happy Reinforcement Learning with ButaChanRL! 🐷
