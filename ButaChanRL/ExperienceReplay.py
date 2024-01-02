import random
import numpy as np 


class ReplayBuffer:
    def __init__(self, buffer_size, minibatch_size, observation_size):
        """
        Args:
            size (integer): The size of the replay buffer.              
            minibatch_size (integer): The sample size.
            seed (integer): The seed for the random number generator. 
        """
        self.buffer = []
        self.minibatch_size = minibatch_size
        #random.seed(seed)
        self.max_size = buffer_size
        self.pos = 0
        self.full = False
        self.states = np.zeros((self.max_size,observation_size))
        self.next_states = np.zeros((self.max_size,observation_size))
        self.actions = np.zeros(self.max_size,dtype=np.int8)
        self.rewards = np.zeros(self.max_size)
        self.terminals = np.zeros(self.max_size,dtype=np.int8)


    def append(self, state, action, reward, terminal, next_state):
        """
        Args:
            state (Numpy array): The state.              
            action (integer): The action.
            reward (float): The reward.
            terminal (integer): 1 if the next state is a terminal state and 0 otherwise.
            next_state (Numpy array): The next state.           
        """
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.terminals[self.pos] = terminal
        self.next_states[self.pos] = next_state
        self.pos += 1
        if(self.pos==self.max_size):
            self.pos = 0
            self.full = True


    def sample(self):
        """
        Returns:
            A list of transition tuples including state, action, reward, terinal, and next_state
        """
        if(self.full):
            idxs = np.random.randint(0,self.max_size,size=self.minibatch_size) 
        else:
            idxs = np.random.randint(0,self.pos,size=self.minibatch_size)
        sample_ = [self.states[idxs],self.actions[idxs],self.rewards[idxs],self.terminals[idxs],
                   self.next_states[idxs]]
        #print(sample_)
        return sample_

    def size(self):
        if(self.full):
            return self.max_size
        else:
            return self.pos
    
    def reset(self):
        self.full = False
        self.pos = 0