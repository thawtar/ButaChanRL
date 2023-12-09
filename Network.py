import torch
import torchvision

class ActorCritic(torch.nn.Module):
    def __init__(self,network_config) -> None:
        super(ActorCritic,self).__init__()
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        self.num_actions = network_config.get("num_actions")

        self.layer1 = torch.nn.Linear(self.state_dim, self.num_hidden_units)
        self.layer2 = torch.nn.Linear(self.num_hidden_units,self.num_hidden_units)
        self.policy_layer = torch.nn.Linear(self.num_hidden_units, self.num_actions)
        self.value_layer = torch.nn.Linear(self.num_hidden_units,1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.actor = torch.nn.Sequential(
            self.layer1,self.relu,self.policy_layer,self.softmax
        )
        self.critic = torch.nn.Sequential(
            self.layer1,self.relu,self.value_layer
        )

    def forward(self, x):
        policy = self.actor(x)
        value = self.critic(x)
        return policy,value
    


class DuelinDQN(torch.nn.Module):
    # Work Required: Yes. Fill in the layer_sizes member variable (~1 Line).
    def __init__(self, network_config):
        
        super(DuelinDQN,self).__init__()
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        self.n_separtae_units = self.num_hidden_units // 2
        self.num_actions = network_config.get("num_actions")
        #random.seed(network_config.get("seed"))
        
        # Specify self.layer_sizes which shows the number of nodes in each layer
        # your code here
        self.layers = [self.state_dim,self.num_hidden_units,self.num_actions]
        
        self.layer1 = torch.nn.Linear(self.state_dim,self.layers[1])
        self.layer2 = torch.nn.Linear(self.layers[1],self.layers[1])
        self.layer3a = torch.nn.Linear(self.layers[1],self.n_separtae_units)
        self.layer4a = torch.nn.Linear(self.n_separtae_units,self.num_actions)

        self.layer3b = torch.nn.Linear(self.layers[1],self.n_separtae_units)
        self.layer4b = torch.nn.Linear(self.n_separtae_units,1)
        

    def forward(self,x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        a = torch.nn.functional.relu(self.layer3a(x))
        a = self.layer4a(a)
        
        v = torch.nn.functional.relu(self.layer3b(x))
        v = self.layer4b(v)
        a = a - a.mean(1).unsqueeze(1)
        q = v+a
        #print(q)
        return q
    
class DQN(torch.nn.Module):
    def __init__(self, network_config):
        super(DQN, self).__init__()
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        
        self.num_actions = network_config.get("num_actions")
        self.layer1 = torch.nn.Linear(self.state_dim, self.num_hidden_units)
        self.layer2 = torch.nn.Linear(self.num_hidden_units,self.num_hidden_units)
        self.layer3 = torch.nn.Linear(self.num_hidden_units, self.num_actions)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.layer_norm = torch.nn.LayerNorm(self.num_hidden_units)
        self.relu = torch.nn.ReLU()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x
    
class LSTMDQN(torch.nn.Module):
    def __init__(self, network_config):
        super(LSTMDQN, self).__init__()
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        
        self.num_actions = network_config.get("num_actions")
        self.lstm_layer = torch.nn.LSTM(self.state_dim, self.num_hidden_units)
        self.layer1 = torch.nn.Linear(self.state_dim, self.num_hidden_units)
        self.layer2 = torch.nn.Linear(self.num_hidden_units,self.num_hidden_units)
        self.layer3 = torch.nn.Linear(self.num_hidden_units, self.num_actions)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.layer_norm = torch.nn.LayerNorm(self.num_hidden_units)
        self.relu = torch.nn.ReLU()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x,_ = self.lstm_layer(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x
    
class CNNDQN(torch.nn.Module):
    def __init__(self, network_config):
        super(CNNDQN, self).__init__()
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        self.conv1 = torch.nn.Conv2d(32,64,8,4)
        self.conv2 = torch.nn.Conv2d(64,64,4,2)
        self.conv3 = torch.nn.Conv2d(64,64,3,1)
        self.num_actions = network_config.get("num_actions")
        self.layer1 = torch.nn.Linear(self.state_dim, self.num_hidden_units)
        self.layer2 = torch.nn.Linear(self.num_hidden_units,self.num_hidden_units)
        self.layer3 = torch.nn.Linear(self.num_hidden_units, self.num_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
def get_td_error_double_dqn(states, next_states, actions, rewards, discount, terminals, target_network, current_q_network):
    with torch.no_grad():
        # The idea of Double DQN is to get max actions from current network
        # and to get Q values from target_network for next states. 
        q_next_mat = current_q_network(next_states)
        max_actions = torch.argmax(q_next_mat,1)
        double_q_mat = target_network(next_states)
    batch_indices = torch.arange(q_next_mat.shape[0])
    double_q_max = double_q_mat[batch_indices,max_actions]
    target_vec = rewards+discount*double_q_max*(torch.ones_like(terminals)-terminals)
    q_mat = current_q_network(states)
    batch_indices = torch.arange(q_mat.shape[0])
    q_vec = q_mat[batch_indices,actions]
    #delta_vec = target_vec - q_vec
    return target_vec,q_vec

def get_td_error_dqn(states, next_states, actions, rewards, discount, terminals, target_network, current_q_network):
    with torch.no_grad():
        q_next_mat = target_network(next_states)
        q_max = torch.max(q_next_mat,1)[0]
    batch_indices = torch.arange(q_next_mat.shape[0])
    target_vec = rewards+discount*q_max*(torch.ones_like(terminals)-terminals)
    q_mat = current_q_network(states)
    batch_indices = torch.arange(q_mat.shape[0])
    q_vec = q_mat[batch_indices,actions]
    return target_vec,q_vec

    
def optimize_network(experiences, discount, optimizer, target_network, current_q_network,device,double_dqn=False):
    """
    Args:
        experiences (Numpy array): The batch of experiences including the states, actions,
                                   rewards, terminals, and next_states.
        discount (float): The discount factor.
        network (ActionValueNetwork): The latest state of the network that is getting replay updates.
        current_q (ActionValueNetwork): The fixed network used for computing the targets,
                                        and particularly, the action-values at the next-states.
    """
    # Get states, action, rewards, terminals, and next_states from experiences
    states, actions, rewards, terminals, next_states = map(list, zip(*experiences))
    states = torch.concatenate(states)
    next_states = torch.concatenate(next_states)
    rewards = torch.tensor(rewards,dtype=torch.float32,device=device)
    terminals = torch.tensor(terminals,dtype=torch.float32,device=device)
    batch_size = states.shape[0]
    # Compute TD error using the get_td_error function
    # Note that q_vec is a 1D array of shape (batch_size)
    if(double_dqn):
        target_vec,q_vec = get_td_error_double_dqn(states, next_states, actions, rewards, discount, terminals, target_network, current_q_network)
    else:
        target_vec,q_vec = get_td_error_dqn(states, next_states, actions, rewards, discount, terminals, target_network, current_q_network)
    loss_fun = torch.nn.MSELoss()
    loss = loss_fun(target_vec,q_vec)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(target_network.parameters(), 10)
    optimizer.step()
    return loss.detach().numpy()


