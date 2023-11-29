import torch


class DuelinDQN(torch.nn.Module):
    # Work Required: Yes. Fill in the layer_sizes member variable (~1 Line).
    def __init__(self, network_config):
        
        super(DuelinDQN,self).__init__()
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        self.n_separtae_units = self.num_hidden_units // 2
        self.num_actions = network_config.get("num_actions")
        random.seed(network_config.get("seed"))
        
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
        #x = torch.nn.functional.relu(self.layer2(x))
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

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x
