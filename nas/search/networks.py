import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dims):
        """
        action_dims: dictionary of parameter name to number of possible values
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dims = action_dims
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # Separate output heads for each architecture parameter
        self.heads = nn.ModuleDict({
            param: nn.Linear(256, dim) 
            for param, dim in action_dims.items()
        })
        
    def forward(self, state):
        shared_features = self.shared(state)
        return {
            param: F.softmax(head(shared_features), dim=-1)
            for param, head in self.heads.items()
        }

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state):
        return self.net(state)