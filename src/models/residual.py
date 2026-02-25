import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from src.models.vla_wrapper import FrozenVLAWrapper

class MicroVLAPolicy(nn.Module):
    """
    Multimodal Actor-Critic neural network for the Micro-VLA.
    Outputs action distributions and state values based on visual and proprioceptive inputs.
    """
    def __init__(self, action_dim=6, state_dim=47):
        super().__init__()
        
        # 1. Instantiate the Frozen VLA Feature Extractor (Outputs 512 dims)
        self.vla_encoder = FrozenVLAWrapper()
        
        # 2. MLP feature extractor for proprioceptive state (Outputs 64 dims)
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        combined_dim = 512 + 64
        
        self.actor_mean = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def get_action_and_value(self, rgb, state, action=None, deterministic=False):
        
        with torch.no_grad():
            vision_features = self.vla_encoder(rgb)  
            
        state_features = self.state_mlp(state)       
        
        combined_features = torch.cat([vision_features, state_features], dim=1)
        
        action_mean = self.actor_mean(combined_features)  
        action_std = self.actor_logstd.expand_as(action_mean).exp()
        probs = Normal(action_mean, action_std)
        
        value = self.critic(combined_features)          
        if action is None:
            if deterministic:
                action = action_mean
            else:
                action = probs.sample()
            
        # Calculate log prob and sum across action dimensions
        log_prob = probs.log_prob(action).sum(dim=1)
        
        entropy = probs.entropy().sum(dim=1)
        
        return action, log_prob, entropy, value
