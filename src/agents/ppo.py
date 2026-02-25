import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class RolloutBuffer:
    """
    On-policy rollout buffer for PPO.
    Stores trajectories (states, actions, rewards, values, logprobs) for a given number of steps
    across multiple parallel environments.
    """
    def __init__(self, num_steps, num_envs, rgb_shape, state_dim, action_dim, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        
        # Pre-allocate memory for trajectories
        # Tensors are shape (num_steps, num_envs, ...)
        self.obs_rgb = torch.zeros((num_steps, num_envs) + rgb_shape, dtype=torch.float32, device=device)
        self.obs_state = torch.zeros((num_steps, num_envs, state_dim), dtype=torch.float32, device=device)
        
        self.actions = torch.zeros((num_steps, num_envs, action_dim), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        
        # Advantages and Returns are calculated at the end of the rollout
        self.advantages = None
        self.returns = None
        
        self.step = 0

    def add(self, rgb, state, action, logprob, reward, done, value):
        """
        Add a transition to the buffer. Inputs should be batched across environments.
        """
        assert self.step < self.num_steps, "Rollout buffer is full!"
        
        self.obs_rgb[self.step] = rgb.to(self.device)
        self.obs_state[self.step] = state.to(self.device)
        self.actions[self.step] = action.to(self.device)
        self.logprobs[self.step] = logprob.to(self.device)
        self.rewards[self.step] = reward.to(self.device)
        self.dones[self.step] = done.to(self.device)
        self.values[self.step] = value.to(self.device).squeeze()
        
        self.step += 1

    def compute_returns_and_advantages(self, agent, next_value, next_done):
        """
        Calls the PPO agent's compute_gae method.
        """
        self.advantages, self.returns = agent.compute_gae(
            self.rewards, self.values, self.dones, next_value, next_done
        )
        # Normalize advantages at the buffer level
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_dataloader(self, batch_size):
        """
        Flattens the num_steps and num_envs dimensions and returns a DataLoader 
        to iterate over mini-batches during PPO optimization.
        """
        assert self.advantages is not None and self.returns is not None, "Must compute advantages first!"
        
        # Flatten the tensors from (num_steps, num_envs, ...) to (num_steps * num_envs, ...)
        b_rgb = self.obs_rgb.view(-1, *self.obs_rgb.shape[2:])
        b_state = self.obs_state.view(-1, self.obs_state.shape[-1])
        b_actions = self.actions.view(-1, self.actions.shape[-1])
        b_logprobs = self.logprobs.view(-1)
        b_returns = self.returns.view(-1)
        b_advantages = self.advantages.view(-1)
        
        dataset = TensorDataset(b_rgb, b_state, b_actions, b_logprobs, b_returns, b_advantages)
        
        # DataLoader handles shuffling and mini-batching
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def reset(self):
        """
        Reset the step counter for the next rollout phase.
        """
        self.step = 0
        self.advantages = None
        self.returns = None

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) training logic for the Micro-VLA Policy.
    """
    def __init__(self, 
                 policy: nn.Module, 
                 lr: float = 3e-4, 
                 gamma: float = 0.99, 
                 gae_lambda: float = 0.95, 
                 clip_coef: float = 0.2, 
                 ent_coef: float = 0.01, 
                 vf_coef: float = 0.5):
        
        self.policy = policy
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        
        # Setup Optimizer for the policy network
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

    def compute_gae(self, rewards, values, dones, next_value, next_done):
        """
        Computes Generalized Advantage Estimation (GAE) and Returns.
        
        Args:
            rewards: Tensor of rewards at each step (T, Batch)
            values: Tensor of state values from the critic at each step (T, Batch)
            dones: Tensor indicating episode termination at each step (T, Batch)
            next_value: The value of the state *after* the final step (Batch)
            next_done: Whether the step *after* the final step is a terminal state (Batch)
        
        Returns:
            advantages: Computed GAE advantages (T, Batch)
            returns: Expected returns (advantages + values) (T, Batch)
        """
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        
        T = len(rewards)
        
        # 1. Calculate GAE (Advantages) and Returns
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - next_done.float()
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1].float()
                nextvalues = values[t + 1]
            
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + values
        
        return advantages, returns

    def update(self, b_rgb, b_state, b_actions, b_old_logprobs, b_returns, b_advantages):
        """
        Performs a single PPO network update step on a batch of transition data.
        
        Args:
            b_rgb: Batch of visual observations (Batch, 3, H, W)
            b_state: Batch of proprioceptive states (Batch, 47)
            b_actions: Batch of actions (Batch, 6)
            b_old_logprobs: Batch of log probabilities from the rollout policy (Batch,)
            b_returns: Batch of computed returns (Batch,)
            b_advantages: Batch of computed advantages (Batch,)
        """
        # Forward pass through the policy with the existing actions to get new probs/values
        _, current_logprobs, entropy, current_values = self.policy.get_action_and_value(b_rgb, b_state, action=b_actions)
        
        current_values = current_values.view(-1)
        
        # 2. Calculate Policy Ratio
        ratio = torch.exp(current_logprobs - b_old_logprobs)

        # 3. Calculate Actor (Surrogate) Loss
        pg_loss1 = -b_advantages * ratio
        pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # 4. Calculate Critic Value Loss
        v_loss = nn.MSELoss()(current_values, b_returns)

        # 5. Calculate Total Loss and Backpropagate
        entropy_loss = entropy.mean()
        loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        return pg_loss.item(), v_loss.item(), entropy_loss.item()
