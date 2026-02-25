import os
import sys
import torch
import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.maniskill_wrapper import ManiSkillVLAWrapper
from src.models.residual import MicroVLAPolicy
from src.agents.ppo import PPOAgent, RolloutBuffer
from src.utils.logger import WandbLogger

def make_env(env_id, kwargs_dict=None):
    """
    Helper function to create a single environment instance.
    Needed for Gymnasium SyncVectorEnv or AsyncVectorEnv.
    """
    if kwargs_dict is None:
        kwargs_dict = {}
        
    def thunk():
        # import mani_skill inside the thunk for multiprocessing compatibility
        import mani_skill.envs
        env = gym.make(
            env_id, 
            obs_mode="rgbd", 
            control_mode="pd_ee_delta_pose",
            **kwargs_dict
        )
        env = ManiSkillVLAWrapper(env)
        return env
    return thunk

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig):
    # --------------------------------------------------------------------------
    # Wandb Setup
    # --------------------------------------------------------------------------
    logger = WandbLogger(config=OmegaConf.to_container(cfg, resolve=True))

    # --------------------------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------------------------
    env_id = cfg.env.name
    num_envs = cfg.agent.num_envs
    num_steps = cfg.agent.num_steps if cfg.agent.name == "ppo" else 128
    total_timesteps = cfg.training.steps
    batch_size = cfg.agent.batch_size
    update_epochs = getattr(cfg.agent, "update_epochs", 4)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --------------------------------------------------------------------------
    # Environment Setup
    # --------------------------------------------------------------------------
    # Create vectorized environment
    envs = gym.vector.AsyncVectorEnv(
        [make_env(env_id) for _ in range(num_envs)]
    )
    
    # Note: In a real run, check the observation space. 
    # For now, using the configured values we defined earlier.
    rgb_shape = (3, 256, 256)
    state_dim = envs.single_observation_space["state"].shape[0]
    action_dim = envs.single_action_space.shape[0]
    
    # --------------------------------------------------------------------------
    # Model and Agent Setup
    # --------------------------------------------------------------------------
    policy = MicroVLAPolicy(action_dim=action_dim).to(device)
    
    if cfg.agent.name == "ppo":
        agent = PPOAgent(policy=policy, lr=cfg.agent.lr)
        buffer = RolloutBuffer(num_steps, num_envs, rgb_shape, state_dim, action_dim, device)
    else:
        raise NotImplementedError(f"Training loop for agent {cfg.agent.name} is not fully integrated yet.")

    # --------------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------------
    global_step = 0
    num_updates = total_timesteps // (num_envs * num_steps)
    
    # Initial Reset
    obs, info = envs.reset()
    next_rgb = torch.tensor(obs["rgb"], device=device, dtype=torch.float32)
    next_state = torch.tensor(obs["state"], device=device, dtype=torch.float32)
    next_done = torch.zeros(num_envs, device=device, dtype=torch.float32)
    
    print(f"Starting Training: {num_updates} total updates.")
    
    for update in range(1, num_updates + 1):
        # 1. Rollout Phase: Collect Data
        policy.eval() # Setting policy to eval disables dropout (if any) during rollout
        
        for step in range(num_steps):
            global_step += num_envs
            
            # Get Action and Value from policy (no gradients needed here)
            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(next_rgb, next_state)
            
            # Step the environment
            action_np = action.cpu().numpy()
            obs, reward, terminated, truncated, info = envs.step(action_np)
            
            # Environment returns numpy arrays, move them to the PyTorch device
            reward = torch.tensor(reward, device=device, dtype=torch.float32)
            done = torch.tensor(terminated | truncated, device=device, dtype=torch.float32)
            
            # Store transition in buffer
            buffer.add(next_rgb, next_state, action, logprob, reward, next_done, value)
            
            # Update the "next" state for the next loop iteration
            next_rgb = torch.tensor(obs["rgb"], device=device, dtype=torch.float32)
            next_state = torch.tensor(obs["state"], device=device, dtype=torch.float32)
            next_done = done
            
            if "final_info" in info:
                for idx, final_info in enumerate(info["final_info"]):
                    if final_info and "episode" in final_info:
                        print(f"Step {global_step} | Env {idx} | Episode Return: {final_info['episode']['r']:.2f} | Length: {final_info['episode']['l']}")
                        logger.log_episode(global_step, final_info['episode']['r'], final_info['episode']['l'])

        # 2. Advantage Calculation Phase
        with torch.no_grad():
            _, _, _, next_value = policy.get_action_and_value(next_rgb, next_state)
            next_value = next_value.squeeze()
        
        buffer.compute_returns_and_advantages(agent, next_value, next_done)
        
        # 3. Optimization Phase
        policy.train()
        
        dataloader = buffer.get_dataloader(batch_size=batch_size)
        
        epoch_pg_loss = 0.0
        epoch_v_loss = 0.0
        epoch_entropy = 0.0
        
        for epoch in range(update_epochs):
            for b_rgb, b_state, b_actions, b_logprobs, b_returns, b_advantages in dataloader:
                # Agent Update Step
                pg_loss, v_loss, entropy = agent.update(
                    b_rgb, b_state, b_actions, b_logprobs, b_returns, b_advantages
                )
                
                epoch_pg_loss += pg_loss
                epoch_v_loss += v_loss
                epoch_entropy += entropy

        # Calculate average metrics over the update
        n_batches = update_epochs * len(dataloader)
        avg_v_loss = epoch_v_loss / n_batches
        avg_pg_loss = epoch_pg_loss / n_batches
        avg_ent = epoch_entropy / n_batches

        print(f"Update {update}/{num_updates} | Value Loss: {avg_v_loss:.4f} | Actor Loss: {avg_pg_loss:.4f} | Entropy: {avg_ent:.4f}")
        logger.log_metrics(global_step, avg_pg_loss, avg_v_loss, avg_ent)

        buffer.reset()
        
        if update % 50 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(policy.state_dict(), f"checkpoints/vla_policy_update_{update}.pt")

    logger.finish()

if __name__ == "__main__":
    train()
