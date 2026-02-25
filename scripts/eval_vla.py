import os
import sys
import torch
import gymnasium as gym

# Add project root to sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.maniskill_wrapper import ManiSkillVLAWrapper
from src.models.residual import MicroVLAPolicy

def eval_policy(weights_path: str, env_id: str = "PushCube-v1", seed: int = 42, max_steps: int = 200, output_video: str = "eval_video.mp4"):
    """
    Evaluates a trained MicroVLAPolicy by running a single rollout and saving a video.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading weights from {weights_path} onto {device}...")

    # Create the environment with video recording enabled
    import mani_skill.envs # Load the registered environments
    raw_env = gym.make(
        env_id, 
        obs_mode="rgbd", 
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array" # Required for video
    )
    raw_env = gym.wrappers.RecordVideo(raw_env, video_folder="eval_videos", name_prefix=output_video.split('.')[0])
    env = ManiSkillVLAWrapper(raw_env)

    # Initialize Model and load weights
    action_dim = env.action_space.shape[0]
    policy = MicroVLAPolicy(action_dim=action_dim).to(device)
    
    # Load state dict
    if os.path.exists(weights_path):
        policy.load_state_dict(torch.load(weights_path, map_location=device))
        print("Weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Could not find weights file: {weights_path}")
    
    policy.eval()

    # Run the episode
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    
    print(f"Starting evaluation rollout in {env_id}...")
    for step in range(max_steps):
        # Format observations
        rgb = torch.tensor(obs["rgb"], device=device, dtype=torch.float32).unsqueeze(0) # Add batch dim
        state = torch.tensor(obs["state"], device=device, dtype=torch.float32).unsqueeze(0)
        
        # Get purely deterministic action from the mean
        with torch.no_grad():
            _, _, _, _ = policy.get_action_and_value(rgb, state)
            # In evaluation, we don't want to sample. 
            # We want to take the actor_mean directly.
            # So, we pass it through the network manually to get the deterministic action.
            
            vision_features = policy.vla_encoder(rgb)
            state_features = policy.state_mlp(state)
            combined_features = torch.cat([vision_features, state_features], dim=1)
            deterministic_action = policy.actor_mean(combined_features)
            
            action_np = deterministic_action.cpu().numpy().squeeze(0) # Remove batch dim

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action_np)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished early at step {step}.")
            break
            
    print(f"Evaluation complete. Total Reward: {total_reward:.2f}")
    env.close()
    print("Video saved to eval_videos/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to the .pt model weights file")
    parser.add_argument("--env", type=str, default="PushCube-v1")
    args = parser.parse_args()
    
    eval_policy(args.weights, args.env)
