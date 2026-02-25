import os
import sys
import gymnasium as gym

# Add project root to sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.maniskill_wrapper import ManiSkillVLAWrapper

def test_rollout_with_random_actions(env_id: str = "PushCube-v1", max_steps: int = 100):
    """
    Sets up the environment exactly as the training script does, but uses
    random actions to collect an episode and record a video.
    This helps visually debug the observation wrapper and camera angles.
    """
    import mani_skill.envs
    
    # Needs to be rgb_array to record videos
    raw_env = gym.make(
        env_id, 
        obs_mode="rgbd", 
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array"
    )
    
    # Record video wrapper before our VLA wrapper so it catches the raw rgb dict structure
    os.makedirs("videos", exist_ok=True)
    raw_env = gym.wrappers.RecordVideo(raw_env, video_folder="videos", name_prefix="random_rollout")
    
    env = ManiSkillVLAWrapper(raw_env)
    
    obs, info = env.reset()
    print(f"Initial RGB shape: {obs['rgb'].shape}")
    print(f"Initial State shape: {obs['state'].shape}")
    
    total_reward = 0.0
    
    for step in range(max_steps):
        # Sample random continuous action
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished at step {step}")
            break
            
    print(f"Total accumulated reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    test_rollout_with_random_actions()
