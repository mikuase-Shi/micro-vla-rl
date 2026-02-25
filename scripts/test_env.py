import os
import sys

# Add the project root to the Python path so it can find the 'src' package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import mani_skill.envs
from src.envs.maniskill_wrapper import ManiSkillVLAWrapper

def test_maniskill_wrapper():
    raw_env=gym.make(
        "PushCube-v1",
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array"
    )
    env=ManiSkillVLAWrapper(raw_env)
    
    obs,info=env.reset()
    print(f"提取出的 RGB 图像形状: {obs['rgb'].shape}")
    print(f"提取出的 State 状态形状: {obs['state'].shape}")
    action=env.action_space.sample()
    obs,reward,terminated,truncated,info=env.step(action)

    print(f"执行动作后的 RGB 图像形状: {obs['rgb'].shape}")
    print(f"执行动作后的 State 状态形状: {obs['state'].shape}")
    print(f"当前动作获得的 Reward (奖励值): {reward}")
    
    env.close()

if __name__ == "__main__":
    test_maniskill_wrapper()