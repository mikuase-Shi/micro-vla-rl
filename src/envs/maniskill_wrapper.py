import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ManiSkillVLAWrapper(gym.ObservationWrapper):
    """
    A custom wrapper for ManiSkill environments to format observations
    specifically for a VLA + Residual RL policy.
    """
    def __init__(self, env):
        super().__init__(env)

        self.observation_space=spaces.Dict({
            "rgb":spaces.Box(
                low=0.0,
                high=1.0,
                shape=(3,256,256),
                dtype=np.float32
            ),
            "state":spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(47,),
                dtype=np.float32
            )
        })
        


    def observation(self, obs):
        """
        Process the raw ManiSkill nested dictionary observation into 
        a flat dict with just {"rgb": image_tensor, "state": flat_1d_tensor}.
        """
        
        raw_image=obs["image"]["base_camera"]["rgb"]
        
        processed_image=np.transpose(raw_image,(2,0,1))
        processed_image=processed_image.astype(np.float32)/255.0

        qpos=obs["agent"]["qpos"].flatten().astype(np.float32)
        qvel=obs["agent"]["qvel"].flatten().astype(np.float32)
        tcp_pose=obs["extra"]["tcp_pose"].flatten().astype(np.float32)
        flat_state=np.concatenate([qpos,qvel,tcp_pose]) 
        
        final_obs={"rgb":processed_image,"state":flat_state}
        
        return final_obs
