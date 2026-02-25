import gymnasium as gym
import numpy as np

class LiberoVLAWrapper(gym.ObservationWrapper):
    """
    Gym wrapper for the LIBERO environment.
    Formats the observation space into 'rgb' and 'state' to match the Micro-VLA requirements.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Standardize observation space to match VLA expectations
        self.observation_space = gym.spaces.Dict({
            "rgb": gym.spaces.Box(low=0, high=255, shape=(3, 256, 256), dtype=np.uint8),
            "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32)
        })
        
    def observation(self, obs):
        # Extract and format the RGB image
        # LIBERO typically provides agentview_image
        if 'agentview_image' in obs:
            rgb = obs['agentview_image']
            # Convert channels-last to channels-first if needed
            if rgb.shape[-1] == 3:
                rgb = np.transpose(rgb, (2, 0, 1))
        else:
            # Fallback shape if rendering is disabled or dictionary keys differ
            rgb = np.zeros((3, 256, 256), dtype=np.float32)
            
        # Extract state details (proprioception)
        state = np.zeros(47, dtype=np.float32)
        if 'robot_state' in obs:
            rs = obs['robot_state']
            # Limit the array to the first 47 dims or pad zeroes
            length = min(len(rs), 47)
            state[:length] = rs[:length]
            
        return {
            "rgb": rgb,
            "state": state
        }
