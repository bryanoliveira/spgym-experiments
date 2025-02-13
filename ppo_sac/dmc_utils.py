import os
os.environ['MUJOCO_GL'] = 'egl'

import gymnasium as gym
import numpy as np
import shimmy


class FrameStackToChannelAxisWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Adjust observation space to account for the new shape
        original_space = self.observation_space
        assert len(original_space.shape) == 4, "Observation must be framestacked (i, w, h, c)."
        frames, width, height, channels = original_space.shape
        self.observation_space = gym.spaces.Box(
            low=original_space.low.min(),
            high=original_space.high.max(),
            shape=(frames * channels, width, height),
            dtype=original_space.dtype,
        )
    def observation(self, observation):
        # Reorder dimensions to merge frames and channels
        return np.array(observation).transpose(0, 3, 1, 2).reshape(-1, *observation.shape[1:3])

class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, repeat):
        super().__init__(env)
        self.repeat = repeat
    def step(self, action):
        obs, total_reward, terminated, truncated, info = self.env.step(action)
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
