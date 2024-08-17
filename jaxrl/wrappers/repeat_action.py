import gymnasium as gym
import numpy as np

from jaxrl.wrappers.common import TimeStep


class RepeatAction(gym.Wrapper):

    def __init__(self, env, action_repeat=4):
        super().__init__(env)
        assert action_repeat >= 1
        self._action_repeat = action_repeat

    def step(self, action: np.ndarray) -> TimeStep:
        total_reward = 0.0
        done = None
        combined_info = {}

        for _ in range(self._action_repeat):
            obs, reward, done, truncate, info = self.env.step(action)
            total_reward += reward
            combined_info.update(info)
            if done:
                break

        return obs, total_reward, done, truncate, combined_info
