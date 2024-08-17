import gymnasium
import numpy as np
from gymnasium import Wrapper
import gymnasium as gym


def make_non_absorbing(observation):
    return np.concatenate([observation, [0.0]], -1)


class AbsorbingStatesWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        low = env.observation_space.low
        high = env.observation_space.high
        self._absorbing_state = np.concatenate([np.zeros_like(low), [1.0]], 0)
        low = np.concatenate([low, [0]], 0)
        high = np.concatenate([high, [1]], 0)

        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        self._done = False
        self._absorbing = False
        obs, info = self.env.reset(**kwargs)
        self._info = info
        return make_non_absorbing(obs), info

    def step(self, action):
        if not self._done:
            observation, reward, done, truncate, info = self.env.step(action)
            observation = make_non_absorbing(observation)
            self._done = done
            self._info = info
            if 'TimeLimit.truncated' in info:
                truncate = info['TimeLimit.truncated']
            return observation, reward, done, truncate, info
        else:
            if not self._absorbing:
                self._absorbing = True
                return self._absorbing_state, 0.0, False, self._info
            else:
                return self._absorbing_state, 0.0, True, self._info


if __name__ == '__main__':
    env = gymnasium.make('Hopper-v4')
    env = AbsorbingStatesWrapper(env)
    env.reset()

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncate, info = env.step(action)
        print(obs, done)
