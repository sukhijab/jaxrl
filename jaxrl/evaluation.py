from typing import Dict

import gymnasium as gym
import numpy as np


def evaluate(agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    successes = None
    for _ in range(num_episodes):
        observation, _ = env.reset()
        finish = False
        while not finish:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, truncate, info = env.step(action)
            finish = done or truncate
        if 'episode' in info:
            for k in stats.keys():
                stats[k].append(info['episode'][k])

        if 'is_success' in info:
            if successes is None:
                successes = 0.0
            successes += info['is_success']

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes
    return stats
