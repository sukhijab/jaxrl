from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper

from jaxrl import wrappers


def make_env(env_name: str,
             seed: int,
             save_folder: Optional[str] = None,
             add_episode_monitor: bool = True,
             action_repeat: int = 1,
             action_cost: float = 0.0,
             frame_stack: int = 1,
             from_pixels: bool = False,
             pixels_only: bool = True,
             image_size: int = 84,
             sticky: bool = False,
             gray_scale: bool = False,
             flatten: bool = True) -> gym.Env:
    # Check if the env is in gym.
    all_envs = gym.envs.registry.values()
    env_ids = [env_spec.id for env_spec in all_envs]

    if from_pixels:
        if env_name in env_ids:
            camera_id = 0
        else:
            domain_name, task_name = env_name.split('-')
            camera_id = 2 if domain_name == 'quadruped' else 0
        render_kwargs = {
            'height': image_size,
            'width': image_size,
            'camera_id': camera_id
        }
    else:
        render_kwargs = {}

    if env_name in env_ids:
        env = gym.make(env_name, **render_kwargs)
    else:
        domain_name, task_name = env_name.split('-')
        env = wrappers.DMCEnv(domain_name=domain_name,
                              task_name=task_name,
                              task_kwargs={'random': seed},
                              **render_kwargs)

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    env = wrappers.ActionCost(env, action_cost=action_cost)

    if add_episode_monitor:
        env = wrappers.EpisodeMonitor(env)

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    env = RescaleAction(env, -1.0, 1.0)

    if save_folder is not None:
        env = gym.wrappers.RecordVideo(env, save_folder)

    if from_pixels:
        env = PixelObservationWrapper(env,
                                      pixels_only=pixels_only)
        env = wrappers.TakeKey(env, take_key='pixels')
        if gray_scale:
            env = wrappers.RGB2Gray(env)
    else:
        env = wrappers.SinglePrecision(env)

    if frame_stack > 1:
        env = wrappers.FrameStack(env, num_stack=frame_stack)

    if sticky:
        env = wrappers.StickyActionEnv(env)

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env
