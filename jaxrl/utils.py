import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper

from jaxrl import wrappers

import os
from typing import Optional, Callable

from gymnasium.wrappers.monitoring import video_recorder


class RecordVideo(gym.wrappers.RecordVideo):

    def __init__(self,
                 env: gym.Env,
                 video_folder: str,
                 episode_trigger: Callable[[int], bool] = None,
                 step_trigger: Callable[[int], bool] = None,
                 video_length: int = 0,
                 name_prefix: str = "rl-video",
                 disable_logger: bool = False,
                 image_size: int = 1024,
                 from_pixels: bool = False,
                 ):
        super().__init__(env=env,
                         video_folder=video_folder,
                         episode_trigger=episode_trigger,
                         step_trigger=step_trigger,
                         video_length=video_length,
                         name_prefix=name_prefix,
                         disable_logger=disable_logger,
                         )
        self.image_size = image_size
        self.from_pixels = from_pixels

    def start_video_recorder(self):
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        base_path = os.path.join(self.video_folder, video_name)
        if not self.from_pixels:
            env = PixelObservationWrapper(self.env,
                                          pixels_only=True)
            env = wrappers.TakeKey(env, take_key='pixels')
            env = gym.wrappers.ResizeObservation(env, self.image_size)
        else:
            env = gym.wrappers.ResizeObservation(self.env, self.image_size)
        self.video_recorder = video_recorder.VideoRecorder(
            env=env,
            base_path=base_path,
            metadata={"step_id": self.step_id, "episode_id": self.episode_id},
            disable_logger=self.disable_logger,
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True


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
             flatten: bool = True,
             recording_image_size: int = 1024,
             ) -> gym.Env:
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
        if env_name in env_ids:
            render_kwargs = {'render_mode': 'rgb_array'}
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
        env = RecordVideo(env, save_folder, image_size=recording_image_size, from_pixels=from_pixels)

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
