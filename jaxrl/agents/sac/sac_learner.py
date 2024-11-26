"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple, List

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.sac import temperature
from jaxrl.agents.sac.actor import update as update_actor
from jaxrl.agents.sac.critic import target_update
from jaxrl.agents.sac.critic import update as update_critic
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey


@functools.partial(jax.jit,
                   static_argnames=('backup_entropy', 'update_target', 'use_log_transform'))
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
    temp: Model, batch: Batch, discount: float, tau: float,
    target_entropy: float, backup_entropy: bool, update_target: bool, use_log_transform: bool,
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            backup_entropy=backup_entropy)
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy, use_log_transform=use_log_transform)

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


class SACLearner(object):

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 backup_entropy: bool = True,
                 init_temperature: float = 1.0,
                 init_mean: Optional[np.ndarray] = None,
                 use_log_transform: bool = True,
                 policy_final_fc_init_scale: float = 1.0,
                 use_bronet: bool = False,
                 reset_period: Optional[int] = None,
                 reset_models: bool = False,
                 ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """
        self.use_log_transform = use_log_transform
        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        if reset_period is None:
            self.reset_period = 250_000
        else:
            self.reset_period = reset_period

        self._reset_models = reset_models
        self.use_bronet = use_bronet

        # kwargs saved for resetting
        self.hidden_dims = hidden_dims
        self.dummy_obs = observations
        self.dummy_acts = actions
        self.policy_final_fc_init_scale = policy_final_fc_init_scale
        self.init_mean = init_mean
        self.init_temperature = init_temperature
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.temp_lr = temp_lr


        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        actor_def = policies.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            init_mean=init_mean,
            final_fc_init_scale=policy_final_fc_init_scale)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        critic_def = critic_net.DoubleCritic(hidden_dims, use_bronet=use_bronet)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=temp_lr))

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.rng = rng

        self.step = 1

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> np.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        if self._reset_models:
            if self.step % self.reset_period == 1 and self.step > 1:
                rng, self.rng = jax.random.split(self.rng)
                self.reset_models(rng=rng)

        self.step += 1
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.target_entropy,
            self.backup_entropy, self.step % self.target_update_period == 0,
            use_log_transform=self.use_log_transform)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info

    def reset_models(self, rng: PRNGKey):
        print(f'resetting model at step: {self.step}')
        actor_key, critic_key, temp_key = jax.random.split(rng, 3)
        action_dim = self.dummy_acts.shape[-1]
        actor_def = policies.NormalTanhPolicy(
            self.hidden_dims,
            action_dim,
            init_mean=self.init_mean,
            final_fc_init_scale=self.policy_final_fc_init_scale)
        actor = Model.create(actor_def,
                             inputs=[actor_key, self.dummy_obs],
                             tx=optax.adam(learning_rate=self.actor_lr))

        critic_def = critic_net.DoubleCritic(self.hidden_dims, use_bronet=self.use_bronet)
        critic = Model.create(critic_def,
                              inputs=[critic_key, self.dummy_obs, self.dummy_acts],
                              tx=optax.adam(learning_rate=self.critic_lr))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, self.dummy_obs, self.dummy_acts])

        temp = Model.create(temperature.Temperature(self.init_temperature),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=self.temp_lr))

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
