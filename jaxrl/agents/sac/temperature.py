from typing import Tuple

import jax.lax
import jax.numpy as jnp
from flax import linen as nn

from jaxrl.networks.common import InfoDict, Model


class Temperature(nn.Module):
    initial_temperature: float = 1.0
    min_temp: float = 1e-8
    max_temp: float = 1e8

    @nn.compact
    def __call__(self) -> [jnp.ndarray, jnp.ndarray]:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        # clip log temp to avoid too large or small values
        clamped_log_temp = jax.lax.stop_gradient(jnp.clip(log_temp,
                                                          min=jnp.log(self.min_temp), max=jnp.log(self.max_temp)))
        # clips gradient but maintains derivative
        log_temp = log_temp - jax.lax.stop_gradient(log_temp) + clamped_log_temp
        # log_temp = jnp.clip(log_temp, min=jnp.log(self.min_temp), max=jnp.log(self.max_temp))
        return jnp.exp(log_temp), log_temp


def update(temp: Model, entropy: float,
           target_entropy: float, use_log_transform: bool = True) -> Tuple[Model, InfoDict]:

    def temperature_loss_fn(temp_params):
        temperature, log_temp = temp.apply_fn({'params': temp_params})
        if use_log_transform:
            temp_loss = log_temp * (entropy - target_entropy).mean()
        else:
            temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss, 'log_temp': log_temp}

    new_temp, info = temp.apply_gradient(temperature_loss_fn)

    return new_temp, info

