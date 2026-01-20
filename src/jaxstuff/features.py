from typing import Tuple
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Float, Array
from jaxstuff.types import Data, Phi


def linear_features(x: Data) -> Phi:
    n = x.shape[0]
    return jnp.c_[jnp.ones((n, 1)), x]


def compute_center_scale(x: Data) -> Tuple[Float[Array, "D"], Float[Array, "D"]]:
    """Compute center and scale of the data's domain.
    center: the center of the data.
    scale: total unsigned width covered by the data.
    """
    max_, min_ = x.max(axis=0), x.min(axis=0)
    return 0.5 * (min_ + max_), max_ - min_


def gaussian_features(x: Data, n_features: int, feat_key, ell: float = 0.2) -> Phi:
    centers = jrd.uniform(
        feat_key,
        shape=(n_features, x.shape[1]),
        minval=x.min(axis=0),
        maxval=x.max(axis=0),
    )
    return jnp.exp(
        -0.5 * jnp.sum(((x[:, None, :] - centers[None, :, :]) ** 2) / (ell**2), axis=-1)
    )
