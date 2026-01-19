"""A logistic regression model for a classification of datapoints into two classes."""

import jax
import jax.numpy as jnp
from sklearn.datasets import make_blobs
from jaxtyping import Array, Int32, Float
from typing import Tuple
from jaxstuff.inference import Labels, Phi, Theta


def make_blobs_two_cls(
    num_points: int, random_state, centers=((-3, -3), (3, 3)), cluster_std=1.5
) -> Tuple[Float[Array, "N M"], Int32[Array, "N"]]:
    X, y = make_blobs(
        num_points,
        2,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return jnp.asarray(X, dtype=jnp.float32), jnp.where(y, jnp.int32(-1), jnp.int32(1))


def logprior_gaussian(w: Theta, alpha=1.0, sigma=2.0) -> Float[Array, ""]:
    return -alpha * w @ w / (2 * sigma**2)


def loglikelihood(features: Phi, w: Theta, y: Labels) -> Float[Array, ""]:
    logits = features @ w  # linear classfier in the feature space
    return jax.nn.log_sigmoid(y * logits).mean()
