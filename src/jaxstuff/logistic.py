"""A logistic regression model for a classification of datapoints into two classes."""

import jax
import jax.numpy as jnp
from sklearn.datasets import make_blobs
from jaxtyping import Array, Int32, Float
from typing import Tuple
from jaxstuff.inference import Labels, Phi, Theta
from jaxstuff.types import Assignments, Data


def make_blobs_two_cls(
    num_points: int, random_state, centers=((-3, -3), (3, 3)), cluster_std=1.5
) -> Tuple[Data, Labels]:
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


def push_forward(features: Phi, w: Theta) -> Assignments:
    """Compute the point prediction for the class +1 from the weights."""
    return jax.nn.sigmoid(features @ w)


def push_forward_samples(
    features: Phi, w_samples: Float[Array, "S D"]
) -> Float[Array, "S N"]:
    """Compute predictions for the class +1 from a bunch of samples of the weights."""
    return jax.nn.sigmoid(jnp.einsum("nd,sd->ns", features, w_samples))
