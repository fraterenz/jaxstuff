import jax
import jax.numpy as jnp
from jaxstuff.inference import find_mode_gd
from jaxstuff.logistic import loglikelihood, logprior_gaussian, make_blobs_two_cls
import pytest
import numpy as np
from datetime import date


class RandState:
    state: int

    def __init__(self):
        # bad idea but ok for now
        self.state = int(date.today().strftime("%Y%m%d"))


@pytest.mark.parametrize("nb_points", [20, 50, 100])
def test_make_blobs_two_cls(nb_points):
    X, y = make_blobs_two_cls(nb_points, RandState().state)
    assert y.shape == (nb_points,)
    assert isinstance(X, jax.Array)
    assert isinstance(y, jax.Array)
    assert X.dtype == jnp.float32
    assert y.dtype == jnp.int32
    assert int(jnp.where(y == -1, 1, y).sum()) == nb_points


@pytest.mark.parametrize("rnd_state", [12121, 50, 1010])
def test_gradient_ascent_linear_features(rnd_state: int):
    D, N, iterations = 3, 50, 10
    rng_key = jax.random.key(rnd_state)
    rng_key, data_key, w_key = jax.random.split(rng_key, 3)
    labels = jnp.ones((N,))
    X = jax.random.uniform(data_key, (N, D - 1))
    initial = jax.random.normal(w_key, (D,))
    features = jnp.c_[jnp.ones((N, 1)), X]
    mode_posterior = find_mode_gd(
        initial,
        labels,
        features,
        log_likelihood=loglikelihood,
        log_prior=logprior_gaussian,
        eps=0.1,
        iterations=iterations,
    )
    assert mode_posterior.inferred_mode.mode.shape == initial.shape
    assert jnp.linalg.vector_norm(
        mode_posterior.inferred_mode.mode
    ) != jnp.linalg.vector_norm(initial)
    assert mode_posterior.inferred_mode.logdensities.shape == (iterations,)
