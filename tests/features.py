import jax.numpy as jnp
import pytest

from jaxstuff import features


testdata = [
    (jnp.array([[0, 0], [2, 2]]), jnp.ones((2,)), 2 * jnp.ones((2,))),
    (jnp.array([[0, 0], [-2, -2]]), -jnp.ones((2,)), 2 * jnp.ones((2,))),
    (jnp.array([[10, 10], [-10, -10]]), jnp.zeros((2,)), 20 * jnp.ones((2,))),
    (
        jnp.array([[1, 10], [-1, -10]]),
        jnp.zeros((2,)),
        jnp.array([2, 20]) * jnp.ones((2,)),
    ),
    (
        jnp.array([[10, 1], [-10, -1]]),
        jnp.zeros((2,)),
        jnp.array([20, 2]) * jnp.ones((2,)),
    ),
]


@pytest.mark.parametrize("x,exp_center,exp_scale", testdata)
def test_computer_center_scale(x, exp_center, exp_scale):
    center, scale = features.compute_center_scale(x)
    assert jnp.array_equal(center, exp_center)
    assert jnp.array_equal(scale, exp_scale)
