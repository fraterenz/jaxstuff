from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from sklearn.datasets import make_blobs
from jaxtyping import Array, Int32, Float


"""Parameters to infer."""
Theta = Float[Array, "D"]
"""The features of the data."""
Phi = Float[Array, "N D"]
"""Labels to infer."""
Labels = Int32[Array, "N"]
LogDensity = Float[Array, ""]

LogLikelihood = Callable[[Phi, Theta, Labels], LogDensity]
LogPrior = Callable[[Theta], LogDensity]


@dataclass
class InferredModePosterior:
    mode: Theta
    logdensity: LogDensity
    initial: Theta
    logdensities: Float[Array, "Iterations"]


@dataclass
class ModePosteriorOptimisation:
    inferred_mode: InferredModePosterior
    # tol: float # don't do early stopping for now
    iterations: int


@dataclass
class ModePosteriorNewton:
    optimisation: ModePosteriorOptimisation
    # Hessian of the neg (?) log likelihood
    curvature: Float[Array, "D D"]


def logdensity(
    theta: Theta,
    log_likelihood: LogLikelihood,
    log_prior: LogPrior,
    features: Phi,
    labels: Labels,
) -> LogDensity:
    return log_likelihood(features, theta, labels) + log_prior(theta)


logdensity_val_grad = jax.value_and_grad(logdensity)


def find_mode_gd(
    initial: Theta,
    labels: Labels,
    features: Phi,
    log_likelihood: LogLikelihood,
    log_prior: LogPrior,
    eps: float,
    iterations: int,
    # tol: float,
) -> ModePosteriorOptimisation:
    def one_step(state, _) -> Tuple[Theta, LogDensity]:
        logdensity, grad_logdensity = logdensity_val_grad(
            state, log_likelihood, log_prior, features, labels
        )
        state = state + eps * grad_logdensity
        return state, logdensity

    @partial(jax.jit, static_argnames=("iterations",))
    def run(initial, iterations):
        # helper to jit the outer function only
        return jax.lax.scan(one_step, initial, xs=None, length=iterations)

    mode, logdensities = run(initial, iterations)
    inferred_mode = InferredModePosterior(
        mode,
        logdensities[-1],
        initial,
        logdensities,
    )
    return ModePosteriorOptimisation(inferred_mode, iterations)
