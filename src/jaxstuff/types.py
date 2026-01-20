from typing import Callable
from jaxtyping import Array, Int32, Float

"""Parameters to infer."""
Theta = Float[Array, "D"]
"""The features of the data."""
Phi = Float[Array, "N D"]
"""Labels to infer."""
Labels = Int32[Array, "N"]
Assignments = Float[Array, "N"]
LogDensity = Float[Array, ""]

LogLikelihood = Callable[[Phi, Theta, Labels], LogDensity]
LogPrior = Callable[[Theta], LogDensity]

Data = Float[Array, "N M"]
