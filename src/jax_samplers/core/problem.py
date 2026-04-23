# src/jax_samplers/core/problem.py
from __future__ import annotations
from typing import Optional, Sequence, Tuple, Callable
import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Any

Array = jnp.ndarray

# Version-agnostic PRNGKey alias
try:
    # Newer JAX releases
    from jax.random import KeyArray as PRNGKey
except Exception:
    # Older JAX: just treat keys as arrays (or Any for typing)
    PRNGKey = Any  # or: PRNGKey = jnp.ndarray

class Problem:
    """
    Unified interface consumed by all samplers.
    Implement loglikelihood/logprior for a flat θ of length `dim`.
    Optionally provide prior_bounds and sample_prior.
    """

    # required
    dim: int

    # optional but recommended
    prior_bounds: Optional[Sequence[Tuple[float, float]]] = None

    # ---- required to implement ----
    def _loglikelihood_single(self, theta: Array) -> Array:
        """Return scalar log-likelihood for θ (shape (dim,))."""
        raise NotImplementedError

    def _logprior_single(self, theta: Array) -> Array:
        """Return scalar log-prior for θ (shape (dim,))."""
        raise NotImplementedError

    # ---- optional to implement ----
    def sample_prior(self, key: PRNGKey, n: int) -> Array:
        """Return (n, dim) θ drawn from the prior. Default: unavailable."""
        raise NotImplementedError("sample_prior not implemented for this Problem.")

    # ---- public, batched-friendly wrappers ----
    def loglikelihood(self, theta: Array) -> Array:
        fn: Callable[[Array], Array] = self._loglikelihood_single
        return fn(theta) if theta.ndim == 1 else jax.vmap(fn)(theta)

    def logprior(self, theta: Array) -> Array:
        fn: Callable[[Array], Array] = self._logprior_single
        return fn(theta) if theta.ndim == 1 else jax.vmap(fn)(theta)
