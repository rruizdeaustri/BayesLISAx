from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Tuple

import jax.numpy as jnp
from ..core.types import PRNGKey

Array = jnp.ndarray


class RJProposalPair(Protocol):
    """
    A reversible pair of proposals between adjacent model sizes (k ↔ k+1).
    Implement *both* directions and return the log proposal density.
    """

    def propose_up(self, key: PRNGKey, k: int, theta_k: Array) -> Tuple[Array, float]:
        """
        From (k, theta_k) propose theta_{k+1} for model k+1.
        Returns (theta_{k+1}, log_q_forward).
        """
        ...

    def propose_down(self, key: PRNGKey, k: int, theta_k1: Array) -> Tuple[Array, float]:
        """
        From (k+1, theta_{k+1}) propose theta_k for model k.
        Returns (theta_k, log_q_reverse).
        NOTE: `k` is the *target* smaller model index.
        """
        ...


@dataclass
class RJGenericConfig:
    steps: int = 20_000
    p_up: float = 0.3        # probability of attempting a trans-dim move
    sigma_rw: float = 0.2    # within-model random-walk std (if using RW)
