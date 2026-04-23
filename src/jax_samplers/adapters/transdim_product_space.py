# src/jax_samplers/adapters/transdim_product_space.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jr

from ..core.types import PRNGKey
from ..core.problem import Problem
from ..core.transdim import ModelFamily


# TransdimProductSpaceProblem (model-agnostic)
class TransdimProductSpaceProblem:
    def __init__(self, family):
        self.family = family
        self.__post_init__()  # so your setup runs
    
    def __post_init__(self):
        self.Ks            = tuple(self.family.Ks)
        self.K_min         = int(self.family.K_min)
        self.K_max         = int(self.family.K_max)
        self.dpa           = int(self.family.dim_per_atom)
        self.D_max         = self.K_max * self.dpa
        self.dim           = 1 + self.D_max

        # cache log p(K) with symmetry correction if needed
        logpK = [self.family.logpmf_K(k) for k in self.Ks]
        if getattr(self.family, "components_exchangeable", False):
            # unordered set correction: -log(K!)
            logpK = [lp - jax.lax.lgamma(k + 1.0) for lp, k in zip(logpK, self.Ks)]
        self.logpK = jnp.array(logpK)  # shape (|Ks|,)

    def sample_prior(self, key, n):
        key, kK, kTheta = jax.random.split(key, 3)
        Ks  = self.family.sample_K(kK, n).astype(jnp.int32)         # (n,)
        #X   = self.family.sample_prior(kTheta, self.K_max, n)        # (n, D_max) active@K_max
        X = self.family.sample_prior_given_K(kTheta, self.K_max, n)

        if X.shape[1] != self.D_max:
            raise RuntimeError(
                f"family.sample_prior_given_K returned shape {X.shape}, expected (_, {self.D_max}). "
                "It must return ONLY the parameter block (no K column)."
            )
        
        Kf  = Ks.astype(X.dtype)
        return jnp.concatenate([Kf[:, None], X], axis=1)             # (n, 1 + D_max)

    def _logpK(self, K: jnp.ndarray) -> jnp.ndarray:
        validK = (K >= self.K_min) & (K <= self.K_max)
        val = jnp.where(validK, self.logpK[K - self.K_min], -jnp.inf)
        return val

    def logprior(self, theta: jnp.ndarray) -> jnp.ndarray:
        Kf, x = theta[0], theta[1:]
        return self._logpK(Kf.astype(jnp.int32)) + self.family.logprior_full(theta)

    def loglikelihood(self, theta: jnp.ndarray) -> jnp.ndarray:
        Kf, x = theta[0], theta[1:]
        validK = (Kf >= self.K_min) & (Kf <= self.K_max)
        ll = self.family.loglikelihood_full(theta)
        return jnp.where(validK, ll, -jnp.inf)

    def logprior_org(self, theta: jnp.ndarray) -> jnp.ndarray:
        Kf, x = theta[0], theta[1:]
        theta_full = jnp.concatenate([jnp.atleast_1d(Kf), x])
        return self._logpK(Kf.astype(jnp.int32)) + self.family.logprior_full(theta_full)
        #return self._logpK(Kf.astype(jnp.int32)) + self.family.logprior_full(x, Kf)

    def loglikelihood_org(self, theta: jnp.ndarray) -> jnp.ndarray:
        Kf, x = theta[0], theta[1:]
        validK = (Kf >= self.K_min) & (Kf <= self.K_max)
        #ll = self.family.loglikelihood_full(x, Kf)
        theta_full = jnp.concatenate([jnp.atleast_1d(Kf), x])
        ll = self.family.loglikelihood_full(theta_full)
        return jnp.where(validK, ll, -jnp.inf)
