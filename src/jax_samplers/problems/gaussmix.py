# src/jax_samplers/problems/gaussmix.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp

from ..core.types import PRNGKey  # cross-version PRNG key alias


@dataclass
class TrueParams:
    w: jnp.ndarray
    mu: jnp.ndarray
    sd: jnp.ndarray


@dataclass
class GaussianMixtureMeans:
    n: int = 400
    lower: float = -10.0
    upper: float = 10.0
    #key: PRNGKey = jr.PRNGKey(0)
    key: any = field(default_factory=lambda: jr.PRNGKey(0))
    
    def __post_init__(self):
        self.true = TrueParams(
            w=jnp.array([0.45, 0.55]),
            mu=jnp.array([-2.0, 2.0]),
            sd=jnp.array([0.4, 0.6]),
        )
        self.key, k1, k2 = jr.split(self.key, 3)
        z = jr.categorical(k1, jnp.log(self.true.w), shape=(self.n,))
        self.y = jr.normal(k2, (self.n,)) * self.true.sd[z] + self.true.mu[z]
        self.dim = 2

    def logprior_old(self, theta: jnp.ndarray) -> jnp.ndarray:
        inb = jnp.all((theta >= self.lower) & (theta <= self.upper))
        return jnp.where(inb, -self.dim * jnp.log(self.upper - self.lower), -jnp.inf)

    def loglikelihood_old(self, theta: jnp.ndarray) -> jnp.ndarray:
        mu0, mu1 = theta
        lp0 = jsp.stats.norm.logpdf(self.y, mu0, self.true.sd[0]) + jnp.log(self.true.w[0])
        lp1 = jsp.stats.norm.logpdf(self.y, mu1, self.true.sd[1]) + jnp.log(self.true.w[1])
        print('here', lp0, lp1, theta)
        sys.exit()
        return jnp.sum(jsp.special.logsumexp(jnp.stack([lp0, lp1], 1), 1))

    def logprior(self, theta: jnp.ndarray) -> jnp.ndarray:
        def single_logprior(t):
            in_bounds = jnp.all((t >= self.lower) & (t <= self.upper))
            return jnp.where(in_bounds, -self.dim * jnp.log(self.upper - self.lower), -jnp.inf)

        if theta.ndim == 1:
            return single_logprior(theta)
        elif theta.ndim == 2:
            return jax.vmap(single_logprior)(theta)
        else:
            raise ValueError(f"Invalid shape for theta: {theta.shape}")
        
    def loglikelihood(self, theta: jnp.ndarray) -> jnp.ndarray:
        def single_loglike(t):
            mu0, mu1 = t
            lp0 = jsp.stats.norm.logpdf(self.y, mu0, self.true.sd[0]) + jnp.log(self.true.w[0])
            lp1 = jsp.stats.norm.logpdf(self.y, mu1, self.true.sd[1]) + jnp.log(self.true.w[1])
            return jnp.sum(jsp.special.logsumexp(jnp.stack([lp0, lp1], axis=1), axis=1))
        if theta.ndim == 1:  # unbatched call (e.g., shape (2,))
            return single_loglike(theta)
        elif theta.ndim == 2:  # batched call (e.g., shape (N, 2))
            return jax.vmap(single_loglike)(theta)
        else:
            raise ValueError(f"Invalid shape for theta: {theta.shape}")
        
        return jax.vmap(single_loglike)(theta)

    
    def sample_prior(self, key: PRNGKey, n: int) -> jnp.ndarray:
        key, sub = jr.split(key)
        return jr.uniform(sub, (n, self.dim), minval=self.lower, maxval=self.upper)

    
    def truths(self) -> Tuple[float, float]:
        return float(self.true.mu[0]), float(self.true.mu[1])

