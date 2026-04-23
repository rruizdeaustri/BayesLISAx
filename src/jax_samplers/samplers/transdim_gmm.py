# src/jax_samplers/problems/transdim_gmm.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp

from ..core.types import PRNGKey


@dataclass
class TransdimGaussianMixtureMeans:
    """Trans-dimensional Gaussian-mixture means problem for Nested Sampling.

    State vector is length ``dim = 1 + K_max`` with layout:
        position[0]   = K encoded as a float (rounded to int inside logdensities)
        position[1:]  = mu_full of shape (K_max,)

    Prior:
      - K ~ Uniform{1, ..., K_max}
      - mu_i ~ Uniform(lower, upper) independently for *active* components (i < K)

    Likelihood (equal-weight mixture over active components):
      y_n ~ sum_{i < K} (1/K) * Normal(mu_i, sd_by_comp[i])

    Notes
    -----
    * This keeps the overall dimension fixed by encoding the discrete K as a float in
      the first coordinate and using a mask `i < K` to select active components.
    * Inside the log-densities we cast `K = int(position[0])` and ignore inactive mu's.
    * For SMC/MCMC kernels that propose continuous moves, this representation avoids
      changing dimensionality while still allowing trans-dimensional inference.
    """

    y: jnp.ndarray                    # observed data, shape (n,)
    K_max: int = 5
    lower: float = -10.0
    upper: float = 10.0
    sd_by_comp: Optional[jnp.ndarray] = None  # shape (K_max,)

    def __post_init__(self):
        if self.sd_by_comp is None:
            # default: moderately wide comps; customize as needed
            self.sd_by_comp = jnp.ones((self.K_max,)) * 0.6
            if self.K_max >= 1:
                self.sd_by_comp = self.sd_by_comp.at[0].set(0.4)
            if self.K_max >= 2:
                self.sd_by_comp = self.sd_by_comp.at[1].set(0.6)
        self.arange_K = jnp.arange(self.K_max)
        self.dim = 1 + self.K_max

    # -------------------- Priors & Likelihood -------------------- #
    def logprior(self, position: jnp.ndarray) -> jnp.ndarray:
        Kf = position[0]
        K = Kf.astype(int)
        mu_full = position[1:]
        # prior over K
        lp_K = jnp.where((1 <= K) & (K <= self.K_max), -jnp.log(self.K_max), -jnp.inf)
        # mask active components
        mask = self.arange_K < K
        # Uniform(lower, upper) for active mus → log-density = -K * log(upper-lower)
        in_bounds = jnp.all(((mu_full >= self.lower) & (mu_full <= self.upper)) | (~mask))
        lp_mu = jnp.where(in_bounds, -K * jnp.log(self.upper - self.lower), -jnp.inf)
        return lp_K + lp_mu

    def loglikelihood(self, position: jnp.ndarray) -> jnp.ndarray:
        Kf = position[0]
        K = Kf.astype(int)
        mu_full = position[1:]
        mask = self.arange_K < K

        # (n, K_max) of logpdfs at all candidate mus with component sds
        def comp_logpdf(i):
            return jsp.stats.norm.logpdf(self.y, mu_full[i], self.sd_by_comp[i])

        lps_full = jnp.stack([comp_logpdf(i) for i in range(self.K_max)], axis=1)
        # zero-out inactive components (log 0 = -inf)
        lps_full = jnp.where(mask[None, :], lps_full, -jnp.inf)
        # equal weights for active comps
        logw = jnp.where(mask, -jnp.log(K), -jnp.inf)  # shape (K_max,)
        lps_full = lps_full + logw[None, :]
        return jnp.sum(jax.scipy.special.logsumexp(lps_full, axis=1))

    # -------------------- Utilities -------------------- #
    def sample_prior(self, key: PRNGKey, n: int) -> jnp.ndarray:
        key, sub = jr.split(key)
        Ks = jr.randint(sub, (n,), 1, self.K_max + 1).astype(float)
        key, sub = jr.split(key)
        mus = jr.uniform(sub, (n, self.K_max), minval=self.lower, maxval=self.upper)
        return jnp.concatenate([Ks[:, None], mus], axis=1)

    def truths(self) -> Tuple[float, ...]:
        # Unknown K in general; return empty/placeholder
        return tuple()

    # -------------------- Constructors -------------------- #
    @classmethod
    def from_synthetic(
        cls,
        key: PRNGKey,
        n: int = 400,
        K_true: int = 2,
        w_true: Optional[jnp.ndarray] = None,  # mixture weights for data gen
        mu_true: Optional[jnp.ndarray] = None,
        sd_true: Optional[jnp.ndarray] = None,
        K_max: int = 5,
        lower: float = -10.0,
        upper: float = 10.0,
    ) -> "TransdimGaussianMixtureMeans":
        """Generate synthetic data and build the problem instance.

        Data are generated from a K_true-component mixture with provided
        `(w_true, mu_true, sd_true)` (defaults to a 2-component example).
        The likelihood used for inference assumes equal weights over active
        components and per-component sds from `sd_by_comp` (which defaults to
        [0.4, 0.6, 0.6, ...]).
        """
        if w_true is None:
            w_true = jnp.array([0.45, 0.55])
        if mu_true is None:
            mu_true = jnp.array([-2.0, 2.0])
        if sd_true is None:
            sd_true = jnp.array([0.4, 0.6])
        assert K_true == len(w_true) == len(mu_true) == len(sd_true)

        key, sub = jr.split(key)
        z = jr.categorical(sub, jnp.log(w_true), shape=(n,))
        key, sub = jr.split(key)
        y = jr.normal(sub, (n,)) * sd_true[z] + mu_true[z]

        # default sd_by_comp: use provided sd_true for first K_true components,
        # and repeat last value for the rest up to K_max
        sd_vec = jnp.ones((K_max,)) * sd_true[-1]
        sd_vec = sd_vec.at[:K_true].set(sd_true)

        return cls(y=y, K_max=K_max, lower=lower, upper=upper, sd_by_comp=sd_vec)
