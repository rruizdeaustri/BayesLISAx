# src/jax_samplers/samplers/elliptical_slice.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax

import blackjax
import blackjax.diagnostics as diag

from ..core.types import PRNGKey
from ..core.problem import Problem
from ..core.result import SamplerResult
from ..registry import register_sampler


@dataclass
class ESConfig:
    """Elliptical Slice Sampling (ESS) configuration.

    Attributes
    ----------
    n_chains : int
        Number of parallel chains to run (vectorized with `vmap`).
    n_samples : int
        Number of samples per chain to draw (kept after burn-in and thinning).
    thin : int
        Keep one every `thin` samples.
    burn_in : int
        Drop this many initial kept samples (after thinning is applied).
    prior_mean : Optional[jnp.ndarray]
        Mean of the Gaussian prior used by ESS (shape = (dim,)). If None, zeros.
    prior_cov : Optional[jnp.ndarray]
        Covariance of the Gaussian prior (shape = (dim, dim)). If None, 25·I.
    init_from_prior : bool
        If True, initial positions are drawn from N(prior_mean, prior_cov). If False, from the Problem's prior via `sample_prior`.

    """

    n_chains: int = 8
    n_samples: int = 1000
    thin: int = 1
    burn_in: int = 0
    prior_mean: Optional[jnp.ndarray] = None
    prior_cov: Optional[jnp.ndarray] = None
    init_from_prior: bool = True



class EllipticalSliceSampler:
    """BlackJAX Elliptical Slice Sampler wrapped to the package interface.

    Targets the posterior with a Gaussian prior N(mean, cov) * Problem.likelihood.
    """

    def __init__(self, problem: Problem, cfg: ESConfig):
        self.problem = problem
        self.cfg = cfg

    def init(self, key: PRNGKey, problem: Problem | None = None, **cfg):
        # Optional overrides
        if problem is not None:
            self.problem = problem
        for k, v in cfg.items():
            setattr(self.cfg, k, v)

        self.key = key
        self.dim = int(self.problem.dim)

        # Prior parameters (defaults if None)
        self.mean = (
            self.cfg.prior_mean
            if self.cfg.prior_mean is not None
            else jnp.zeros((self.dim,))
        )
        self.cov = (
            self.cfg.prior_cov
            if self.cfg.prior_cov is not None
            else jnp.eye(self.dim) * 25.0
        )

        # Build ESS kernel using only the likelihood; Gaussian prior is internal to ESS
        self.es = blackjax.elliptical_slice(
            loglikelihood_fn=self.problem.loglikelihood,
            mean=self.mean,
            cov=self.cov,
        )
        self.es_init, self.es_step = self.es.init, self.es.step

        # Initial positions
        self.key, sub = jr.split(self.key)
        if self.cfg.init_from_prior:
            # Gaussian prior init (consistent with ESS auxiliary direction)
            init_positions = jr.multivariate_normal(sub, self.mean, self.cov, (self.cfg.n_chains,))
        else:
            init_positions = self.problem.sample_prior(sub, self.cfg.n_chains)

        # Vectorized init across chains
        self.states0 = jax.vmap(self.es_init)(init_positions)
        return self

    def run(self, key: PRNGKey) -> SamplerResult:
        n_chains = int(self.cfg.n_chains)
        n_samples = int(self.cfg.n_samples)

        es_step = self.es_step  # local for speed

        @jax.jit
        def multi_step(states, k):
            keys = jr.split(k, n_chains)
            new_states, infos = jax.vmap(es_step)(keys, states)
            return new_states, infos

        @jax.jit
        def run_chains(states, k):
            def body(carry, _):
                states, k = carry
                k, sub = jr.split(k)
                states, _ = multi_step(states, sub)
                return (states, k), states.position  # positions has shape (n_chains, dim)

            (final_states, _), positions = lax.scan(body, (states, k), None, length=n_samples)
            return final_states, positions  # positions: (n_samples, n_chains, dim)

        final_states, positions = run_chains(self.states0, key)

        # Diagnostics expect (chains, draws, dim)
        samples_c_d_d = jnp.transpose(positions, (1, 0, 2))
        rhat = diag.potential_scale_reduction(samples_c_d_d)
        ess = diag.effective_sample_size(samples_c_d_d)

        # Flatten to (draws_total, dim)
        flat = np.array(positions).reshape(-1, self.dim)

        # Thinning & burn-in (apply on the flattened sequence in draw-major order)
        if self.cfg.thin > 1:
            flat = flat[:: self.cfg.thin]
        if self.cfg.burn_in > 0:
            flat = flat[self.cfg.burn_in :]

        diagnostics: Dict[str, Any] = {
            "rhat": np.asarray(rhat),
            "ess": np.asarray(ess),
            "n_kept": int(flat.shape[0]),
            "n_chains": n_chains,
        }
        return SamplerResult(samples=flat, weights=None, diagnostics=diagnostics)


# Register under two convenient names
register_sampler("es")(EllipticalSliceSampler)
register_sampler("elliptical-slice")(EllipticalSliceSampler)
