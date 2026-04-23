# src/jax_samplers/samplers/smc_hmc.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax

import blackjax
from blackjax.smc.adaptive_tempered import as_top_level_api
import blackjax.smc.resampling as resampling

from ..core.types import PRNGKey
from ..core.problem import Problem
from ..core.result import SamplerResult
from ..registry import register_sampler


@dataclass
class HMCSMCConfig:
    """Adaptive-tempered SMC with **HMC** rejuvenation (BlackJAX).

    Parameters
    ----------
    n_particles : int
        Number of SMC particles.
    step_size : float
        HMC integrator step size.
    inv_mass_diag : float | jnp.ndarray
        Diagonal entries of the inverse mass matrix (scalar or array of shape (dim,)).
    n_leapfrog : int
        Number of leapfrog steps per HMC transition.
    target_ess : float
        Relative ESS threshold in (0,1]; resample when ESS < target_ess * n_particles.
    num_mcmc_steps : int
        Number of HMC transitions per tempering step.
    resampler : str
        One of {"systematic", "multinomial", "stratified"}.
    """

    n_particles: int = 1000
    step_size: float = 1e-3
    inv_mass_diag: float | jnp.ndarray = 1.0
    n_leapfrog: int = 10
    target_ess: float = 0.5
    num_mcmc_steps: int = 5
    resampler: str = "systematic"
    verbose: bool = False

class AdaptiveTemperedSMC_HMC:
    def __init__(self, problem: Problem, cfg: HMCSMCConfig):
        self.problem = problem
        self.cfg = cfg

    def _resampler(self):
        name = self.cfg.resampler.lower()
        if name == "systematic":
            return resampling.systematic
        if name == "multinomial":
            return resampling.multinomial
        if name == "stratified":
            return resampling.stratified
        raise ValueError(f"Unknown resampler: {self.cfg.resampler}")

    def init(self, key: PRNGKey, problem: Problem | None = None, **cfg):
        # allow runtime overrides
        if problem is not None:
            self.problem = problem
        for k, v in cfg.items():
            setattr(self.cfg, k, v)

        self.key = key
        dim = int(self.problem.dim)

        # Posterior log target for MCMC transitions
        def logtarget(theta):
            return self.problem.logprior(theta) + self.problem.loglikelihood(theta)

        inv_mass = (
            jnp.ones((dim,)) * self.cfg.inv_mass_diag
            if np.isscalar(self.cfg.inv_mass_diag)
            else jnp.asarray(self.cfg.inv_mass_diag)
        )

        hmc_init, hmc_step = blackjax.hmc(
            logtarget,
            self.cfg.step_size,
            inv_mass,
            self.cfg.n_leapfrog,
        )

        def mcmc_step_fn(rkey, mstate, mparams):
            # ignore mparams; all hyper-params closed over above
            return hmc_step(rkey, mstate)

        self.smc = as_top_level_api(
            logprior_fn=self.problem.logprior,
            loglikelihood_fn=self.problem.loglikelihood,
            mcmc_step_fn=mcmc_step_fn,
            mcmc_init_fn=hmc_init,
            mcmc_parameters={},
            resampling_fn=self._resampler(),
            target_ess=self.cfg.target_ess,
            num_mcmc_steps=self.cfg.num_mcmc_steps,
        )

        # Initial particles from the prior
        self.key, sub = jr.split(self.key)
        self.init_particles = self.problem.sample_prior(sub, self.cfg.n_particles)
        self.state = self.smc.init(self.init_particles)
        return self


    def run(self, key: PRNGKey) -> SamplerResult:
        @jax.jit
        def run_smc(k, state):
            def cond(carry):
                s, _ = carry
                return s.lmbda < 1.0 - 1e-6   # scalar bool

            def body(carry):
                s, k = carry
                k, sub = jr.split(k)
                s_new, info = self.smc.step(sub, s)  # info can be logged separately if needed
                return (s_new, k)                    # <-- return SAME structure as carry

            # init carry and run
            init_carry = (state, k)
            final_state, _ = jax.lax.while_loop(cond, body, init_carry)
            return final_state
    
        final_state = run_smc(key, self.state)

        parts = np.array(final_state.particles)
        wts = np.array(final_state.weights)
        wts = wts / (wts.sum() + 1e-300)
        neff = 1.0 / float((wts**2).sum() + 1e-300)

        diags = {
            "lambda_final": float(final_state.lmbda),
            "ess_weights": neff,
        }

        if self.cfg.verbose:
          print(diags)
        return SamplerResult(samples=parts, weights=wts, diagnostics=diags)


# Register with the plugin registry
register_sampler("smc-hmc")(AdaptiveTemperedSMC_HMC)
