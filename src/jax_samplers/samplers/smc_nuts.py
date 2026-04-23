# src/jax_samplers/samplers/smc_nuts.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax

import blackjax
from blackjax.smc.adaptive_tempered import as_top_level_api
import blackjax.smc.resampling as resampling
import blackjax.smc.base as smc_base

from ..core.types import PRNGKey
from ..core.problem import Problem
from ..core.result import SamplerResult
from ..registry import register_sampler


@dataclass
class SMCConfig:
    n_particles: int = 1000
    step_size: float = 1e-3
    inv_mass_diag: float = 1.0
    target_ess: float = 0.2
    num_mcmc_steps: int = 5


class AdaptiveTemperedSMC_old:
    def __init__(self, problem: Problem, cfg: SMCConfig):
        self.problem = problem
        self.cfg = cfg

    def init(self, key: PRNGKey, problem: Problem | None = None, **cfg):
        if problem is not None:
            self.problem = problem
        for k, v in cfg.items():
            setattr(self.cfg, k, v)

        self.key = key

        def logtarget(theta):
            return self.problem.logprior(theta) + self.problem.loglikelihood(theta)

        nuts_init, nuts_step = blackjax.nuts(
            logtarget,
            step_size=self.cfg.step_size,
            inverse_mass_matrix=jnp.ones(self.problem.dim) * self.cfg.inv_mass_diag,
        )

        def mcmc_step_fn(rkey, mstate, mparams):
            return nuts_step(rkey, mstate)

        self.smc = as_top_level_api(
            logprior_fn=self.problem.logprior,
            loglikelihood_fn=self.problem.loglikelihood,
            mcmc_step_fn=mcmc_step_fn,
            mcmc_init_fn=nuts_init,
            mcmc_parameters={},
            resampling_fn=resampling.systematic,
            target_ess=self.cfg.target_ess,
            num_mcmc_steps=self.cfg.num_mcmc_steps,
        )

        self.key, sub = jr.split(self.key)
        self.init_particles = self.problem.sample_prior(sub, self.cfg.n_particles)
        self.state = self.smc.init(self.init_particles)
        return self

    def run(self, key: PRNGKey) -> SamplerResult:
        def cond(carry):
            s, _ = carry
            return s.lmbda < 1.0 - 1e-6

        def body(carry):
            s, k = carry
            k, sub = jr.split(k)
            s, _ = self.smc.step(sub, s)
            return s, k

        final_state, _ = lax.while_loop(cond, body, (self.state, key))
        parts = np.array(final_state.particles)
        wts = np.array(final_state.weights)
        return SamplerResult(samples=parts, weights=wts, diagnostics={})

class AdaptiveTemperedSMC:
    def __init__(self, problem: Problem, cfg: SMCConfig):
        self.problem = problem
        self.cfg = cfg

    def init(self, key: PRNGKey, problem: Problem | None = None, **cfg):
        if problem is not None:
            self.problem = problem
        for k, v in cfg.items():
            setattr(self.cfg, k, v)

        self.key = key
        dim = int(self.problem.dim)

        inverse_mass_matrix = jnp.ones((dim,)) * self.cfg.inv_mass_diag

        # Base NUTS kernel (takes logdensity_fn at call time)
        nuts_kernel = blackjax.nuts.build_kernel()

        def mcmc_init_fn(position, logdensity_fn):
            return blackjax.nuts.init(position, logdensity_fn)

        def mcmc_step_fn(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix):
            return nuts_kernel(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix)

        # IMPORTANT: params must be arrays AND have leading dim 1 to mean “shared”
        mcmc_parameters = smc_base.extend_params({
            "step_size": jnp.asarray(self.cfg.step_size),          # scalar -> (1,)
            "inverse_mass_matrix": inverse_mass_matrix,            # (dim,) -> (1, dim)
        })

        
       # mcmc_parameters = dict(
       #     step_size=self.cfg.step_size,
       #     inverse_mass_matrix=inverse_mass_matrix,
       # )

        self.smc = as_top_level_api(
            logprior_fn=self.problem.logprior,
            loglikelihood_fn=self.problem.loglikelihood,
            mcmc_step_fn=mcmc_step_fn,
            mcmc_init_fn=mcmc_init_fn,
            mcmc_parameters=mcmc_parameters,
            resampling_fn=resampling.systematic,
            target_ess=self.cfg.target_ess,
            num_mcmc_steps=self.cfg.num_mcmc_steps,
        )

        self.key, sub = jr.split(self.key)
        self.init_particles = self.problem.sample_prior(sub, self.cfg.n_particles)

        # DEBUG: is the prior sample diverse?
        #p0 = np.array(self.init_particles)
        #print("init_particles shape:", p0.shape)
        #print("init std:", p0.std(axis=0))
        #print("init unique rows:", np.unique(np.round(p0, 12), axis=0).shape[0])
        #print("first 3 rows:\n", p0[:3])
        #sys.exit()
        self.state = self.smc.init(self.init_particles)

        
        return self

    def run(self, key: PRNGKey) -> SamplerResult:
        def cond(carry):
            s, _ = carry
            return s.lmbda < 1.0 - 1e-6

        def body(carry):
            s, k = carry
            k, sub = jr.split(k)
            s, _ = self.smc.step(sub, s)
            return s, k

        final_state, _ = lax.while_loop(cond, body, (self.state, key))

        parts = np.array(final_state.particles)
        wts = np.array(final_state.weights)
        return SamplerResult(samples=parts, weights=wts, diagnostics={})
    

# Register with the plugin registry
register_sampler("smc-nuts")(AdaptiveTemperedSMC)
