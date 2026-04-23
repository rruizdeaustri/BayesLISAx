from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from ..core.types import PRNGKey
from ..core.problem import Problem
from ..core.result import SamplerResult
from ..registry import register_sampler


@dataclass
class NumPyroNUTSConfig:
    num_warmup: int = 1000
    num_samples: int = 2000
    num_chains: int = 1

    # NUTS tuning
    target_accept_prob: float = 0.8
    dense_mass: bool = False          # True => adapt dense mass matrix (can be expensive)
    max_tree_depth: int = 10

    # UX
    progress_bar: bool = True
    verbose: bool = True

def _build_numpyro_model_from_problem(problem: Problem):
    """
    Builds a NumPyro model using:
      - Uniform priors from problem.prior_bounds for parameters not in problem._gauss
      - Normal priors for parameters in problem._gauss
      - Likelihood via numpyro.factor("loglike", problem.loglikelihood(theta))

    Assumes the Problem exposes:
      - problem.dim
      - problem.prior_bounds: list[(low, high)] length dim
      - problem._names: list[str] length dim
      - problem._gauss: dict{name: (mu, sigma)} for gaussian priors (optional)
      - problem.loglikelihood(theta): theta shape (dim,)
    """
    names: List[str] = list(problem._names)
    bounds = list(problem.prior_bounds)
    gauss: Dict[str, Any] = getattr(problem, "_gauss", {})

    def model():
        params = []
        for i, name in enumerate(names):
            lo, hi = bounds[i]
            if name in gauss:
                mu, sig = gauss[name]
                x = numpyro.sample(name, dist.Normal(jnp.asarray(mu), jnp.asarray(sig)))
            else:
                x = numpyro.sample(name, dist.Uniform(jnp.asarray(lo), jnp.asarray(hi)))
            params.append(x)

        theta = jnp.stack(params)  # (dim,)
        numpyro.factor("loglike", problem.loglikelihood(theta))

    return model, names


class NumPyroNUTSBackend:
    def __init__(self, problem: Problem, cfg: NumPyroNUTSConfig):
        self.problem = problem
        self.cfg = cfg

    def init(self, key: PRNGKey, problem: Problem | None = None, **cfg):
        if problem is not None:
            self.problem = problem
        for k, v in cfg.items():
            setattr(self.cfg, k, v)

        self.key = key
        self.model, self.names = _build_numpyro_model_from_problem(self.problem)
        return self

    def run(self, key: PRNGKey) -> SamplerResult:
        # Build NUTS kernel
        kernel = NUTS(
            self.model,
            target_accept_prob=self.cfg.target_accept_prob,
            dense_mass=self.cfg.dense_mass,
            max_tree_depth=self.cfg.max_tree_depth,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=self.cfg.num_warmup,
            num_samples=self.cfg.num_samples,
            num_chains=self.cfg.num_chains,
            progress_bar=self.cfg.progress_bar,
        )

        mcmc.run(key)
        if self.cfg.verbose:
         mcmc.print_summary()
        
        # NumPyro returns dict: {name: array[chains, draws] or [draws]} depending on chains
        samples_dict = mcmc.get_samples(group_by_chain=True)  # => [chains, draws, ...]
        # Stack into (chains*draws, dim) in the same order as problem._names
        cols = [np.array(samples_dict[n]).reshape(-1) for n in self.names]
        samples = np.stack(cols, axis=1)  # (N, dim)

        # uniform weights
        wts = np.ones(samples.shape[0], dtype=np.float64) / float(samples.shape[0])

        extra = mcmc.get_extra_fields()
        diags = {
            "num_warmup": self.cfg.num_warmup,
            "num_samples": self.cfg.num_samples,
            "num_chains": self.cfg.num_chains,
        }
        # extra fields vary by version, but often include accept_prob, diverging, etc.
        for k2, v2 in extra.items():
            try:
                diags[f"extra_{k2}"] = np.array(v2)
            except Exception:
                pass

        return SamplerResult(samples=samples, weights=wts, diagnostics=diags)


register_sampler("numpyro-nuts")(NumPyroNUTSBackend)
