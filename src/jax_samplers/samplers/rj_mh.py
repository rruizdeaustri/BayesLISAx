# src/jax_samplers/samplers/rj_mh.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import tqdm

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp

from ..core.types import PRNGKey
from ..core.problem import Problem
from ..core.result import SamplerResult
from ..registry import register_sampler


@dataclass
class RJConfig:
    steps: int = 20000
    sigma_rw: float = 0.2
    sigma_split: float = 0.6
    p_birth: float = 0.3


class RJMixture12:
    """Toy RJ-MH toggling K∈{1,2} for the Gaussian-mixture means problem."""

    def __init__(self, problem: Problem, cfg: RJConfig):
        self.problem = problem
        self.cfg = cfg

    def init(self, key: PRNGKey, problem: Problem | None = None, **cfg):
        if problem is not None:
            self.problem = problem
        for k, v in cfg.items():
            setattr(self.cfg, k, v)
        self.key = key
        self.theta = jnp.array([-1.5, 1.5])  # start at K=2
        self.K = 2
        return self

    def _logpost(self, theta, K):
        lpK = jnp.log(0.5)  # prior on K
        if K == 1:
            th = jnp.array([theta[0], theta[0]])
        else:
            th = theta
        return lpK + self.problem.logprior(th) + self.problem.loglikelihood(th)

    def run(self, key: PRNGKey) -> SamplerResult:
        samples = []
        acc_rw = acc_bd = 0
        k = key
        for _ in tqdm.tqdm(range(self.cfg.steps), desc="RJ-MH", unit="step"):
            k, sub = jr.split(k)

            # Random-walk within current model
            if self.K == 1:
                prop = self.theta + self.cfg.sigma_rw * jr.normal(sub, (1,))
                cur = self._logpost(self.theta, 1)
                prp = self._logpost(prop, 1)
                if jnp.log(jr.uniform(sub)) < (prp - cur):
                    self.theta = prop
                    acc_rw += 1
            else:
                prop = self.theta + self.cfg.sigma_rw * jr.normal(sub, (2,))
                cur = self._logpost(self.theta, 2)
                prp = self._logpost(prop, 2)
                if jnp.log(jr.uniform(sub)) < (prp - cur):
                    self.theta = prop
                    acc_rw += 1

            # Birth/Death move
            k, sub = jr.split(k)
            if jr.uniform(sub) < self.cfg.p_birth:
                if self.K == 1:
                    k, subn = jr.split(k)
                    delta = self.cfg.sigma_split * jr.normal(subn)
                    theta_new = jnp.array([self.theta[0] + delta, self.theta[0] - delta])
                    cur = self._logpost(self.theta, 1)
                    new = self._logpost(theta_new, 2)
                    log_q_fwd = jsp.stats.norm.logpdf(delta, 0.0, self.cfg.sigma_split)
                    log_acc = (new - cur) + jnp.log(2.0) - log_q_fwd
                    if jnp.log(jr.uniform(k)) < log_acc:
                        self.theta, self.K = theta_new, 2
                        acc_bd += 1
                else:
                    mu = jnp.mean(self.theta)
                    delta = 0.5 * (self.theta[0] - self.theta[1])
                    theta_new = jnp.array([mu])
                    cur = self._logpost(self.theta, 2)
                    new = self._logpost(theta_new, 1)
                    log_q_rev = jsp.stats.norm.logpdf(delta, 0.0, self.cfg.sigma_split)
                    log_acc = (new - cur) - jnp.log(2.0) + log_q_rev
                    if jnp.log(jr.uniform(k)) < log_acc:
                        self.theta, self.K = theta_new, 1
                        acc_bd += 1

            # Store as 2D (duplicate when K=1)
            if self.K == 1:
                samples.append([float(self.theta[0]), float(self.theta[0])])
            else:
                samples.append([float(self.theta[0]), float(self.theta[1])])

        samples = np.asarray(samples)
        return SamplerResult(
            samples=samples,
            weights=None,
            diagnostics={
                "acc_rw": acc_rw / self.cfg.steps,
                "acc_bd": acc_bd / max(1, int(self.cfg.steps * self.cfg.p_birth)),
            },
        )


# Register with the plugin registry
register_sampler("rj")(RJMixture12)
