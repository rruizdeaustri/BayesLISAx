from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import tqdm

import jax
import jax.numpy as jnp
import jax.random as jr

from ..core.types import PRNGKey
from ..core.result import SamplerResult
from ..core.transdim import ModelFamily
from ..core.rj import RJGenericConfig, RJProposalPair
from ..registry import register_sampler


@dataclass
class RJGeneric:
    """
    Model-agnostic RJ-MCMC driver.
    - Uses a ModelFamily for log pmfs/densities and prior sampling per model size.
    - Uses an RJProposalPair for birth/death (k ↔ k±1) proposals.
    - Default within-model move is Gaussian RW; you can override via subclassing
      or add an ES/HMC/NUTS step inside `_within_model_step`.
    """
    family: ModelFamily
    cfg: RJGenericConfig
    proposal: RJProposalPair

    # keep constructor signature uniform with other samplers (problem unused)
    def __init__(self, problem, cfg: RJGenericConfig, family: ModelFamily, proposal: RJProposalPair):
        self.family = family
        self.cfg = cfg
        self.proposal = proposal

    def init(self, key: PRNGKey, **overrides):
        for k, v in overrides.items():
            setattr(self.cfg, k, v)
        self.key = key
        # start at smallest model index with a prior draw
        self.k = int(min(self.family.Ks))
        key, sub = jr.split(key)
        self.theta = self.family.sample_prior_k(sub, self.k, 1)[0]
        return self

    # ----- target -----
    def _logpost(self, k: int, theta_k: jnp.ndarray) -> jnp.ndarray:
        return ( self.family.logpmf_K(k)
               + self.family.logprior_k(k, theta_k)
               + self.family.loglikelihood_k(k, theta_k) )

    # ----- within-model move (RW by default) -----
    def _within_model_step(self, key: PRNGKey, k: int, theta_k: jnp.ndarray) -> tuple[jnp.ndarray, bool]:
        d = self.family.dim_of(k)
        key, sub = jr.split(key)
        prop = theta_k + self.cfg.sigma_rw * jr.normal(sub, (d,))
        logA = self._logpost(k, prop) - self._logpost(k, theta_k)
        key, subu = jr.split(key)
        accept = jnp.log(jr.uniform(subu)) < logA
        theta_new = jax.lax.select(accept, prop, theta_k)
        return theta_new, bool(accept)

    def run(self, key: PRNGKey) -> SamplerResult:
        k = key
        samples = []
        acc_within = 0
        acc_trans = 0

        Ks = list(self.family.Ks)
        k_min, k_max = min(Ks), max(Ks)

        for _ in tqdm.tqdm(range(self.cfg.steps), desc="RJ-MCMC", unit="step"):
            k, sub = jr.split(k)

            # choose within-model vs trans-dim move
            if jr.uniform(sub) >= self.cfg.p_up:
                theta_new, acc = self._within_model_step(k, self.k, self.theta)
                self.theta = theta_new
                acc_within += int(acc)
            else:
                # decide direction given boundaries
                k, sub = jr.split(k)
                go_up = jnp.where(self.k <= k_min, True,
                                  jnp.where(self.k >= k_max, False, jr.uniform(sub) < 0.5))

                if bool(go_up) and self.k < k_max:
                    # k -> k+1
                    k_next = self.k + 1
                    k, subp = jr.split(k)
                    theta_next, logq_fwd = self.proposal.propose_up(subp, self.k, self.theta)
                    # reverse proposal density
                    k, subr = jr.split(k)
                    _, logq_rev = self.proposal.propose_down(subr, self.k, theta_next)
                    # MH ratio
                    logA = ( self._logpost(k_next, theta_next) - self._logpost(self.k, self.theta)
                           + logq_rev - logq_fwd )
                    k, subu = jr.split(k)
                    if jnp.log(jr.uniform(subu)) < logA:
                        self.k, self.theta = k_next, theta_next
                        acc_trans += 1

                elif (not bool(go_up)) and self.k > k_min:
                    # k -> k-1
                    k_prev = self.k - 1
                    k, subp = jr.split(k)
                    theta_prev, logq_rev = self.proposal.propose_down(subp, k_prev, self.theta)
                    k, subr = jr.split(k)
                    _, logq_fwd = self.proposal.propose_up(subr, k_prev, theta_prev)
                    logA = ( self._logpost(k_prev, theta_prev) - self._logpost(self.k, self.theta)
                           + logq_fwd - logq_rev )
                    k, subu = jr.split(k)
                    if jnp.log(jr.uniform(subu)) < logA:
                        self.k, self.theta = k_prev, theta_prev
                        acc_trans += 1

            # store padded (K, theta, zeros...) for uniform analysis
            dmax = max(self.family.dim_of(Ki) for Ki in Ks)
            buf = jnp.zeros((1 + dmax,))
            buf = buf.at[0].set(self.k)
            dcur = self.family.dim_of(self.k)
            buf = buf.at[1:1+dcur].set(self.theta)
            samples.append(np.asarray(buf))

        samples = np.asarray(samples)
        steps_within = max(1, int(self.cfg.steps * (1 - self.cfg.p_up)))
        steps_trans  = max(1, int(self.cfg.steps * self.cfg.p_up))
        return SamplerResult(
            samples=samples,
            weights=None,
            diagnostics={
                "acc_within": acc_within / steps_within,
                "acc_trans": acc_trans / steps_trans,
            },
        )


# Register with the plugin registry
register_sampler("rj-generic")(RJGeneric)
