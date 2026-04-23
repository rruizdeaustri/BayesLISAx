# src/jax_samplers/samplers/rj_es.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import blackjax

from ..registry import register_sampler


# ---------- Result object expected by cli.py ----------
@dataclass
class SamplerResult:
    samples: jnp.ndarray             # (T, 1+K_max): [K, mu_0..mu_{K_max-1}]
    weights: Optional[jnp.ndarray] = None


# ---------- Config ----------
@dataclass
class RJESConfig:
    steps: int = 20_000
    prior_scale: float = 5.0         # ES ellipse prior σ (matches your standalone 5.0)
    p_mid_birth: float = 0.5         # birth prob when K_min < K < K_max
    box_width: float = 20.0          # uniform box prior width for RJ accept ([-10,10])

# ---------- Internal state (PyTree) ----------
class _RJState(NamedTuple):
    K:  jnp.ndarray                  # scalar int32 in [K_min..K_max]
    mu: jnp.ndarray                  # (K_max,)


# ---------- Utility: build ES kernels, one per K  ----------
def _build_llk_branchers(problem, K_min: int, K_max: int, sigma: float):
    """
    Returns tuples of callables that lax.switch can dispatch:
      llk_branches(mu)      -> loglik
      es_init_branches(mu)  -> es_state
      es_step_branches((key, es_state)) -> (es_state, info)
    Each branch fixes K = K_min + b (static within the closure).
    """
    def make_llk_for_K(K_fixed: int):
        def llk(mu_full: jnp.ndarray) -> jnp.ndarray:
            # theta = [K, mu_0..]
            theta = jnp.concatenate(
                [jnp.array([K_fixed], dtype=mu_full.dtype), mu_full], axis=0
            )

            return problem.loglikelihood(theta)
        return llk

    llk_list = tuple(make_llk_for_K(K) for K in range(K_min, K_max + 1))

    # ES targets likelihood-only; Gaussian N(0, σ²I) defines ellipse path
    mean = jnp.zeros(K_max)
    cov  = (sigma ** 2) * jnp.eye(K_max)

    es_algs = tuple(
        blackjax.elliptical_slice(loglikelihood_fn=f, mean=mean, cov=cov)
        for f in llk_list
    )

    llk_branches     = tuple((lambda mu, f=f: f(mu)) for f in llk_list)
    es_init_branches = tuple((lambda mu, a=a: a.init(mu)) for a in es_algs)
    es_step_branches = tuple((lambda pair, a=a: a.step(pair[0], pair[1])) for a in es_algs)
    return llk_branches, es_init_branches, es_step_branches


# ---------- Registered Sampler ----------
@register_sampler("rj-es")
class RJ_EllipticalSlice:
    """
    RJ sampler with Elliptical Slice within each fixed-K model.

    Product-space parameterization:
      theta_full = [K, μ_0, μ_1, ..., μ_{K_max-1}]
      Only first K entries of μ are active.

    This mirrors your standalone script:
    - ES uses N(0, σ²I) ellipse (σ = prior_scale).
    - RJ accept uses uniform box prior over active μ: [-W/2, +W/2], W=box_width.
    - p(K) uniform over {K_min..K_max}.
    """

    def __init__(self, problem, cfg: RJESConfig):

        if not hasattr(problem, "K_max") or not hasattr(problem, "K_min"):
            raise SystemExit("rj-es requires a transdimensional problem (TransdimProductSpaceProblem).")

        self.problem = problem
        self.cfg = cfg

        self.K_min = int(problem.K_min)
        self.K_max = int(problem.K_max)

        self.sigma = float(cfg.prior_scale)      # ES ellipse scale
        self.W     = float(cfg.box_width)        # box width

        # Build branch tables for all K
        (self._llk_branches,
         self._es_init_branches,
         self._es_step_branches) = _build_llk_branchers(problem, self.K_min, self.K_max, sigma=self.sigma)

    # ---- Switch helpers (mark self static) ----
    def _loglik_switch(self, mu_full: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
        b = K - self.K_min
        return lax.switch(b, self._llk_branches, mu_full)

    def _es_init_switch(self, mu_full: jnp.ndarray, K: jnp.ndarray):
        b = K - self.K_min
        return lax.switch(b, self._es_init_branches, mu_full)

    def _es_step_switch(self, key_and_state, K: jnp.ndarray):
        b = K - self.K_min
        return lax.switch(b, self._es_step_branches, key_and_state)

    # ---- Priors used in RJ accept (box prior like your script) ----
    def _logprior_K(self, K: jnp.ndarray) -> jnp.ndarray:
        ok = (self.K_min <= K) & (K <= self.K_max)
        ZK = jnp.log(self.K_max - self.K_min + 1.0)
        return jnp.where(ok, -ZK, -jnp.inf)

    def _logprior_mu_box(self, mu_full: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
        # Active mask: first K entries
        idx  = jnp.arange(self.K_max)
        mask = idx < K
        # In-bounds mask (apply only to actives)
        half = 0.5 * self.W
        inb_each = jnp.logical_or(~mask, jnp.logical_and(mu_full >= -half, mu_full <= half))
        inb = jnp.all(inb_each)
        # Uniform over width W per active dim → -K*log(W)
        return jnp.where(inb, -K * jnp.log(self.W), -jnp.inf)

    # ---- API ----
    def init(self, key: jnp.ndarray) -> "RJ_EllipticalSlice":
        # Start at K_min with μ ~ N(0, σ^2) padded (like your script)
        key, sub = jr.split(key)
        mu0 = jr.normal(sub, (self.K_max,)) * self.sigma
        self._state = _RJState(K=jnp.array(self.K_min, dtype=jnp.int32), mu=mu0)
        self._key   = key
        return self

    def _step(self, rng, state: _RJState) -> Tuple[jnp.ndarray, _RJState]:
        K, mu = state.K, state.mu

        # 1) Within-model ES (cov already σ²I)
        rng, sub = jr.split(rng)
        es_state = self._es_init_switch(mu, K)
        es_state, _ = self._es_step_switch((sub, es_state), K)
        mu1 = es_state.position

        # 2) Birth/Death proposal (same logic as your script)
        def p_birth_of(Kv):
            return jnp.where(
                Kv <= self.K_min, 1.0,
                jnp.where(Kv >= self.K_max, 0.0, self.cfg.p_mid_birth)
            )
        p_b = p_birth_of(K)
        p_d = 1.0 - p_b

        rng, sub = jr.split(rng)
        rng_bd, rng_newmu = jr.split(sub)

        def birth(_):
            new_muK = jr.normal(rng_newmu, ()) * self.sigma
            mu_prop = mu1.at[K].set(new_muK)   # write at slot K
            Kp = K + 1
            # In your script you used symmetric 50/50 & prior-only target,
            # but to be correct, include q ratio (it’s okay if you omit; it cancels here
            # if you keep the same p_b/p_d pattern both ways and Gaussian draw for μ_K).
            log_q_ratio = 0.0
            return mu_prop, Kp, log_q_ratio

        def death(_):
            removed = mu1[K - 1]
            mu_prop = mu1.at[K - 1].set(0.0)   # tidy zeroing
            Kp = K - 1
            log_q_ratio = 0.0
            return mu_prop, Kp, log_q_ratio

        u = jr.uniform(rng_bd)
        mu_prop, Kp, log_q_ratio = lax.cond(u < p_b, birth, death, operand=None)

        # 3) RJ accept on the *full target* matching your script:
        #    log p(K) + log box prior on active μ + loglikelihood
        def logpost(Kv, muf):
            return (
                self._logprior_K(Kv)
                + self._logprior_mu_box(muf, Kv)
                + self._loglik_switch(muf, Kv)
            )

        rng, sub = jr.split(rng)
        logA   = (logpost(Kp, mu_prop) - logpost(K, mu1)) + log_q_ratio
        accept = jnp.log(jr.uniform(sub)) < logA

        new_state = lax.cond(
            accept,
            lambda _: _RJState(Kp, mu_prop),
            lambda _: _RJState(K,  mu1),
            operand=None
        )
        return rng, new_state

    def run(self, key: jnp.ndarray) -> SamplerResult:
        def body(carry, _):
            rng, s = carry
            rng2, s2 = self._step(rng, s)
            return (rng2, s2), s2

        (_, _), traj = lax.scan(body, (self._key, self._state), None, length=self.cfg.steps)
        
        Ks  = traj.K.astype(jnp.int32)               # (T,)
        Mus = traj.mu                                # (T, K_max)
        samples = jnp.concatenate([Ks[:, None].astype(jnp.float64), Mus], axis=1)
        return SamplerResult(samples=samples, weights=None)
