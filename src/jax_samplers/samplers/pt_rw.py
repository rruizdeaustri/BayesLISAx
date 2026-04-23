# src/jax_samplers/samplers/pt_rw.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import blackjax

from ..core.types import PRNGKey
from ..core.problem import Problem
from ..core.result import SamplerResult
from ..registry import register_sampler

from ..core.precision import DTYPE

@dataclass
class PTRWConfig:
    n_replicas: int = 5
    beta_min: float = 0.1
    beta_max: float = 1.0
    schedule: Literal["linear", "geometric"] = "linear"

    rw_sigma: float = 0.3              # initial step std
    adapt_steps: int = 1000            # 0 = disable adaptation
    adapt_target_acc: float = 0.25     # Robbins–Monro target acceptance
    adapt_eta: float = 0.01            # learning rate for adaptation

    n_sweeps: int = 5000
    steps_per_sweep: int = 1
    thin: int = 1
    burn_in: int = 0
    record_replica: int = 0            # or set to -1 to return full ladder

class PTRW:
    NAME   = "pt-rw"
    Config = PTRWConfig

    def __init__(self, problem: Problem, cfg: PTRWConfig):
        self.problem = problem
        self.cfg = cfg

        if cfg.schedule == "linear":
            self.betas = jnp.linspace(cfg.beta_max, cfg.beta_min, num=cfg.n_replicas)
        elif cfg.schedule == "geometric":
            r = (cfg.beta_min / cfg.beta_max) ** (1.0 / (cfg.n_replicas - 1))
            self.betas = cfg.beta_max * (r ** jnp.arange(cfg.n_replicas))
        else:
            raise ValueError(f"Unknown schedule {cfg.schedule!r}")

        self.betas = jnp.asarray(self.betas, dtype=DTYPE)
        
    def init(self, key: PRNGKey):
        R = self.cfg.n_replicas
        key, sub = jr.split(key)
        thetas0 = self.problem.sample_prior(sub, R)  # (R, dim)
        if thetas0.ndim == 1:
            thetas0 = jnp.repeat(thetas0[None, :], R, axis=0)

        ll0 = jax.vmap(self.problem.loglikelihood)(thetas0).astype(DTYPE)
        rw_sigmas0 = jnp.full((R,), self.cfg.rw_sigma, dtype=DTYPE)

        
        self.state0 = (jnp.asarray(thetas0), jnp.asarray(ll0), rw_sigmas0)
        return self

    def run(self, key: PRNGKey) -> SamplerResult:
        cfg = self.cfg
        betas = self.betas
        R = cfg.n_replicas

        even_pairs = jnp.stack([jnp.arange(0, R - 1, 2), jnp.arange(1, R, 2)], axis=1)
        odd_pairs  = jnp.stack([jnp.arange(1, R - 1, 2), jnp.arange(2, R, 2)], axis=1)

        def rw_all(key, thetas, ll, rw_sigmas):
            keys = jr.split(key, R)
            _, dim = thetas.shape

            def propose(k, theta, sigma):
                return theta + sigma * jr.normal(k, shape=(dim,))
            theta_prop = jax.vmap(propose)(keys, thetas, rw_sigmas)

            lp_cur  = jax.vmap(self.problem.logprior)(thetas)
            lp_prop = jax.vmap(self.problem.logprior)(theta_prop)
            ll_prop = jax.vmap(self.problem.loglikelihood)(theta_prop)

            logp_cur  = lp_cur  + betas * ll
            logp_prop = lp_prop + betas * ll_prop

            key_u = jr.split(key, 1)[0]
            u = jr.uniform(key_u, shape=(R,))
            accept = jnp.log(u) < (logp_prop - logp_cur)

            thetas_new = jnp.where(accept[:, None], theta_prop, thetas)
            ll_new     = jnp.where(accept, ll_prop, ll)
            return thetas_new, ll_new, accept.astype(DTYPE)

        def adapt_sigmas(rw_sigmas, acc):
            err = jnp.asarray(cfg.adapt_target_acc, DTYPE) - acc
            eta = jnp.asarray(cfg.adapt_eta, DTYPE)
            out = rw_sigmas * jnp.exp(eta * err)
            return jnp.clip(out, jnp.asarray(1e-6, DTYPE), jnp.asarray(1e6, DTYPE))

        
        def swap_pass(rng, thetas, ll, pairs):
            def swap_one(c, ij):
                rng, th, llc, acc = c
                i, j = ij
                rng, sub = jr.split(rng)
                bi, bj = betas[i], betas[j]
                logR = (bi - bj) * (llc[j] - llc[i])
                accept = jnp.log(jr.uniform(sub)) < logR

                def do_swap(th_ll):
                    th_, ll_ = th_ll
                    ti, tj = th_[i], th_[j]
                    li, lj = ll_[i], ll_[j]
                    th_ = th_.at[i].set(tj); th_ = th_.at[j].set(ti)
                    ll_ = ll_.at[i].set(lj); ll_ = ll_.at[j].set(li)
                    return th_, ll_

                th, llc = lax.cond(accept, do_swap, lambda x: x, (th, llc))
                acc = acc + accept.astype(jnp.int32)
                return (rng, th, llc, acc), None

            (rng, thetas, ll, acc), _ = lax.scan(swap_one, (rng, thetas, ll, jnp.int32(0)), pairs)
            return rng, thetas, ll, acc

        def one_sweep(carry, t):
            rng, thetas, ll, rw_sigmas, acc_counters, swap_counters = carry
            rng, sub = jr.split(rng)

            # multiple MH steps per sweep
            def mh_body(i, x):
                rng_i, th_i, ll_i, acc_sum = x
                rng_i, k = jr.split(rng_i)
                th_i, ll_i, acc = rw_all(k, th_i, ll_i, rw_sigmas)
                return (rng_i, th_i, ll_i, acc_sum + acc)

            acc0 = jnp.zeros((R,), dtype=DTYPE)
            rng_mh, thetas, ll, acc_sum = lax.fori_loop(
                0, int(cfg.steps_per_sweep), mh_body, (sub, thetas, ll, acc0)
            )
            acc = acc_sum / float(max(1, int(cfg.steps_per_sweep)))

            rw_sigmas = lax.cond(
                t < cfg.adapt_steps,
                lambda s: adapt_sigmas(s, acc),
                lambda s: s,
                rw_sigmas
            )

            rng, thetas, ll, acc_even = swap_pass(rng, thetas, ll, even_pairs)
            rng, thetas, ll, acc_odd  = swap_pass(rng, thetas, ll, odd_pairs)
            swap_acc = acc_even + acc_odd

            acc_counters  = acc_counters + acc
            swap_counters = swap_counters + swap_acc
            return (rng, thetas, ll, rw_sigmas, acc_counters, swap_counters), (thetas, rw_sigmas)

        # initial carry
        thetas0, ll0, rw_sigmas0 = self.state0
        acc_counters0  = jnp.zeros((R,), dtype=DTYPE)
        swap_counters0 = jnp.int32(0)
        init = (key, thetas0, ll0, rw_sigmas0, acc_counters0, swap_counters0)

        @jax.jit
        def run_pt(carry):
            (rng_f, thetas_f, ll_f, rw_sigmas_f, acc_c_f, swap_c_f), (traj_thetas, traj_sigmas) = \
                lax.scan(one_sweep, carry, jnp.arange(cfg.n_sweeps))
            return (rng_f, thetas_f, ll_f, rw_sigmas_f, acc_c_f, swap_c_f), (traj_thetas, traj_sigmas)

        (rng_f, thetas_f, ll_f, rw_sigmas_f, acc_c_f, swap_c_f), (traj_thetas, traj_sigmas) = run_pt(init)

        # output
        start = int(cfg.burn_in)
        thin  = int(cfg.thin) if cfg.thin and cfg.thin > 0 else 1

        if cfg.record_replica >= 0:
            r = int(cfg.record_replica)  # 0 == beta_max == 1
            samples = traj_thetas[start::thin, r, :]
        else:
            samples = traj_thetas[start::thin, :, :]

        # diagnostics
        diags = {
            "betas": jax.device_get(self.betas).tolist(),
            "n_kept": int(samples.shape[0]),
            "avg_swaps_per_sweep": float(int(swap_c_f) / max(1, int(cfg.n_sweeps))),
            "avg_accept_by_replica": (jax.device_get(acc_c_f) / max(1, float(cfg.n_sweeps))).tolist(),
            "final_rw_sigma_by_replica": jax.device_get(rw_sigmas_f).tolist(),
        }
        return SamplerResult(samples=jax.device_get(samples), weights=None, diagnostics=diags)
    


# Register with the plugin registry (matches your other samplers)
register_sampler("pt-rw")(PTRW)
