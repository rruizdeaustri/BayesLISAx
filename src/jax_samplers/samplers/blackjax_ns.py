# src/jax_samplers/samplers/blackjax_ns.py
import sys
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import tqdm
import jax
import jax.random as jr
import jax.numpy as jnp

import blackjax
from blackjax.ns.utils import finalise

import anesthetic

from ..core.types import PRNGKey
from ..core.problem import Problem
from ..core.result import SamplerResult
from ..registry import register_sampler

import matplotlib.pyplot as plt


def _evidence_view(state):
    if hasattr(state, "logZ") and hasattr(state, "logZ_live"):
        return state
    integ = getattr(state, "integrator", None)
    if integ is not None and hasattr(integ, "logZ") and hasattr(integ, "logZ_live"):
        return integ
    raise NotImplementedError(
        f"Could not find logZ/logZ_live on state={type(state)} "
        f"or integrator={type(integ)}"
    )

@dataclass
class NSConfig:
    n_live: int = 500
    num_delete_ratio: float = 0.3
    num_inner_steps: int = 3
    tol: float = 3.0
    log_every: int = 10
    profile: bool = False
    profile_dir: str = "./jax_profile"
    # Hamiltonian NS specific
    dt_ini: float = 0.3
    min_reflections: int = 2
    max_reflections: int = 10
    sigma_vel: float = 0.0
    ham_max_steps: int = 150
    # GGNS-specific conservative defaults
    ggns_step_size: float = 0.001
    ggns_num_inner_steps: int = 1
    # Bounds for Hamiltonian NS reflections (set from CLI or problem)
    lower: list | None = None   # list of floats, length = dim
    upper: list | None = None   # list of floats, length = dim

class BlackJAXNestedSampler:
    def __init__(self, problem: Problem, cfg: NSConfig):
        self.problem = problem
        self.cfg = cfg
        self.state = None
        self.algo = None
        self.key = None
        self.d = None
        self.num_delete = None
        self.num_inner = None

    def _scalar(self, x):
        return jnp.reshape(jnp.asarray(x), ())  # enforce scalar ()

    def _make_scalar_fns(self):
        d = int(self.problem.dim)

        def logprior_1(theta):
            theta = jnp.asarray(theta).reshape((d,))
            return self._scalar(self.problem.logprior(theta))

        def loglike_1(theta):
            theta = jnp.asarray(theta).reshape((d,))
            return self._scalar(self.problem.loglikelihood(theta))

        return logprior_1, loglike_1

    def _setup_common(self, key: PRNGKey, problem: Problem | None, cfg_overrides: dict) -> tuple:
        """Shared setup: update problem/cfg, compute num_delete/num_inner, sample prior.

        Returns (logprior_fn, loglike_fn, init_pts).
        """
        if problem is not None:
            self.problem = problem
        for k, v in cfg_overrides.items():
            setattr(self.cfg, k, v)

        self.key = key
        self.d = int(self.problem.dim)
        self.num_delete = max(1, int(self.cfg.num_delete_ratio * self.cfg.n_live))

        if self.cfg.num_inner_steps is None or self.cfg.num_inner_steps <= 0:
            self.num_inner = 3 * self.d
        else:
            self.num_inner = int(self.cfg.num_inner_steps)

        logprior_fn, loglike_fn = self._make_scalar_fns()

        self.key, sub = jr.split(self.key)
        init_pts = self.problem.sample_prior(sub, self.cfg.n_live)

        expected = (self.cfg.n_live, self.d)
        if init_pts.shape != expected:
            raise ValueError(f"sample_prior returned {init_pts.shape}, expected {expected}")

        _ = self.problem.logprior(jnp.asarray(init_pts[0]).reshape((self.d,)))
        _ = self.problem.loglikelihood(jnp.asarray(init_pts[0]).reshape((self.d,)))

        return logprior_fn, loglike_fn, init_pts

    def init(self, key: PRNGKey, problem: Problem | None = None, **cfg):
        logprior_fn, loglike_fn, init_pts = self._setup_common(key, problem, cfg)

        self.algo = blackjax.nss(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglike_fn,
            num_delete=self.num_delete,
            num_inner_steps=self.num_inner,
        )

        self.state = self.algo.init(init_pts)
        return self
    
    def ns_report(self, nested_samples, nsamples_err=200):
        """
        Return a JSON-serialisable report dict from an anesthetic.NestedSamples.
        """
        rep = {}

        # Point estimates
        s = nested_samples.stats()  # logZ, D_KL, logL_P, d_G  (beta=1)  :contentReference[oaicite:2]{index=2}

        rep["logZ"]   = float(s["logZ"])
        rep["D_KL"]   = float(s["D_KL"])
        rep["logL_P"] = float(s["logL_P"])
        rep["d_G"]    = float(s["d_G"])

        # Uncertainty estimates by drawing from the NS-induced distribution
        # (gives you a distribution over logZ, etc.)
        try:
            s_draws = nested_samples.stats(nsamples=int(nsamples_err))
            rep["logZ_std"]   = float(s_draws["logZ"].std())
            rep["D_KL_std"]   = float(s_draws["D_KL"].std())
            rep["logL_P_std"] = float(s_draws["logL_P"].std())
            rep["d_G_std"]    = float(s_draws["d_G"].std())
        except Exception:
            # If something goes weird, skip error bars rather than crashing the sampler
            pass

        # Effective sample size of the weighted set
        # (this is the useful ESS; posterior_points() are equal-weight so ESS=N there)
        try:
            rep["ESS"] = float(nested_samples.neff())
        except Exception:
            pass

        # Basic run sizes (often handy in a report)
        rep["n_total"] = int(len(nested_samples))

        return rep

    def run(self, key: PRNGKey | None = None) -> SamplerResult:
        if self.state is None or self.algo is None:
            raise RuntimeError("Sampler not initialized. Call init(...) first.")
        if key is None:
            key = self.key

        @jax.jit
        def step(state, key):
            key, sub = jr.split(key)
            state, dead_pt = self.algo.step(sub, state)
            return state, key, dead_pt

        dead = []

        if self.cfg.profile:
            import os
            os.makedirs(self.cfg.profile_dir, exist_ok=True)
            jax.profiler.start_trace(self.cfg.profile_dir)

        state_fields = {x for x in dir(self.state) if not x.startswith("_")}
        has_logz = ("logZ" in state_fields) and ("logZ_live" in state_fields)

        #print("[ns debug] algo type:", type(self.algo))
        #print("[ns debug] state type:", type(self.state))
        #print("[ns debug] state fields:", sorted(state_fields))

        log_every = max(1, int(self.cfg.log_every))
        it = 0

        #print(type(self.state.integrator))
        #print([x for x in dir(self.state.integrator) if not x.startswith("_")])        
        ev = _evidence_view(self.state)
        with tqdm.tqdm(desc="NSS dead points", unit="pts") as pbar:
            while (ev.logZ_live - ev.logZ) >= -float(self.cfg.tol):
                #while (self.state.logZ_live - self.state.logZ) >= -float(self.cfg.tol):
                    self.state, key, dead_pt = step(self.state, key)
                    dead.append(dead_pt)
                    pbar.update(self.num_delete)
                    it += 1

                    ev = _evidence_view(self.state)
                    
                    if it % log_every == 0:
                        #gap = float(self.state.logZ_live - self.state.logZ)
                        gap = float(ev.logZ_live - ev.logZ)
                        pbar.set_postfix({
                            "iter": it,
                            "gap": f"{gap:.3f}",
                            "logZ": f"{float(ev.logZ):.3f}",
                            "logZ_live": f"{float(ev.logZ_live):.3f}",
                            "dead": it * self.num_delete,
                        })

            jax.block_until_ready(ev.logZ)

        if self.cfg.profile:
            jax.profiler.stop_trace()

        out = finalise(self.state, dead)

        #print("[ns debug] out type:", type(out))
        #print("[ns debug] out fields:", [x for x in dir(out) if not x.startswith("#_")])
        #print("[ns debug] out repr:", out)
        #raise SystemExit
        
        diags: Dict[str, Any] = {
            "n_live": int(self.cfg.n_live),
            "num_delete": int(self.num_delete),
            "num_inner_steps": int(self.num_inner),
        }
        if has_logz:
            diags["logZ"] = float(self.state.logZ)
            diags["logZ_live"] = float(self.state.logZ_live)

       # Convert to posterior samples (equal-weight) if anesthetic works

        posterior_df = None
        
        #logL = np.asarray(out.loglikelihood)
        #imax = int(np.argmax(logL))
        #best_logL = float(logL[imax])
        #best_u = np.asarray(out.particles)[imax, :int(self.problem.dim)]

        logL = np.asarray(out.particles.loglikelihood)
        imax = int(np.argmax(logL))
        best_logL = float(logL[imax])
        best_u = np.asarray(out.particles.position)[imax, :int(self.problem.dim)]
        
        tqdm.tqdm.write(f"[NS] best logL = {best_logL:.3f}  (at dead index {imax})")

        """
        dd, quad, ddmquad, logdet, scale, jitter, dmin, dmax = self.problem._marg_diag(jnp.asarray(best_u))

        print("dd =", float(dd))
        print("quad =", float(quad))
        print("dd-quad =", float(ddmquad))
        print("implied logL =", -0.5 * float(ddmquad))
        """

        p_active_min = getattr(self.problem, "p_active_min", 1e-3)

        decode_fn = getattr(self.problem, "decode_batch", None)
        if callable(decode_fn):
            best_phys = decode_fn(best_u[None, :], p_active_min=float(p_active_min), sort_by="f0")
            #summary = {k: best_phys.get(k) for k in ("K_hat","pK_vals","pK_probs")}
            f0_best = np.asarray(best_phys.get("f0"))
            tqdm.tqdm.write(f"[NS] best f0 = {f0_best}")            

            summary = {k: best_phys.get(k) for k in ("K_hard","K_soft","pK_vals","pK_probs")}
            
            tqdm.tqdm.write(f"[NS] best decoded summary: {summary}")
            #g = np.asarray(best_phys.get("g_u", []))
            #if g.size:
            #    tqdm.tqdm.write(f"[NS] best g_u min/max = {g.min():.3f}/{g.max():.3f}")
            
            if getattr(self.problem, "use_gates", False):
                g_raw = best_phys.get("g_u", None)
                if g_raw is not None:
                    g = np.asarray(g_raw)
                    if g.size:
                        tqdm.tqdm.write(f"[NS] best g_u min/max = {g.min():.3f}/{g.max():.3f}")

        """
        if callable(decode_fn):
            pbest = np.asarray(best_phys.get("p"))
            if pbest is not None:
                tqdm.tqdm.write(f"[NS] best p min/mean/max = {pbest.min():.3f}/{pbest.mean():.3f}/{pbest.max():.3f}")        
        """
        
        if callable(decode_fn):
            pbest_raw = best_phys.get("p", None)
            if pbest_raw is not None:
                pbest = np.asarray(pbest_raw)
                if pbest.size:
                    tqdm.tqdm.write(
                        f"[NS] best p min/mean/max = {pbest.min():.3f}/{pbest.mean():.3f}/{pbest.max():.3f}"
                    )

                
        # --- force gates ON test: does likelihood improve? ---
        try:
            K = int(self.problem.Kmax)
            per = int(self.problem.per)
            u3 = best_u.reshape(K, per)

            if getattr(self.problem, "use_gates", False):
                u_forced = u3.copy()
                u_forced[:, -1] = 5.0  # g_u -> p ~ 0.993
                u_forced = u_forced.reshape(-1)
                forced_ll = float(self.problem.loglikelihood(jnp.asarray(u_forced)))
                tqdm.tqdm.write(f"[NS] forced gates ON logL = {forced_ll:.3f}  (delta={forced_ll - best_logL:+.3f})")
        except Exception as e:
            tqdm.tqdm.write(f"[NS] forced-gate check failed: {e}")

        try:
            #nested_samples = anesthetic.NestedSamples(
            #    data=out.particles,
            #    logL=out.loglikelihood,
            #    logL_birth=out.loglikelihood_birth,
            #)

            nested_samples = anesthetic.NestedSamples(
                data=np.asarray(out.particles.position),
                logL=np.asarray(out.particles.loglikelihood),
                logL_birth=np.asarray(out.particles.loglikelihood_birth),
            )
            
            diags["ns_report"] = self.ns_report(nested_samples)
            r = diags["ns_report"]

            ess = float(r.get("ESS", float("nan")))
            diags["ns_report"]["ESS_over_nlive"] = ess / float(self.cfg.n_live)
            diags["ns_report"]["ESS_over_ndead"] = ess / max(1.0, float(len(out.particles)))
            msg = f"[NS report] logZ={r['logZ']:.3f}±{r.get('logZ_std', float('nan')):.3f}  D_KL={r['D_KL']:.3f}  ESS={r.get('ESS','?'):.0f}"
            tqdm.tqdm.write(msg)
                
            posterior_df = nested_samples.posterior_points()
            parts_raw = posterior_df.to_numpy()
        except Exception:
            parts_raw = np.asarray(out.particles)

            
        if posterior_df is not None and hasattr(posterior_df, "columns"):
            tqdm.tqdm.write(f"[debug] posterior_df columns ({len(posterior_df.columns)}): {list(posterior_df.columns)}")
        else:
            pr = np.asarray(parts_raw)
            tqdm.tqdm.write(f"[debug] posterior type={type(parts_raw)}, shape={pr.shape}")
            
        # 2) convert to physical if decoder exists
        parts = parts_raw
        extra = {"samples_raw": parts_raw}

        decode_fn = getattr(self.problem, "decode_batch", None)
        if callable(decode_fn):
            d = int(self.problem.dim)

            # If anesthetic returned a DataFrame, select param columns cleanly
            if hasattr(parts_raw, "columns") and hasattr(parts_raw, "to_numpy"):
                # columns are [0..d-1, 'logL', 'logL_birth', 'nlive', ...]
                cols = list(range(d))
                missing = [c for c in cols if c not in parts_raw.columns]
                if missing:
                    raise ValueError(f"Posterior missing param cols {missing}. Got: {list(parts_raw.columns)}")
                theta_u = parts_raw[cols].to_numpy()
                extra["samples_raw"] = parts_raw  # keep DF
            else:
                parts_raw = np.asarray(parts_raw)
                if parts_raw.ndim != 2:
                    raise ValueError(f"Expected 2D samples array, got shape {parts_raw.shape}")
                if parts_raw.shape[1] < d:
                    raise ValueError(f"posterior table has D={parts_raw.shape[1]} < problem.dim={d}; cannot decode.")
                theta_u = parts_raw[:, :d]
                extra["samples_raw"] = parts_raw

            dec = decode_fn(theta_u, p_active_min=float(p_active_min), sort_by="f0")
            parts = dec.get("flat_phys", theta_u)

            p = np.asarray(dec["p"])
            f0 = np.asarray(dec["f0"])

            tqdm.tqdm.write(f"[NS summary] posterior mean p per source = {p.mean(axis=0)}")
            tqdm.tqdm.write(f"[NS summary] frac active per source = {(p > p_active_min).mean(axis=0)}")
            tqdm.tqdm.write(f"[NS summary] f0 mean per source = {f0.mean(axis=0)}")
            tqdm.tqdm.write(f"[NS summary] f0 std per source = {f0.std(axis=0)}")
            
            extra["samples_u"] = theta_u
            if "pK_vals" in dec and "pK_probs" in dec:
                diags["pK_vals"] = np.asarray(dec["pK_vals"]).tolist()
                diags["pK_probs"] = np.asarray(dec["pK_probs"]).tolist()
            extra["decoded"] = dec
        else:
            parts = np.asarray(parts_raw)
            extra["samples_raw"] = parts_raw

        
        #print(out.particles.shape, parts.shape, parts[:10], np.array(dec.get("g_u")).min(), np.array(dec.get("g_u")).max())
        #sys.exit()
        return SamplerResult(samples=parts, weights=None, diagnostics=diags)


    
    def run_old(self, key: PRNGKey | None = None) -> SamplerResult:
        if self.state is None or self.algo is None:
            raise RuntimeError("Sampler not initialized. Call init(...) first.")
        if key is None:
            key = self.key

        @jax.jit
        def step(state, key):
            key, sub = jr.split(key)
            state, dead_pt = self.algo.step(sub, state)
            return state, key, dead_pt

        dead = []

        if self.cfg.profile:
            import os
            os.makedirs(self.cfg.profile_dir, exist_ok=True)
            jax.profiler.start_trace(self.cfg.profile_dir)

        # Continue while remaining live evidence is not yet negligible:
        # stop when (logZ_live - logZ) < -tol
        #max_iters = 100
        log_every = max(1, int(self.cfg.log_every))
        #log_every = int(_pick(cfg, "model", "log_every", default=10))
        it = 0

        print(type(self.algo))
        print(self.algo)
        print(type(self.state))
        print([x for x in dir(self.state) if not x.startswith("_")])


        with tqdm.tqdm(desc="NSS dead points", unit="pts") as pbar:
            while (self.state.logZ_live - self.state.logZ) >= -float(self.cfg.tol):
                self.state, key, dead_pt = step(self.state, key)
                dead.append(dead_pt)
                pbar.update(self.num_delete)
                it += 1

                if it % log_every == 0:
                    gap = float(self.state.logZ_live - self.state.logZ)
                    pbar.set_postfix({
                        "iter": it,
                        "gap": f"{gap:.3f}",
                        "logZ": f"{float(self.state.logZ):.3f}",
                        "logZ_live": f"{float(self.state.logZ_live):.3f}",
                        "dead": it * self.num_delete,
                    })

                #if it >= max_iters:   # pilot stop
                #    tqdm.tqdm.write(f"[pilot stop] iter={it}, gap={gap:.3f}")
                    #print(f"[pilot stop] iter={it}, gap={float(self.state.logZ_live - self.state.logZ):.3f}")
                    #sys.exit()
                    #break

        jax.block_until_ready(self.state.logZ)

        if self.cfg.profile:
            jax.profiler.stop_trace()

        out = finalise(self.state, dead)

        diags: Dict[str, Any] = {
            "logZ": float(self.state.logZ),
            "logZ_live": float(self.state.logZ_live),
            "n_live": int(self.cfg.n_live),
            "num_delete": int(self.num_delete),
            "num_inner_steps": int(self.num_inner),
        }
        
        # Convert to posterior samples (equal-weight) if anesthetic works

        posterior_df = None
        
        logL = np.asarray(out.loglikelihood)
        imax = int(np.argmax(logL))
        best_logL = float(logL[imax])
        best_u = np.asarray(out.particles)[imax, :int(self.problem.dim)]

        tqdm.tqdm.write(f"[NS] best logL = {best_logL:.3f}  (at dead index {imax})")

        """
        dd, quad, ddmquad, logdet, scale, jitter, dmin, dmax = self.problem._marg_diag(jnp.asarray(best_u))

        print("dd =", float(dd))
        print("quad =", float(quad))
        print("dd-quad =", float(ddmquad))
        print("implied logL =", -0.5 * float(ddmquad))
        """

        p_active_min = getattr(self.problem, "p_active_min", 1e-3)

        decode_fn = getattr(self.problem, "decode_batch", None)
        if callable(decode_fn):
            best_phys = decode_fn(best_u[None, :], p_active_min=float(p_active_min), sort_by="f0")
            #summary = {k: best_phys.get(k) for k in ("K_hat","pK_vals","pK_probs")}
            f0_best = np.asarray(best_phys.get("f0"))
            tqdm.tqdm.write(f"[NS] best f0 = {f0_best}")            

            summary = {k: best_phys.get(k) for k in ("K_hard","K_soft","pK_vals","pK_probs")}
            
            tqdm.tqdm.write(f"[NS] best decoded summary: {summary}")
            #g = np.asarray(best_phys.get("g_u", []))
            #if g.size:
            #    tqdm.tqdm.write(f"[NS] best g_u min/max = {g.min():.3f}/{g.max():.3f}")
            
            if getattr(self.problem, "use_gates", False):
                g_raw = best_phys.get("g_u", None)
                if g_raw is not None:
                    g = np.asarray(g_raw)
                    if g.size:
                        tqdm.tqdm.write(f"[NS] best g_u min/max = {g.min():.3f}/{g.max():.3f}")

        """
        if callable(decode_fn):
            pbest = np.asarray(best_phys.get("p"))
            if pbest is not None:
                tqdm.tqdm.write(f"[NS] best p min/mean/max = {pbest.min():.3f}/{pbest.mean():.3f}/{pbest.max():.3f}")        
        """
        
        if callable(decode_fn):
            pbest_raw = best_phys.get("p", None)
            if pbest_raw is not None:
                pbest = np.asarray(pbest_raw)
                if pbest.size:
                    tqdm.tqdm.write(
                        f"[NS] best p min/mean/max = {pbest.min():.3f}/{pbest.mean():.3f}/{pbest.max():.3f}"
                    )

                
        # --- force gates ON test: does likelihood improve? ---
        try:
            K = int(self.problem.Kmax)
            per = int(self.problem.per)
            u3 = best_u.reshape(K, per)

            if getattr(self.problem, "use_gates", False):
                u_forced = u3.copy()
                u_forced[:, -1] = 5.0  # g_u -> p ~ 0.993
                u_forced = u_forced.reshape(-1)
                forced_ll = float(self.problem.loglikelihood(jnp.asarray(u_forced)))
                tqdm.tqdm.write(f"[NS] forced gates ON logL = {forced_ll:.3f}  (delta={forced_ll - best_logL:+.3f})")
        except Exception as e:
            tqdm.tqdm.write(f"[NS] forced-gate check failed: {e}")

        try:
            nested_samples = anesthetic.NestedSamples(
                data=out.particles,
                logL=out.loglikelihood,
                logL_birth=out.loglikelihood_birth,
            )

            diags["ns_report"] = self.ns_report(nested_samples)
            r = diags["ns_report"]

            ess = float(r.get("ESS", float("nan")))
            diags["ns_report"]["ESS_over_nlive"] = ess / float(self.cfg.n_live)
            diags["ns_report"]["ESS_over_ndead"] = ess / max(1.0, float(len(out.particles)))
            msg = f"[NS report] logZ={r['logZ']:.3f}±{r.get('logZ_std', float('nan')):.3f}  D_KL={r['D_KL']:.3f}  ESS={r.get('ESS','?'):.0f}"
            tqdm.tqdm.write(msg)
                
            posterior_df = nested_samples.posterior_points()
            parts_raw = posterior_df.to_numpy()
        except Exception:
            parts_raw = np.asarray(out.particles)

            
        if posterior_df is not None and hasattr(posterior_df, "columns"):
            tqdm.tqdm.write(f"[debug] posterior_df columns ({len(posterior_df.columns)}): {list(posterior_df.columns)}")
        else:
            pr = np.asarray(parts_raw)
            tqdm.tqdm.write(f"[debug] posterior type={type(parts_raw)}, shape={pr.shape}")
            
        # 2) convert to physical if decoder exists
        parts = parts_raw
        extra = {"samples_raw": parts_raw}

        decode_fn = getattr(self.problem, "decode_batch", None)
        if callable(decode_fn):
            d = int(self.problem.dim)

            # If anesthetic returned a DataFrame, select param columns cleanly
            if hasattr(parts_raw, "columns") and hasattr(parts_raw, "to_numpy"):
                # columns are [0..d-1, 'logL', 'logL_birth', 'nlive', ...]
                cols = list(range(d))
                missing = [c for c in cols if c not in parts_raw.columns]
                if missing:
                    raise ValueError(f"Posterior missing param cols {missing}. Got: {list(parts_raw.columns)}")
                theta_u = parts_raw[cols].to_numpy()
                extra["samples_raw"] = parts_raw  # keep DF
            else:
                parts_raw = np.asarray(parts_raw)
                if parts_raw.ndim != 2:
                    raise ValueError(f"Expected 2D samples array, got shape {parts_raw.shape}")
                if parts_raw.shape[1] < d:
                    raise ValueError(f"posterior table has D={parts_raw.shape[1]} < problem.dim={d}; cannot decode.")
                theta_u = parts_raw[:, :d]
                extra["samples_raw"] = parts_raw

            dec = decode_fn(theta_u, p_active_min=float(p_active_min), sort_by="f0")
            parts = dec.get("flat_phys", theta_u)

            p = np.asarray(dec["p"])
            f0 = np.asarray(dec["f0"])

            tqdm.tqdm.write(f"[NS summary] posterior mean p per source = {p.mean(axis=0)}")
            tqdm.tqdm.write(f"[NS summary] frac active per source = {(p > p_active_min).mean(axis=0)}")
            tqdm.tqdm.write(f"[NS summary] f0 mean per source = {f0.mean(axis=0)}")
            tqdm.tqdm.write(f"[NS summary] f0 std per source = {f0.std(axis=0)}")
            
            extra["samples_u"] = theta_u
            if "pK_vals" in dec and "pK_probs" in dec:
                diags["pK_vals"] = np.asarray(dec["pK_vals"]).tolist()
                diags["pK_probs"] = np.asarray(dec["pK_probs"]).tolist()
            extra["decoded"] = dec
        else:
            parts = np.asarray(parts_raw)
            extra["samples_raw"] = parts_raw

        
        #print(out.particles.shape, parts.shape, parts[:10], np.array(dec.get("g_u")).min(), np.array(dec.get("g_u")).max())
        #sys.exit()
        return SamplerResult(samples=parts, weights=None, diagnostics=diags)

    
class BlackJAXNestedSamplerTD:
    def __init__(self, problem: Problem, cfg: NSConfig):
        self.problem = problem
        self.cfg = cfg

    def init(self, key: PRNGKey, problem: Problem | None = None, **cfg):
        if problem is not None:
            self.problem = problem
        for k, v in cfg.items():
            setattr(self.cfg, k, v)

        self.key = key
        self.d = int(self.problem.dim)

        self.num_delete = int(self.cfg.num_delete_ratio * self.cfg.n_live)

        if self.cfg.num_inner_steps is None or self.cfg.num_inner_steps <= 0:
            self.cfg.num_inner_steps = 3 * self.d
        self.num_inner = int(self.cfg.num_inner_steps)

        def _scalar(x):
            # forces “size==1”; if not, JAX will throw a shape error (good!)
            return jnp.reshape(jnp.asarray(x), ())

        def logprior_1(theta):
            theta = jnp.asarray(theta).reshape((self.d,))
            return _scalar(self.problem.logprior(theta))

        def loglike_1(theta):
            theta = jnp.asarray(theta).reshape((self.d,))
            return _scalar(self.problem.loglikelihood(theta))

        self.algo = blackjax.nss(
            logprior_fn=logprior_1,
            loglikelihood_fn=loglike_1,
            num_delete=self.num_delete,
            num_inner_steps=self.num_inner,
        )

        self.key, sub = jr.split(self.key)
        init_pts = self.problem.sample_prior(sub, self.cfg.n_live)
        if init_pts.shape != (self.cfg.n_live, self.d):
            raise ValueError(f"sample_prior returned {init_pts.shape}, expected {(self.cfg.n_live, self.d)}")

        # Optional: early non-jitted sanity check on one point (Python-side)
        _ = self.problem.logprior(jnp.asarray(init_pts[0]).reshape((self.d,)))
        _ = self.problem.loglikelihood(jnp.asarray(init_pts[0]).reshape((self.d,)))

        self.state = self.algo.init(init_pts)
        return self
    

class BlackJAXNestedSamplerFD:
    def __init__(self, problem: Problem, cfg: NSConfig):
        self.problem = problem
        self.cfg = cfg

    def init(self, key: PRNGKey, problem: Problem | None = None, **cfg):
        if problem is not None:
            self.problem = problem
        for k, v in cfg.items():
            setattr(self.cfg, k, v)
        self.key = key
        self.d = self.problem.dim
        self.num_delete = int(self.cfg.num_delete_ratio * self.cfg.n_live)
        self.num_inner = self.cfg.num_inner_steps
        self.algo = blackjax.nss(
            logprior_fn=self.problem.logprior,
            loglikelihood_fn=self.problem.loglikelihood,
            num_delete=self.num_delete,
            num_inner_steps=self.num_inner
        )
        self.key, sub = jr.split(self.key)
        init_pts = self.problem.sample_prior(sub, self.cfg.n_live)

        #print("Initial particles shape:", init_pts.shape)
        #print("Example particle:", init_pts[0])

        self.state = self.algo.init(init_pts)
        return self

    def run(self, key: PRNGKey) -> SamplerResult:
        @jax.jit
        def step(state, key):
            key, sub = jr.split(key)
            state, dead_pt = self.algo.step(sub, state)
            return state, key, dead_pt

        dead = []
        if self.cfg.profile:
            import os
            os.makedirs(self.cfg.profile_dir, exist_ok=True)
            jax.profiler.start_trace(self.cfg.profile_dir)

        with tqdm.tqdm(desc="NSS dead points", unit="pts") as pbar:
            while (self.state.logZ_live - self.state.logZ) > -self.cfg.tol:
                self.state, key, dead_pt = step(self.state, key)
                dead.append(dead_pt)
                pbar.update(self.num_delete)
                
        jax.block_until_ready(self.state.logZ)
        if self.cfg.profile:
            jax.profiler.stop_trace()

        out = finalise(self.state, dead)

        parts = None
        used_anesthetic = False
        try:
            nested_samples = anesthetic.NestedSamples(
                data=out.particles,
                logL=out.loglikelihood,
                logL_birth=out.loglikelihood_birth,
            )

            #prior = nested_samples.set_beta(0.0).plot_2d(np.arange(9), label="prior")
            #post = nested_samples.plot_2d(prior, label= "posterior")
            #prior.iloc[-1, 0].legend(bbox_to_anchor=(len(prior), len(prior)), loc='lower right')
        

            
            # Equal-weight posterior draws (default nsamples is fine; tweak if you want)
            posterior_df = nested_samples.posterior_points()
            

            parts_u = posterior_df.to_numpy() # np.asarray(posterior_df.to_numpy())
            #weights = np.ones(parts.shape[0], dtype=float) / max(1, parts.shape[0])
            used_anesthetic = True
        except Exception:
            # Fallback: raw NS particles (dead+live finalized). Plots may warn about contours.
            parts_u = np.asarray(out.particles)
            #weights = None


        """
        samples = posterior_df.to_numpy()               # columns: [K, mu0, mu1, ..., mu_{K_max-1}]
    
        K_max = 5
        # 1) Plot p(K|y) separately:
        counts = np.bincount(samples[:,0].astype(int), minlength=K_max+1)[1:]
        pmf    = counts / counts.sum()
        plt.figure()
        plt.bar(np.arange(1, K_max+1), pmf, align="center")
        plt.xlabel("K"); plt.ylabel("p(K|y)")
        plt.show()
        sys.exit() 
        """   
        diags: Dict[str, Any] = {
            "logZ": float(self.state.logZ),
            "logZ_live": float(self.state.logZ_live),
        }

        # decode to physical for output
        if hasattr(self.problem, "decode_batch"):
            dec = self.problem.decode_batch(parts_u, p_active_min=float(p_active_min))
            parts = dec["flat_phys"]          # this becomes the returned chain
            diags["pK_vals"] = dec["pK_vals"].tolist()
            diags["pK_probs"] = dec["pK_probs"].tolist()
            # optionally keep raw in diagnostics if not too large:
            # diags["samples_u_shape"] = list(parts_u.shape)
        else:
            parts = parts_u
        
        #extra: Dict[str, Any] = {
        #    "loglikelihood": np.array(out.loglikelihood),
        #    "loglikelihood_birth": np.array(out.loglikelihood_birth),
        #    "nested_out": out,
        #}
        #return SamplerResult(samples=parts, weights=weights, diagnostics=diags)
        return SamplerResult(samples=parts, weights=None, diagnostics=diags)
        #return SamplerResult(samples=parts, weights=None, diagnostics=diags, extra=extras)
        
    
# Register in the plugin registry
register_sampler("ns")(BlackJAXNestedSampler)


class BlackJAXDynamicNSS(BlackJAXNestedSampler):
    """Dynamic nested sampling using blackjax.dynamic_nss."""

    def init(self, key: PRNGKey, problem: Problem | None = None, **cfg):
        logprior_fn, loglike_fn, init_pts = self._setup_common(key, problem, cfg)

        self.algo = blackjax.dynamic_nss(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglike_fn,
            num_delete=self.num_delete,
            num_inner_steps=self.num_inner,
        )

        self.state = self.algo.init(init_pts)
        return self


class BlackJAXHamiltonianNS(BlackJAXNestedSampler):
    """Hamiltonian nested sampling using blackjax.ns_hamiltonian."""

    def init(self, key: PRNGKey, problem: Problem | None = None, **cfg):
        logprior_fn, loglike_fn, init_pts = self._setup_common(key, problem, cfg)

        # Resolve lower/upper bounds
        lower = self.cfg.lower
        upper = self.cfg.upper
        if lower is None:
            lower = getattr(self.problem, "lower", None)
        if upper is None:
            upper = getattr(self.problem, "upper", None)
        if lower is None or upper is None:
            raise ValueError(
                "BlackJAXHamiltonianNS requires lower and upper bounds for reflections. "
                "Pass --ham-lower / --ham-upper on the CLI or set them in the config, "
                "or ensure your Problem exposes .lower and .upper attributes."
            )

        lower_arr = jnp.array(lower, dtype=jnp.float64 if jnp.array(lower).dtype == jnp.float64 else jnp.float32)
        upper_arr = jnp.array(upper, dtype=lower_arr.dtype)

        self.algo = blackjax.ns_hamiltonian(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglike_fn,
            num_delete=self.num_delete,
            num_inner_steps=self.num_inner,
            dt_ini=float(self.cfg.dt_ini),
            min_reflections=int(self.cfg.min_reflections),
            max_reflections=int(self.cfg.max_reflections),
            sigma_vel=float(self.cfg.sigma_vel),
            max_steps=int(self.cfg.ham_max_steps),
            lower=lower_arr,
            upper=upper_arr,
        )

        self.state = self.algo.init(init_pts)
        return self




def _resolve_ggns_ctor(dynamic: bool = False):
    """Resolve GGNS constructor name from BlackJAX fork variants."""
    candidates = (
        ("dynamic_ggns", "ggns_dynamic") if dynamic else ("ggns", "static_ggns"),
    )[0]
    for name in candidates:
        ctor = getattr(blackjax, name, None)
        if callable(ctor):
            return ctor, name
    mode = "dynamic" if dynamic else "static"
    tried = ", ".join(candidates)
    raise AttributeError(f"BlackJAX {mode} GGNS constructor not found. Tried: {tried}")


class BlackJAXGGNS(BlackJAXNestedSampler):
    """Static GGNS using BlackJAX fork API (if available)."""

    def init(self, key: PRNGKey, problem: Problem | None = None, **cfg):
        logprior_fn, loglike_fn, init_pts = self._setup_common(key, problem, cfg)
        ctor, _ = _resolve_ggns_ctor(dynamic=False)
        self.algo = ctor(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglike_fn,
            num_delete=self.num_delete,
            num_inner_steps=int(self.cfg.ggns_num_inner_steps),
            step_size=float(self.cfg.ggns_step_size),
        )
        self.state = self.algo.init(init_pts)
        return self


class BlackJAXDynamicGGNS(BlackJAXNestedSampler):
    """Dynamic GGNS using BlackJAX fork API (if available)."""

    def init(self, key: PRNGKey, problem: Problem | None = None, **cfg):
        logprior_fn, loglike_fn, init_pts = self._setup_common(key, problem, cfg)
        ctor, _ = _resolve_ggns_ctor(dynamic=True)
        self.algo = ctor(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglike_fn,
            num_delete=self.num_delete,
            num_inner_steps=int(self.cfg.ggns_num_inner_steps),
            step_size=float(self.cfg.ggns_step_size),
        )
        self.state = self.algo.init(init_pts)
        return self


register_sampler("dynamic_nss")(BlackJAXDynamicNSS)
register_sampler("ns_hamiltonian")(BlackJAXHamiltonianNS)
register_sampler("ggns")(BlackJAXGGNS)
register_sampler("dynamic_ggns")(BlackJAXDynamicGGNS)
