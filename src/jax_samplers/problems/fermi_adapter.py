# src/jax_samplers/problems/fermi_adapter.py
from __future__ import annotations
import os, json
import jax
import jax.numpy as jnp
import jax.random as jr
from jax_samplers.core.problem import Problem
from jax_samplers.core.precision import resolve
from jax_samplers.problems.fermi_ps import FermiPointSourcesProblem

# reuse your implementation
from .fermi_ps import (
    FermiPointSourcesProblem,
    forward_model,
    poisson_loglik,
    debug_poisson_terms
)

class FermiFixedAdapter(Problem):
    def __init__(self, base: FermiPointSourcesProblem):
        assert base.mode == "fixed"
        self.base = base
        self.DT = getattr(base, "DT", jnp.float32)  # precision selected in problem

        self.K = int(base.K_fixed)
        self.fit_bg = bool(base.cfg.allow_bg_fit)   # <- keep/omit bg dims
        self.H, self.W = base.patch.iso.shape

        # compute bounds in the chosen dtype
        self.LOG_FMIN = jnp.log(self.DT(base.cfg.flux_min))
        self.LOG_FMAX = jnp.log(self.DT(base.cfg.flux_max))

        # parameter dimension: drop bg if not fitting
        self.dim = 3 * self.K + (2 if self.fit_bg else 0)

        # optional: prior box for plotting
        self.prior_bounds = (
            [(0.0, self.W-1.0)] * self.K +
            [(0.0, self.H-1.0)] * self.K +
            [(float(self.LOG_FMIN), float(self.LOG_FMAX))] * self.K +
            ([] if not self.fit_bg else [(-jnp.inf, jnp.inf), (-jnp.inf, jnp.inf)])
        )

    # ---- helpers ----
    def _theta_bad(self, theta: jnp.ndarray) -> jnp.ndarray:
        theta = theta.astype(self.DT)
        K = self.K
        x    = theta[0:K]
        y    = theta[K:2*K]
        logF = theta[2*K:3*K]
        finite = jnp.all(jnp.isfinite(theta))
        in_pos = jnp.all((x >= 0) & (x < self.DT(self.W)) &
                         (y >= 0) & (y < self.DT(self.H)))
        in_lf  = jnp.all((logF >= self.LOG_FMIN) & (logF <= self.LOG_FMAX))
        return ~(finite & in_pos & in_lf)

    def _unpack(self, theta: jnp.ndarray):
        theta = theta.astype(self.DT)
        K = self.K
        x    = theta[0:K]
        y    = theta[K:2*K]
        logF = jnp.clip(theta[2*K:3*K], self.LOG_FMIN, self.LOG_FMAX)  # clip BEFORE exp
        pos  = jnp.stack([x, y], axis=-1).astype(self.DT)
        flx  = jnp.exp(logF).astype(self.DT)
        if self.fit_bg:
            bg = theta[3*K:3*K+2].astype(self.DT)
        else:
            bg = jnp.array([1.0, 1.0], dtype=self.DT)
        return {"K": self.K, "positions": pos, "fluxes": flx, "bg": bg}

    def _logprior_single(self, theta):
        dt = theta.dtype
        minus_inf = jnp.asarray(-jnp.inf, dt)
        def ok():
            K = self.K
            lp_pos  = -K * jnp.log(jnp.asarray(self.H * self.W, dt))
            logf_min = jnp.log(jnp.asarray(self.base.cfg.flux_min, dt))
            logf_max = jnp.log(jnp.asarray(self.base.cfg.flux_max, dt))
            lp_flux = -K * jnp.log(logf_max - logf_min)
            lp_bg   = jnp.asarray(0.0, dt)
            if self.fit_bg:
                bg  = theta[3*K:3*K+2].astype(dt)
                mu  = jnp.asarray(self.base.cfg.bg_scale_mu, dt)
                sig = jnp.asarray(self.base.cfg.bg_scale_sigma, dt)
                lp_bg = -0.5 * jnp.sum(((bg - mu)/sig)**2) - jnp.sum(jnp.log(sig * jnp.sqrt(jnp.asarray(2.0*jnp.pi, dt))))
            return (lp_pos + lp_flux + lp_bg).astype(dt)
        return jax.lax.cond(self._theta_bad(theta), lambda: minus_inf, ok)

    
    # ---- single-point logprior/likelihood with dtype-stable cond ----
    def _logprior_single_old(self, theta: jnp.ndarray) -> jnp.ndarray:
        dt = theta.dtype
        minus_inf = jnp.asarray(-jnp.inf, dt)
        
        def ok():
            K = self.K
            lp_pos  = -K * jnp.log(self.DT(self.H * self.W))           # uniform in (x,y)
            lp_flux = -K * jnp.log(self.LOG_FMAX - self.LOG_FMIN)      # uniform in log f
            lp_bg   = self.DT(0.0)
            if self.fit_bg:
                bg  = theta[3*K:3*K+2].astype(self.DT)
                mu  = jnp.asarray(self.base.cfg.bg_scale_mu, self.DT)
                sig = jnp.asarray(self.base.cfg.bg_scale_sigma, self.DT)
                lp_bg = -0.5 * jnp.sum(((bg - mu) / sig) ** 2) - jnp.sum(jnp.log(sig * jnp.sqrt(self.DT(2.0*jnp.pi))))
            return (lp_pos + lp_flux + lp_bg).astype(self.DT)
        return jax.lax.cond(self._theta_bad(theta), lambda: minus_inf, ok)

    def _loglikelihood_single(self, theta: jnp.ndarray) -> jnp.ndarray:
        # dtype that the NS state (and slice sampler carry) is using
        ns_dt = theta.dtype

        minus_inf = jnp.asarray(-jnp.inf, ns_dt)

        def ok():
            # choose model dtype: default to ns_dt; or use self.base.MODEL_DT if you support it
            model_dt = getattr(self.base, "MODEL_DT", ns_dt)

            # unpack IN model dtype (avoid doing work before the guard)
            th = self._unpack(jax.lax.convert_element_type(theta, model_dt))

            lam = forward_model(
                self.base.patch.iso.astype(model_dt),
                self.base.patch.iem.astype(model_dt),
                self.base.patch.Fker.astype(model_dt),
                th["positions"], th["fluxes"], th["bg"]
            )
            ll = poisson_loglik(lam, self.base.patch.counts.astype(model_dt))
            return ll.astype(ns_dt)  # cast scalar back to NS dtype

        # EARLY REJECT: if any NaN / OOB / logF out-of-range → skip model, return -inf
        return jax.lax.cond(self._theta_bad(theta), lambda: minus_inf, ok)

    
    def _loglikelihood_single_old(self, theta: jnp.ndarray) -> jnp.ndarray:
        dt = theta.dtype
        minus_inf = jnp.asarray(-jnp.inf, dt)
        
        th = self._unpack(theta)  # already clipped & cast
        
        """
        pos = th["positions"]
        flx = th["fluxes"]
        pos = pos.at[0, 0].set(7.51671448e+01)
        pos = pos.at[0, 1].set(44.745322)
        pos = pos.at[1, 0].set(72.748227)
        pos = pos.at[1, 1].set(103.06816)
        pos = pos.at[2, 0].set(96.8931122)
        pos = pos.at[2, 1].set(120.60628)
        flx = flx.at[0].set(3.28665308e-08)
        flx = flx.at[1].set(2.29986807e-09)
        flx = flx.at[2].set(2.17807106e-09)
  
        print(th["positions"], th["fluxes"])
        #lam = forward_model(self.base.patch.iso, self.base.patch.iem,
        #                        self.base.patch.Fker, pos, flx, th["bg"])
        lam = forward_model(self.base.patch.iso, self.base.patch.iem,
                                self.base.patch.Fker, th["positions"], th["fluxes"], th["bg"])       
        loglike =  poisson_loglik(lam, self.base.patch.counts).astype(dt)
        
        #loglike =  poisson_loglik(lam, self.base.patch.counts).astype(self.DT)

        return loglike
        #sys.exit()    
        """
        def ok():
            th = self._unpack(theta)  # already clipped & cast
            
            lam = forward_model(self.base.patch.iso, self.base.patch.iem,
                                self.base.patch.Fker, th["positions"], th["fluxes"], th["bg"])
            loglike =  poisson_loglik(lam, self.base.patch.counts).astype(self.DT)

            return poisson_loglik(lam, self.base.patch.counts).astype(dt)
        return jax.lax.cond(self._theta_bad(theta), lambda: minus_inf, ok)

    # ---- public API used by samplers ----
    def logprior(self, theta: jnp.ndarray) -> jnp.ndarray:
        theta = theta.astype(self.DT)
        return self._logprior_single(theta)

    def loglikelihood(self, theta: jnp.ndarray) -> jnp.ndarray:
        theta = theta.astype(self.DT)
        return self._loglikelihood_single(theta)

    def sample_prior(self, key: jr.PRNGKey, n: int) -> jnp.ndarray:
        K, DT = self.K, self.DT
        def draw(k):
            th  = self.base.sample_prior(k)          # dict from base (already DT)
            pos = th["positions"][:K].astype(DT)
            x, y = pos[:, 0], pos[:, 1]
            # build θ with logF parameters (like old fermi_fixed)
            logF = jnp.log(th["fluxes"][:K].astype(DT))
            parts = [x, y, logF]
            if self.fit_bg:
                parts.append(th["bg"].astype(DT))
            return jnp.concatenate(parts, axis=0).astype(DT)
        return jax.vmap(draw)(jr.split(key, n))      # (n, dim)


class FermiFixedAdapter_old(Problem):
    def __init__(self, base: FermiPointSourcesProblem):
        assert base.mode == "fixed"
        self.base = base
        self.DT = getattr(base, "DT", jnp.float32)  # use problem’s dtype
        
        self.K = int(base.K_fixed)
        self.dim = 3*self.K + 2  # [x1..xK, y1..yK, logF1..K, bg0,bg1]

        H, W = base.patch.iso.shape
        LOG_FMIN = jnp.log(base.cfg.flux_min)
        LOG_FMAX = jnp.log(base.cfg.flux_max)
        self.prior_bounds = (
            [(0.0, W-1.0)] * self.K +
            [(0.0, H-1.0)] * self.K +
            [(float(LOG_FMIN), float(LOG_FMAX))] * self.K +
            [(-jnp.inf, jnp.inf), (-jnp.inf, jnp.inf)]
        )

    def _unpack(self, theta: jnp.ndarray):
        x    = theta[:self.K]
        y    = theta[self.K:2*self.K]
        logF = theta[2*self.K:3*self.K]
        bg   = theta[3*self.K:3*self.K+2]
        pos  = jnp.stack([x, y], axis=-1)
        flx  = jnp.exp(logF)
        return {"K": self.K, "positions": pos, "fluxes": flx, "bg": bg}

    def _loglikelihood_single(self, theta: jnp.ndarray) -> jnp.ndarray:
        return self.base.loglik(self._unpack(theta))

    #def _logprior_single(self, theta: jnp.ndarray) -> jnp.ndarray:
    #    return self.base.log_prior(self._unpack(theta))

    def _logprior_single(self, theta: jnp.ndarray) -> jnp.ndarray:
        # theta layout: [x(0..K-1), y(0..K-1), logF(0..K-1), bg0, bg1]
        K = self.K
        W = self.base.cfg.W
        H = self.base.cfg.H
        logf_min = jnp.log(self.base.cfg.flux_min)
        logf_max = jnp.log(self.base.cfg.flux_max)

        x    = theta[0:K]
        y    = theta[K:2*K]
        logF = theta[2*K:3*K]
        bg   = theta[3*K:3*K+2]

        in_box  = jnp.all((x >= 0) & (x < W) & (y >= 0) & (y < H))
        in_flux = jnp.all((logF >= logf_min) & (logF <= logf_max))

        lp_pos  = -K * jnp.log(float(H * W))                       # uniform in (x,y)
        lp_flux = -K * jnp.log(logf_max - logf_min)               # uniform in log f (θ-space)
        lp_bg   = 0.0
        if self.base.cfg.allow_bg_fit:
            mu  = jnp.array(self.base.cfg.bg_scale_mu, dtype=theta.dtype)
            sig = jnp.array(self.base.cfg.bg_scale_sigma, dtype=theta.dtype)
            lp_bg = -0.5 * jnp.sum(((bg - mu) / sig) ** 2) - jnp.sum(jnp.log(sig * jnp.sqrt(2 * jnp.pi)))

        lp = lp_pos + lp_flux + lp_bg
        return jnp.where(in_box & in_flux, lp, -jnp.inf)

    
    def sample_prior_old(self, key: jr.PRNGKey, n: int) -> jnp.ndarray:
        keys = jr.split(key, n)
        def draw(k):
            th = self.base.sample_prior(k)
            pos, flx, bg = th["positions"][:self.K], th["fluxes"][:self.K], th["bg"]
            x, y = pos[:,0], pos[:,1]
            logF = jnp.log(flx)
            return jnp.concatenate([x, y, logF, bg], axis=0)
        return jnp.stack([draw(k) for k in keys], axis=0)


    def sample_prior(self, key: jr.PRNGKey, n: int) -> jnp.ndarray:
        K, DT = self.K, getattr(self.base, "DT", jnp.float32)

        def draw(k):
            th  = self.base.sample_prior(k)              # already in DT if you wired it
            pos = th["positions"][:K].astype(DT)
            flx = th["fluxes"][:K].astype(DT)
            x, y  = pos[:, 0], pos[:, 1]
            logF  = jnp.log(flx).astype(DT)
            parts = [x, y, logF]
            if getattr(self, "fit_bg", bool(self.base.cfg.allow_bg_fit)):
                parts.append(th["bg"].astype(DT))
                return jnp.concatenate(parts, axis=0).astype(DT)

        keys = jr.split(key, n)
        return jax.vmap(draw)(keys)

    
    def sample_prior_old(self, key: jr.PRNGKey, n: int) -> jnp.ndarray:
        def draw(k):
            th  = self.base.sample_prior(k)
            # dict from base
            pos = th["positions"][:self.K]
            flx = th["fluxes"][:self.K]
            x, y = pos[:, 0], pos[:, 1]
            logF = jnp.log(flx)
            bg   = th.get("bg", jnp.array([1.0, 1.0], dtype=logF.dtype))
            return jnp.concatenate([x, y, logF, bg], axis=0).astype(jnp.float32)

        keys = jr.split(key, n)
        return jax.vmap(draw)(keys)                      # (n, dim)
    

def make(*, config_path: str = "", extras=None, **overrides):
    extras = extras or {}
    config_path = config_path or extras.get("config_path") \
                  or os.environ.get("FERMI_CONFIG_PATH", "")

    if not config_path:
        raise ValueError("Provide config_path or FERMI_CONFIG_PATH")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    cfg.update(extras)
    cfg.update(overrides)

    # Pass strings through; problem will resolve
    dtype_name       = str(cfg.get("dtype", "float64"))        # NS/state dtype
    model_dtype_name = str(cfg.get("model_dtype", dtype_name))  # model math dtype
    
    base = FermiPointSourcesProblem(
        dtype=dtype_name,
        model_dtype=model_dtype_name,
        iso_path=cfg.get("iso_path"),
        iem_path=cfg.get("iem_path"),
        exposure_path=cfg.get("exposure_path"),
        counts_path=cfg.get("counts_path"),
        psf_size=int(cfg.get("psf_size", 41)),
        psf_sigma=float(cfg.get("psf_sigma", 2)),
        psf_gamma=float(cfg.get("psf_gamma", 30)),
        flux_min=float(cfg.get("flux_min", 1e-11)),
        flux_max=float(cfg.get("flux_max", 1e-7)),
        mode="fixed",
        K_fixed=int(cfg.get("K_fixed", 3)),
        allow_bg_fit=bool(cfg.get("allow_bg_fit", False)),
        bg_scale_mu=tuple(cfg.get("bg_scale_mu", (1.0, 1.0))),
        bg_scale_sigma=tuple(cfg.get("bg_scale_sigma", (0.1, 0.1))),
    )
    return FermiFixedAdapter(base)

    
