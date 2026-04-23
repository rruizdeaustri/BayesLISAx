# problems/fermi_fixed.py
from __future__ import annotations
import os, json
import jax  #
import jax.numpy as jnp
import jax.random as jr

# reuse your implementation
from .fermi_ps import (
    FermiPointSourcesProblem,
    forward_model,
    poisson_loglik,
    debug_poisson_terms
)

# ---------- config at import time ----------
# Set this before running:
#   export FERMI_FIXED_CONFIG=/path/to/fermi_fixed.json
_cfg_path = os.environ.get("FERMI_FIXED_CONFIG")
if _cfg_path is None:
    raise RuntimeError("Set FERMI_FIXED_CONFIG to a JSON file via FERMI_FIXED_CONFIG")

with open(_cfg_path, "r") as f:
    CFG = json.load(f)

K_FIXED: int = int(CFG.get("K_fixed", 3))
ALLOW_BG: bool = bool(CFG.get("allow_bg_fit", False))

# build the underlying problem in fixed mode
PROB = FermiPointSourcesProblem(
    iso_path=CFG.get("iso_path"),
    iem_path=CFG.get("iem_path"),
    exposure_path=CFG.get("exposure_path"),
    counts_path=CFG.get("counts_path"),
    psf_size=int(CFG.get("psf_size", 40)),
    psf_sigma=float(CFG.get("psf_sigma", 2)),
    psf_gamma=float(CFG.get("psf_gamma", 30)),
    flux_min=float(CFG.get("flux_min", 1e-11)),
    flux_max=float(CFG.get("flux_max", 1e-7)),
    mode="fixed",
    K_fixed=K_FIXED,
    allow_bg_fit=ALLOW_BG,
    bg_scale_mu=tuple(CFG.get("bg_scale_mu", (1.0, 1.0))),
    bg_scale_sigma=tuple(CFG.get("bg_scale_sigma", (0.1, 0.1))),
)

H, W = PROB.patch.shape
LOG_FMIN = jnp.log(PROB.cfg.flux_min)
LOG_FMAX = jnp.log(PROB.cfg.flux_max)

# parameterization: [y1,x1,logf1, y2,x2,logf2, ..., yK,xK,logfK, (a_iso,a_iem if ALLOW_BG)]
DIM = 3 * K_FIXED + (2 if ALLOW_BG else 0)

def _vec_to_params(theta: jnp.ndarray):
    """Decode flat theta -> (positions[K,2], fluxes[K], bg[2])"""
    theta = theta.reshape((-1,))
    pos = theta[: 2 * K_FIXED].reshape((K_FIXED, 2))
    logf = theta[2 * K_FIXED : 3 * K_FIXED]
    flx = jnp.exp(logf)
    if ALLOW_BG:
        bg = theta[3 * K_FIXED : 3 * K_FIXED + 2]
    else:
        bg = jnp.array([1.0, 1.0])
    return pos, flx, bg

def _in_support(pos: jnp.ndarray, logf: jnp.ndarray) -> bool:
    y_ok = (pos[:, 0] >= 0.0) & (pos[:, 0] < float(H))
    x_ok = (pos[:, 1] >= 0.0) & (pos[:, 1] < float(W))
    f_ok = (logf >= LOG_FMIN) & (logf <= LOG_FMAX)
    return jnp.all(y_ok & x_ok) & jnp.all(f_ok)
    #return bool(jnp.all(y_ok & x_ok) & jnp.all(f_ok))

# ---------- entrypoints for your GenericProblem ----------
def _sample_one(key: jnp.ndarray) -> jnp.ndarray:
    """One draw → (DIM,)"""
    th  = PROB.sample_prior(key)   # fixed-K
    pos = th["positions"][:K_FIXED]
    logf = jnp.log(th["fluxes"][:K_FIXED])
    parts = [pos.reshape(-1), logf]
    if ALLOW_BG:
        parts.append(th["bg"])
    return jnp.concatenate(parts).astype(jnp.float32)

def sample_prior(key: jnp.ndarray, n: int) -> jnp.ndarray:
    """Batch of n draws → (n, DIM)"""
    keys = jr.split(key, n)
    return jax.vmap(_sample_one)(keys)

@jax.jit
def _logprior_one(theta: jnp.ndarray) -> jnp.ndarray:
    K = K_FIXED; H, W = PROB.patch.iso.shape
    x    = theta[0:K]
    y    = theta[K:2*K]
    logF = theta[2*K:3*K]

    in_box  = jnp.all((x >= 0) & (x < W) & (y >= 0) & (y < H))
    in_flux = jnp.all((logF >= LOG_FMIN) & (logF <= LOG_FMAX))

    lp_pos  = -K * jnp.log(float(H * W))
    lp_flux = -K * jnp.log(LOG_FMAX - LOG_FMIN)
    lp_bg   = 0.0
    if ALLOW_BG:
        bg  = theta[3*K:3*K+2]
        mu  = jnp.array(PROB.cfg.bg_scale_mu)
        sig = jnp.array(PROB.cfg.bg_scale_sigma)
        lp_bg = -0.5 * jnp.sum(((bg - mu) / sig) ** 2) - jnp.sum(jnp.log(sig * jnp.sqrt(2 * jnp.pi)))

    lp = lp_pos + lp_flux + lp_bg
    return jnp.where(in_box & in_flux, lp, -jnp.inf)

@jax.jit
def _logprior_one_old(theta: jnp.ndarray) -> jnp.ndarray:
    pos = theta[: 2 * K_FIXED].reshape((K_FIXED, 2))
    logf = theta[2 * K_FIXED : 3 * K_FIXED]
    in_supp = _in_support(pos, logf)
    # Use where to avoid branching under JIT
    lp_pos  = -K_FIXED * jnp.log(float(H * W))
    lp_flux = -K_FIXED * jnp.log(LOG_FMAX - LOG_FMIN)
    lp_bg = 0.0
    if ALLOW_BG:
        bg  = theta[3 * K_FIXED : 3 * K_FIXED + 2]
        mu  = jnp.array(PROB.cfg.bg_scale_mu)
        sig = jnp.array(PROB.cfg.bg_scale_sigma)
        lp_bg = -0.5 * jnp.sum(((bg - mu) / sig) ** 2) - jnp.sum(jnp.log(sig * jnp.sqrt(2 * jnp.pi)))
    lp = lp_pos + lp_flux + lp_bg
    return jnp.where(in_supp, lp, -jnp.inf)

def logprior(theta: jnp.ndarray) -> jnp.ndarray:
    return _logprior_one(theta) if theta.ndim == 1 else jax.vmap(_logprior_one)(theta)

def _loglik_one_old(theta: jnp.ndarray) -> jnp.ndarray:
    pos, flx, bg = _vec_to_params(theta)             # expect bg shape (2,)
    lam = forward_model(PROB.patch, pos, flx, bg)    # <- pass bg as JAX array
    return poisson_loglik(lam, PROB.patch.counts)

def loglikelihood(theta: jnp.ndarray) -> jnp.ndarray:
    pos, flx, bg = _vec_to_params(theta)          # bg is (2,) array
    ps = PROB.patch

    #print("theta dtype:", theta.dtype)
    #sys.exit()

    """    
    pos = pos.at[0, 0].set(7.51671448e+01)
    pos = pos.at[0, 1].set(44.745322)
    pos = pos.at[1, 0].set(72.748227)
    pos = pos.at[1, 1].set(103.06816)
    pos = pos.at[2, 0].set(96.8931122)
    pos = pos.at[2, 1].set(120.60628)
    flx = flx.at[0].set(3.28665308e-08)
    flx = flx.at[1].set(2.29986807e-09)
    flx = flx.at[2].set(2.17807106e-09)
    """
    lam = forward_model(ps.iso, ps.iem, ps.Fker, pos, flx, bg)
    #lam = forward_model(ps.iso, ps.iem, ps.Fker, pos, flx, bg)
    
    loglike = poisson_loglik(lam, ps.counts)
    #debug_poisson_terms(lam, ps.counts)
    print(pos)
    print(loglike)
    sys.exit()
    return loglike

