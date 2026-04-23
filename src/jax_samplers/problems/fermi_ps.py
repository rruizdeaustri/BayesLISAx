"""
Transdimensional Fermi-LAT point‑source model for your sampling framework (JAX).

Drop this file under your repository's `problems/` folder and refer to it as

    --problem problems.fermi_ps:FermiPointSourcesProblem

in your CLI, e.g.

    python -m jax_samplers.cli \
      --algo ns \
      --problem problems.fermi_ps:FermiPointSourcesProblem \
      --n-live 6s00 --tol 3 --num-inner-steps 16 \
      --problem-kwargs '{
          "iso_path":"/path/iso.fits",
          "iem_path":"/path/iem.fits",
          "exposure_path":"/path/exposure.fits",
          "counts_path":"/path/counts.fits",
          "psf_size":41, "psf_sigma":1.8,
          "flux_min":1e-11, "flux_max":1e-7,
          "K_max":30
      }'

This implements:
  * A forward model: positions & fluxes -> convolved counts map + backgrounds
  * A Poisson log-likelihood for the observed counts
  * A transdimensional prior over the number of sources K, their positions, and fluxes
  * RJ-style local proposals: birth, death, move, and flux-rescale (optional for RJ-MCMC)

If you are using **nested sampling** with your transdim API, only `sample_prior`
(and `log_prior` if your runner needs it) plus `loglik` are strictly necessary.
The RJ proposals are provided for RJ-MCMC / SMC tempering pipelines as well.

All arrays are JAX arrays. No Python control flow leaks into JIT hot paths.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.signal import convolve
from jax.scipy.special import gammaln, xlogy

#from .core.utils import JAX_FLOAT, as_jax, as_np

from jax_samplers.core.precision import resolve as _resolve_dtype
from jax_samplers.core.precision import as_dtype as _asdt

try:
    # Optional FITS i/o. If you prefer to pass numpy arrays directly, that's fine.
    from astropy.io import fits  # type: ignore
    _HAS_ASTROPY = True
except Exception:  # pragma: no cover
    _HAS_ASTROPY = False

Array = jnp.ndarray


# ───────────────────────────── Utilities ─────────────────────────────
#eps = jnp.asarray(1e-12, jnp.float64)
eps = jnp.asarray(1e-9, jnp.float32)

def _load_fits(path):
    return jnp.array(fits.open(path)[0].data.astype(jnp.float32))

def _normalize_psf(psf: Array) -> Array:
    s = jnp.sum(psf)
    return psf / jnp.maximum(s, 1e-32)


def gaussian_psf(size: int = 41, sigma: float = 1.8) -> Array:
    """Returns a square, normalized Gaussian PSF kernel of shape (size, size).
    `size` should be odd.
    """
    ax = jnp.arange(size) - (size // 2)
    X, Y = jnp.meshgrid(ax, ax, indexing="xy")
    g = jnp.exp(-(X**2 + Y**2) / (2.0 * sigma**2))
    return _normalize_psf(g)

def generate_king_psf(size: int = 40, sigma: float = 2, gamma: float = 30):
        x = jnp.arange(-size//2, size//2 + 1)
        y = jnp.arange(-size//2, size//2 + 1)
        X, Y = jnp.meshgrid(x, y)
        r2 = X**2 + Y**2
        psf = (1 / (2 * jnp.pi * sigma**2)) * (1 - 1/gamma) * (1 + (r2 / (2 * gamma * sigma**2)))**(-gamma)
        return psf / psf.sum()


def _conv2_same(img, ker):
    H,W = img.shape; Kh,Kw = ker.shape
    padH, padW = H+Kh-1, W+Kw-1
    Fimg = jnp.fft.rfftn(img, s=(padH, padW))
    Fker = jnp.fft.rfftn(ker, s=(padH, padW))
    full = jnp.fft.irfftn(Fimg * Fker, s=(padH, padW))
    sh, sw = (Kh-1)//2, (Kw-1)//2
    return full[sh:sh+H, sw:sw+W]

def prepare_psf_fft(psf, H, W):
    Kh, Kw = psf.shape
    padH, padW = H + Kh - 1, W + Kw - 1
    Fker = jnp.fft.rfftn(psf, s=(padH, padW))
    sh, sw = (Kh - 1) // 2, (Kw - 1) // 2
    return Fker, padH, padW, sh, sw

#@jax.jit
#def conv2_same_with_Fker(img, Fker, padH, padW, sh, sw):
#    Fimg = jnp.fft.rfftn(img, s=(padH, padW))
#    full = jnp.fft.irfftn(Fimg * Fker, s=(padH, padW))
#    return full[sh:sh + img.shape[0], sw:sw + img.shape[1]]


@jax.jit
def conv2_same_with_Fker(img: jnp.ndarray, Fker: jnp.ndarray) -> jnp.ndarray:
    # Fker has shape (padH, padW_rfft) where padW = 2*(padW_rfft-1)
    padH = Fker.shape[0]
    padW = 2 * (Fker.shape[1] - 1)

    Fimg = jnp.fft.rfftn(img, s=(padH, padW))
    full = jnp.fft.irfftn(Fimg * Fker, s=(padH, padW))

    H, W = img.shape
    sh = (padH - H) // 2
    sw = (padW - W) // 2
    return full[sh:sh + H, sw:sw + W]


# ───────────────────────────── Data holder ─────────────────────────────

@dataclass
class FermiPatch:
    """Maps needed by the forward model.

    All maps are 2D arrays of the same shape (H, W).
    - iso, iem: background templates in counts (already exposure-folded) or in rate; see `bg_scale`.
    - exposure: exposure map (can be all ones if backgrounds already in counts).
    - counts: observed counts (for likelihood).
    - psf: 2D kernel, normalized to sum=1.
    - bg_scale: tuple (a_iso, a_iem) scaling coefficients for backgrounds (optional, can be fixed or sampled).
    """
    iso: Array
    iem: Array
    exposure: Array
    counts: Array
    psf: Array
    bg_scale: Tuple[float, float] = (1.0, 1.0)

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.counts.shape)  # type: ignore


# ───────────────────────────── Model pieces ─────────────────────────────

@jax.jit
def rasterize_sources(positions: jnp.ndarray,
                           fluxes: jnp.ndarray,
                           template: jnp.ndarray) -> jnp.ndarray:
    H, W = template.shape
    y = jnp.clip(jnp.round(positions[:, 0]).astype(jnp.int32), 0, H - 1)
    x = jnp.clip(jnp.round(positions[:, 1]).astype(jnp.int32), 0, W - 1)
    canvas = jnp.zeros_like(template, dtype=jnp.float32)
    scale = jnp.asarray(1e11, jnp.float32)
    return canvas.at[(y, x)].add(fluxes.astype(jnp.float32) * scale)


@jax.jit
def forward_model(iso, iem, Fker, positions_xy, fluxes, bg_scale):
    H, W = iso.shape
    x = positions_xy[:, 0]
    y = positions_xy[:, 1]

    # in-bounds mask
    inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    w   = inb.astype(fluxes.dtype)

    # clip positions into valid range (so rasterizer never sees OOB)
    # subtract tiny epsilon so max index never hits W/H exactly
    eps = 1e-6
    x_safe = jnp.clip(x, 0.0, W - 1.0 - eps)
    y_safe = jnp.clip(y, 0.0, H - 1.0 - eps)

    # swap once: (x,y) -> (row=y, col=x)
    pos_rc = jnp.stack([y_safe, x_safe], axis=-1)

    # zero flux for OOB sources; shapes stay (K,)
    src_map     = rasterize_sources(pos_rc, fluxes * w, iso)
    src_blurred = conv2_same_with_Fker(src_map, Fker)

    a_iso, a_iem = bg_scale
    lam = src_blurred + a_iso * iso + a_iem * iem
    return lam



#@jax.jit
def forward_model_old(iso: jnp.ndarray,
                  iem: jnp.ndarray,
                  Fker: jnp.ndarray,       # precomputed PSF FFT
                  positions: jnp.ndarray,  # (K,2)
                  fluxes: jnp.ndarray,     # (K,)
                  bg_scale: jnp.ndarray    # (2,)
                  ) -> jnp.ndarray:

    positions_rc = positions_xy[:, ::-1]  # (row=y, col=x)
    
    #H, W = map(int, iso.shape)        # iso is your (H,W) array
    src_map = rasterize_sources(positions_rc, fluxes, iso)

    src_blurred = conv2_same_with_Fker(src_map, Fker)
    a_iso, a_iem = bg_scale
    lam = src_blurred + a_iso * iso + a_iem * iem
    eps = jnp.asarray(1e-9, jnp.float32)
    return jnp.clip(lam, a_min=eps)


@jax.jit
def poisson_loglik(expected: Array, observed: Array) -> Array:
    """Pixel-wise Poisson log-likelihood, summed over the map.
    log P(D|λ) = Σ [ D log λ − λ − log Γ(D+1) ]
    Constant terms (log Γ) can be kept for exactness or dropped for nested sampling
    if you only care about relative values.
    """
    # Using gammaln for numerical stability.

    D = observed
    lam = expected
    return jnp.sum(D * jnp.log(lam) - lam) #- gammaln(D + 1.0))


def debug_poisson_terms(lam: jnp.ndarray, k: jnp.ndarray):
    """
    expected = λ (model counts), observed = D (data counts)
    Returns a dict with the key Poisson pieces so you can see what's off.
    """
    lam = jnp.clip(lam, a_min=0.0)  # never negative for logs
    sum_k        = jnp.sum(k)
    sum_lam      = jnp.sum(lam)
    sum_kloglam  = jnp.sum(xlogy(k, lam))
    sum_logkfac  = jnp.sum(gammaln(k + 1.0))
    ll_noconst   = sum_kloglam - sum_lam
    ll_full      = ll_noconst - sum_logkfac
    n_lam_le_0   = jnp.sum(lam <= 0)
    n_bad        = jnp.sum((k > 0) & (lam <= 0))
    jax.debug.print(
        "sum_k={:.3f} sum_lam={:.3f} sum_kloglam={:.3f} sum_logk!={:.3f} "
        "ll_nc={:.3f} ll_full={:.3f} n_lam<=0={} n(D>0 & lam<=0)={}",
        sum_k, sum_lam, sum_kloglam, sum_logkfac, ll_noconst, ll_full,
        n_lam_le_0, n_bad
    )

# ───────────────────────────── Transdimensional prior ─────────────────────────────

@dataclass
class TDConfig:
    H: int
    W: int
    flux_min: float = 1e-11
    flux_max: float = 1e-7
    K_max: int = 50
    p_K_geom: float = 0.5  # geometric prior over K: P(K) ∝ (1-p) p^K, truncated at K_max
    allow_bg_fit: bool = False
    bg_scale_mu: Tuple[float, float] = (1.0, 1.0)
    bg_scale_sigma: Tuple[float, float] = (0.1, 0.1)


def sample_K(key: Array, cfg: TDConfig) -> Tuple[Array, int]:
    # Truncated geometric on K = 0..K_max
    key, sub = jr.split(key)
    u = jr.uniform(sub)
    p = cfg.p_K_geom
    # Inverse CDF of geometric truncated at K_max
    # P(K>=k) = p^k; sample by counting how many times we pass u>p^k.
    ks = jnp.arange(cfg.K_max + 1)
    cdf = 1.0 - p ** (ks + 1)  # CDF of shifted geometric; simple, monotone
    K = int(jnp.searchsorted(cdf, u, side="right"))
    K = int(jnp.minimum(K, cfg.K_max))
    return key, K


#def sample_positions(key: Array, K: int, H: int, W: int) -> Tuple[Array, Array]:
#    key, ky, kx = jr.split(key, 3)
#    ys = jr.uniform(ky, (K,), minval=0.0, maxval=float(H))
#    xs = jr.uniform(kx, (K,), minval=0.0, maxval=float(W))
#    return key, jnp.stack([ys, xs], axis=-1)

def sample_positions(key, K, H, W, dtype=jnp.float32):
    key, ky, kx = jr.split(key, 3)
    ys = jr.uniform(ky, (K,), minval=0.0, maxval=float(H), dtype=dtype)
    xs = jr.uniform(kx, (K,), minval=0.0, maxval=float(W), dtype=dtype)
    return key, jnp.stack([xs, ys], axis=-1).astype(dtype)

def sample_fluxes(key, K, fmin, fmax, dtype=jnp.float32):
    key, k = jr.split(key)
    u = jr.uniform(k, (K,), dtype=dtype)
    logf = jnp.log(dtype(fmin)) + u * (jnp.log(dtype(fmax)) - jnp.log(dtype(fmin)))
    return key, jnp.exp(logf).astype(dtype)  # or return logf if you prefer log-param

def log_prior_K(K: int, cfg: TDConfig) -> Array:
    p = cfg.p_K_geom
    # Truncated geometric normalization constant Z = 1 - p^{K_max+1}
    Z = 1.0 - p ** (cfg.K_max + 1)
    return jnp.log1p(-p) + K * jnp.log(p) - jnp.log(Z)


def log_prior_positions(positions: Array, H: int, W: int) -> Array:
    # Uniform in the rectangle
    area = float(H * W)
    K = positions.shape[0]
    return -K * jnp.log(area)


def log_prior_fluxes(fluxes: Array, fmin: float, fmax: float) -> Array:
    # log-uniform prior: p(f) ∝ 1/f on [fmin, fmax]
    K = fluxes.shape[0]
    Z = jnp.log(fmax) - jnp.log(fmin)
    return -K * Z - jnp.sum(jnp.log(jnp.clip(fluxes, fmin, fmax))) + 0.0 * jnp.sum(
        (fluxes >= fmin) & (fluxes <= fmax)
    )


def log_prior_bg_old(bg_scale: Tuple[float, float], mu: Tuple[float, float], sigma: Tuple[float, float]) -> Array:
    a_iso, a_iem = bg_scale
    m1, m2 = mu
    s1, s2 = sigma
    lp = -0.5 * ((a_iso - m1) / s1) ** 2 - jnp.log(s1 * jnp.sqrt(2 * jnp.pi))
    lp += -0.5 * ((a_iem - m2) / s2) ** 2 - jnp.log(s2 * jnp.sqrt(2 * jnp.pi))
    return lp


def log_prior_bg(bg_scale: Array,
                 mu: Tuple[float, float],
                 sigma: Tuple[float, float]) -> Array:
    a_iso, a_iem = bg_scale  # works with a JAX array
    m1, m2 = mu
    s1, s2 = sigma
    lp = -0.5 * ((a_iso - m1) / s1) ** 2 - jnp.log(s1 * jnp.sqrt(2 * jnp.pi))
    lp += -0.5 * ((a_iem - m2) / s2) ** 2 - jnp.log(s2 * jnp.sqrt(2 * jnp.pi))
    return lp


# ───────────────────────────── Problem class ─────────────────────────────

class FermiPointSourcesProblem:
    """Problem wrapper exposing the minimal API used by your samplers.

    Modes:
      - mode="fixed": use a fixed number of sources K_fixed (constant-dim parameters)
      - mode="transdim": infer K with a truncated geometric prior up to K_max

    Required problem-kwargs (either file paths OR direct arrays):
      - iso_path | iso
      - iem_path | iem
      - exposure_path | exposure
      - counts_path | counts
      - psf_size (int, odd) and psf_sigma (float) OR psf (2D array)

    Optional:
      - mode: "fixed" or "transdim" (default: "transdim")
      - K_fixed: int (required if mode="fixed")
      - flux_min, flux_max, K_max, p_K_geom
      - allow_bg_fit, bg_scale_mu, bg_scale_sigma
    """

    def __init__(self, **kwargs: Any) -> None:
        # Load maps (either from paths or directly from kwargs)

        # ── precision flag (default float32) ─────────────────────────
        #self.DT: jnp.dtype = _resolve_dtype(kwargs.get("dtype", "float32"))

        self.DT = _resolve_dtype(kwargs.get("dtype", "float64"))          # NS / state dtype
        self.MODEL_DT = _resolve_dtype(kwargs.get("model_dtype", self.DT))
        
        iso = kwargs.get("iso")
        iem = kwargs.get("iem")
        exposure = kwargs.get("exposure")
        counts = kwargs.get("counts")
        
        if iso is None:
            iso = _load_fits(kwargs["iso_path"])
        if iem is None:
            iem = _load_fits(kwargs["iem_path"])
        if exposure is None:
            exposure = _load_fits(kwargs["exposure_path"])
        if counts is None:
            counts = _load_fits(kwargs["counts_path"])

        # Cast + scale in chosen dtype
        iso     = _asdt(iso,     self.DT) * _asdt(1e-9, self.DT)
        iem     = _asdt(iem,     self.DT) * _asdt(1e-9, self.DT)
        exposure= _asdt(exposure,self.DT)
        counts  = _asdt(counts,  self.DT)
            
        H, W = counts.shape

        psf = kwargs.get("psf")
        if psf is None:
             psf = generate_king_psf(
                 size=int(kwargs.get("psf_size", 40)),
                 sigma=float(kwargs.get("psf_sigma", 2)),
                 gamma=float(kwargs.get("psf_gamma", 30)),
             ).astype("float32")
            
            #psf = gaussian_psf(
            #    size=int(kwargs.get("psf_size", 40)),
            #    sigma=float(kwargs.get("psf_sigma", 2)),
            #).astype("float32")

        psf = _asdt(psf, self.DT)

            
        self.patch = FermiPatch(
            iso=jnp.asarray(iso),
            iem=jnp.asarray(iem),
            exposure=jnp.asarray(exposure),
            counts=jnp.asarray(counts),
            psf=jnp.asarray(psf),
        )

        # Precompute FFT kernel for PSF convolution
        self.patch.Fker, self.patch.padH, self.patch.padW, self.patch.sh, self.patch.sw = prepare_psf_fft(self.patch.psf, *self.patch.shape)
        
        # Mode & dimensionality
        self.mode: str = str(kwargs.get("mode", "transdim"))
        if self.mode not in ("fixed", "transdim"):
            raise ValueError(f"mode must be 'fixed' or 'transdim', got {self.mode}")
        self.K_fixed: int | None = int(kwargs["K_fixed"]) if (self.mode == "fixed") else None

        self.cfg = TDConfig(
            H=H,
            W=W,
            flux_min=float(kwargs.get("flux_min", 1e-11)),
            flux_max=float(kwargs.get("flux_max", 1e-7)),
            K_max=int(kwargs.get("K_max", 50)),
            p_K_geom=float(kwargs.get("p_K_geom", 0.5)),
            allow_bg_fit=bool(kwargs.get("allow_bg_fit", False)),
            bg_scale_mu=tuple(kwargs.get("bg_scale_mu", (1.0, 1.0))),
            bg_scale_sigma=tuple(kwargs.get("bg_scale_sigma", (0.1, 0.1))),
        )

        # Sanity: if fixed, ensure K_fixed ≤ K_max so packing works
        if self.mode == "fixed":
            if self.K_fixed is None:
                raise ValueError("K_fixed must be provided when mode='fixed'")
            if not (0 <= self.K_fixed <= self.cfg.K_max):
                raise ValueError(f"K_fixed={self.K_fixed} must be in [0, {self.cfg.K_max}]")

    # —— API expected by samplers ——
    def sample_prior(self, key: Array) -> Dict[str, Array]:
        DT = self.DT
        if self.mode == "fixed":
            K = int(self.K_fixed)  # type: ignore[arg-type]
        else:
            key, K = sample_K(key, self.cfg)

        key, pos = sample_positions(key, K, self.cfg.H, self.cfg.W, dtype=DT)
        key, flx = sample_fluxes(key, K, self.cfg.flux_min, self.cfg.flux_max, dtype=DT)

        if self.cfg.allow_bg_fit:
            key, kb1, kb2 = jr.split(key, 3)
            a_iso = jr.normal(kb1, (), dtype=DT) * DT(self.cfg.bg_scale_sigma[0]) + DT(self.cfg.bg_scale_mu[0])
            a_iem = jr.normal(kb2, (), dtype=DT) * DT(self.cfg.bg_scale_sigma[1]) + DT(self.cfg.bg_scale_mu[1])
            bg = jnp.stack([a_iso, a_iem]).astype(DT)
        else:
            bg = _asdt([1.0, 1.0], DT)

        return {
            "K": jnp.asarray(K),                 # K as scalar (int), OK
            "positions": pos.astype(DT),         # (K,2) DT
            "fluxes": flx.astype(DT),            # (K,)  DT   (adapter can log if needed)
            "bg": bg.astype(DT),                 # (2,)  DT
        }
        
    def log_prior_old(self, theta: Dict[str, Array]) -> Array:
        K = int(theta["K"]) if not isinstance(theta["K"], (int,)) else theta["K"]
        # K prior: delta(K=K_fixed) in fixed mode (constant -> 0 contribution); geometric otherwise
        if self.mode == "fixed":
            lp = 0.0
        else:
            lp = log_prior_K(K, self.cfg)
        lp += log_prior_positions(theta["positions"], self.cfg.H, self.cfg.W)
        lp += log_prior_fluxes(theta["fluxes"], self.cfg.flux_min, self.cfg.flux_max)
        if self.cfg.allow_bg_fit:
            lp += log_prior_bg(tuple(theta["bg"].tolist()), self.cfg.bg_scale_mu, self.cfg.bg_scale_sigma)
        return lp

    def log_prior(self, theta: Dict[str, Array]) -> Array:
        # Avoid int(theta["K"]) inside traced code. For fixed mode it's constant anyway.
        if self.mode == "fixed":
            lp = 0.0
        else:
            # If you keep a transdim path here later, make log_prior_K accept a JAX scalar or
            # compute it outside jit. For now this is fine if log_prior isn't jitted.
            lp = log_prior_K(int(theta["K"]), self.cfg)

        lp += log_prior_positions(theta["positions"], self.cfg.H, self.cfg.W)
        lp += log_prior_fluxes(theta["fluxes"], self.cfg.flux_min, self.cfg.flux_max)

        if self.cfg.allow_bg_fit:
            bg = jnp.asarray(theta.get("bg", jnp.array([1.0, 1.0], dtype=theta["positions"].dtype)))
            lp += log_prior_bg(bg, self.cfg.bg_scale_mu, self.cfg.bg_scale_sigma)

        return lp    

    def loglik_old(self, theta: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        # 1) Use a static (Python) int for K in fixed runs
        #    Put K_fixed on self (or self.patch) at construction time as a Python int.
        K = int(self.patch.K_fixed)  # <- Python int known at trace time

        # 2) Slice with static K (OK under jit/vmap)
        pos = theta["positions"][:K]   # (K, 2)
        flx = theta["fluxes"][:K]      # (K,)

        # 3) Keep bg as a JAX array; no .tolist()/tuple()
        bg = jnp.asarray(theta.get("bg", jnp.array([1.0, 1.0], dtype=pos.dtype)))  # (2,)

        # 4) Call forward_model with the correct signature (as you use elsewhere)
        lam = forward_model(self.patch.iso, self.patch.iem, self.patch.Fker, pos, flx, bg)
        
        loglike = poisson_loglik(lam, self.patch.counts)
        print(lam, self.patch.counts)
        sys.exit()
        return loglike
    
        #return poisson_loglik(lam, self.patch.counts)

    def loglik(self, theta: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        # Fixed-K path: keep K as a Python int to make slicing static/JIT-friendly.
        if self.mode != "fixed":
            # if you later want transdim here, use a masking strategy; for now:
            raise RuntimeError("loglik currently implemented for mode='fixed' only")
        K = self.K_fixed  # set in __init__, Python int

        pos = theta["positions"][:K].astype(self.DT)
        flx = theta["fluxes"][:K].astype(self.DT)
        bg  = jnp.asarray(theta.get("bg", jnp.array([1.0, 1.0], dtype=pos.dtype))).astype(self.DT)

        
        print(pos, flx)

        lam = forward_model(self.patch.iso, self.patch.iem, self.patch.Fker, pos, flx, bg)
        #print(lam)
    
        loglike = poisson_loglik(lam, self.patch.counts)
        print('loglike', loglike)
        #sys.exit()
        return loglike
        #return poisson_loglik(lam, self.patch.counts)

    
    def loglik_old(self, theta: Dict[str, Array]) -> Array:
        K = int(theta["K"]) if not isinstance(theta["K"], (int,)) else theta["K"]
        pos = theta["positions"][:K]
        flx = theta["fluxes"][:K]
        bg = tuple(theta.get("bg", jnp.array([1.0, 1.0])).tolist())
        lam = forward_model(self.patch, pos, flx, self.patch.counts)
        return poisson_loglik(lam, self.patch.counts)

    # Optional: proposals for RJMCMC / SMC
    def rj_propose_birth(self, key: Array, theta: Dict[str, Array]) -> Tuple[Array, Dict[str, Array], Array]:
        if self.mode == "fixed":
            return key, theta, -jnp.inf
        K = int(theta["K"]) if not isinstance(theta["K"], (int,)) else theta["K"]
        if K >= self.cfg.K_max:
            return key, theta, -jnp.inf
        key, pos_new = sample_positions(key, 1, self.cfg.H, self.cfg.W)
        key, flx_new = sample_fluxes(key, 1, self.cfg.flux_min, self.cfg.flux_max)
        new_theta = {
            "K": jnp.array(K + 1),
            "positions": jnp.concatenate([theta["positions"], pos_new], axis=0),
            "fluxes": jnp.concatenate([theta["fluxes"], flx_new], axis=0),
            "bg": theta.get("bg", jnp.array([1.0, 1.0])),
        }
        q = 0.0
        return key, new_theta, q

    def rj_propose_death(self, key: Array, theta: Dict[str, Array]) -> Tuple[Array, Dict[str, Array], Array]:
        if self.mode == "fixed":
            return key, theta, -jnp.inf
        K = int(theta["K"]) if not isinstance(theta["K"], (int,)) else theta["K"]
        if K <= 0:
            return key, theta, -jnp.inf
        key, ksub = jr.split(key)
        idx = jr.randint(ksub, (), 0, K)
        keep = jnp.arange(K) != idx
        new_theta = {
            "K": jnp.array(K - 1),
            "positions": theta["positions"][keep],
            "fluxes": theta["fluxes"][keep],
            "bg": theta.get("bg", jnp.array([1.0, 1.0])),
        }
        q = 0.0
        return key, new_theta, q

    def rj_propose_move(self, key: Array, theta: Dict[str, Array], pos_sigma: float = 0.75) -> Tuple[Array, Dict[str, Array], Array]:
        K = int(theta["K"]) if not isinstance(theta["K"], (int,)) else theta["K"]
        if K == 0:
            return key, theta, 0.0
        key, kidx, kstep = jr.split(key, 3)
        idx = jr.randint(kidx, (), 0, K)
        step = jr.normal(kstep, (2,)) * pos_sigma
        pos = theta["positions"].at[idx].add(step)
        pos = pos.at[:, 0].clip(0.0, self.cfg.H - 1.0)
        pos = pos.at[:, 1].clip(0.0, self.cfg.W - 1.0)
        new_theta = {**theta, "positions": pos}
        return key, new_theta, 0.0

    def rj_propose_flux(self, key: Array, theta: Dict[str, Array], log_step: float = 0.3) -> Tuple[Array, Dict[str, Array], Array]:
        K = int(theta["K"]) if not isinstance(theta["K"], (int,)) else theta["K"]
        if K == 0:
            return key, theta, 0.0
        key, kidx, kstep = jr.split(key, 3)
        idx = jr.randint(kidx, (), 0, K)
        # Log-space random-walk that preserves log-uniform prior (symmetric in log)
        delta = jr.normal(kstep) * log_step
        flx = theta["fluxes"].at[idx].multiply(jnp.exp(delta))
        flx = jnp.clip(flx, self.cfg.flux_min, self.cfg.flux_max)
        new_theta = {**theta, "fluxes": flx}
        return key, new_theta, 0.0

    # —— Compatibility bridge: pack/unpack to fixed arrays of size K_max ——
    def pack(self, theta: Dict[str, Array]) -> Tuple[Array, Array]:
        """Pack (K, positions[K,2], fluxes[K]) into fixed-size (vec, mask).
        vec layout = [positions_pad[K_max,2], fluxes_pad[K_max], bg[2]] flattened.
        mask shape = (K_max,) with 1 for active slots, 0 for padding.
        """
        K = int(theta["K"]) if not isinstance(theta["K"], (int,)) else theta["K"]
        Kmax = self.cfg.K_max
        pos = theta["positions"]
        flx = theta["fluxes"]
        bg  = theta.get("bg", jnp.array([1.0, 1.0]))
        pos_pad = jnp.zeros((Kmax, 2))
        flx_pad = jnp.zeros((Kmax,))
        pos_pad = pos_pad.at[:K].set(pos)
        flx_pad = flx_pad.at[:K].set(flx)
        mask = jnp.zeros((Kmax,))
        mask = mask.at[:K].set(1.0)
        vec = jnp.concatenate([pos_pad.reshape(-1), flx_pad, bg], axis=0)
        return vec, mask

    def unpack(self, vec: Array, mask: Array) -> Dict[str, Array]:
        Kmax = self.cfg.K_max
        npos = Kmax * 2
        pos_flat = vec[:npos]
        flx = vec[npos:npos + Kmax]
        bg  = vec[npos + Kmax:npos + Kmax + 2]
        K = int(jnp.sum(mask))
        pos = pos_flat.reshape((Kmax, 2))[:K]
        flx = flx[:K]
        return {"K": jnp.array(K), "positions": pos, "fluxes": flx, "bg": bg}

# ───────────────────────────── Transdimensional "Family" wrapper for --problem transdim

class FermiPointSourcesFamily:
#class FermiPointSourcesProblem:
    """Family wrapper compatible with your CLI mode `--problem transdim`.

    Use this when your driver expects a *family* that exposes K-varying
    priors/likelihoods, while the driver itself governs birth/death moves.

    Example CLI:
      python -m sbi_samplers.cli \
        --algo ns \
        --problem transdim \
        --family problems.fermi_ps:FermiPointSourcesFamily \
        --family-kwargs '{
          "iso_path": "/path/iso.fits",
          "iem_path": "/path/iem.fits",
          "exposure_path": "/path/exposure.fits",
          "counts_path": "/path/counts.fits",
          "psf_size": 41,
          "psf_sigma": 1.8,
          "flux_min": 1e-11,
          "flux_max": 1e-7,
          "K_max": 30,
          "p_K_geom": 0.5,
          "allow_bg_fit": false
        }'

    Exposed attributes/methods expected by the transdim driver:
      - max_K (int): upper bound on K
      - sample_params(key, K) -> theta dict with keys {K, positions, fluxes, bg}
      - log_prior(K, theta) -> scalar
      - loglik(theta) -> scalar

    Notes:
      * If your transdim driver handles the prior over K itself, it will *not* call
        our geometric K prior — that's fine. We still expose `p_K_geom` in case
        your driver wants to query it or re-use it.
    """

    def __init__(self, **kwargs: Any) -> None:
        # We reuse the same loading logic as the Problem class
        #self.problem = FermiPointSourcesProblem(**{**kwargs, "mode": "transdim"})
        self.problem = FermiPointSourcesProblem(**{**kwargs, "mode": "transdim"})
        self.max_K: int = self.problem.cfg.K_max
        self.p_K_geom: float = self.problem.cfg.p_K_geom

    # --- API ---
    @property
    def max_K(self) -> int:  # type: ignore[no-redef]
        return self._max_K

    @max_K.setter
    def max_K(self, value: int) -> None:
        self._max_K = int(value)

    def sample_params(self, key: Array, K: int) -> Dict[str, Array]:
        # Sample exactly K sources (driver chooses K)
        key, pos = sample_positions(key, K, self.problem.cfg.H, self.problem.cfg.W)
        key, flx = sample_fluxes(key, K, self.problem.cfg.flux_min, self.problem.cfg.flux_max)
        if self.problem.cfg.allow_bg_fit:
            key, kb1, kb2 = jr.split(key, 3)
            a_iso = jr.normal(kb1) * self.problem.cfg.bg_scale_sigma[0] + self.problem.cfg.bg_scale_mu[0]
            a_iem = jr.normal(kb2) * self.problem.cfg.bg_scale_sigma[1] + self.problem.cfg.bg_scale_mu[1]
            bg = jnp.array([a_iso, a_iem])
        else:
            bg = jnp.array([1.0, 1.0])
        return {"K": jnp.array(K), "positions": pos, "fluxes": flx, "bg": bg}

    def log_prior(self, K: int, theta: Dict[str, Array]) -> Array:
        # Prior over parameters conditioned on K (driver may add its own prior over K)
        lp = log_prior_positions(theta["positions"], self.problem.cfg.H, self.problem.cfg.W)
        lp += log_prior_fluxes(theta["fluxes"], self.problem.cfg.flux_min, self.problem.cfg.flux_max)
        if self.problem.cfg.allow_bg_fit:
            lp += log_prior_bg(tuple(theta["bg"].tolist()), self.problem.cfg.bg_scale_mu, self.problem.cfg.bg_scale_sigma)
        return lp

    def loglik(self, theta: Dict[str, Array]) -> Array:
        return self.problem.loglik(theta)


# ───────────────────────────── Convenience helpers ─────────────────────────────

def make_problem_from_paths(
    iso_path: str,
    iem_path: str,
    exposure_path: str,
    counts_path: str,
    psf_size: int = 41,
    psf_sigma: float = 1.8,
    **kwargs: Any,
) -> FermiPointSourcesProblem:
    """Tiny helper if you prefer constructing from file paths programmatically."""
    return FermiPointSourcesProblem(
        iso_path=iso_path,
        iem_path=iem_path,
        exposure_path=exposure_path,
        counts_path=counts_path,
        psf_size=psf_size,
        psf_sigma=psf_sigma,
        **kwargs,
    )
