# lisa_ongb_problem.py  -- narrow-band LISA GB backend (GBJAX, amp-profiled)

from __future__ import annotations
import os, json, sys
from typing import List, Tuple
from functools import partial
from typing import Dict, Tuple, Any
import math

import h5py
import numpy as np
from numpy.fft import rfft, rfftfreq

# -------------------------------------------------------------------
# Precision must be set BEFORE importing jax
# -------------------------------------------------------------------
from ..core.precision import DTYPE as _DTYPE, CDTYPE as _CDTYPE, EPS as _EPS
#from ..core.precision import jnp_real as _jnp_real, jnp_cplx as _jnp_cplx

#from jax import config as jax_config
#jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.random as jr
from jax_samplers.core.problem import Problem

from jaxlisa.tdi import xyz_to_aet
from jaxlisa.noise import AnalyticNoiseModel, NoisePreset
from jaxlisa.tdi_transfer import TDIChannel, TDIGeneration

from gbjax import (
    GBJAXConfig,
    GBJAXParameters,
    make_simulator,
    estimate_n_f_bins,
)

# Use global precision policy
DTYPE  = _DTYPE
CDTYPE = _CDTYPE

# Pick a safe epsilon for the chosen dtype
# (1e-300 underflows to 0 in float32; use the dtype’s tiny instead)
EPS_LIKE = jnp.finfo(DTYPE).tiny
EPS_LIKE = jnp.asarray(EPS_LIKE, dtype=DTYPE)

# Fix types for this backend (64-bit)
#DTYPE  = jnp.float64
#CDTYPE = jnp.complex128
#EPS_LIKE = jnp.asarray(1e-300, DTYPE)

EPS = EPS_LIKE

def jnp_real(x): return jnp.asarray(x, DTYPE).real
def jnp_cplx(x): return jnp.asarray(x, CDTYPE)

Array  = jax.Array
Bounds = List[Tuple[float, float]]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def as_1d(x):
    """Flatten to 1D NumPy array (host-side)."""
    a = np.asarray(x)
    return a.reshape(-1)


def load_single_vgb_or_fail(f, cat_idx: int):
    """
    Return (t, X, Y, Z) for exactly one noiseless VGB source.

    Supports:
      - Case A: separate 2D per-channel datasets:
          sky/vgb/tdi/X  shape (N_src, N_time)
          sky/vgb/tdi/Y
          sky/vgb/tdi/Z
          sky/vgb/tdi/t  shape (N_time,)
      - Case B: compound per-source dataset:
          sky/vgb/tdi  shape (N_src,), fields include 't','X','Y','Z'
    """
    # Case A
    if "sky/vgb/tdi/X" in f and "sky/vgb/tdi/Y" in f and "sky/vgb/tdi/Z" in f:
        Xds, Yds, Zds = f["sky/vgb/tdi/X"], f["sky/vgb/tdi/Y"], f["sky/vgb/tdi/Z"]
        t = np.asarray(f["sky/vgb/tdi/t"][:]).reshape(-1)
        X = np.asarray(Xds[cat_idx, :]).reshape(-1)
        Y = np.asarray(Yds[cat_idx, :]).reshape(-1)
        Z = np.asarray(Zds[cat_idx, :]).reshape(-1)
        return t, X, Y, Z

    # Case B
    if "sky/vgb/tdi" in f:
        TDI = f["sky/vgb/tdi"][()]
        t = as_1d(TDI["t"])
        X = as_1d(TDI["X"])
        Y = as_1d(TDI["Y"])
        Z = as_1d(TDI["Z"])
        return t, X, Y, Z

    raise ValueError("No per-source noiseless VGB time series in this file.")

def read_xyz_compound(ds):
    rec = ds[()]

    # (i) time-series compound: shape (Nt,) OR (Nt,1)
    if hasattr(rec, "dtype") and rec.dtype.fields:
        t_field = rec["t"]
        # accept (Nt,) or (Nt,1)
        if getattr(t_field, "ndim", 0) in (1, 2):
            if t_field.shape == rec.shape or (t_field.ndim == 2 and t_field.shape[-1] == 1):
                t = np.asarray(rec["t"]).reshape(-1).astype(np.float64)
                X = np.asarray(rec["X"]).reshape(-1).astype(np.float64)
                Y = np.asarray(rec["Y"]).reshape(-1).astype(np.float64)
                Z = np.asarray(rec["Z"]).reshape(-1).astype(np.float64)
                return t, X, Y, Z

    # (ii) scalar compound: rec.shape == ()
    if getattr(rec, "shape", ()) == ():
        t = as_1d(rec["t"]).astype(np.float64)
        X = as_1d(rec["X"]).astype(np.float64)
        Y = as_1d(rec["Y"]).astype(np.float64)
        Z = as_1d(rec["Z"]).astype(np.float64)
        return t, X, Y, Z

    # (iii) per-source compound: shape (Nsrc,), each element has vector fields
    if getattr(rec, "ndim", 0) == 1 and rec.shape[0] > 0:
        t0 = rec[0]["t"]
        if np.ndim(t0) > 0 and not np.isscalar(t0):
            t = as_1d(t0).astype(np.float64)
            X = np.zeros_like(t, dtype=np.float64)
            Y = np.zeros_like(t, dtype=np.float64)
            Z = np.zeros_like(t, dtype=np.float64)
            for i in range(rec.shape[0]):
                X += as_1d(rec[i]["X"]).astype(np.float64)
                Y += as_1d(rec[i]["Y"]).astype(np.float64)
                Z += as_1d(rec[i]["Z"]).astype(np.float64)
            return t, X, Y, Z

    raise ValueError(f"Unsupported compound layout for dataset {ds.name}: shape={getattr(rec,'shape',None)}")


def read_xyz(f, base):
    """
    Read XYZ(+t) from either:
      - group datasets base/t, base/X, base/Y, base/Z
      - or compound dataset at `base`
    """
    # group layout
    if f"{base}/X" in f and f"{base}/Y" in f and f"{base}/Z" in f:
        t = as_1d(f[f"{base}/t"][:]) if f"{base}/t" in f else None
        X = as_1d(f[f"{base}/X"][:]).astype(np.float64)
        Y = as_1d(f[f"{base}/Y"][:]).astype(np.float64)
        Z = as_1d(f[f"{base}/Z"][:]).astype(np.float64)
        return t, X, Y, Z

    # compound layout
    if base in f:
        return read_xyz_compound(f[base])

    raise ValueError(f"Could not find {base} as group or compound dataset.")

def load_instrument_noise_xyz(f, subtract_vgb=True):
    """
    Build instrument noise:
      n = obs - (dgb + igb + mbhb [+ vgb if subtract_vgb])

    Returns (t, nX, nY, nZ).
    """
    # IMPORTANT: in your file obs/tdi is compound => check "obs/tdi"
    if "obs/tdi" not in f:
        raise ValueError("This H5 does not contain obs/tdi; cannot build instrument noise.")

    t, X, Y, Z = read_xyz(f, "obs/tdi")
    if t is None:
        raise ValueError("obs/tdi has no time array 't' (unexpected for Sangria).")

    # subtract other sky components
    _, dX, dY, dZ = read_xyz(f, "sky/dgb/tdi")
    _, iX, iY, iZ = read_xyz(f, "sky/igb/tdi")
    _, mX, mY, mZ = read_xyz(f, "sky/mbhb/tdi")

    X = X - (dX + iX + mX)
    Y = Y - (dY + iY + mY)
    Z = Z - (dZ + iZ + mZ)

    if subtract_vgb:
        # subtract *total* VGB sky if available
        # - if sky/vgb/tdi/X exists and is (Nsrc,Nt): sum(axis=0) handled by read_xyz_compound? no, that's group layout.
        #   So read_xyz() will return t and arrays; if group layout is per-source you must sum manually here:
        if "sky/vgb/tdi/X" in f:
            t_v = as_1d(f["sky/vgb/tdi/t"][:])
            vX = as_1d(f["sky/vgb/tdi/X"][:].sum(axis=0)).astype(np.float64)
            vY = as_1d(f["sky/vgb/tdi/Y"][:].sum(axis=0)).astype(np.float64)
            vZ = as_1d(f["sky/vgb/tdi/Z"][:].sum(axis=0)).astype(np.float64)
        else:
            # compound (scalar or per-source): our reader sums per-source automatically
            t_v, vX, vY, vZ = read_xyz(f, "sky/vgb/tdi")

        if t_v.size != t.size:
            raise ValueError("Time grids differ between obs and vgb; cannot subtract safely.")

        X = X - vX
        Y = Y - vY
        Z = Z - vZ

    return t, X, Y, Z


def xyz_to_AE_freq(X, Y, Z, dt):
    """
    X,Y,Z(t) -> one-sided rFFT × dt -> A,E in frequency (complex CDTYPE).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    Xf = rfft(X) * dt
    Yf = rfft(Y) * dt
    Zf = rfft(Z) * dt
    Af = (Zf - Xf) / np.sqrt(2.0)
    Ef = (Xf - 2.0 * Yf + Zf) / np.sqrt(6.0)
    return jnp_cplx(Af), jnp_cplx(Ef)


def load_theta(h5path, idx, flip_phi0=True) -> GBJAXParameters:
    """
    Load catalogue parameters for one GB and return GBJAXParameters.
    """
    with h5py.File(h5path, "r") as f:
        row = f["sky/vgb/cat"][()][idx]

    def sc(v):
        a = np.asarray(v)
        return a.ravel()[0].item()

    phi0 = sc(row["InitialPhase"])
    if flip_phi0:
        phi0 = -phi0

    return GBJAXParameters(
        amp  = jnp.asarray([sc(row["Amplitude"])], DTYPE),
        f0   = jnp.asarray([sc(row["Frequency"])], DTYPE),
        fdot = jnp.asarray([sc(row["FrequencyDerivative"])], DTYPE),
        fddot= jnp.zeros((1,), DTYPE),
        phi0 = jnp.asarray([phi0], DTYPE),
        iota = jnp.asarray([sc(row["Inclination"])], DTYPE),
        psi  = jnp.asarray([sc(row["Polarization"])], DTYPE),
        lam  = jnp.asarray([sc(row["EclipticLongitude"])], DTYPE),
        beta = jnp.asarray([sc(row["EclipticLatitude"])], DTYPE),
    )


def psd_AE(f_band, preset, gen):
    """
    Return one-sided PSD arrays (SA, SE) on f_band in DTYPE.
    """
    f_band = jnp.asarray(f_band, DTYPE)

    nmA = AnalyticNoiseModel.from_preset(preset, TDIChannel.A, gen)
    SA  = jnp.asarray(nmA.psd(f_band))
    try:
        nmE = AnalyticNoiseModel.from_preset(preset, TDIChannel.E, gen)
        SE  = jnp.asarray(nmE.psd(f_band))
    except Exception:
        SE = SA.copy()

    # Avoid zero PSD at DC
    if SA.size and SA[0] == 0:
        SA = SA.at[0].set(jnp.inf)
    if SE.size and SE[0] == 0:
        SE = SE.at[0].set(jnp.inf)

    return jnp_real(SA), jnp_real(SE)

def grids_compatible(t1, t2):
    t1 = np.asarray(t1).reshape(-1)
    t2 = np.asarray(t2).reshape(-1)
    if t1.size != t2.size:
        return False
    dt1 = float(np.median(np.diff(t1)))
    dt2 = float(np.median(np.diff(t2)))
    if not np.isclose(dt1, dt2, rtol=0.0, atol=1e-9):
        return False
    # allow rounding at large t; require starts within half a sample
    if abs(float(t1[0]) - float(t2[0])) > 0.5 * dt1:
        return False
    return True

def delta_t_align(hA, hE, dA, dE, SA, SE, f_band, df, dt_range=200.0, dt_steps=801):
    """
    Small bulk Δt alignment to kill a linear phase ramp between data and template.

    Returns:
      hA_aligned, hE_aligned, dt_star, coh_max
    """
    f_band = jnp.asarray(f_band, DTYPE)
    SA = jnp.asarray(SA, DTYPE)
    SE = jnp.asarray(SE, DTYPE)
    hA = jnp.asarray(hA, CDTYPE)
    hE = jnp.asarray(hE, CDTYPE)
    dA = jnp.asarray(dA, CDTYPE)
    dE = jnp.asarray(dE, CDTYPE)

    grid = jnp.linspace(-dt_range, dt_range, dt_steps)

    def coh(dtau):
        ph = jnp.exp(-2j * jnp.pi * f_band * dtau)
        num = 4 * df * (
            jnp.vdot(hA * ph, dA / (SA + EPS)) +
            jnp.vdot(hE * ph, dE / (SE + EPS))
        )
        den = 4 * df * (
            jnp.vdot(hA * ph, hA * ph / (SA + EPS)) +
            jnp.vdot(hE * ph, hE * ph / (SE + EPS))
        )
        DD = 4 * df * (
            jnp.vdot(dA, dA / (SA + EPS)) +
            jnp.vdot(dE, dE / (SE + EPS))
        )
        return (jnp.abs(num) ** 2) / (den.real * DD.real + 1e-300)

    vals = jax.vmap(coh)(grid)
    kmax = jnp.argmax(vals)
    dt_star = grid[kmax]
    coh_max = vals[kmax]
    phase   = jnp.exp(-2j * jnp.pi * f_band * dt_star)
    return hA * phase, hE * phase, dt_star, coh_max

def _to_float_pair(v, name: str) -> Tuple[float, float]:
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        raise ValueError(f"Prior bound for '{name}' must be a 2-element list/tuple, got: {v}")
    a, b = float(v[0]), float(v[1])
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"Prior bound for '{name}' must be finite, got: ({a}, {b})")
    if not (a < b):
        raise ValueError(f"Prior bound for '{name}' must satisfy a < b, got: ({a}, {b})")
    return a, b


def build_merged_box(
    *,
    marg_ap: bool,
    f0_true: float,
    df: float,
    f0_hard_min: float = 1e-4,
    f0_hard_max: float = 5e-2,
    prior_box_in: Dict[str, Any] | None = None,
    f_band_min: float | None = None,   # analysis/simulator band lower edge
    f_band_max: float | None = None,   # analysis/simulator band upper edge
) -> Dict[str, Tuple[float, float]]:
    """
    Build final prior box from defaults + user overrides + optional band intersection.
    """

    # --- defaults by mode ---
    if marg_ap:
        # 6D: (f0, fdot, iota, psi, lam, beta)
        box = {
            "f0":   (max(f0_hard_min, f0_true - 50.0 * df), min(f0_hard_max, f0_true + 50.0 * df)),
            "fdot": (-1.0e-7, 1.0e-7),
            "iota": (-math.pi/2.0, math.pi/2.0),
            "psi":  (-math.pi, math.pi),
            "lam":  (-math.pi, math.pi),
            "beta": (-math.pi/2.0, math.pi/2.0),
        }
    else:
        # 8D explicit: (lnA, f0, fdot, phi0, iota, psi, lam, beta)
        box = {
            "lnA":  (-35.0, -5.0),
            "f0":   (max(f0_hard_min, f0_true - 50.0 * df), min(f0_hard_max, f0_true + 50.0 * df)),
            "fdot": (-1.0e-7, 1.0e-7),
            "phi0": (0.0, 2.0 * math.pi),
            "iota": (-math.pi/2.0, math.pi/2.0),
            "psi":  (-math.pi, math.pi),
            "lam":  (-math.pi, math.pi),
            "beta": (-math.pi/2.0, math.pi/2.0),
        }

    # --- apply user overrides only for active keys ---
    prior_box_in = prior_box_in or {}
    for k, v in prior_box_in.items():
        if k in box:
            box[k] = _to_float_pair(v, k)

    # --- validate all bounds ---
    for k, v in box.items():
        box[k] = _to_float_pair(v, k)

    # --- enforce consistency of f0 prior with analysis band ---
    if f_band_min is not None and f_band_max is not None:
        f_band_min = float(f_band_min)
        f_band_max = float(f_band_max)
        if not (math.isfinite(f_band_min) and math.isfinite(f_band_max) and f_band_min < f_band_max):
            raise ValueError(f"Invalid analysis band: [{f_band_min}, {f_band_max}]")

        a, b = box["f0"]
        a2, b2 = max(a, f_band_min), min(b, f_band_max)
        if not (a2 < b2):
            raise ValueError(
                f"f0 prior [{a:.9e}, {b:.9e}] does not overlap "
                f"analysis band [{f_band_min:.9e}, {f_band_max:.9e}]"
            )
        box["f0"] = (a2, b2)

    return box


def filter_gaussian_priors_for_mode(
    gaussian_priors: Dict[str, Any] | None,
    marg_ap: bool,
) -> Dict[str, Tuple[float, float]]:
    """
    Keep only Gaussian priors for active parameters and validate (mu, sigma).
    """
    gp_in = gaussian_priors or {}

    active = {"f0", "fdot", "iota", "psi", "lam", "beta"} if marg_ap else \
             {"lnA", "f0", "fdot", "phi0", "iota", "psi", "lam", "beta"}

    gp_out: Dict[str, Tuple[float, float]] = {}
    for k, v in gp_in.items():
        if k not in active:
            continue
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            raise ValueError(f"Gaussian prior for '{k}' must be [mu, sigma], got {v}")
        mu, sig = float(v[0]), float(v[1])
        if not (math.isfinite(mu) and math.isfinite(sig) and sig > 0.0):
            raise ValueError(f"Invalid Gaussian prior for '{k}': mu={mu}, sigma={sig}")
        gp_out[k] = (mu, sig)

    return gp_out


# -------------------------------------------------------------------
# Context: holds data, PSD, band, narrow-band simulator, likelihoods
# -------------------------------------------------------------------

class _Ctx:
    """
    Fixed data, PSDs, band and a narrow-band GBJAX simulator.

    Exposes:
      - loglike_explicit(theta8) for 8D (lnA, f0, fdot, phi0, iota, psi, lam, beta)
      - loglike_marg_ap(theta6) for 6D (f0, fdot, iota, psi, lam, beta)
    """

    def __init__(self,
                 sim_band,
                 df,
                 f_band,
                 dA,
                 dE,
                 SA,
                 SE,
                 dt_star):
        self.sim_band = sim_band

        self.df     = float(df)
        self.f_band = jnp.asarray(f_band, DTYPE)

        self.dA = jnp.asarray(dA, CDTYPE)
        self.dE = jnp.asarray(dE, CDTYPE)
        self.SA = jnp.asarray(SA, DTYPE)
        self.SE = jnp.asarray(SE, DTYPE)

        self.dt_star = jnp.asarray(dt_star, DTYPE)
        self.M = self.f_band.size

        # (d|d) constant on the band
        self.d_d_const = (4.0 * self.df * (
            jnp.vdot(self.dA, self.dA / (self.SA + EPS)) +
            jnp.vdot(self.dE, self.dE / (self.SE + EPS))
        )).real

    @staticmethod
    def _ip(x, y, S, df):
        """(x|y) = 4 df Σ x* y / S."""
        return 4.0 * df * jnp.vdot(x, y / (S + EPS))

    # ----------------- templates -----------------

    def _template_AE_explicit(self, theta8):
        """
        Template (hA, hE) for θ = (lnA, f0, fdot, phi0, iota, psi, lam, beta)
        with explicit amplitude and phase.
        """
        lnA, f0, fdot, phi0, iota, psi, lam, beta = theta8

        amp  = jnp.exp(lnA)
        iota = jnp.clip(iota, -jnp.pi / 2, jnp.pi / 2)
        psi  = (psi + jnp.pi) % (2 * jnp.pi) - jnp.pi
        lam  = (lam + jnp.pi) % (2 * jnp.pi) - jnp.pi
        beta = jnp.clip(beta, -jnp.pi / 2, jnp.pi / 2)

        pars = GBJAXParameters(
            amp  = jnp.asarray([amp],  DTYPE),
            f0   = jnp.asarray([f0],   DTYPE),
            fdot = jnp.asarray([fdot], DTYPE),
            fddot= jnp.zeros((1,),     DTYPE),
            phi0 = jnp.asarray([phi0], DTYPE),
            iota = jnp.asarray([iota], DTYPE),
            psi  = jnp.asarray([psi],  DTYPE),
            lam  = jnp.asarray([lam],  DTYPE),
            beta = jnp.asarray([beta], DTYPE),
        )

        xyzf = self.sim_band(pars)
        A, E, _ = xyz_to_aet(xyzf)
        hA = jnp.asarray(jnp.squeeze(A), CDTYPE)[: self.M]
        hE = jnp.asarray(jnp.squeeze(E), CDTYPE)[: self.M]

        phase = jnp.exp(-2j * jnp.pi * self.f_band * self.dt_star).astype(CDTYPE)
        hA = hA * phase
        hE = hE * phase

        return hA, hE

    def _template_AE_single(self, theta6):
        """
        Template (hA, hE) for θ = (f0, fdot, iota, psi, lam, beta)
        with amp = 1, phi0 = 0 (for amplitude-profiled likelihood).
        """
        f0, fdot, iota, psi, lam, beta = theta6

        iota = jnp.clip(iota, -jnp.pi / 2, jnp.pi / 2)
        psi  = (psi + jnp.pi) % (2 * jnp.pi) - jnp.pi
        lam  = (lam + jnp.pi) % (2 * jnp.pi) - jnp.pi
        beta = jnp.clip(beta, -jnp.pi / 2, jnp.pi / 2)

        pars = GBJAXParameters(
            amp  = jnp.asarray([1.0],  DTYPE),
            f0   = jnp.asarray([f0],   DTYPE),
            fdot = jnp.asarray([fdot], DTYPE),
            fddot= jnp.zeros((1,),     DTYPE),
            phi0 = jnp.asarray([0.0],  DTYPE),
            iota = jnp.asarray([iota], DTYPE),
            psi  = jnp.asarray([psi],  DTYPE),
            lam  = jnp.asarray([lam],  DTYPE),
            beta = jnp.asarray([beta], DTYPE),
        )

        xyzf = self.sim_band(pars)
        A, E, _ = xyz_to_aet(xyzf)
        hA = jnp.asarray(jnp.squeeze(A), CDTYPE)[: self.M]
        hE = jnp.asarray(jnp.squeeze(E), CDTYPE)[: self.M]

        phase = jnp.exp(-2j * jnp.pi * self.f_band * self.dt_star).astype(CDTYPE)
        hA = hA * phase
        hE = hE * phase

        return hA, hE

    # ----------------- explicit-amp likelihood -----------------

    def loglike_explicit_core(self, theta8: Array) -> Array:
        """
        θ = (lnA, f0, fdot, phi0, iota, psi, lam, beta)

        Standard Gaussian log-likelihood with explicit amplitude:

          log L(θ) = -0.5 (d - h | d - h)
                    = -0.5[(d|d) - 2 Re(d|h) + (h|h)].
        """
        
        theta8 = jnp.asarray(theta8, DTYPE).reshape(-1)
        hA, hE = self._template_AE_explicit(theta8)

        d_h = self._ip(hA, self.dA, self.SA, self.df) + \
              self._ip(hE, self.dE, self.SE, self.df)
        h_h = self._ip(hA, hA,  self.SA, self.df) + \
              self._ip(hE, hE,  self.SE, self.df)

        return -0.5 * (
            self.d_d_const
            - 2.0 * jnp.real(d_h)
            + jnp.maximum(h_h.real, 0.0)
        )

    @partial(jax.jit, static_argnums=0)
    def loglike_explicit(self, theta8: Array) -> Array:
        return self.loglike_explicit_core(theta8)

    # ----------------- amp-profiled likelihood -----------------

    def loglike_marg_ap_core(self, theta_vec: Array) -> Array:
        """
        Profiled / 'marginalised' likelihood over amplitude A:

        θ = (f0, fdot, iota, psi, lam, beta)

        log L(θ) ∝ -0.5 [ (d|d) - |(d|h)|^2 / (h|h) ]

        up to a θ-independent constant.
        """
        theta = jnp.asarray(theta_vec, DTYPE)

        hA, hE = self._template_AE_single(theta)
        d_h = self._ip(hA, self.dA, self.SA, self.df) + \
                  self._ip(hE, self.dE, self.SE, self.df)
        h_h = self._ip(hA, hA,  self.SA, self.df) + \
                  self._ip(hE, hE,  self.SE, self.df)

        frac = (jnp.abs(d_h) ** 2) / jnp.maximum(h_h.real, EPS)

        loglike = -0.5 * (self.d_d_const - frac)

        #jax.debug.print("[loglike_marg_ap] ll.shape = {}", loglike)
        return loglike
        
    @partial(jax.jit, static_argnums=0)
    def loglike_marg_ap(self, theta_vec: Array) -> Array:
        return self.loglike_marg_ap_core(theta_vec)


# -------------------------------------------------------------------
# LISAOneGBProblem: interface to jax-samplers / SBI
# -------------------------------------------------------------------

class LISAOneGBProblem(Problem):
    def __init__(self,
                 ctx: _Ctx,
                 prior_box: dict | None,
                 gaussian_priors: dict | None,
                 marg_ap: bool = True):
        """
        LISA one-GB problem for jax-samplers.

        - If marg_ap=True:   6D problem with amp-profiled ('marginalised') likelihood
                             over A, with fixed phi0=0 internally.

             θ = (f0, fdot, iota, psi, lam, beta)

        - If marg_ap=False:  8D problem with explicit amplitude and phase:

             θ = (lnA, f0, fdot, phi0, iota, psi, lam, beta)
        """
        self.ctx   = ctx
        self._gauss = gaussian_priors or {}
        self._marg_ap = bool(marg_ap)

        if self._marg_ap:
            # 6D, amplitude-profiled likelihood
            self.dim    = 6
            self._names = ['f0', 'fdot', 'iota', 'psi', 'lam', 'beta']

            base_box_6 = {
                'f0'  : (1.0e-4, 5.0e-2),
                'fdot': (-1.0e-7, 1.0e-7),
                'iota': (-jnp.pi/2, jnp.pi/2),
                'psi' : (-jnp.pi, jnp.pi),
                'lam' : (-jnp.pi, jnp.pi),
                'beta': (-jnp.pi/2, jnp.pi/2),
            }
            box = prior_box or base_box_6
            self._ll_single = self.ctx.loglike_marg_ap

        else:
            # 8D, explicit lnA and phi0
            self.dim    = 8
            self._names = ['lnA', 'f0', 'fdot', 'phi0', 'iota', 'psi', 'lam', 'beta']

            base_box_8 = {
                'lnA' : (-35.0,  -5.0),
                'f0'  : (1.0e-4, 5.0e-2),
                'fdot': (-1.0e-7, 1.0e-7),
                'phi0': (0.0, 2*jnp.pi),
                'iota': (-jnp.pi/2, jnp.pi/2),
                'psi' : (-jnp.pi, jnp.pi),
                'lam' : (-jnp.pi, jnp.pi),
                'beta': (-jnp.pi/2, jnp.pi/2),
            }
            box = prior_box or base_box_8
            self._ll_single = self.ctx.loglike_explicit

        # NOTE: ctx.loglike_* are already jitted; avoid double-jitting.
        # self._ll_single = jax.jit(self._ll_single)

        # store prior box and bounds in the order of self._names
        self._box = box
        self.prior_bounds: Bounds = [self._box[k] for k in self._names]

    # -------- likelihood --------
    """
    def loglikelihood(self, theta: Array) -> Array:
        θ can be shape (dim,) or (B, dim).
        We vmap the single-point jitted loglike over axis 0 if batched.
        
        theta = jnp.asarray(theta, DTYPE)

        
        if theta.ndim == 1:
            return self._ll_single(theta)
        elif theta.ndim == 2:
            return jax.vmap(self._ll_single, in_axes=0)(theta)
        else:
            raise ValueError(f"loglikelihood: theta must be (dim,) or (B,dim); got {theta.shape}")
    """
    # -------- prior --------

    def _single_logprior(self, t1d: Array) -> Array:
        t1d = jnp.asarray(t1d, DTYPE).reshape(-1)


        lowers = jnp.array([a for (a, _) in self.prior_bounds], DTYPE)
        uppers = jnp.array([b for (_, b) in self.prior_bounds], DTYPE)

        inside = jnp.logical_and(t1d >= lowers, t1d <= uppers)
        all_inside = jnp.all(inside)

        lp   = jnp.array(0.0, DTYPE)
        twopi = jnp.array(2 * jnp.pi, DTYPE)

        for i, name in enumerate(self._names):
            v = t1d[i]
            a, b = self.prior_bounds[i]

            if name in self._gauss:
                mu, sig = self._gauss[name]
                mu  = jnp.asarray(mu, DTYPE)
                sig = jnp.asarray(sig, DTYPE)
                z = (v - mu) / sig
                lp = lp + (-0.5 * z * z - jnp.log(sig * jnp.sqrt(twopi)))
            else:
                lp = lp - jnp.log(jnp.asarray(b - a, DTYPE))

        lp = jnp.where(all_inside, lp, -jnp.inf)
        return lp

    """
    def logprior(self, theta: Array) -> Array:
        theta = jnp.asarray(theta, DTYPE)
        if theta.ndim == 1:
            return self._single_logprior(theta)
        elif theta.ndim == 2:
            return jax.vmap(self._single_logprior, in_axes=0)(theta)
        else:
            raise ValueError(f"logprior: theta must be (dim,) or (B,dim); got {theta.shape}")
    """
    def loglikelihood(self, theta: Array) -> Array:
        theta = jnp.asarray(theta, DTYPE).reshape(-1)
        # Expect shape (dim,)
        return self._ll_single(theta)

    def logprior(self, theta: Array) -> Array:
        theta = jnp.asarray(theta, DTYPE).reshape(-1)

        return self._single_logprior(theta)


        
    # -------- prior sampling --------

    def sample_prior(self, key: Array, n: int) -> Array:
        ks = jr.split(key, self.dim)
        b  = self._box

        def U(k, a, bnd):
            return jr.uniform(
                k,
                (n,),
                minval=float(a),
                maxval=float(bnd),
                dtype=DTYPE,
            )

        cols = [U(ks[i], *b[name]) for i, name in enumerate(self._names)]
        return jnp.stack(cols, axis=1)


# -------------------------------------------------------------------
# Factory: used by jax-samplers to build the Problem
# -------------------------------------------------------------------

def make(args=None):
    """
    Build a LISAOneGBProblem from:
      - H5 training file (Sangria),
      - one GB source (cat_idx),
      - band-limited GBJAX likelihood (narrow-band, amp-profiled).
    """

    # Optional JSON cfg pointed by env var
    CFG_PATH = os.environ.get("LISA_CFG", "")
    cfg_in = {}
    if CFG_PATH and os.path.exists(CFG_PATH):
        with open(CFG_PATH, "r") as fh:
            cfg_in = json.load(fh)

    # Inputs with env fallbacks
    H5      = cfg_in.get("h5", os.environ.get(
                 "LISA_H5",
                 "/r5/home/rruiz/projects/sbi/lisa/data/LDC2_sangria_training_v2.h5"))
    CATIDX  = int(cfg_in.get("cat_idx",  os.environ.get("LISA_CATIDX", "0")))
    HALFB   = int(cfg_in.get("half_bins", os.environ.get("LISA_HALFBINS", "300")))
    TDI2    = bool(cfg_in.get("use_tdi2", bool(int(os.environ.get("LISA_TDI2", "0")))))
    ALIGN   = bool(cfg_in.get("align_dt", bool(int(os.environ.get("LISA_ALIGN_DT", "1")))))
    FLIPPHI = bool(cfg_in.get("flip_phi0", bool(int(os.environ.get("LISA_FLIP_PHI0", "0")))))

    PRESET = getattr(NoisePreset,    cfg_in.get("noise_preset",  "MRDv1"))
    TGEN   = getattr(TDIGeneration,  cfg_in.get("tdi_generation", "TDI1"))
    NOISE  = bool(cfg_in.get("noise_det", bool(int(os.environ.get("LISA_NOISE_DT", "0")))))
    DEBUG =  bool(cfg_in.get("debug", bool(int(os.environ.get("DEBUG", "0")))))
    
    prior_box_in    = cfg_in.get("prior_box", {}) or {}
    gaussian_priors = cfg_in.get("gaussian_priors", {}) or {}

    marg_ap = cfg_in.get("marginalize", {}).get("amp_phase", True)
    
    # ----------------------- load noiseless data -----------------------
    #with h5py.File(H5, "r") as f:
    #    t, X, Y, Z = load_single_vgb_or_fail(f, CATIDX)

    with h5py.File(H5, "r") as f:
        # signal: single noiseless VGB
        t_sig, X_sig, Y_sig, Z_sig = load_single_vgb_or_fail(f, CATIDX)

        if NOISE:
            t_n, nX, nY, nZ = load_instrument_noise_xyz(
                f, subtract_vgb=True,
            )

            # sanity: same time grid (tolerance)
            #if t_n.shape != t_sig.shape or not np.allclose(t_n, t_sig, rtol=0.0, atol=0.0):
                # if you want a safer tolerance:
                # not np.allclose(t_n, t_sig, rtol=0.0, atol=1e-12)
            #    raise RuntimeError("Time grids for signal and noise do not match.")

            if not grids_compatible(t_sig, t_n):
                raise RuntimeError("Time grids for signal and noise are not compatible (dt/t0 mismatch).")

            
            t = t_sig
            X = X_sig + nX
            Y = Y_sig + nY
            Z = Z_sig + nZ
            print("[data] mode = single_vgb + instrument_noise")
        else:
            t, X, Y, Z = t_sig, X_sig, Y_sig, Z_sig
            print("[data] mode = single_vgb (noiseless)")
        
    t = as_1d(t)
    dt   = float(t[1] - t[0])
    N    = t.size
    Tobs = N * dt

    if DEBUG:
     jax.debug.print(f"Original grid: dt = {dt:.6f} s, N = {N}, Tobs = {Tobs:.3f} s")

    # Frequency grid (positive rFFT freqs)
    f_pos = rfftfreq(N, d=dt)

    # Exact FFT spacing (prefer this over float diffs)
    df = 1.0 / Tobs  # should match f_pos[1] - f_pos[0]

    # data A/E on full positive grid
    dA_full, dE_full = xyz_to_AE_freq(X, Y, Z, dt)

    # ----------------------- truth + band -----------------------
    theta_true = load_theta(H5, CATIDX, flip_phi0=FLIPPHI)
    f0_true    = float(theta_true.f0[0])

    if DEBUG:
     jax.debug.print(f"Catalogue f0 = {f0_true:.9e} Hz")

    # pick central bin
    k0 = int(round(f0_true * Tobs))

    # choose i1..i2 inclusive, BUT keep i2 <= len(f_pos)-2 so (i2+1) is valid
    i1 = max(1, k0 - HALFB)
    i2 = min(f_pos.size - 2, k0 + HALFB)

    idx_band = np.arange(i1, i2 + 1, dtype=int)
    M_data   = idx_band.size

    # band arrays from the FFT grid
    f_band   = f_pos[idx_band]
    dA_band  = dA_full[idx_band]
    dE_band  = dE_full[idx_band]
    SA_band, SE_band = psd_AE(f_band, preset=PRESET, gen=TGEN)

    # ----------------------- narrow-band GBJAX sim -----------------------
    n_f_bins_arr = estimate_n_f_bins(theta_true.amp, theta_true.f0, Tobs)
    n_f_bins = int(np.max(np.asarray(n_f_bins_arr)))
    n_f_bins = max(n_f_bins, 256)

    if DEBUG:
     jax.debug.print(f"Estimated n_f_bins = {n_f_bins}")

    # BIN-EXACT config edges:
    # include bins i1..i2 inclusive => length = i2-i1+1
    f_min_cfg = float(i1 * df)
    f_max_cfg = float(np.nextafter((i2 + 1) * df, np.inf))  # exclusive upper edge nudged up

    cfg = GBJAXConfig(
        t_obs=Tobs,
        dt=dt,
        n_f_bins=n_f_bins,
        tdi2=TDI2,
        f_min=f_min_cfg,
        f_max=f_max_cfg,
    )

    if DEBUG:
     jax.debug.print(f"GBJAX band: f_min = {cfg.f_min:.9e}, f_max = {cfg.f_max:.9e}, cfg.length = {cfg.length}")
     jax.debug.print(f"M_data = {M_data}")
    assert cfg.length == M_data, (cfg.length, M_data)

    sim_band = make_simulator(cfg)

    # ----------------------- Δt* alignment (optional) ------------------
    dt_star = 0.0
    if ALIGN:
        pars_true = GBJAXParameters(
            amp  = jnp.array([1.0], DTYPE),
            f0   = theta_true.f0,
            fdot = theta_true.fdot,
            fddot= jnp.zeros((1,), DTYPE),
            phi0 = jnp.array([0.0], DTYPE),
            iota = theta_true.iota,
            psi  = theta_true.psi,
            lam  = theta_true.lam,
            beta = theta_true.beta,
        )
        xyzf_true = sim_band(pars_true)
        A_full, E_full, _ = xyz_to_aet(xyzf_true)
        #A_full = jnp.asarray(jnp.squeeze(A_full), CDTYPE)[:M]
        #E_full = jnp.asarray(jnp.squeeze(E_full), CDTYPE)[:M]

        A_full = jnp.asarray(jnp.squeeze(A_full), CDTYPE)[:M_data]
        E_full = jnp.asarray(jnp.squeeze(E_full), CDTYPE)[:M_data]

        
        _, _, dt_star_val, coh_max = delta_t_align(
            A_full, E_full,
            dA_band, dE_band,
            SA_band, SE_band,
            f_band, df,
        )
        dt_star = float(dt_star_val)

        if DEBUG:
            jax.debug.print(f"[align] Δt* = {dt_star:.3f} s | coh ≈ {float(coh_max):.4f}")
    else:
        if DEBUG:
            jax.debug.print("[align] Δt* disabled; using dt_star = 0")

    # ----------------------- build context -----------------------
    ctx = _Ctx(
        sim_band=sim_band,
        df=df,
        f_band=f_band,
        dA=dA_band,
        dE=dE_band,
        SA=SA_band,
        SE=SE_band,
        dt_star=dt_star,
    )

    
    # ----------------------- quick sanity check -----------------------
    if DEBUG:
        theta6_true = jnp.array([
            float(theta_true.f0[0]),
            float(theta_true.fdot[0]),
            float(theta_true.iota[0]),
            float(theta_true.psi[0]),
            float(theta_true.lam[0]),
            float(theta_true.beta[0]),
        ], dtype=DTYPE)

        L_marg_true = float(ctx.loglike_marg_ap(theta6_true))
        print("[DEBUG] logL_marg(theta_true) =", L_marg_true)

            

    # user priors from config
    # supports either {"prior_box": {...}} or {"priors":{"uniform":{...}}}
    prior_box_in = cfg_in.get("prior_box", None)
    if prior_box_in is None:
        prior_box_in = (cfg_in.get("priors", {}) or {}).get("uniform", {}) or {}

    gaussian_priors_in = cfg_in.get("gaussian_priors", None)
    if gaussian_priors_in is None:
        gaussian_priors_in = (cfg_in.get("priors", {}) or {}).get("gaussian", {}) or {}

        marg_ap = bool(cfg_in.get("marginalize", {}).get("amp_phase", True))

    merged_box = build_merged_box(
        marg_ap=marg_ap,
        f0_true=f0_true,
        df=df,
        prior_box_in=prior_box_in,
        f_band_min=f_min_cfg,  # enforce overlap with actual analysis band
        f_band_max=f_max_cfg,
    )

    gaussian_priors = filter_gaussian_priors_for_mode(gaussian_priors_in, marg_ap)

    return LISAOneGBProblem(
        ctx,
        prior_box=merged_box,
        gaussian_priors=gaussian_priors,
        marg_ap=marg_ap,   # <-- IMPORTANT: don't hardcode True
    )
    
        

