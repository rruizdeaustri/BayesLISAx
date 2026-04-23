#!/usr/bin/env python3
"""
lisa_tdgb_problem.py  -- transdimensional narrow-band LISA GB backend (GBJAX, gates)

Implements a fixed-Kmax model with optional gates:
  amp_k = exp(lnA_k) * sigmoid(gate_logit_k)

Parameters live in unconstrained space theta_u ∈ R^D, D = Kmax*(8 or 9).

Data modes:
  - synthetic_from_catalogue: sum K_true catalogue sources with GBJAX (truth known),
    optionally add extracted instrument noise from Sangria.
  - sangria_total_band: take TOTAL VGB time series in band from Sangria (truth unknown),
    optionally add extracted instrument noise.

This backend returns a jax_samplers.core.problem.Problem with:
  loglikelihood(theta_u), logprior(theta_u), sample_prior(key,n)

Author: adapted from your validated test_recons_msource scripts.
"""

from __future__ import annotations

import os, json
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np

# precision policy (match your project)
from ..core.precision import DTYPE as _DTYPE, CDTYPE as _CDTYPE
import jax
import jax.numpy as jnp
import jax.random as jr

from jax_samplers.core.problem import Problem

from jaxlisa.tdi import xyz_to_aet, TDIXYZ
from jaxlisa.noise import AnalyticNoiseModel, NoisePreset
from jaxlisa.tdi_transfer import TDIChannel, TDIGeneration

from gbjax import (
    GBJAXConfig,
    GBJAXParameters,
    make_simulator,
    estimate_n_f_bins,
)

DTYPE  = _DTYPE
CDTYPE = _CDTYPE
EPS = jnp.asarray(jnp.finfo(DTYPE).tiny, dtype=DTYPE)

Array  = jax.Array
Bounds = List[Tuple[float, float]]


# -------------------- basic utilities -------------------- #

def as_1d(x):
    return np.asarray(x).reshape(-1)

def grids_compatible(t1, t2):
    t1 = np.asarray(t1).reshape(-1)
    t2 = np.asarray(t2).reshape(-1)
    if t1.size != t2.size:
        return False
    dt1 = float(np.median(np.diff(t1)))
    dt2 = float(np.median(np.diff(t2)))
    if not np.isclose(dt1, dt2, rtol=0.0, atol=1e-9):
        return False
    if abs(float(t1[0]) - float(t2[0])) > 0.5 * dt1:
        return False
    return True

def sc(row, key):
    a = np.asarray(row[key])
    return a.ravel()[0].item()

def sigmoid(x):
    return jax.nn.sigmoid(x)

def wrap_pm_pi(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi

def clamp_iota(x):
    return jnp.clip(x, -jnp.pi/2, jnp.pi/2)

def clamp_beta(x):
    return jnp.clip(x, -jnp.pi/2, jnp.pi/2)

# --- add near your globals ---
def unpack_theta_td(theta, Kmax, use_gates):
    """theta = [Kf, theta_u...]"""
    theta = jnp.asarray(theta, dtype=DTYPE).reshape(-1)
    Kf = theta[0]
    theta_u = theta[1:]
    per = 9 if use_gates else 8
    th = theta_u.reshape((Kmax, per))
    if use_gates:
        lnA_u, f0_u, fdot_u, phi0_u, iota_u, psi_u, lam_u, beta_u, g_u = th.T
        return Kf, lnA_u, f0_u, fdot_u, phi0_u, iota_u, psi_u, lam_u, beta_u, g_u
    else:
        lnA_u, f0_u, fdot_u, phi0_u, iota_u, psi_u, lam_u, beta_u = th.T
        return Kf, lnA_u, f0_u, fdot_u, phi0_u, iota_u, psi_u, lam_u, beta_u, None


def theta_to_gbjax_pars_td(theta, Kmax, use_gates, order_f0, f_min, f_max, df_bin):
    Kf, lnA_u, f0_u, fdot_u, phi0_u, iota_u, psi_u, lam_u, beta_u, g_u = unpack_theta_td(theta, Kmax, use_gates)

    # Discrete K in [K_min, K_max]
    K = jnp.rint(Kf).astype(jnp.int32)
    mask = (jnp.arange(Kmax) < K).astype(DTYPE)  # (Kmax,)

    amp = jnp.exp(lnA_u)
    if use_gates:
        amp = amp * sigmoid(g_u)
    amp = amp * mask  # hard-off inactive slots

    # frequency mapping (ordering can still be across Kmax; inactive amps are 0 anyway)
    if order_f0:
        f0 = u_to_f0_ordered(f0_u[0], f0_u[1:], f_min, f_max, df_min=df_bin, guard_bins=2.0)
    else:
        f0 = u_to_f0_unordered(f0_u, f_min, f_max)

    fdot = fdot_u
    phi0 = wrap_pm_pi(phi0_u)
    iota = clamp_iota(iota_u)
    psi  = wrap_pm_pi(psi_u)
    lam  = wrap_pm_pi(lam_u)
    beta = clamp_beta(beta_u)

    return K, GBJAXParameters(
        amp=amp.astype(DTYPE),
        f0=f0.astype(DTYPE),
        fdot=fdot.astype(DTYPE),
        fddot=jnp.zeros((Kmax,), dtype=DTYPE),
        phi0=phi0.astype(DTYPE),
        iota=iota.astype(DTYPE),
        psi=psi.astype(DTYPE),
        lam=lam.astype(DTYPE),
        beta=beta.astype(DTYPE),
    )

def psd_AE(f_phys, preset=NoisePreset.MRDv1, gen=TDIGeneration.TDI1):
    f_abs = jnp.asarray(np.abs(f_phys), dtype=DTYPE)
    nmA = AnalyticNoiseModel.from_preset(preset, TDIChannel.A, gen)
    SA = np.asarray(nmA.psd(f_abs), dtype=np.float64)
    try:
        nmE = AnalyticNoiseModel.from_preset(preset, TDIChannel.E, gen)
        SE = np.asarray(nmE.psd(f_abs), dtype=np.float64)
    except Exception:
        SE = SA.copy()
    if SA.size and SA[0] == 0:
        SA[0] = np.inf
    if SE.size and SE[0] == 0:
        SE[0] = np.inf
    return SA.astype(np.float64), SE.astype(np.float64)


# -------------------- Sangria I/O -------------------- #

def read_total_xyz_compound(ds: h5py.Dataset):
    """
    Your file: sky/vgb/tdi is compound dataset shape (Nt,1), fields t,X,Y,Z.
    """
    rec = ds[()]
    t = np.asarray(rec["t"]).reshape(-1).astype(np.float64)
    X = np.asarray(rec["X"]).reshape(-1).astype(np.float64)
    Y = np.asarray(rec["Y"]).reshape(-1).astype(np.float64)
    Z = np.asarray(rec["Z"]).reshape(-1).astype(np.float64)
    return t, X, Y, Z

def read_xyz_any(f: h5py.File, base: str):
    """
    Read XYZ(+t) from either group datasets base/X,Y,Z,(t) or a compound dataset at base.
    """
    if f"{base}/X" in f and f"{base}/Y" in f and f"{base}/Z" in f:
        t = as_1d(f[f"{base}/t"][:]) if f"{base}/t" in f else None
        X = as_1d(f[f"{base}/X"][:]).astype(np.float64)
        Y = as_1d(f[f"{base}/Y"][:]).astype(np.float64)
        Z = as_1d(f[f"{base}/Z"][:]).astype(np.float64)
        return t, X, Y, Z

    if base in f and isinstance(f[base], h5py.Dataset):
        ds = f[base]
        rec = ds[()]
        if not (hasattr(rec, "dtype") and rec.dtype.fields):
            raise ValueError(f"{base} is a dataset but not compound; unsupported.")
        t = np.asarray(rec["t"]).reshape(-1).astype(np.float64) if "t" in rec.dtype.fields else None
        X = np.asarray(rec["X"]).reshape(-1).astype(np.float64)
        Y = np.asarray(rec["Y"]).reshape(-1).astype(np.float64)
        Z = np.asarray(rec["Z"]).reshape(-1).astype(np.float64)
        return t, X, Y, Z

    raise ValueError(f"Could not read {base} as group or compound dataset.")

def load_total_vgb_xyz(h5path: str):
    with h5py.File(h5path, "r") as f:
        if "sky/vgb/tdi" not in f:
            raise ValueError("Missing sky/vgb/tdi in file.")
        t, X, Y, Z = read_total_xyz_compound(f["sky/vgb/tdi"])
    return t, X, Y, Z

def load_instrument_noise_xyz(h5path: str, subtract_vgb: bool):
    """
    n = obs - (dgb + igb + mbhb [+ vgb_total if subtract_vgb])
    """
    with h5py.File(h5path, "r") as f:
        if "obs/tdi" not in f:
            raise ValueError("This H5 does not contain obs/tdi; cannot build instrument noise.")
        t, X, Y, Z = read_xyz_any(f, "obs/tdi")
        if t is None:
            raise ValueError("obs/tdi has no time array 't' (unexpected).")

        _, dX, dY, dZ = read_xyz_any(f, "sky/dgb/tdi")
        _, iX, iY, iZ = read_xyz_any(f, "sky/igb/tdi")
        _, mX, mY, mZ = read_xyz_any(f, "sky/mbhb/tdi")

        X = X - (dX + iX + mX)
        Y = Y - (dY + iY + mY)
        Z = Z - (dZ + iZ + mZ)

        if subtract_vgb:
            _, vX, vY, vZ = read_xyz_any(f, "sky/vgb/tdi")
            X = X - vX
            Y = Y - vY
            Z = Z - vZ

    return t, X, Y, Z

def load_catalogue_rows(h5path: str, idxs: list[int]):
    with h5py.File(h5path, "r") as f:
        cat = f["sky/vgb/cat"][()]
        rows = [cat[i] for i in idxs]
    return rows

def pick_sources_in_band_from_catalogue(h5path: str, K: int, fmin: float, fmax: float, seed: int):
    rng = np.random.default_rng(seed)
    with h5py.File(h5path, "r") as f:
        cat = f["sky/vgb/cat"][()]
        f0s = np.array([np.asarray(row["Frequency"]).ravel()[0].item() for row in cat], dtype=np.float64)
    ok = np.where((f0s >= fmin) & (f0s < fmax))[0]
    if ok.size < K:
        raise RuntimeError(f"Not enough catalogue sources in band: have {ok.size}, need {K}")
    pick = rng.choice(ok, size=K, replace=False)
    return [int(x) for x in pick]

def gbjax_params_from_rows(rows, flip_phi0: bool) -> GBJAXParameters:
    amps  = np.array([sc(r, "Amplitude") for r in rows], dtype=np.float64)
    f0s   = np.array([sc(r, "Frequency") for r in rows], dtype=np.float64)
    fdots = np.array([sc(r, "FrequencyDerivative") for r in rows], dtype=np.float64)
    phi0s = np.array([sc(r, "InitialPhase") for r in rows], dtype=np.float64)
    if flip_phi0:
        phi0s = -phi0s
    iotas = np.array([sc(r, "Inclination") for r in rows], dtype=np.float64)
    psis  = np.array([sc(r, "Polarization") for r in rows], dtype=np.float64)
    lams  = np.array([sc(r, "EclipticLongitude") for r in rows], dtype=np.float64)
    betas = np.array([sc(r, "EclipticLatitude") for r in rows], dtype=np.float64)

    K = len(rows)
    return GBJAXParameters(
        amp=jnp.asarray(amps, dtype=DTYPE),
        f0=jnp.asarray(f0s, dtype=DTYPE),
        fdot=jnp.asarray(fdots, dtype=DTYPE),
        fddot=jnp.zeros((K,), dtype=DTYPE),
        phi0=jnp.asarray(phi0s, dtype=DTYPE),
        iota=jnp.asarray(iotas, dtype=DTYPE),
        psi=jnp.asarray(psis, dtype=DTYPE),
        lam=jnp.asarray(lams, dtype=DTYPE),
        beta=jnp.asarray(betas, dtype=DTYPE),
    )


# -------------------- band building -------------------- #

def build_band_from_time_series(
    t: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    decim: int,
    f_min_user: float | None,
    f_max_user: float | None,
    half_bins: int,
    f0_refs: list[float] | None,
):
    """
    FFT -> A/E -> choose integer-bin band.
    Returns: (t_dec, dt_dec, Tobs, f_band, Af_band, Ef_band, df_full, i1, i2)
    """
    t = np.asarray(t).reshape(-1)
    X = np.asarray(X).reshape(-1)
    Y = np.asarray(Y).reshape(-1)
    Z = np.asarray(Z).reshape(-1)

    dt0 = float(t[1] - t[0])
    if decim > 1:
        t = t[::decim]
        X = X[::decim]
        Y = Y[::decim]
        Z = Z[::decim]
        dt = dt0 * decim
    else:
        dt = dt0

    N = t.size
    Tobs = N * dt

    Xf = np.fft.rfft(X) * dt
    Yf = np.fft.rfft(Y) * dt
    Zf = np.fft.rfft(Z) * dt
    freqs = np.fft.rfftfreq(N, d=dt)

    Af = (Zf - Xf) / np.sqrt(2.0)
    Ef = (Xf - 2.0 * Yf + Zf) / np.sqrt(6.0)

    df_full = float(freqs[1] - freqs[0])

    if f_min_user is not None and f_max_user is not None:
        fmin = float(f_min_user); fmax = float(f_max_user)
        i1 = int(np.ceil(fmin / df_full))
        i2 = int(np.floor(np.nextafter(fmax, -np.inf) / df_full))
    else:
        if f0_refs is None or len(f0_refs) == 0:
            raise ValueError("Auto-band requires f0_refs (provide --ref-cat-idx or explicit f-min/f-max).")
        f0_min = float(np.min(f0_refs))
        f0_max = float(np.max(f0_refs))
        k1 = int(round(f0_min * Tobs))
        k2 = int(round(f0_max * Tobs))
        i1 = k1 - half_bins
        i2 = k2 + half_bins

    i1 = max(1, i1)
    i2 = min(freqs.size - 2, i2)
    if i2 < i1:
        raise RuntimeError("Empty band after clipping.")

    idx = np.arange(i1, i2 + 1, dtype=np.int64)
    f_band = freqs[idx]
    A_band = Af[idx]
    E_band = Ef[idx]
    return t, dt, Tobs, f_band, A_band, E_band, df_full, i1, i2


# -------------------- simulator wrapper (robust TDIXYZ) -------------------- #

def build_simulator(cfg: GBJAXConfig):
    base = make_simulator(cfg)

    def sim_multi(pars: GBJAXParameters):
        xyz = base(pars)
        K = int(pars.amp.shape[0])

        if isinstance(xyz, TDIXYZ):
            x = getattr(xyz, "x", None)
            y = getattr(xyz, "y", None)
            z = getattr(xyz, "z", None)
            if x is None:
                x = getattr(xyz, "X")
                y = getattr(xyz, "Y")
                z = getattr(xyz, "Z")

            if hasattr(x, "ndim") and x.ndim == 2 and x.shape[0] == K:
                return TDIXYZ(jnp.sum(x, axis=0), jnp.sum(y, axis=0), jnp.sum(z, axis=0))
            return xyz

        if hasattr(xyz, "ndim") and xyz.ndim == 3 and xyz.shape[0] == K:
            return jnp.sum(xyz, axis=0)

        return xyz

    return sim_multi


# -------------------- theta_u -> GBJAXParameters -------------------- #

def u_to_f0_unordered(u, f_min, f_max):
    return f_min + sigmoid(u) * (f_max - f_min)

def u_to_f0_ordered(u0, v_incr, f_min, f_max, df_min, guard_bins=2.0):
    band = f_max - f_min
    guard = guard_bins * df_min
    Km1 = v_incr.shape[0]
    min_span = df_min * Km1
    free = band - 2.0 * guard - min_span
    free = jnp.maximum(free, 10.0 * df_min)

    offset = sigmoid(u0) * free
    f1 = f_min + guard + offset

    w = jax.nn.softplus(v_incr) + 1e-12
    w = w / jnp.sum(w)
    remaining = jnp.maximum(free - offset, 0.0)

    deltas = df_min + remaining * w
    f0s = f1 + jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(deltas)])
    return f0s

def unpack_theta_u(theta_u, Kmax, use_gates):
    per = 9 if use_gates else 8
    th = theta_u.reshape((Kmax, per))
    if use_gates:
        lnA_u, f0_u, fdot_u, phi0_u, iota_u, psi_u, lam_u, beta_u, g_u = th.T
        return lnA_u, f0_u, fdot_u, phi0_u, iota_u, psi_u, lam_u, beta_u, g_u
    else:
        lnA_u, f0_u, fdot_u, phi0_u, iota_u, psi_u, lam_u, beta_u = th.T
        return lnA_u, f0_u, fdot_u, phi0_u, iota_u, psi_u, lam_u, beta_u, None

def theta_u_to_gbjax_pars(theta_u, Kmax, use_gates, order_f0, f_min, f_max, df_bin):
    lnA_u, f0_u, fdot_u, phi0_u, iota_u, psi_u, lam_u, beta_u, g_u = unpack_theta_u(theta_u, Kmax, use_gates)

    amp = jnp.exp(lnA_u)
    if use_gates:
        amp = amp * sigmoid(g_u)

    if order_f0:
        f0 = u_to_f0_ordered(f0_u[0], f0_u[1:], f_min, f_max, df_min=df_bin, guard_bins=2.0)
    else:
        f0 = u_to_f0_unordered(f0_u, f_min, f_max)

    fdot = fdot_u
    phi0 = wrap_pm_pi(phi0_u)
    iota = clamp_iota(iota_u)
    psi  = wrap_pm_pi(psi_u)
    lam  = wrap_pm_pi(lam_u)
    beta = clamp_beta(beta_u)

    return GBJAXParameters(
        amp=amp.astype(DTYPE),
        f0=f0.astype(DTYPE),
        fdot=fdot.astype(DTYPE),
        fddot=jnp.zeros((Kmax,), dtype=DTYPE),
        phi0=phi0.astype(DTYPE),
        iota=iota.astype(DTYPE),
        psi=psi.astype(DTYPE),
        lam=lam.astype(DTYPE),
        beta=beta.astype(DTYPE),
    )


# -------------------- Context: loglike/logprior -------------------- #

class _Ctx:
    def __init__(self, sim, f_band, dA, dE, SA, SE, df_band, Kmin, 
                 Kmax, use_gates, order_f0, f_min_cfg, f_max_cfg,
                 lnA_mu, lnA_sigma, fdot_sigma, gate_mu, gate_sigma):
        self.sim = sim
        self.f_band = jnp.asarray(f_band, dtype=DTYPE)
        self.dA = jnp.asarray(dA, dtype=CDTYPE)
        self.dE = jnp.asarray(dE, dtype=CDTYPE)
        self.SA = jnp.asarray(SA, dtype=DTYPE)
        self.SE = jnp.asarray(SE, dtype=DTYPE)
        self.df = float(df_band)

        self.K_min = int(Kmin)  # or 1 if you never want K=0
        self.Kmax = int(Kmax)
        self.K_max = self.Kmax  # convenience
        self.use_gates = bool(use_gates)
        self.order_f0 = bool(order_f0)
        self.f_min_cfg = float(f_min_cfg)
        self.f_max_cfg = float(f_max_cfg)
        
        self.lnA_mu = float(lnA_mu)
        self.lnA_sigma = float(lnA_sigma)
        self.fdot_sigma = float(fdot_sigma)
        self.gate_mu = float(gate_mu)
        self.gate_sigma = float(gate_sigma)

        self.M = int(np.asarray(dA).size)

    @staticmethod
    def _ip(x, y, S, df):
        return 4.0 * df * jnp.vdot(x, y / (S + EPS))

    @jax.jit
    def loglike(self, theta: Array) -> Array:
        theta = jnp.asarray(theta, dtype=DTYPE).reshape(-1)

        K, pars = theta_to_gbjax_pars_td(
            theta=theta,
            Kmax=self.Kmax,
            use_gates=self.use_gates,
            order_f0=self.order_f0,
            f_min=self.f_min_cfg,
            f_max=self.f_max_cfg,
            df_bin=self.df,
        )

        # Reject K outside support (fast)
        badK = (K < self.K_min) | (K > self.Kmax)
        def _neg_inf():
            return jnp.asarray(-jnp.inf, dtype=DTYPE)

        def _ok():
            xyz = self.sim(pars)
            aet = xyz_to_aet(xyz)
            A_sim, E_sim = (jnp.squeeze(ch) for ch in aet[:-1])
            A_sim = A_sim[:self.M]
            E_sim = E_sim[:self.M]
            rA = self.dA - A_sim
            rE = self.dE - E_sim
            chi2 = self._ip(rA, rA, self.SA, self.df) + self._ip(rE, rE, self.SE, self.df)
            return -0.5 * jnp.real(chi2)

        return jax.lax.cond(badK, _neg_inf, _ok)

    @jax.jit
    def logprior(self, theta: Array) -> Array:
        theta = jnp.asarray(theta, dtype=DTYPE).reshape(-1)
        Kf, lnA_u, f0_u, fdot_u, phi0_u, iota_u, psi_u, lam_u, beta_u, g_u = unpack_theta_td(
            theta, self.Kmax, self.use_gates
        )
        K = jnp.rint(Kf).astype(jnp.int32)
        mask = jnp.arange(self.Kmax) < K  # bool

        # discrete Uniform over K_min..K_max
        lp_K = jnp.where(
            (K >= self.K_min) & (K <= self.Kmax),
            -jnp.log(self.Kmax - self.K_min + 1.0),
            -jnp.inf
        )

        twopi = jnp.asarray(2.0 * jnp.pi, dtype=DTYPE)
        def normal_lp(x, mu, sig):
            mu = jnp.asarray(mu, dtype=DTYPE)
            sig = jnp.asarray(sig, dtype=DTYPE)
            z = (x - mu) / sig
            return -0.5 * jnp.sum(z*z) - x.size * jnp.log(sig * jnp.sqrt(twopi))

        # Priors: only "count" active slots for physics-y params if you want
        # (Inactive don't matter because amp is masked, but this helps NS geometry.)
        def masked_normal_lp(x, mu, sig, mask):
            x   = jnp.asarray(x)
            mu  = jnp.asarray(mu, dtype=DTYPE)
            sig = jnp.asarray(sig, dtype=DTYPE)

            m = mask.astype(DTYPE)                      # 0/1
            z = (x - mu) / sig
            # sum only active
            quad = jnp.sum(m * z * z)
            nact = jnp.sum(m)
            twopi = jnp.asarray(2.0 * jnp.pi, dtype=DTYPE)
            return -0.5 * quad - nact * jnp.log(sig * jnp.sqrt(twopi))

        
        lp = lp_K
        lp += masked_normal_lp(lnA_u, self.lnA_mu, self.lnA_sigma, mask)
        lp += masked_normal_lp(fdot_u, 0.0, self.fdot_sigma, mask)

        ang_sigma = 10.0
        lp += masked_normal_lp(phi0_u, 0.0, ang_sigma, mask)
        lp += masked_normal_lp(iota_u, 0.0, ang_sigma, mask)
        lp += masked_normal_lp(psi_u,  0.0, ang_sigma, mask)
        lp += masked_normal_lp(lam_u,  0.0, ang_sigma, mask)
        lp += masked_normal_lp(beta_u, 0.0, ang_sigma, mask)

        f0_sigma = 5.0
        lp += masked_normal_lp(f0_u, 0.0, f0_sigma, mask)

        if self.use_gates:
            # Recommended: active gates ~ "on", inactive gates ~ "off"
            # (These are logits, so "off" means negative, "on" means near 0 or positive.)
            g_on_mu,  g_on_sig  = 0.0, 2.0
            g_off_mu, g_off_sig = self.gate_mu, self.gate_sigma  # e.g. -6 ± 2
            lp += normal_lp(jnp.where(mask, g_u, 0.0), g_on_mu,  g_on_sig)
            lp += normal_lp(jnp.where(~mask, g_u, 0.0), g_off_mu, g_off_sig)

        return lp


# -------------------- jax_samplers Problem wrapper -------------------- #
class LISATDGBProblem(Problem):
    def __init__(self, ctx: _Ctx):
        self.ctx = ctx
        self.K_min = ctx.K_min
        self.K_max = ctx.Kmax
        self.use_gates = ctx.use_gates
        self.dim_per_atom = 9 if self.use_gates else 8

        self.dim = 1 + self.K_max * self.dim_per_atom  # <-- transdim layout

        self.prior_bounds = [(-50.0, 50.0)] * self.dim

    def loglikelihood(self, theta: Array) -> Array:
        return self.ctx.loglike(theta)

    def logprior(self, theta: Array) -> Array:
        return self.ctx.logprior(theta)

    def sample_prior(self, key: Array, n: int) -> Array:
        ctx = self.ctx
        Kmax = ctx.Kmax
        per  = self.dim_per_atom

        k0, k1 = jr.split(key, 2)

        # Sample discrete K (stored as float)
        Ks = jr.randint(k0, shape=(n,), minval=ctx.K_min, maxval=ctx.Kmax + 1).astype(DTYPE)

        # Sample per-source unconstrained params
        th = jr.normal(k1, shape=(n, Kmax, per), dtype=DTYPE)

        ang_sigma = 10.0
        f0_sigma = 5.0

        th = th.at[:, :, 0].set(ctx.lnA_mu + ctx.lnA_sigma * th[:, :, 0])   # lnA
        th = th.at[:, :, 1].set(f0_sigma * th[:, :, 1])                     # f0_u
        th = th.at[:, :, 2].set(ctx.fdot_sigma * th[:, :, 2])               # fdot
        th = th.at[:, :, 3].set(ang_sigma * th[:, :, 3])                    # phi0_u
        th = th.at[:, :, 4].set(ang_sigma * th[:, :, 4])                    # iota_u
        th = th.at[:, :, 5].set(ang_sigma * th[:, :, 5])                    # psi_u
        th = th.at[:, :, 6].set(ang_sigma * th[:, :, 6])                    # lam_u
        th = th.at[:, :, 7].set(ang_sigma * th[:, :, 7])                    # beta_u

        if self.use_gates:
            # draw "off-ish" by default, then overwrite active slots to "on-ish"
            g = ctx.gate_mu + ctx.gate_sigma * th[:, :, 8]
            # make first K slots on-ish per sample
            # (vectorized by comparing index to Ks)
            idx = jnp.arange(Kmax)[None, :]
            active = idx < Ks[:, None].astype(jnp.int32)
            g = jnp.where(active, 0.0 + 2.0 * th[:, :, 8], g)  # on: N(0,2), off: N(gate_mu,gate_sigma)
            th = th.at[:, :, 8].set(g)

        theta_u = th.reshape((n, Kmax * per))
        return jnp.concatenate([Ks[:, None], theta_u], axis=1)

# -------------------- factory make() -------------------- #

def make(args=None):
    """
    Build LISATDGBProblem. Configure via env var JSON, similar to your one-GB backend.
    """

    CFG_PATH = os.environ.get("LISA_TD_CFG", "")
    cfg_in = {}
    if CFG_PATH and os.path.exists(CFG_PATH):
        with open(CFG_PATH, "r") as fh:
            cfg_in = json.load(fh)

    H5 = cfg_in.get("h5", os.environ.get(
        "LISA_H5",
        "/r5/home/rruiz/projects/sbi/lisa/data/LDC2_sangria_training_v2.h5"
    ))

    # ---- main knobs ----
    DATA_MODE = cfg_in.get("data_mode", os.environ.get("LISA_TD_MODE", "synthetic_from_catalogue"))
    KMIN = int(cfg_in.get("K_min", 0))
    KMAX = int(cfg_in.get("Kmax", os.environ.get("LISA_TD_KMAX", "2")))
    USE_GATES = bool(cfg_in.get("use_gates", bool(int(os.environ.get("LISA_TD_GATES", "1")))))
    ORDER_F0 = bool(cfg_in.get("order_f0", bool(int(os.environ.get("LISA_TD_ORDER_F0", "0")))))

    # band
    HALF_BINS = int(cfg_in.get("half_bins", os.environ.get("LISA_HALFBINS", "300")))
    DECIM = int(cfg_in.get("decim", os.environ.get("LISA_DECIM", "1")))
    F_MIN = cfg_in.get("f_min", None)
    F_MAX = cfg_in.get("f_max", None)
    if F_MIN is None:
        F_MIN = os.environ.get("LISA_FMIN", "")
        F_MIN = float(F_MIN) if F_MIN else None
    if F_MAX is None:
        F_MAX = os.environ.get("LISA_FMAX", "")
        F_MAX = float(F_MAX) if F_MAX else None

    # noise
    WITH_NOISE = bool(cfg_in.get("with_noise", bool(int(os.environ.get("LISA_WITH_NOISE", "0")))))
    NOISE_SUBTRACT_VGB = bool(cfg_in.get("noise_subtract_vgb", bool(int(os.environ.get("LISA_NOISE_SUB_VGB", "1")))))

    # catalogue selection
    CAT_IDXS = cfg_in.get("cat_idxs", os.environ.get("LISA_TD_CATIDXS", "0,35"))
    K_TRUE = int(cfg_in.get("K_true", os.environ.get("LISA_TD_KTRUE", "2")))
    SEED = int(cfg_in.get("seed", os.environ.get("LISA_TD_SEED", "0")))
    REF_CATIDX = int(cfg_in.get("ref_cat_idx", os.environ.get("LISA_TD_REF_CATIDX", "0")))
    FLIP_PHI0 = bool(cfg_in.get("flip_phi0", bool(int(os.environ.get("LISA_FLIP_PHI0", "0")))))

    # GBJAX / noise model
    PRESET = getattr(NoisePreset, cfg_in.get("noise_preset", "MRDv1"))
    TGEN = getattr(TDIGeneration, cfg_in.get("tdi_generation", "TDI1"))
    USE_TDI2 = bool(cfg_in.get("use_tdi2", bool(int(os.environ.get("LISA_TDI2", "0")))))

    # prior knobs
    lnA_mu = float(cfg_in.get("lnA_mu", -45.0))
    lnA_sigma = float(cfg_in.get("lnA_sigma", 6.0))
    fdot_sigma = float(cfg_in.get("fdot_sigma", 1e-12))
    gate_mu = float(cfg_in.get("gate_mu", -6.0))
    gate_sigma = float(cfg_in.get("gate_sigma", 2.0))

    # --------- build data (dA,dE) and cfg --------- #

    if DATA_MODE == "synthetic_from_catalogue":
        # parse cat_idxs
        cat_idxs = [int(x.strip()) for x in CAT_IDXS.split(",") if x.strip() != ""]
        if len(cat_idxs) == 0:
            if F_MIN is None or F_MAX is None:
                raise ValueError("synthetic_from_catalogue without cat_idxs requires f_min/f_max to pick sources.")
            cat_idxs = pick_sources_in_band_from_catalogue(H5, K_TRUE, float(F_MIN), float(F_MAX), seed=SEED)

        rows = load_catalogue_rows(H5, cat_idxs)
        theta_true = gbjax_params_from_rows(rows, flip_phi0=FLIP_PHI0)
        f0_refs = [float(x) for x in np.asarray(theta_true.f0)]

        # Use Sangria TOTAL VGB time grid for consistent FFT binning
        t_sig, _, _, _ = load_total_vgb_xyz(H5)
        t_sig = np.asarray(t_sig).reshape(-1)

        # dummy zeros to define bin edges
        X0 = np.zeros_like(t_sig)
        Y0 = np.zeros_like(t_sig)
        Z0 = np.zeros_like(t_sig)

        _, dt_dec, Tobs_dec, f_band, _, _, df_full, i1, i2 = build_band_from_time_series(
            t=t_sig, X=X0, Y=Y0, Z=Z0,
            decim=DECIM,
            f_min_user=F_MIN,
            f_max_user=F_MAX,
            half_bins=HALF_BINS,
            f0_refs=f0_refs,
        )

        # choose n_f_bins from truth (stable)
        n_f_bins_arr = estimate_n_f_bins(theta_true.amp, theta_true.f0, Tobs_dec)
        n_f_bins = int(np.max(np.asarray(n_f_bins_arr)))
        n_f_bins = max(n_f_bins, 256)

        f_min_cfg = float(i1 * df_full)
        f_max_cfg = float(np.nextafter((i2 + 1) * df_full, np.inf))

        cfg = GBJAXConfig(
            t_obs=Tobs_dec, dt=dt_dec, n_f_bins=n_f_bins,
            tdi2=USE_TDI2, f_min=f_min_cfg, f_max=f_max_cfg
        )
        assert cfg.length == f_band.size, (cfg.length, f_band.size)

        sim = build_simulator(cfg)

        # synthetic signal in frequency domain (already narrow-band)
        xyz_syn = sim(theta_true)
        aet_syn = xyz_to_aet(xyz_syn)
        A_syn, E_syn = (np.asarray(jnp.squeeze(ch)) for ch in aet_syn[:-1])
        A_syn = A_syn[:f_band.size]
        E_syn = E_syn[:f_band.size]

        # add instrument noise in frequency domain by FFTing extracted noise time series
        if WITH_NOISE:
            t_n, nX, nY, nZ = load_instrument_noise_xyz(H5, subtract_vgb=NOISE_SUBTRACT_VGB)
            if not grids_compatible(t_sig[::DECIM], t_n[::DECIM]):
                raise RuntimeError("Noise time grid mismatch after decimation.")
            q = int(DECIM)
            nX = np.asarray(nX)[::q]; nY = np.asarray(nY)[::q]; nZ = np.asarray(nZ)[::q]
            Xf = np.fft.rfft(nX) * dt_dec
            Yf = np.fft.rfft(nY) * dt_dec
            Zf = np.fft.rfft(nZ) * dt_dec
            Af = (Zf - Xf) / np.sqrt(2.0)
            Ef = (Xf - 2.0 * Yf + Zf) / np.sqrt(6.0)
            idx = np.arange(i1, i2 + 1, dtype=np.int64)
            dA = A_syn + Af[idx]
            dE = E_syn + Ef[idx]
        else:
            dA, dE = A_syn, E_syn

        SA, SE = psd_AE(f_band, preset=PRESET, gen=TGEN)
        df_band = float(f_band[1] - f_band[0])

    elif DATA_MODE == "sangria_total_band":
        t_sig, X_sig, Y_sig, Z_sig = load_total_vgb_xyz(H5)

        if WITH_NOISE:
            t_n, nX, nY, nZ = load_instrument_noise_xyz(H5, subtract_vgb=NOISE_SUBTRACT_VGB)
            if not grids_compatible(t_sig, t_n):
                raise RuntimeError("Time grids for signal and noise do not match.")
            X_dat = X_sig + nX
            Y_dat = Y_sig + nY
            Z_dat = Z_sig + nZ
        else:
            X_dat, Y_dat, Z_dat = X_sig, Y_sig, Z_sig

        # band center if not explicit
        if F_MIN is None or F_MAX is None:
            row = load_catalogue_rows(H5, [REF_CATIDX])[0]
            f0_refs = [float(sc(row, "Frequency"))]
        else:
            f0_refs = None

        t_dec, dt_dec, Tobs_dec, f_band, dA, dE, df_full, i1, i2 = build_band_from_time_series(
            t=t_sig, X=X_dat, Y=Y_dat, Z=Z_dat,
            decim=DECIM,
            f_min_user=F_MIN,
            f_max_user=F_MAX,
            half_bins=HALF_BINS,
            f0_refs=f0_refs,
        )

        SA, SE = psd_AE(f_band, preset=PRESET, gen=TGEN)
        df_band = float(f_band[1] - f_band[0])

        # n_f_bins heuristic (we don't know truth here)
        n_f_bins = max(256, int(2 * f_band.size))

        f_min_cfg = float(i1 * df_full)
        f_max_cfg = float(np.nextafter((i2 + 1) * df_full, np.inf))

        cfg = GBJAXConfig(
            t_obs=Tobs_dec, dt=dt_dec, n_f_bins=n_f_bins,
            tdi2=USE_TDI2, f_min=f_min_cfg, f_max=f_max_cfg
        )
        assert cfg.length == f_band.size, (cfg.length, f_band.size)

        sim = build_simulator(cfg)

    else:
        raise ValueError(f"Unknown DATA_MODE={DATA_MODE}")

    # --------- build context + Problem --------- #
    ctx = _Ctx(
        sim=sim,
        f_band=f_band,
        dA=dA,
        dE=dE,
        SA=SA,
        SE=SE,
        df_band=df_band,
        kmin=KMIN,
        Kmax=KMAX,
        use_gates=USE_GATES,
        order_f0=ORDER_F0,
        f_min_cfg=f_min_cfg,
        f_max_cfg=f_max_cfg,
        lnA_mu=lnA_mu,
        lnA_sigma=lnA_sigma,
        fdot_sigma=fdot_sigma,
        gate_mu=gate_mu,
        gate_sigma=gate_sigma,
    )

    # expose ctx in sample_prior (slightly hacky but convenient)
    global ctx  # noqa: PLW0603
    return LISATDGBProblem(ctx)
