# lisa_gb_transdim_problem.py
# ------------------------------------------------------------
# Transdimensional narrow-band LISA GB backend for jax_samplers
#   - K is inferred via continuous gates p_k in [0,1]
#   - Flat (box) priors on all *physical* parameters (including p_k)
#   - Analytic profiling/integration over per-source complex coeffs
#       c_k = A_k e^{i phi_k}  (optional; default ON)
#
# Parameterization (default, marg_Aphi=True):
#   For k=1..Kmax:
#     f0_k   in [f_min, f_max]
#     fdot_k in [fdot_min, fdot_max]
#     iota_k in [-pi/2, +pi/2]
#     psi_k  in [0, 2pi]
#     lam_k  in [0, 2pi]
#     beta_k in [-pi/2, +pi/2]
#     p_k    in [0,1]   (gate amplitude scale, also used to infer K)
#
# K inference:
#   define K_eff(theta) = sum_k [p_k > p_active_min]
#   and report posterior for K_eff from posterior samples.
#
# Notes:
# - You MUST choose a band [f_min, f_max] (Hz) in a JSON file (see bottom).
# - Keep bands narrow enough that only a few sources are expected;
#   otherwise dimensionality explodes (Kmax * 7 parameters).
#
# ------------------------------------------------------------

from __future__ import annotations
import os, json
import h5py
import numpy as np

from ..core.precision import DTYPE as _DTYPE, CDTYPE as _CDTYPE, EPS as _EPS

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from jaxlisa.noise import AnalyticNoiseModel, NoisePreset
from jaxlisa.tdi_transfer import TDIChannel, TDIGeneration
from jaxlisa.tdi import TDIXYZ

from jax_samplers.core.problem import Problem

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
LOG2PI = jnp.log(2.0 * jnp.pi)
 
def jnp_real(x): return jnp.asarray(x, DTYPE).real
def jnp_cplx(x): return jnp.asarray(x, CDTYPE)


# import *the same helpers you already have* (or move them into a utils module)
# - load_total_vgb_xyz, load_instrument_noise_xyz
# - load_catalogue_rows (Sangria uses /sky/vgb/cat)
# - gbjax_params_from_rows
# - build_band_from_time_series
# - psd_AE
# - build_simulators
# - make_loglike_and_logprior  (your big function)
# - etc.

def _cfg_path():
    # You already use JAX_SAMPLERS_CONFIG; keep fallback to LISA_CFG if you want
    return os.environ.get("JAX_SAMPLERS_CONFIG") or os.environ.get("LISA_CFG") or ""

def _pick(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None or k not in cur:
            return default
        cur = cur[k]
    return cur

def _parse_cat_idxs(x):
    # allow list in JSON, or string "0,35"
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    if isinstance(x, str):
        return [int(v) for v in x.split(",") if v.strip() != ""]
    return [int(x)]

def _set_dtype_from_cfg(cfg):
    # Your CLI already sets env var before importing JAX.
    # This is just to read/print; not to toggle x64 here.
    return _pick(cfg, "data", "dtype", default=None) or cfg.get("dtype", None)

def load_catalogue_rows(h5path: str, idxs: list[int]):
    with h5py.File(h5path, "r") as f:
        cat = f["sky/vgb/cat"][()]
        rows = [cat[i] for i in idxs]
    return rows


# -------------------- small utilities -------------------- #

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

def psd_AE(f_phys, preset=NoisePreset.MRDv1, gen=TDIGeneration.TDI1):
    f_abs = jnp.asarray(np.abs(f_phys))
    nmA = AnalyticNoiseModel.from_preset(preset, TDIChannel.A, gen)
    SA = np.asarray(nmA.psd(f_abs))
    try:
        nmE = AnalyticNoiseModel.from_preset(preset, TDIChannel.E, gen)
        SE = np.asarray(nmE.psd(f_abs))
    except Exception:
        SE = SA.copy()
    if SA.size and SA[0] == 0:
        SA[0] = np.inf
    if SE.size and SE[0] == 0:
        SE[0] = np.inf

    return jnp_real(SA), jnp_real(SE)


# -------------------- Sangria I/O -------------------- #

def read_total_xyz_compound(ds: h5py.Dataset):
    rec = ds[()]
    t = np.asarray(rec["t"]).reshape(-1).astype(np.float64)
    X = np.asarray(rec["X"]).reshape(-1).astype(np.float64)
    Y = np.asarray(rec["Y"]).reshape(-1).astype(np.float64)
    Z = np.asarray(rec["Z"]).reshape(-1).astype(np.float64)
    return t, X, Y, Z

def read_xyz_any(f: h5py.File, base: str):
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

def load_obs_xyz(h5path: str):
    """Load the observed Sangria TDI stream from obs/tdi."""
    with h5py.File(h5path, "r") as f:
        if "obs/tdi" not in f:
            raise ValueError("Missing obs/tdi in file.")
        t, X, Y, Z = read_xyz_any(f, "obs/tdi")
        if t is None:
            raise ValueError("obs/tdi has no time array 't' (unexpected).")
    return t, X, Y, Z


def load_total_vgb_xyz(h5path: str):
    with h5py.File(h5path, "r") as f:
        if "sky/vgb/tdi" not in f:
            raise ValueError("Missing sky/vgb/tdi in file.")
        t, X, Y, Z = read_total_xyz_compound(f["sky/vgb/tdi"])
    return t, X, Y, Z

def load_instrument_noise_xyz(h5path: str, subtract_vgb: bool):
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

def sc(row, key):
    a = np.asarray(row[key])
    return a.ravel()[0].item()

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
        amp=jnp.asarray(amps),
        f0=jnp.asarray(f0s),
        fdot=jnp.asarray(fdots),
        fddot=jnp.zeros((K,)),
        phi0=jnp.asarray(phi0s),
        iota=jnp.asarray(iotas),
        psi=jnp.asarray(psis),
        lam=jnp.asarray(lams),
        beta=jnp.asarray(betas),
    )

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


# -------------------- parameter transforms (unconstrained -> physical) -------------------- #

def sigmoid(x):
    return jax.nn.sigmoid(x)

def wrap_pm_pi(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi

def clamp_iota(x):
    return jnp.clip(x, -jnp.pi/2, jnp.pi/2)

def clamp_beta(x):
    return jnp.clip(x, -jnp.pi/2, jnp.pi/2)

def u_to_phi(x):
    return wrap_pm_pi(x)

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

def unpack_theta_u(theta_u, Kmax, use_gates, marg_Aphi):
    theta_u = jnp.asarray(theta_u).reshape(-1)

    n_phys = 6 if marg_Aphi else 8
    per = n_phys + (1 if use_gates else 0)
    expected = Kmax * per

    if theta_u.size != expected:
        raise ValueError(
            f"size mismatch: got {theta_u.size}, expected {expected} "
            f"(Kmax={Kmax}, n_phys={n_phys}, use_gates={use_gates})"
        )

    th = theta_u.reshape((Kmax, per))

    if marg_Aphi:
        f0_u   = th[:, 0]
        fdot_u = th[:, 1]
        iota_u = th[:, 2]
        psi_u  = th[:, 3]
        lam_u  = th[:, 4]
        beta_u = th[:, 5]
        lnA_u  = None
        phi0_u = None
    else:
        lnA_u  = th[:, 0]
        f0_u   = th[:, 1]
        fdot_u = th[:, 2]
        phi0_u = th[:, 3]
        iota_u = th[:, 4]
        psi_u  = th[:, 5]
        lam_u  = th[:, 6]
        beta_u = th[:, 7]

    g_u = th[:, -1] if use_gates else None
    return lnA_u, f0_u, fdot_u, phi0_u, iota_u, psi_u, lam_u, beta_u, g_u

def theta_u_to_gbjax_pars(theta_u, Kmax, use_gates, marg_Aphi, order_f0, f_min, f_max, df_bin):
    lnA_u, f0_u, fdot_u, phi0_u, iota_u, psi_u, lam_u, beta_u, g_u = \
        unpack_theta_u(theta_u, Kmax, use_gates, marg_Aphi)

    # A/phi handling
    if marg_Aphi:
        amp = jnp.ones((Kmax,), dtype=DTYPE)
        phi0 = jnp.zeros((Kmax,), dtype=DTYPE)
    else:
        amp = jnp.exp(lnA_u)
        phi0 = u_to_phi(phi0_u)

    if use_gates and (not marg_Aphi):
        amp = amp * sigmoid(g_u)

    if order_f0:
        f0 = u_to_f0_ordered(f0_u[0], f0_u[1:], f_min, f_max, df_min=df_bin, guard_bins=2.0)
    else:
        f0 = u_to_f0_unordered(f0_u, f_min, f_max)

    fdot = fdot_u
    iota = clamp_iota(iota_u)
    psi  = wrap_pm_pi(psi_u)
    lam  = wrap_pm_pi(lam_u)
    beta = clamp_beta(beta_u)

    return GBJAXParameters(
        amp=amp, f0=f0, fdot=fdot, fddot=jnp.zeros((Kmax,), dtype=DTYPE),
        phi0=phi0, iota=iota, psi=psi, lam=lam, beta=beta
    )

def theta_u_to_pars_unit_amp_phi0(theta_u, Kmax, use_gates, marg_Aphi, order_f0, f_min, f_max, df_bin):
    """
    Like theta_u_to_gbjax_pars but sets amp=1 (times gate if enabled) and phi0=0.
    Used for marginalization over complex coefficient c_k = A_k e^{i phi_k}.
    """
    lnA_u, f0_u, fdot_u, phi0_u, iota_u, psi_u, lam_u, beta_u, g_u = unpack_theta_u(theta_u, Kmax, use_gates, marg_Aphi)

    amp = jnp.ones((Kmax,), dtype=DTYPE)
    #if use_gates:
    #    amp = amp * sigmoid(g_u)  # gate scales the template itself

    if order_f0:
        f0 = u_to_f0_ordered(f0_u[0], f0_u[1:], f_min, f_max, df_min=df_bin, guard_bins=2.0)
    else:
        f0 = u_to_f0_unordered(f0_u, f_min, f_max)

    fdot = fdot_u
    phi0 = jnp.zeros((Kmax,), dtype=DTYPE)  # fixed template phase
    iota = clamp_iota(iota_u)
    psi  = wrap_pm_pi(psi_u)
    lam  = wrap_pm_pi(lam_u)
    beta = clamp_beta(beta_u)

    return GBJAXParameters(
        amp=amp, f0=f0, fdot=fdot, fddot=jnp.zeros((Kmax,)),
        phi0=phi0, iota=iota, psi=psi, lam=lam, beta=beta
    )


# -------------------- simulator wrappers -------------------- #

def _get_XYZ_fields(xyz):
    if isinstance(xyz, TDIXYZ):
        x = getattr(xyz, "x", None)
        y = getattr(xyz, "y", None)
        z = getattr(xyz, "z", None)
        if x is None:
            x = getattr(xyz, "X")
            y = getattr(xyz, "Y")
            z = getattr(xyz, "Z")
        return x, y, z
    raise TypeError("Simulator did not return TDIXYZ; unsupported for this script.")

def xyz_to_AE_arrays(X, Y, Z):
    """
    Convert XYZ -> A,E in frequency domain using standard linear combos.
    Works for shapes (M,) or (K,M).
    """
    A = (Z - X) / jnp.sqrt(2.0)
    E = (X - 2.0 * Y + Z) / jnp.sqrt(6.0)
    return A, E

def build_simulators(cfg: GBJAXConfig):
    base = make_simulator(cfg)

    def sim_sum(pars: GBJAXParameters):
        xyz = base(pars)
        return xyz

    def sim_per_source(pars: GBJAXParameters):
        """
        Always returns per-source XYZ with fields shape (K, M),
        without dynamic slicing (JAX-safe under jit/vmap).
        """
        def one_source(amp1, f01, fdot1, phi01, iota1, psi1, lam1, beta1):
            p1 = GBJAXParameters(
                amp=amp1[None],
                f0=f01[None],
                fdot=fdot1[None],
                fddot=jnp.zeros((1,), dtype=amp1.dtype),
                phi0=phi01[None],
                iota=iota1[None],
                psi=psi1[None],
                lam=lam1[None],
                beta=beta1[None],
            )
            xyz1 = base(p1)
            X1, Y1, Z1 = _get_XYZ_fields(xyz1)

            # Works whether base returns (M,) or (1,M)
            X1 = jnp.reshape(X1, (-1,))
            Y1 = jnp.reshape(Y1, (-1,))
            Z1 = jnp.reshape(Z1, (-1,))
            return X1, Y1, Z1

        Xk, Yk, Zk = jax.vmap(one_source)(
        pars.amp, pars.f0, pars.fdot, pars.phi0, pars.iota, pars.psi, pars.lam, pars.beta
        )
        return TDIXYZ(Xk, Yk, Zk)
    
    return sim_sum, sim_per_source


# -------------------- likelihoods -------------------- #

def make_loglike_and_logprior(
    sim_sum,
    sim_per_source,
    f_band: np.ndarray,
    dA: np.ndarray,
    dE: np.ndarray,
    SA: np.ndarray,
    SE: np.ndarray,
    df_band: float,
    Kmax: int,
    use_gates: bool,
    order_f0: bool,
    f_min_cfg: float,
    f_max_cfg: float,
    marg_Aphi: bool,
    marg_mode: str,
    prior_box: dict,
    gaussian_priors: dict,
    p_active_min: float,    
    # prior hyperparams
    lnA_mu: float = -45.0,
    lnA_sigma: float = 6.0,
    fdot_sigma: float = 1e-12,
    gate_mu: float = -6.0,
    gate_sigma: float = 2.0,
):
    M = dA.size
    dA_c = jnp.asarray(dA)
    dE_c = jnp.asarray(dE)
    SA_c = jnp.asarray(SA)
    SE_c = jnp.asarray(SE)
    df_c = float(df_band)
    fmin = float(f_min_cfg)
    fmax = float(f_max_cfg)
    df_bin = float(df_band)

    def ip_chan(x, y, S):
        return 4.0 * df_c * jnp.vdot(x, y / (S + EPS))

    def ip_twochan(Ax, Ex, Ay, Ey):
        return ip_chan(Ax, Ay, SA_c) + ip_chan(Ex, Ey, SE_c)

    @jax.jit
    def loglike_plain(theta_u):

        #pars = theta_u_to_pars_unit_amp_phi0(
        #    theta_u, Kmax=Kmax, use_gates=use_gates, marg_Aphi=marg_Aphi,
        #    order_f0=order_f0, f_min=fmin, f_max=fmax, df_bin=df_bin
        #)

        pars = theta_u_to_gbjax_pars(
            theta_u, Kmax=Kmax, use_gates=use_gates, marg_Aphi=marg_Aphi,
            order_f0=order_f0, f_min=fmin, f_max=fmax, df_bin=df_bin
        )
        
        xyz = sim_sum(pars)
        X, Y, Z = _get_XYZ_fields(xyz)
        A_sim, E_sim = xyz_to_AE_arrays(X, Y, Z)

        A_sim = A_sim[:M]
        E_sim = E_sim[:M]
        rA = dA_c - A_sim
        rE = dE_c - E_sim
        chi2 = ip_chan(rA, rA, SA_c) + ip_chan(rE, rE, SE_c)
        return -0.5 * jnp.real(chi2)

    @jax.jit
    def loglike_marg(theta_u):
        """
        Profile or integrate over complex coefficients c_k = A_k e^{i phi_k}
        using per-source templates with amp=1, phi0=0.

        Variant B:
        - gates scale templates continuously
        - no hard p_active_min threshold in the likelihood
        - no renormalization that washes out the gate amplitude effect
        - only numerical protection for nearly-zero templates
        """

        pars0 = theta_u_to_pars_unit_amp_phi0(
            theta_u,
            Kmax=Kmax,
            use_gates=use_gates,
            marg_Aphi=marg_Aphi,
            order_f0=order_f0,
            f_min=fmin,
            f_max=fmax,
            df_bin=df_bin,
        )

        xyz_k = sim_per_source(pars0)   # TDIXYZ with fields (K,M)
        Xk, Yk, Zk = _get_XYZ_fields(xyz_k)
        Ak, Ek = xyz_to_AE_arrays(Xk, Yk, Zk)

        Ak = Ak[:, :M]
        Ek = Ek[:, :M]

        """
        # Continuous gate scaling
        if use_gates:
            _, _, _, _, _, _, _, _, g_u = unpack_theta_u(
                theta_u, Kmax, use_gates, marg_Aphi
            )
            gate = jax.nn.sigmoid(g_u).astype(DTYPE)   # shape (K,)
            gate = jnp.clip(gate, 1e-3, 1.0)
        else:
            gate = jnp.ones((Kmax,), dtype=DTYPE)

        Ak = gate[:, None] * Ak
        Ek = gate[:, None] * Ek
        """
        
        if use_gates:
            _, _, _, _, _, _, _, _, g_u = unpack_theta_u(
                theta_u, Kmax, use_gates, marg_Aphi
            )
            gate = jax.nn.sigmoid(g_u).astype(DTYPE)
        else:
            gate = jnp.ones((Kmax,), dtype=DTYPE)

        # diagnostic: do not scale templates by gate in integrate mode
        if marg_mode != "integrate":
            Ak = gate[:, None] * Ak
            Ek = gate[:, None] * Ek
        
        # Data norm
        dd = ip_twochan(dA_c, dE_c, dA_c, dE_c)

        # Template norms, only for numerical checks
        nk2 = jax.vmap(
            lambda hA, hE: jnp.real(ip_twochan(hA, hE, hA, hE))
        )(Ak, Ek)

        # Numerical "alive" mask: only kill truly negligible templates
        # This is NOT a model-selection threshold.
        alive = (nk2 > 1e-30).astype(DTYPE)

        # b_k = (h_k|d)
        b = jax.vmap(
            lambda hA, hE: ip_twochan(hA, hE, dA_c, dE_c)
        )(Ak, Ek)
        b = b * alive

        # M_ij = (h_i|h_j)
        def M_row(hAi, hEi):
            return jax.vmap(
                lambda hAj, hEj: ip_twochan(hAi, hEi, hAj, hEj)
            )(Ak, Ek)

        Mmat = jax.vmap(M_row)(Ak, Ek)
        Mmat = 0.5 * (Mmat + jnp.conjugate(jnp.swapaxes(Mmat, 0, 1)))

        # Zero rows/cols only for numerically dead templates
        Mmat = Mmat * (alive[:, None] * alive[None, :])

        # Small diagonal jitter for stability
        K = Mmat.shape[0]
        diag = jnp.real(jnp.diag(Mmat))
        scale = jnp.maximum(jnp.max(diag), 1.0)
        
        if marg_mode == "integrate":
            jitter = jnp.asarray(1e-8, dtype=DTYPE) * scale
        else:
            jitter = jnp.asarray(1e-10, dtype=DTYPE) * scale
        
        Mmat_j = Mmat + jitter * jnp.eye(K, dtype=Mmat.dtype)

        # Cholesky solve
        L = jnp.linalg.cholesky(Mmat_j)
        y = jax.scipy.linalg.solve_triangular(L, b, lower=True)
        x = jax.scipy.linalg.solve_triangular(jnp.conjugate(L.T), y, lower=False)

        quad = jnp.vdot(b, x)
        ll = -0.5 * jnp.real(dd - quad)

        if marg_mode == "integrate":
            logdet = 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diag(L)) + EPS))
            ll = ll - 0.5 * logdet   # constants dropped

        return ll
    
    @jax.jit
    def loglike_marg_old(theta_u):
        """
        Profile (or integrate) over complex coefficients c_k = A_k e^{i phi_k}
        using per-source templates with amp=1, phi0=0 (and optional gate scaling).

        Robust for Kmax~5 with gates:
        - normalize templates so Gram matrix is O(1)
        - smoothly deactivate near-zero (gated-out) templates
        - use small relative jitter
        """

        pars0 = theta_u_to_pars_unit_amp_phi0(
            theta_u, Kmax=Kmax, use_gates=use_gates, marg_Aphi=marg_Aphi, order_f0=order_f0,
            f_min=fmin, f_max=fmax, df_bin=df_bin
        )

        """
        # ---- run at truth ----
        pars0 = GBJAXParameters(
            amp=jnp.array([0.9999, 0.99999], dtype=jnp.float64),
            f0=jnp.array([0.00181373, 0.00184213], dtype=jnp.float64),
            fdot=jnp.array([2.52496382e-18, 3.85613147e-18], dtype=jnp.float64),
            fddot=jnp.array([0.0, 0.0], dtype=jnp.float64),
            phi0=jnp.array([0., 0.], dtype=jnp.float64),
            iota=jnp.array([0.52359878, 0.26179939], dtype=jnp.float64),
            psi=jnp.array([6.06022346, 3.74191773], dtype=jnp.float64),
            lam=jnp.array([4.1029979, 5.204804 ], dtype=jnp.float64),
            beta=jnp.array([0.08656634, 1.07257767], dtype=jnp.float64),
)
        """
        xyz_k = sim_per_source(pars0)  # TDIXYZ with fields (K,M)

        Xk, Yk, Zk = _get_XYZ_fields(xyz_k)
        Ak, Ek = xyz_to_AE_arrays(Xk, Yk, Zk)

        Ak = Ak[:, :M]
        Ek = Ek[:, :M]
        
        dd = ip_twochan(dA_c, dE_c, dA_c, dE_c)

        # --- template norms for normalization ---
        nk2 = jax.vmap(lambda hA, hE: jnp.real(ip_twochan(hA, hE, hA, hE)))(Ak, Ek)

        # floor avoids division by 0 when a gate kills a component
        nk2_floor = 1e-60
        nk = jnp.sqrt(nk2 + nk2_floor)

        AkN = Ak / nk[:, None]
        EkN = Ek / nk[:, None]

        # active mask: 1 for templates with non-negligible norm, else 0
        # (threshold can be tuned; this is conservative for float64)
        # --- gate-based activity mask (recommended when use_gates=True) ---
        # unpack returns g_u only when use_gates=True
        if use_gates:
            _, _, _, _, _, _, _, _, g_u = unpack_theta_u(theta_u, Kmax, use_gates, marg_Aphi)
            gate = jax.nn.sigmoid(g_u)
            #active = (gate > 1e-3).astype(AkN.dtype)
            active = (gate > p_active_min).astype(AkN.dtype)
        else:
            active = jnp.ones((Kmax,), dtype=AkN.dtype)
        
        #active = (gate > 1e-3).astype(AkN.dtype)

        #active = (nk2 > 1e-40).astype(AkN.dtype)

        # b_k = (h_k|d) in normalized basis
        b = jax.vmap(lambda hA, hE: ip_twochan(hA, hE, dA_c, dE_c))(AkN, EkN)
        b = b * active  # deactivate near-zero templates

        # M_ij = (h_i|h_j) in normalized basis
        def M_row(hAi, hEi):
            return jax.vmap(lambda hAj, hEj: ip_twochan(hAi, hEi, hAj, hEj))(AkN, EkN)

        Mmat = jax.vmap(M_row)(AkN, EkN)
        Mmat = 0.5 * (Mmat + jnp.conjugate(jnp.swapaxes(Mmat, 0, 1)))

        # deactivate rows/cols for inactive templates
        Mmat = Mmat * (active[:, None] * active[None, :])

        # --- small relative jitter (now M is O(1)) ---
        K = Mmat.shape[0]
        diag = jnp.real(jnp.diag(Mmat))
        jitter = (1e-10 * (jnp.max(diag) + 1.0)).astype(Mmat.dtype)
        Mmat_j = Mmat + jitter * jnp.eye(K, dtype=Mmat.dtype)

        # Cholesky solve
        L = jnp.linalg.cholesky(Mmat_j)
        y = jax.scipy.linalg.solve_triangular(L, b, lower=True)
        xN = jax.scipy.linalg.solve_triangular(jnp.conjugate(L.T), y, lower=False)

        quad = jnp.vdot(b, xN)  # b^H M^{-1} b (normalized basis, invariant)
        ll = -0.5 * jnp.real(dd - quad)
        
        #print(M, df_band, dd)
        #sys.exit()
        if marg_mode == "integrate":
            # logdet(M) in normalized basis
            logdet = 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diag(L)) + EPS))
            ll = ll - 0.5 * logdet  # constants dropped

        #print('ll', ll)
        
        return ll

    @jax.jit
    def marg_diagnostics(theta_u):
        pars0 = theta_u_to_pars_unit_amp_phi0(
            theta_u,
            Kmax=Kmax,
            use_gates=use_gates,
            marg_Aphi=marg_Aphi,
            order_f0=order_f0,
            f_min=fmin,
            f_max=fmax,
            df_bin=df_bin,
        )

        xyz_k = sim_per_source(pars0)
        Xk, Yk, Zk = _get_XYZ_fields(xyz_k)
        Ak, Ek = xyz_to_AE_arrays(Xk, Yk, Zk)

        Ak = Ak[:, :M]
        Ek = Ek[:, :M]

        if use_gates:
            _, _, _, _, _, _, _, _, g_u = unpack_theta_u(
                theta_u, Kmax, use_gates, marg_Aphi
            )
            gate = jax.nn.sigmoid(g_u).astype(DTYPE)
        else:
            gate = jnp.ones((Kmax,), dtype=DTYPE)

        Ak = gate[:, None] * Ak
        Ek = gate[:, None] * Ek

        dd = ip_twochan(dA_c, dE_c, dA_c, dE_c)

        nk2 = jax.vmap(
            lambda hA, hE: jnp.real(ip_twochan(hA, hE, hA, hE))
        )(Ak, Ek)
        alive = (nk2 > 1e-40).astype(DTYPE)

        b = jax.vmap(
            lambda hA, hE: ip_twochan(hA, hE, dA_c, dE_c)
        )(Ak, Ek)
        b = b * alive

        def M_row(hAi, hEi):
            return jax.vmap(
                lambda hAj, hEj: ip_twochan(hAi, hEi, hAj, hEj)
            )(Ak, Ek)

        Mmat = jax.vmap(M_row)(Ak, Ek)
        Mmat = 0.5 * (Mmat + jnp.conjugate(jnp.swapaxes(Mmat, 0, 1)))
        Mmat = Mmat * (alive[:, None] * alive[None, :])

        K = Mmat.shape[0]
        diag = jnp.real(jnp.diag(Mmat))
        scale = jnp.maximum(jnp.max(diag), 1.0)
        jitter = jnp.asarray(1e-12, dtype=DTYPE) * scale

        Mmat_j = Mmat + jitter * jnp.eye(K, dtype=Mmat.dtype)
        L = jnp.linalg.cholesky(Mmat_j)
        y = jax.scipy.linalg.solve_triangular(L, b, lower=True)
        x = jax.scipy.linalg.solve_triangular(jnp.conjugate(L.T), y, lower=False)

        quad = jnp.real(jnp.vdot(b, x))
        logdet = 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diag(L)) + EPS))

        return (
            jnp.real(dd),
            quad,
            jnp.real(dd - quad),
            logdet,
            scale,
            jitter,
            jnp.min(diag),
            jnp.max(diag),
        )

    @jax.jit
    def marg_diagnostics_old(theta_u):
        pars0 = theta_u_to_pars_unit_amp_phi0(
            theta_u, Kmax=Kmax, use_gates=use_gates, marg_Aphi=marg_Aphi, order_f0=order_f0,
            f_min=fmin, f_max=fmax, df_bin=df_bin
        )

        xyz_k = sim_per_source(pars0)
        Xk, Yk, Zk = _get_XYZ_fields(xyz_k)
        Ak, Ek = xyz_to_AE_arrays(Xk, Yk, Zk)
        Ak = Ak[:, :M]
        Ek = Ek[:, :M]

        dd = ip_twochan(dA_c, dE_c, dA_c, dE_c)

        nk2 = jax.vmap(lambda hA, hE: jnp.real(ip_twochan(hA, hE, hA, hE)))(Ak, Ek)
        nk = jnp.sqrt(nk2 + 1e-60)

        AkN = Ak / nk[:, None]
        EkN = Ek / nk[:, None]

        #active = (nk2 > 1e-40).astype(AkN.dtype)    

        if use_gates:
            _, _, _, _, _, _, _, _, g_u = unpack_theta_u(theta_u, Kmax, use_gates, marg_Aphi)
            gate = jax.nn.sigmoid(g_u)
            #active = (gate > 1e-3).astype(AkN.dtype)
            active = (gate > p_active_min).astype(AkN.dtype)
        else:
            active = jnp.ones((Kmax,), dtype=AkN.dtype)
            
        b = jax.vmap(lambda hA, hE: ip_twochan(hA, hE, dA_c, dE_c))(AkN, EkN)
        b = b * active

        def M_row(hAi, hEi):
            return jax.vmap(lambda hAj, hEj: ip_twochan(hAi, hEi, hAj, hEj))(AkN, EkN)

        Mmat = jax.vmap(M_row)(AkN, EkN)
        Mmat = 0.5 * (Mmat + jnp.conjugate(jnp.swapaxes(Mmat, 0, 1)))
        Mmat = Mmat * (active[:, None] * active[None, :])

        K = Mmat.shape[0]
        diag = jnp.real(jnp.diag(Mmat))
        scale = jnp.real(jnp.trace(Mmat)) / jnp.maximum(K, 1)
        jitter = 1e-10 * (jnp.max(diag) + 1.0)

        Mmat_j = Mmat + jitter * jnp.eye(K, dtype=Mmat.dtype)
        L = jnp.linalg.cholesky(Mmat_j)
        y = jax.scipy.linalg.solve_triangular(L, b, lower=True)
        xN = jax.scipy.linalg.solve_triangular(jnp.conjugate(L.T), y, lower=False)

        quad = jnp.real(jnp.vdot(b, xN))
        logdet = 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diag(L)) + EPS))

        return jnp.real(dd), quad, jnp.real(dd - quad), logdet, scale, jitter, jnp.min(diag), jnp.max(diag)

    
    @jax.jit
    def loglike_old(theta_u):
        return jax.lax.cond(
            jnp.asarray(marg_Aphi),
            lambda th: loglike_marg(th),
            lambda th: loglike_plain(th),
            theta_u,
        )

    @jax.jit
    def loglike(theta_u):
        if marg_Aphi:
            return loglike_marg(theta_u)
        else:
            return loglike_plain(theta_u)
    
    def _gauss_logpdf(x, mu, sig):
        z = (x - mu) / sig
        return -0.5 * z * z - jnp.log(sig) - 0.5 * LOG2PI

    @jax.jit
    def logprior_old(theta_u):
        # unpack unconstrained coords
        lnA_u, f0_u, fdot_u, phi0_u, iota_u, psi_u, lam_u, beta_u, g_u = \
            unpack_theta_u(theta_u, Kmax, use_gates, marg_Aphi)

        # ---------- map to physical variables ----------
        if order_f0:
            f0 = u_to_f0_ordered(
                f0_u[0], f0_u[1:], f_min_cfg, f_max_cfg, df_min=df_bin, guard_bins=2.0
            )
        else:
            f0 = u_to_f0_unordered(f0_u, f_min_cfg, f_max_cfg)

        if order_f0 and Kmax > 1:
            df_sep_min = 5.0 * df_bin   # start with 5 bins; test 3, 5, 10
            sep_ok = jnp.all(jnp.diff(f0) > df_sep_min)
            all_inside = all_inside & sep_ok
            
        fdot = fdot_u
        iota = clamp_iota(iota_u)
        psi  = wrap_pm_pi(psi_u)
        lam  = wrap_pm_pi(lam_u)
        beta = clamp_beta(beta_u)

        if not marg_Aphi:
            lnA  = lnA_u
            phi0 = u_to_phi(phi0_u)

        lp = jnp.array(0.0, dtype=theta_u.dtype)
        all_inside = jnp.array(True)

        def add_param_prior(vals, name, lo, hi):
            inside = jnp.all((vals >= lo) & (vals <= hi))
            if name in gaussian_priors:
                mu, sig = gaussian_priors[name]
                mu = jnp.asarray(mu, dtype=theta_u.dtype)
                sig = jnp.asarray(sig, dtype=theta_u.dtype)
                lp_local = jnp.sum(_gauss_logpdf(vals, mu, sig))
            else:
                width = jnp.asarray(hi - lo, dtype=theta_u.dtype)
                lp_local = -vals.size * jnp.log(width)
            return inside, lp_local

        # ---- physical priors ----
        i, add = add_param_prior(f0,   "f0",   prior_box["f0"][0],   prior_box["f0"][1]);   all_inside &= i; lp += add
        i, add = add_param_prior(fdot, "fdot", prior_box["fdot"][0], prior_box["fdot"][1]); all_inside &= i; lp += add
        i, add = add_param_prior(iota, "iota", prior_box["iota"][0], prior_box["iota"][1]); all_inside &= i; lp += add
        i, add = add_param_prior(psi,  "psi",  prior_box["psi"][0],  prior_box["psi"][1]);  all_inside &= i; lp += add
        i, add = add_param_prior(lam,  "lam",  prior_box["lam"][0],  prior_box["lam"][1]);  all_inside &= i; lp += add
        i, add = add_param_prior(beta, "beta", prior_box["beta"][0], prior_box["beta"][1]); all_inside &= i; lp += add

        if not marg_Aphi:
            i, add = add_param_prior(lnA,  "lnA",  prior_box["lnA"][0],  prior_box["lnA"][1]);  all_inside &= i; lp += add
            i, add = add_param_prior(phi0, "phi0", prior_box["phi0"][0], prior_box["phi0"][1]); all_inside &= i; lp += add

        # ---- gates ----
        if use_gates:
            if "g" in gaussian_priors:
                mu, sig = gaussian_priors["g"]
                mu = jnp.asarray(mu, dtype=theta_u.dtype)
                sig = jnp.asarray(sig, dtype=theta_u.dtype)
                lp += jnp.sum(_gauss_logpdf(g_u, mu, sig))
            else:
                # p ~ Uniform(0,1), with g_u = logit(p): add log|dp/dg| = log(p(1-p))
                p = jax.nn.sigmoid(g_u)
                lp += jnp.sum(jnp.log(p + EPS) + jnp.log1p(-p + EPS))

        return jnp.where(all_inside, lp, -jnp.inf)

    @jax.jit
    def logprior(theta_u):
        # unpack unconstrained coords
        lnA_u, f0_u, fdot_u, phi0_u, iota_u, psi_u, lam_u, beta_u, g_u = \
            unpack_theta_u(theta_u, Kmax, use_gates, marg_Aphi)

        # ---------- map to physical variables ----------
        if order_f0:
            f0 = u_to_f0_ordered(
                f0_u[0], f0_u[1:], f_min_cfg, f_max_cfg, df_min=df_bin, guard_bins=2.0
            )
        else:
            f0 = u_to_f0_unordered(f0_u, f_min_cfg, f_max_cfg)

        fdot = fdot_u
        iota = clamp_iota(iota_u)
        psi  = wrap_pm_pi(psi_u)
        lam  = wrap_pm_pi(lam_u)
        beta = clamp_beta(beta_u)

        if not marg_Aphi:
            lnA  = lnA_u
            phi0 = u_to_phi(phi0_u)

        lp = jnp.array(0.0, dtype=theta_u.dtype)
        all_inside = jnp.array(True)

        def add_param_prior(vals, name, lo, hi):
            inside = jnp.all((vals >= lo) & (vals <= hi))
            if name in gaussian_priors:
                mu, sig = gaussian_priors[name]
                mu = jnp.asarray(mu, dtype=theta_u.dtype)
                sig = jnp.asarray(sig, dtype=theta_u.dtype)
                lp_local = jnp.sum(_gauss_logpdf(vals, mu, sig))
            else:
                width = jnp.asarray(hi - lo, dtype=theta_u.dtype)
                lp_local = -vals.size * jnp.log(width)
            return inside, lp_local

        # ---- physical priors ----
        i, add = add_param_prior(f0,   "f0",   prior_box["f0"][0],   prior_box["f0"][1]);   all_inside &= i; lp += add
        i, add = add_param_prior(fdot, "fdot", prior_box["fdot"][0], prior_box["fdot"][1]); all_inside &= i; lp += add
        i, add = add_param_prior(iota, "iota", prior_box["iota"][0], prior_box["iota"][1]); all_inside &= i; lp += add
        i, add = add_param_prior(psi,  "psi",  prior_box["psi"][0],  prior_box["psi"][1]);  all_inside &= i; lp += add
        i, add = add_param_prior(lam,  "lam",  prior_box["lam"][0],  prior_box["lam"][1]);  all_inside &= i; lp += add
        i, add = add_param_prior(beta, "beta", prior_box["beta"][0], prior_box["beta"][1]); all_inside &= i; lp += add

        if not marg_Aphi:
            i, add = add_param_prior(lnA,  "lnA",  prior_box["lnA"][0],  prior_box["lnA"][1]);  all_inside &= i; lp += add
            i, add = add_param_prior(phi0, "phi0", prior_box["phi0"][0], prior_box["phi0"][1]); all_inside &= i; lp += add

        # ---- minimum separation prior ----
        if order_f0 and Kmax > 1:
            df_sep_min = 3.0 * df_bin
            sep_ok = jnp.all(jnp.diff(f0) > df_sep_min)
            all_inside = all_inside & sep_ok

        # ---- gates ----
        if use_gates:
            if "g" in gaussian_priors:
                mu, sig = gaussian_priors["g"]
                mu = jnp.asarray(mu, dtype=theta_u.dtype)
                sig = jnp.asarray(sig, dtype=theta_u.dtype)
                lp += jnp.sum(_gauss_logpdf(g_u, mu, sig))
            else:
                p = jax.nn.sigmoid(g_u)
                lp += jnp.sum(jnp.log(p + EPS) + jnp.log1p(-p + EPS))

        return jnp.where(all_inside, lp, -jnp.inf)
    
    @jax.jit
    def logpost(theta_u):
        return loglike(theta_u) + logprior(theta_u)

    @jax.jit
    def template_diagnostics(theta_u):
        # build the full waveform h(theta)
        pars = theta_u_to_gbjax_pars(
            theta_u, Kmax=Kmax, use_gates=use_gates, marg_Aphi=marg_Aphi,
            order_f0=order_f0, f_min=fmin, f_max=fmax, df_bin=df_bin, 
        )
        xyz = sim_sum(pars)
        X, Y, Z = _get_XYZ_fields(xyz)
        A_sim, E_sim = xyz_to_AE_arrays(X, Y, Z)

        A_sim = A_sim[:M]
        E_sim = E_sim[:M]

        # inner products
        dd = ip_twochan(dA_c, dE_c, dA_c, dE_c)         # (d|d)
        hh = ip_twochan(A_sim, E_sim, A_sim, E_sim)     # (h|h)
        dh = ip_twochan(dA_c, dE_c, A_sim, E_sim)       # (d|h)

        chi2 = dd - 2.0 * dh + hh
        ll = -0.5 * jnp.real(chi2)
        return ll, jnp.real(dd), jnp.real(hh), jnp.real(dh)

    return loglike, logprior, logpost, template_diagnostics, marg_diagnostics

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
        if not (fmax > fmin):
            raise ValueError("Require f_max > f_min.")
        i1 = int(np.ceil(fmin / df_full))
        i2 = int(np.floor(np.nextafter(fmax, -np.inf) / df_full))
    else:
        if f0_refs is not None and len(f0_refs) > 0:
            f0_min = float(np.min(f0_refs))
            f0_max = float(np.max(f0_refs))
            k1 = int(round(f0_min * Tobs))
            k2 = int(round(f0_max * Tobs))
            i1 = k1 - half_bins
            i2 = k2 + half_bins
        else:
            k0 = int(round(0.5 * freqs[-1] * Tobs))
            i1 = k0 - half_bins
            i2 = k0 + half_bins

    i1 = max(1, i1)
    i2 = min(freqs.size - 2, i2)
    if i2 < i1:
        raise RuntimeError("Empty band after clipping.")

    idx = np.arange(i1, i2 + 1, dtype=np.int64)
    f_band = freqs[idx]
    dA = Af[idx]
    dE = Ef[idx]

    return t, dt, Tobs, freqs, f_band, dA, dE, df_full, i1, i2



class LISAGBTransdimProblem(Problem):
    """
    Fixed-size product space with gates.
    theta_u shape: (Kmax * per,)
    per = 9 if use_gates else 8
    """

    def __init__(self, *, Kmax, use_gates, marg_Aphi, loglike_fn, logprior_fn, sample_prior_fn):
        self.Kmax = int(Kmax)
        self.use_gates = bool(use_gates)
        self.marg_Aphi = bool(marg_Aphi)

        # choose parameter layout
        if self.marg_Aphi:
            # no lnA, no phi0
            self._names = ["f0", "fdot", "iota", "psi", "lam", "beta"]
        else:
            self._names = ["lnA", "f0", "fdot", "phi0", "iota", "psi", "lam", "beta"]
            
        if self.use_gates:
            self._names += ["p"]   # or "g_u" if you store unconstrained gates

        self.per = len(self._names)
        self.dim = self.Kmax * self.per

        self._loglike = loglike_fn
        self._logprior = logprior_fn
        self._sample_prior = sample_prior_fn

   # --- REQUIRED overrides ---
    def loglikelihood(self, theta: Array) -> Array:
        return self._loglike(theta)

    def logprior(self, theta: Array) -> Array:
        theta = jnp.asarray(theta, DTYPE).reshape(-1)
        return self._logprior(theta)

    def sample_prior(self, key: Array, n: int) -> Array:
        return self._sample_prior(key, n)        

    def decode_batch(self, samples_u, p_active_min: float = 0.5, sort_by: str = "p"):
        """
        Decode internal samples (N,d) -> physical dict.

        Key changes vs previous version:
        - Computes both hard and soft "number of active sources":
        * K_hard = sum(p > p_active_min)
        * K_soft = sum(p)   (expected number of active components)
        - Optionally sorts per-sample source slots to reduce label-switching in plots:
        sort_by="p"  (default) sorts by descending gate probability p
        sort_by="f0" sorts by increasing frequency f0
        sort_by=None disables sorting
        - Keeps returning a fixed-size flat_phys of shape (N, K*7) with p included.
        """
        X = jnp.asarray(samples_u, dtype=DTYPE)
        if X.ndim != 2:
            raise ValueError(f"Expected (N,d) array, got shape {X.shape}")
        N, d = X.shape

        K = int(self.Kmax)
        per = int(self.per)

        if d != K * per:
            raise ValueError(f"decode_batch: d={d} but expected K*per={K*per} (K={K}, per={per})")

        X3 = X.reshape((N, K, per))

        if self.marg_Aphi:
            # [f0_u, fdot, iota, psi, lam, beta, g_u?]
            f0_u   = X3[:, :, 0]
            fdot   = X3[:, :, 1]
            iota_u = X3[:, :, 2]
            psi_u  = X3[:, :, 3]
            lam_u  = X3[:, :, 4]
            beta_u = X3[:, :, 5]
            g_u    = X3[:, :, 6] if self.use_gates else None
        else:
            # [lnA, f0_u, fdot, phi0, iota, psi, lam, beta, g_u?]
            f0_u   = X3[:, :, 1]
            fdot   = X3[:, :, 2]
            iota_u = X3[:, :, 4]
            psi_u  = X3[:, :, 5]
            lam_u  = X3[:, :, 6]
            beta_u = X3[:, :, 7]
            g_u    = X3[:, :, 8] if self.use_gates else None

        # Decode f0 (ordered vs unordered)
        if getattr(self, "order_f0", False):
            def _f0_one(fu):
                return u_to_f0_ordered(
                    fu[0], fu[1:],
                    self.f_min_cfg, self.f_max_cfg,
                    df_min=self.df_bin, guard_bins=2.0
                )
            f0 = jax.vmap(_f0_one)(f0_u)
        else:
            f0 = u_to_f0_unordered(f0_u, self.f_min_cfg, self.f_max_cfg)

        # Canonicalize angles
        iota = clamp_iota(iota_u)
        psi  = wrap_pm_pi(psi_u)
        lam  = wrap_pm_pi(lam_u)
        beta = clamp_beta(beta_u)

        # Gates -> probabilities
        if self.use_gates:
            p = jax.nn.sigmoid(g_u)  # (N,K) in (0,1)
            K_soft = jnp.sum(p, axis=1)  # float, expected number of actives
            K_hard = jnp.sum(p > p_active_min, axis=1).astype(jnp.int32)  # int
        else:
            p = jnp.ones((N, K), dtype=DTYPE)
            K_soft = jnp.full((N,), float(K), dtype=DTYPE)
            K_hard = jnp.full((N,), K, dtype=jnp.int32)
            g_u = None  # for cleaner downstream

        # Optional sorting to reduce label-switching in plots
        if sort_by is not None:
            if sort_by == "p":
                idx = jnp.argsort(-p, axis=1)  # descending p
            elif sort_by == "f0":
                idx = jnp.argsort(f0, axis=1)  # ascending f0
            else:
                raise ValueError(f"Unknown sort_by={sort_by!r}; use 'p', 'f0', or None")
            
            f0   = jnp.take_along_axis(f0,   idx, axis=1)
            fdot = jnp.take_along_axis(fdot, idx, axis=1)
            iota = jnp.take_along_axis(iota, idx, axis=1)
            psi  = jnp.take_along_axis(psi,  idx, axis=1)
            lam  = jnp.take_along_axis(lam,  idx, axis=1)
            beta = jnp.take_along_axis(beta, idx, axis=1)
            p    = jnp.take_along_axis(p,    idx, axis=1)
            if self.use_gates:
                g_u = jnp.take_along_axis(g_u, idx, axis=1)

        # Flat physical-like chain: [f0, fdot, iota, psi, lam, beta, p]
        flat = jnp.stack([f0, fdot, iota, psi, lam, beta, p], axis=-1).reshape((N, K * 7))

        # Posterior mass over K_hard
        K_np = np.asarray(K_hard)
        vals, cnts = np.unique(K_np, return_counts=True)

        return {
            "flat_phys": np.asarray(flat),

            "f0": np.asarray(f0),
            "fdot": np.asarray(fdot),
            "iota": np.asarray(iota),
            "psi": np.asarray(psi),
            "lam": np.asarray(lam),
            "beta": np.asarray(beta),

            # gate internals
            "g_u": np.asarray(g_u) if self.use_gates else None,
            "p": np.asarray(p),

            # K summaries
            "K_hard": K_np,
            "K_soft": np.asarray(K_soft),
            "K_eff": K_np,  # keep compatibility with your existing code

            # histogram of K_hard across the batch
            "pK_vals": vals,
            "pK_probs": cnts / cnts.sum(),
        }

    
def make(args=None):
    """
    Factory used by:
      --problem factory --problem-factory jax_samplers.problems.lisa_transdim_gb_problem:make
    """
    cfgp = _cfg_path()
    if not cfgp or not os.path.exists(cfgp):
        raise RuntimeError("Set JAX_SAMPLERS_CONFIG (or LISA_CFG) to the JSON config path.")

    with open(cfgp, "r") as fh:
        cfg = json.load(fh)

    # ---------------- data config ----------------
    h5 = _pick(cfg, "data", "h5") or cfg.get("h5")
    if not h5:
        raise RuntimeError("Config missing data.h5")

    data_mode = _pick(cfg, "data", "data_mode", default="obs")
    cat_idxs  = _parse_cat_idxs(_pick(cfg, "data", "cat_idxs", default=None))
    with_noise = bool(_pick(cfg, "data", "with_noise", default=False))
    subtract_vgb = bool(_pick(cfg, "data", "noise_subtract_vgb", default=True))
    decim = int(_pick(cfg, "data", "decim", default=1))

    # ---------------- model config ----------------
    Kmax = int(_pick(cfg, "model", "Kmax", default=2))
    use_gates = bool(_pick(cfg, "model", "use_gates", default=True))
    order_f0  = bool(_pick(cfg, "model", "order_f0", default=False))
    marg_Aphi = bool(_pick(cfg, "model", "marg_Aphi", default=True))
    marg_mode = str(_pick(cfg, "model", "marg_mode", default="profile"))

    # ---------------- priors config ----------------
    # IMPORTANT: for f0 you should *not* use prior_box blindly: you must
    # build a band first, then set f0 prior to [f_min_cfg, f_max_cfg]
    #pr = _pick(cfg, "prior", "prior_box", default={}) or _pick(cfg, "priors", default={}) or {}

    uniform_box = _pick(cfg, "priors", "uniform", default={}) or {}
 
    gaussian_priors = (
        _pick(cfg, "priors", "gaussian", default={})
        or _pick(cfg, "prior", "gaussian_priors", default={})
        or {}
    )
    
    # required: user band (when not using cat_idxs you still need it)

    band_cfg = _pick(cfg, "band", default={}) or {}
    f_min_user = band_cfg.get("f_min", None)
    f_max_user = band_cfg.get("f_max", None)
    f_min_user = None if f_min_user is None else float(f_min_user)
    f_max_user = None if f_max_user is None else float(f_max_user)
    half_bins = int(band_cfg.get("half_bins", 300))
    
    # ---------------- load / synthesize timeseries ----------------
    # You already discovered Sangria truth catalogue is /sky/vgb/cat,
    # so "verification catalog" loaders won't work here.
    if data_mode == "synthetic_from_catalogue":
        if not cat_idxs:
            raise RuntimeError("data_mode=synthetic_from_catalogue requires data.cat_idxs like [0,35].")

        rows = load_catalogue_rows(h5, cat_idxs)
        theta_true = gbjax_params_from_rows(rows, flip_phi0=bool(_pick(cfg, "data", "flip_phi0", default=False)))

        f0_refs = [float(x) for x in np.asarray(theta_true.f0)]

        # use vgb time grid (or obs grid; but your standalone used total_vgb time grid)
        t_sig, _, _, _ = load_total_vgb_xyz(h5)
        t_sig = np.asarray(t_sig).reshape(-1)
        X0 = np.zeros_like(t_sig); Y0 = np.zeros_like(t_sig); Z0 = np.zeros_like(t_sig)

        _, dt_dec, Tobs_dec, freqs_pos, f_band, _, _, df_full, i1, i2 = build_band_from_time_series(
            t=t_sig, X=X0, Y=Y0, Z=Z0,
            decim=decim,
            f_min_user=f_min_user,
            f_max_user=f_max_user,
            half_bins=half_bins,
            f0_refs=f0_refs,
        )

        # choose n_f_bins like you did
        n_f_bins_arr = estimate_n_f_bins(theta_true.amp, theta_true.f0, Tobs_dec)
        n_f_bins = int(np.max(np.asarray(n_f_bins_arr)))
        n_f_bins = max(n_f_bins, 256)

        f_min_cfg = float(i1 * df_full)
        f_max_cfg = float(np.nextafter((i2 + 1) * df_full, np.inf))

        gb_cfg = GBJAXConfig(
            t_obs=float(Tobs_dec),
            dt=float(dt_dec),
            n_f_bins=int(n_f_bins),
            tdi2=bool(_pick(cfg, "model", "use_tdi2", default=False)),
            f_min=float(f_min_cfg),
            f_max=float(f_max_cfg),
        )
        sim_sum, sim_per = build_simulators(gb_cfg)

        # synthesize signal
        xyz_syn = sim_sum(theta_true)
        Xs, Ys, Zs = _get_XYZ_fields(xyz_syn)
        A_syn, E_syn = xyz_to_AE_arrays(Xs, Ys, Zs)
        A_syn = np.asarray(A_syn); E_syn = np.asarray(E_syn)

        if with_noise:
            t_n, nX, nY, nZ = load_instrument_noise_xyz(h5, subtract_vgb=subtract_vgb)
            q = int(decim)
            nX = np.asarray(nX)[::q]; nY = np.asarray(nY)[::q]; nZ = np.asarray(nZ)[::q]
            Xf = np.fft.rfft(nX) * dt_dec
            Yf = np.fft.rfft(nY) * dt_dec
            Zf = np.fft.rfft(nZ) * dt_dec
            Af = (Zf - Xf) / np.sqrt(2.0)
            Ef = (Xf - 2.0 * Yf + Zf) / np.sqrt(6.0)
            idx = np.arange(i1, i2 + 1, dtype=np.int64)
            dA = A_syn[:f_band.size] + Af[idx]
            dE = E_syn[:f_band.size] + Ef[idx]
        else:
            dA = A_syn[:f_band.size]
            dE = E_syn[:f_band.size]

        SA, SE = psd_AE(f_band, preset=NoisePreset.MRDv1, gen=TDIGeneration.TDI1)
        df_band = float(f_band[1] - f_band[0])

        def band_signature(tag, f_band, dA, dE, SA, SE, df_band, i1, i2, dt, Tobs):
            import numpy as np
            print(f"\n[{tag}] signature")
            print(" i1,i2 =", i1, i2, " M =", len(f_band))
            print(" dt =", dt, " Tobs =", Tobs, " df_band =", df_band)
            print(" f_first,last =", float(f_band[0]), float(f_band[-1]))
            # quick norms (no PSD)
            print(" |dA|^2 sum =", float(np.sum(np.abs(dA)**2)))
            print(" |dE|^2 sum =", float(np.sum(np.abs(dE)**2)))
            # dd computed exactly like likelihood
            dd = 4.0*df_band*(np.vdot(dA, dA/(SA+1e-300)).real + np.vdot(dE, dE/(SE+1e-300)).real)
            print(" dd =", float(dd))
            sys.exit()

        #band_signature("TEST", f_band, dA, dE, SA, SE, df_band, i1, i2, dt_dec, Tobs_dec)    

    elif data_mode in ("obs", "sangria_obs"):
        t_obs, X_obs, Y_obs, Z_obs = load_obs_xyz(h5)

        _, dt_dec, Tobs_dec, freqs_pos, f_band, dA, dE, df_full, i1, i2 = build_band_from_time_series(
            t=t_obs, X=X_obs, Y=Y_obs, Z=Z_obs,
            decim=decim,
            f_min_user=f_min_user,
            f_max_user=f_max_user,
            half_bins=half_bins,
            f0_refs=None,
        )

        f_min_cfg = float(i1 * df_full)
        f_max_cfg = float(np.nextafter((i2 + 1) * df_full, np.inf))

        n_f_bins = int(_pick(cfg, "model", "n_f_bins", default=max(256, len(f_band))))
        n_f_bins = max(n_f_bins, int(len(f_band)))

        gb_cfg = GBJAXConfig(
            t_obs=float(Tobs_dec),
            dt=float(dt_dec),
            n_f_bins=int(n_f_bins),
            tdi2=bool(_pick(cfg, "model", "use_tdi2", default=False)),
            f_min=float(f_min_cfg),
            f_max=float(f_max_cfg),
        )
        sim_sum, sim_per = build_simulators(gb_cfg)

        SA, SE = psd_AE(f_band, preset=NoisePreset.MRDv1, gen=TDIGeneration.TDI1)
        df_band = float(f_band[1] - f_band[0])

        dd = 4.0 * df_band * (
            np.vdot(dA, dA / (np.asarray(SA) + 1e-300)).real
            + np.vdot(dE, dE / (np.asarray(SE) + 1e-300)).real
        )
        print(f"\n[{data_mode}] observed TDI band diagnostics")
        print(" i1,i2 =", int(i1), int(i2))
        print(" n_f_bins =", int(len(f_band)))
        print(" f_first,last =", float(f_band[0]), float(f_band[-1]))
        print(" dt =", float(dt_dec), " Tobs =", float(Tobs_dec), " df =", float(df_full))
        print(" |dA|^2 sum =", float(np.sum(np.abs(dA) ** 2)))
        print(" |dE|^2 sum =", float(np.sum(np.abs(dE) ** 2)))
        print(" (d|d) =", float(dd))

    else:
        raise NotImplementedError(f"Unsupported data_mode={data_mode!r}; expected synthetic_from_catalogue, obs, or sangria_obs.")

    # ---- merge prior box with defaults (physical-space bounds) ----
    default_box = {
        "f0":   (float(f_min_cfg), float(f_max_cfg)),
        "fdot": (-1e-13, 1e-13),
        "iota": (-np.pi/2, np.pi/2),
        "psi":  (-np.pi, np.pi),
        "lam":  (-np.pi, np.pi),
        "beta": (-np.pi/2, np.pi/2),
    }
    
    if not marg_Aphi:
        default_box["lnA"]  = (-60.0, -40.0)
        default_box["phi0"] = (-np.pi, np.pi)

    if use_gates:
        # choose one convention; here: p in [0,1], sampled then converted to g_u
        default_box["p"] = (0.0, 1.0)

    merged_box = dict(default_box)
    for k, v in uniform_box.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            merged_box[k] = (float(v[0]), float(v[1]))

    # ---------------- build loglike/logprior ----------------

    p_active_min = float(_pick(cfg, "model", "p_active_min", default=1e-3))
    
    loglike, logprior, logpost, template_diag, marg_diag = make_loglike_and_logprior(
        sim_sum=sim_sum,
        sim_per_source=sim_per,
        f_band=f_band, dA=dA, dE=dE,
        SA=SA, SE=SE,
        df_band=df_band,
        Kmax=Kmax,
        use_gates=use_gates,
        order_f0=order_f0,
        f_min_cfg=f_min_cfg,
        f_max_cfg=f_max_cfg,
        marg_Aphi=marg_Aphi,
        marg_mode=marg_mode,
        prior_box=merged_box,
        gaussian_priors=gaussian_priors,
        p_active_min=p_active_min
        # you can pass hyperparams from cfg too
    )

    # ---------------- optional debug: compare backend loglike at truth ----------------
    """
    if bool(_pick(cfg, "debug", "check_truth_loglike", default=False)):
        if data_mode != "synthetic_from_catalogue":
            raise RuntimeError("truth check currently implemented for synthetic_from_catalogue only.")

        K_true = int(theta_true.amp.shape[0])

        if K_true > Kmax:
            raise RuntimeError(f"K_true={K_true} > Kmax={Kmax}")

        # backend layout:
        # marg_Aphi=True,  use_gates=True  -> [f0_u, fdot, iota, psi, lam, beta, g_u]
        # marg_Aphi=True,  use_gates=False -> [f0_u, fdot, iota, psi, lam, beta]
        # marg_Aphi=False, use_gates=True  -> [lnA, f0_u, fdot, phi0, iota, psi, lam, beta, g_u]
        # marg_Aphi=False, use_gates=False -> [lnA, f0_u, fdot, phi0, iota, psi, lam, beta]

        per = (6 + (1 if use_gates else 0)) if marg_Aphi else (8 + (1 if use_gates else 0))
        th0 = np.zeros((Kmax, per), dtype=np.float64)

        # ---- fill truth for active sources ----
        f0_true = np.asarray(theta_true.f0, dtype=np.float64)
        x = (f0_true - f_min_cfg) / (f_max_cfg - f_min_cfg)
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        f0_u_true = np.log(x / (1.0 - x))

        if marg_Aphi:
            # columns: [f0_u, fdot, iota, psi, lam, beta, (g_u)]
            th0[:K_true, 0] = f0_u_true
            th0[:K_true, 1] = np.asarray(theta_true.fdot, dtype=np.float64)
            th0[:K_true, 2] = np.asarray(theta_true.iota, dtype=np.float64)
            th0[:K_true, 3] = np.asarray(theta_true.psi, dtype=np.float64)
            th0[:K_true, 4] = np.asarray(theta_true.lam, dtype=np.float64)
            th0[:K_true, 5] = np.asarray(theta_true.beta, dtype=np.float64)

            if use_gates:
                # active slots "on", inactive slots "off"
                th0[:K_true, 6] = 10.0
                th0[K_true:, 6] = -20.0

            else:
                # columns: [lnA, f0_u, fdot, phi0, iota, psi, lam, beta, (g_u)]
                th0[:K_true, 0] = np.log(np.asarray(theta_true.amp, dtype=np.float64))
                th0[:K_true, 1] = f0_u_true
                th0[:K_true, 2] = np.asarray(theta_true.fdot, dtype=np.float64)
                th0[:K_true, 3] = np.asarray(theta_true.phi0, dtype=np.float64)
                th0[:K_true, 4] = np.asarray(theta_true.iota, dtype=np.float64)
                th0[:K_true, 5] = np.asarray(theta_true.psi, dtype=np.float64)
                th0[:K_true, 6] = np.asarray(theta_true.lam, dtype=np.float64)
                th0[:K_true, 7] = np.asarray(theta_true.beta, dtype=np.float64)

                if use_gates:
                    th0[:K_true, 8] = 10.0
                    th0[K_true:, 8] = -20.0
                    th0[K_true:, 0] = -60.0

            theta_u_truth = jnp.asarray(th0.reshape(-1), dtype=DTYPE)

            ll_truth = float(loglike(theta_u_truth))
            ll_plain_truth, dd_truth, hh_truth, dh_truth = template_diag(theta_u_truth)
            ll_plain_truth = float(ll_plain_truth)

            print(f"[debug] backend loglike(theta_truth)       = {ll_truth:.6f}")
            print(f"[debug] backend loglike_plain(theta_truth) = {ll_plain_truth:.6f}")
            print(f"[debug] (d|d) = {float(dd_truth):.6e}")
            print(f"[debug] (h|h) = {float(hh_truth):.6e}")
            print(f"[debug] (d|h) = {float(dh_truth):.6e}")

            if marg_Aphi:
                print(f"[debug] delta(profile-plain) = {ll_truth - ll_plain_truth:.6e}")
            #sys.exit()    

        if bool(_pick(cfg, "debug", "check_truth_loglike", default=False)):
            if data_mode != "synthetic_from_catalogue":
                raise RuntimeError("truth check currently implemented for synthetic_from_catalogue only.")

            if order_f0:
                raise RuntimeError("Truth-coordinate debug currently assumes order_f0=False.")

            if not marg_Aphi or use_gates:
                raise RuntimeError("This debug block currently supports only marg_Aphi=True and use_gates=False.")

            K_true = int(theta_true.amp.shape[0])
            if K_true > Kmax:
                raise RuntimeError(f"K_true={K_true} > Kmax={Kmax}")

            per = 6
            th0 = np.zeros((Kmax, per), dtype=np.float64)

            f0_true = np.asarray(theta_true.f0, dtype=np.float64)
            x = (f0_true - f_min_cfg) / (f_max_cfg - f_min_cfg)
            x = np.clip(x, 1e-6, 1.0 - 1e-6)
            f0_u_true = np.log(x / (1.0 - x))

            # layout: [f0_u, fdot, iota, psi, lam, beta]
            th0[:K_true, 0] = f0_u_true
            th0[:K_true, 1] = np.asarray(theta_true.fdot, dtype=np.float64)
            th0[:K_true, 2] = np.asarray(theta_true.iota, dtype=np.float64)
            th0[:K_true, 3] = np.asarray(theta_true.psi, dtype=np.float64)
            th0[:K_true, 4] = np.asarray(theta_true.lam, dtype=np.float64)
            th0[:K_true, 5] = np.asarray(theta_true.beta, dtype=np.float64)

            theta_u_truth = jnp.asarray(th0.reshape(-1), dtype=DTYPE)

            ll_truth = float(loglike(theta_u_truth))
            print(f"[debug] backend loglike(theta_truth) = {ll_truth:.6f}")
            sys.exit()

    """
    
    if bool(_pick(cfg, "debug", "check_truth_loglike", default=False)):
        if data_mode != "synthetic_from_catalogue":
            raise RuntimeError("truth check currently implemented for synthetic_from_catalogue only.")

        K_true = int(theta_true.amp.shape[0])

        if K_true > Kmax:
            raise RuntimeError(f"K_true={K_true} > Kmax={Kmax}")

        per = (6 + (1 if use_gates else 0)) if marg_Aphi else (8 + (1 if use_gates else 0))
        th0 = np.zeros((Kmax, per), dtype=np.float64)

        f0_true = np.asarray(theta_true.f0, dtype=np.float64)
        x = (f0_true - f_min_cfg) / (f_max_cfg - f_min_cfg)
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        f0_u_true = np.log(x / (1.0 - x))

        if marg_Aphi:
            # [f0_u, fdot, iota, psi, lam, beta, (g_u)]
            th0[:K_true, 0] = f0_u_true
            th0[:K_true, 1] = np.asarray(theta_true.fdot, dtype=np.float64)
            th0[:K_true, 2] = np.asarray(theta_true.iota, dtype=np.float64)
            th0[:K_true, 3] = np.asarray(theta_true.psi, dtype=np.float64)
            th0[:K_true, 4] = np.asarray(theta_true.lam, dtype=np.float64)
            th0[:K_true, 5] = np.asarray(theta_true.beta, dtype=np.float64)

            if use_gates:
                th0[:K_true, 6] = 10.0
                th0[K_true:, 6] = -20.0

        else:
            # [lnA, f0_u, fdot, phi0, iota, psi, lam, beta, (g_u)]
            th0[:K_true, 0] = np.log(np.asarray(theta_true.amp, dtype=np.float64))
            th0[:K_true, 1] = f0_u_true
            th0[:K_true, 2] = np.asarray(theta_true.fdot, dtype=np.float64)
            th0[:K_true, 3] = np.asarray(theta_true.phi0, dtype=np.float64)
            th0[:K_true, 4] = np.asarray(theta_true.iota, dtype=np.float64)
            th0[:K_true, 5] = np.asarray(theta_true.psi, dtype=np.float64)
            th0[:K_true, 6] = np.asarray(theta_true.lam, dtype=np.float64)
            th0[:K_true, 7] = np.asarray(theta_true.beta, dtype=np.float64)

            if use_gates:
                th0[:K_true, 8] = 10.0
                th0[K_true:, 8] = -20.0
                th0[K_true:, 0] = -60.0

        theta_u_truth = jnp.asarray(th0.reshape(-1), dtype=DTYPE)

        ll_truth = float(loglike(theta_u_truth))
        print(f"[debug] backend loglike(theta_truth) = {ll_truth:.6f}")

        if not marg_Aphi:
            ll_plain_truth, dd_truth, hh_truth, dh_truth = template_diag(theta_u_truth)
            ll_plain_truth = float(ll_plain_truth)

            print(f"[debug] backend loglike_plain(theta_truth) = {ll_plain_truth:.6f}")
            print(f"[debug] (d|d) = {float(dd_truth):.6e}")
            print(f"[debug] (h|h) = {float(hh_truth):.6e}")
            print(f"[debug] (d|h) = {float(dh_truth):.6e}")

        # gate tests for K=2,1,0
        if marg_Aphi and use_gates and Kmax == 2:
            th_2 = np.array(theta_u_truth, copy=True)
            th_1a = np.array(theta_u_truth, copy=True)
            th_1b = np.array(theta_u_truth, copy=True)
            th_0 = np.array(theta_u_truth, copy=True)

            # gate entries are last slot of each 7-parameter block: 6 and 13
            th_2[6],  th_2[13]  = 10.0, 10.0
            th_1a[6], th_1a[13] = 10.0, -10.0
            th_1b[6], th_1b[13] = -10.0, 10.0
            th_0[6],  th_0[13]  = -10.0, -10.0

            print(f"[debug] loglike K=2 = {float(loglike(jnp.asarray(th_2, dtype=DTYPE))):.6f}")
            print(f"[debug] loglike K=1a = {float(loglike(jnp.asarray(th_1a, dtype=DTYPE))):.6f}")
            print(f"[debug] loglike K=1b = {float(loglike(jnp.asarray(th_1b, dtype=DTYPE))):.6f}")
            print(f"[debug] loglike K=0 = {float(loglike(jnp.asarray(th_0, dtype=DTYPE))):.6f}")

        sys.exit()
    
            
    # ---------------- sample_prior: FLAT priors in your physical box ----------------
    # This is the key: implement a prior sampler consistent with your intended priors.
    # Since your logprior in the standalone is Gaussian in unconstrained coords,
    # here we implement flat priors in *physical* parameters by sampling in physical
    # and mapping back to theta_u. (Or: define logprior in theta_u accordingly.)
    prior_box = merged_box

    def sample_prior(key, n):
        # merged prior box must already be validated and include band-consistent f0
        f0_lo, f0_hi = merged_box["f0"]
        fdot_lo, fdot_hi = merged_box["fdot"]
        iota_lo, iota_hi = merged_box["iota"]
        psi_lo, psi_hi = merged_box["psi"]
        lam_lo, lam_hi = merged_box["lam"]
        beta_lo, beta_hi = merged_box["beta"]

        if not marg_Aphi:
            lnA_lo, lnA_hi = merged_box["lnA"]
            phi_lo, phi_hi = merged_box["phi0"]

        #if use_gates:
        #    p_lo, p_hi = merged_box.get("g", merged_box.get("p", (0.0, 1.0)))

        if marg_Aphi:
            per = 7 if use_gates else 6
        else:
            per = 9 if use_gates else 8

        K = int(Kmax)

        # count RNG blocks
        n_blocks = 6 + (0 if marg_Aphi else 2) + (1 if use_gates else 0)
        subs = jr.split(key, n_blocks)

        def U(k, lo, hi):
            return lo + (hi - lo) * jr.uniform(k, (n, K), dtype=DTYPE)

        idx = 0
        if not marg_Aphi:
            lnA = U(subs[idx], lnA_lo, lnA_hi); idx += 1

        f0   = U(subs[idx], f0_lo, f0_hi); idx += 1
        fdot = U(subs[idx], fdot_lo, fdot_hi); idx += 1
        iota = U(subs[idx], iota_lo, iota_hi); idx += 1
        psi  = U(subs[idx], psi_lo, psi_hi); idx += 1
        lam  = U(subs[idx], lam_lo, lam_hi); idx += 1
        beta = U(subs[idx], beta_lo, beta_hi); idx += 1


        if not marg_Aphi:
            phi0 = U(subs[idx], phi_lo, phi_hi); idx += 1


        if use_gates:
            if "g" in gaussian_priors:
                mu, sig = gaussian_priors["g"]          # (mean, std) in logit space
                mu  = jnp.asarray(mu, dtype=DTYPE)
                sig = jnp.asarray(sig, dtype=DTYPE)
                g_u = mu + sig * jr.normal(subs[idx], (n, K), dtype=DTYPE)
                idx += 1
            else:
                # fallback: uniform in p in [0,1] (or from merged_box["p"] if provided)
                p_lo, p_hi = merged_box.get("p", (0.0, 1.0))
                p = U(subs[idx], p_lo, p_hi); idx += 1
                p = jnp.clip(p, 1e-6, 1 - 1e-6)
                g_u = jnp.log(p / (1 - p))

        """    
        if use_gates:
            p = U(subs[idx], p_lo, p_hi); idx += 1
            p = jnp.clip(p, 1e-6, 1 - 1e-6)
            g_u = jnp.log(p / (1 - p))
        """
        # physical -> unconstrained for f0 slot
        x = (f0 - f_min_cfg) / (f_max_cfg - f_min_cfg)
        x = jnp.clip(x, 1e-6, 1 - 1e-6)
        f0_u = jnp.log(x / (1 - x))

        if marg_Aphi:
            if use_gates:
                th = jnp.stack([f0_u, fdot, iota, psi, lam, beta, g_u], axis=-1)
            else:
                th = jnp.stack([f0_u, fdot, iota, psi, lam, beta], axis=-1)
        else:
            if use_gates:
                th = jnp.stack([lnA, f0_u, fdot, phi0, iota, psi, lam, beta, g_u], axis=-1)
            else:
                th = jnp.stack([lnA, f0_u, fdot, phi0, iota, psi, lam, beta], axis=-1)
        
        return th.reshape((n, K * per))
    
    def sample_prior_flat(key, n):
        # f0 MUST be within [f_min_cfg, f_max_cfg] (the band)
        # for other params use cfg ranges if present; else reasonable defaults
        f0_lo, f0_hi = f_min_cfg, f_max_cfg
        fdot_lo, fdot_hi = prior_box.get("fdot", [-1e-13, 1e-13])
        iota_lo, iota_hi = prior_box.get("iota", [-np.pi/2, np.pi/2])
        psi_lo,  psi_hi  = prior_box.get("psi",  [0.0, 2*np.pi])
        lam_lo,  lam_hi  = prior_box.get("lam",  [0.0, 2*np.pi])
        beta_lo, beta_hi = prior_box.get("beta", [-np.pi/2, np.pi/2])

        # lnA/phi0 only matter if NOT marg_Aphi. If marg_Aphi, can keep weak sampling.
        lnA_lo, lnA_hi = prior_box.get("lnA", [-60.0, -40.0])   # example
        phi_lo, phi_hi = prior_box.get("phi0", [-np.pi, np.pi])

        # gate “p” prior: easiest is sample gate probability p~U(0,1) and convert to logit.
        # If your gate variable is g_u with sigmoid(g_u)=p, then g_u = logit(p).
        p_lo, p_hi = prior_box.get("p", [0.0, 1.0])

        Kmax_loc = Kmax
        per = 9 if use_gates else 8

        key, *subs = jr.split(key, 1 + 10)
        subs = list(subs)

        def U(sub, shape, lo, hi):
            return lo + (hi - lo) * jr.uniform(sub, shape=shape)

        # sample physical arrays with shape (n,Kmax)
        lnA  = U(subs[0], (n, Kmax_loc), lnA_lo, lnA_hi)
        f0   = U(subs[1], (n, Kmax_loc), f0_lo,  f0_hi)
        fdot = U(subs[2], (n, Kmax_loc), float(fdot_lo), float(fdot_hi))
        phi0 = U(subs[3], (n, Kmax_loc), phi_lo, phi_hi)
        iota = U(subs[4], (n, Kmax_loc), iota_lo, iota_hi)
        psi  = U(subs[5], (n, Kmax_loc), psi_lo,  psi_hi)
        lam  = U(subs[6], (n, Kmax_loc), lam_lo,  lam_hi)
        beta = U(subs[7], (n, Kmax_loc), beta_lo, beta_hi)

        if use_gates:
            p = U(subs[8], (n, Kmax_loc), float(p_lo), float(p_hi))
            p = jnp.clip(p, 1e-6, 1-1e-6)
            g_u = jnp.log(p/(1-p))
        else:
            g_u = None

        # map physical -> theta_u (inverse of your transforms)
        # f0_u uses logit((f0-fmin)/(fmax-fmin)) in unordered mode
        x = (f0 - f_min_cfg) / (f_max_cfg - f_min_cfg)
        x = jnp.clip(x, 1e-6, 1-1e-6)
        f0_u = jnp.log(x/(1-x))

        # for angles you currently do wrap/clamp in forward map;
        # simplest “inverse” for a flat prior backend is to just store them as-is:
        # phi0_u=phi0, iota_u=iota, etc. (since forward map clamps anyway)
        # same for fdot_u=fdot, lnA_u=lnA

        if use_gates:
            th = jnp.stack([lnA, f0_u, fdot, phi0, iota, psi, lam, beta, g_u], axis=-1)
        else:
            th = jnp.stack([lnA, f0_u, fdot, phi0, iota, psi, lam, beta], axis=-1)

        return th.reshape((n, Kmax_loc*per))

    problem = LISAGBTransdimProblem(
        Kmax=Kmax,
        use_gates=use_gates,
        marg_Aphi=marg_Aphi,      # <-- add this
        loglike_fn=loglike,
        logprior_fn=logprior,
        sample_prior_fn=sample_prior,
    )

    p_active_min = float(_pick(cfg, "model", "p_active_min", default=1e-3))
    problem.p_active_min = p_active_min
    
    problem._marg_diag = marg_diag
    problem._templ_diag = template_diag
    
    problem.f_min_cfg = float(f_min_cfg)
    problem.f_max_cfg = float(f_max_cfg)
    problem.df_bin = float(df_band)
    problem.order_f0 = bool(order_f0)

    return problem

    """ 
    return LISAGBTransdimProblem(
        Kmax=Kmax,
        use_gates=use_gates,
        marg_Aphi=marg_Aphi,      # <-- add this
        loglike_fn=loglike,
        logprior_fn=logprior,
        sample_prior_fn=sample_prior,
    )
    """
