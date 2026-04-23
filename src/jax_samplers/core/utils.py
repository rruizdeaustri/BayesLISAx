# jax_samplers/core/utils.py
from __future__ import annotations
import jax
import jax.numpy as jnp

import re
import numpy as np
import matplotlib.pyplot as plt
import corner

JAX_FLOAT = jnp.array(0.).dtype
NP_FLOAT  = np.float64 if JAX_FLOAT == jnp.float64 else np.float32

def as_jax(x): return jnp.asarray(x, dtype=JAX_FLOAT)
def as_np(x):  return np.asarray(x,  dtype=NP_FLOAT)

def _norm_weights(w):
    if w is None: return None
    w = np.asarray(w, float); return w / (w.sum() + 1e-300)

def _parse_aliases(s: str) -> dict:
    if not s.strip(): return {}
    out = {}
    for item in s.split(","):
        if not item.strip(): continue
        k, v = item.split(":", 1)
        out[k.strip()] = v.strip()
    return out

def _parse_truth(s: str):
    s = s.strip()
    if not s:
        return None

    # remove wrapping quotes if user wrote "--truth '...'"
    s = s.strip("\"'")

    # split by commas OR any whitespace
    parts = [p for p in re.split(r"[,\s]+", s) if p]
    vals = [float(p) for p in parts]
    return np.asarray(vals, dtype=float)


def _labels(fields, K, *, style, template, aliases):
    if style == "plain":
        return [f"{aliases.get(f, f)}_{k}" for k in range(1, K+1) for f in fields]

    if style == "latex":
        return [rf"${aliases.get(f, f)}_{{{k}}}$"
                for k in range(1, K+1) for f in fields]

    return [template.format(name=aliases.get(f, f), k=k)
            for k in range(1, K+1) for f in fields]

def extract_component_major(samples, *, K, start_col, fields, sort_by=None):
    # layout: [..., f1_1,f2_1,..., fM_1,  f1_2,f2_2,..., fM_2, ...]
    N = samples.shape[0]; M = len(fields)
    block = samples[:, start_col:start_col + K*M].reshape(N, K, M)  # (N,K,M)
    arrs = {f: block[:,:,i] for i,f in enumerate(fields)}           # each (N,K)
    if sort_by:
        order = np.argsort(arrs[sort_by], axis=1)
        idx = (np.arange(N)[:,None], order)
        for f in fields: arrs[f] = arrs[f][idx]
    X = np.concatenate([arrs[f] for f in fields], axis=1).reshape(N, K*M, order='F')
    # (the reshape with order='F' interleaves by component)
    return X

def extract_field_major(samples, *, K, start_col, fields, extras=0, sort_by=None):
    # layout: [..., f1_1..f1_K, f2_1..f2_K, ...] then extras
    N = samples.shape[0]; M = len(fields)
    blocks = {f: samples[:, start_col + i*K : start_col + (i+1)*K] for i,f in enumerate(fields)}  # each (N,K)
    if sort_by:
        desc = sort_by.lower() in ("w","weight","amp","amplitude","f","flux","logf")
        key = blocks[sort_by]
        order = np.argsort(-key, axis=1) if desc else np.argsort(key, axis=1)
        idx = (np.arange(N)[:,None], order)
        for f in fields: blocks[f] = blocks[f][idx]
    X = np.stack([blocks[f][:,k] for k in range(K) for f in fields], axis=1)  # (N, K*M)
    return X


def _posterior_K(samples, weights=None, *, K_col=0, K_min=None, K_max=None):
    K_raw = samples[:, K_col]
    if K_min is None: K_min = int(np.floor(np.nanmin(K_raw)))
    if K_max is None: K_max = int(np.ceil(np.nanmax(K_raw)))
    K_vals = np.rint(K_raw).astype(int)
    K_vals = np.clip(K_vals, K_min, K_max)
    support = np.arange(K_min, K_max + 1)
    if weights is None:
        counts = np.bincount(K_vals, minlength=K_max + 1)[K_min:]
        pmf = counts / (counts.sum() + 1e-300)
    else:
        w = _norm_weights(weights)
        wcounts_full = np.bincount(K_vals, weights=w, minlength=K_max + 1)
        pmf = wcounts_full[K_min:]
        pmf = pmf / (pmf.sum() + 1e-300)
    K_mode = int(support[np.argmax(pmf)])
    return support, pmf, K_mode

def _extract_per_component(
    samples_K: np.ndarray, *, K: int, start_col: int,
    fields: list[str], stride: int | None = None,
    transforms: dict[str, callable] | None = None, sort_by: str | None = None,
) -> np.ndarray:
    if stride is None: stride = len(fields)
    N = samples_K.shape[0]; M = len(fields)
    need = start_col + K*stride
    if samples_K.shape[1] < need:
        raise ValueError(f"Need ≥{need} cols for K={K}, got {samples_K.shape[1]}")
    block = samples_K[:, start_col:start_col + K*stride].reshape(N, K, stride)
    idx_map = {f: i for i, f in enumerate(fields)}
    arrs = {}
    for f in fields:
        A = block[..., idx_map[f]]
        if transforms and f in transforms: A = transforms[f](A)
        arrs[f] = A  # (N,K)
    if sort_by:
        if sort_by not in fields:
            raise ValueError(f"sort_by='{sort_by}' not in fields={fields}")
        # common magnitudes descend; positions ascend
        desc = sort_by.lower() in ("f","logf","weight","w","amp","amplitude")
        order = np.argsort(-arrs[sort_by], axis=1) if desc else np.argsort(arrs[sort_by], axis=1)
        idx = (np.arange(N)[:, None], order)
        for f in fields:
            arrs[f] = arrs[f][idx]
    X = np.stack([arrs[f] for f in fields], axis=-1).reshape(N, K*M)
    return X

def _resolve_schema(args):
    # preset or generic
    if args.schema == "fermi":
        fields = ["y","x","logF"]
        start_col = 1
        sort_by = "logF" if not args.sort_by else args.sort_by
    else:
        fields = [t.strip() for t in args.fields.split(",") if t.strip()]
        start_col = args.start_col
        sort_by = args.sort_by or None
    # optional transform: show linear F if we stored logF
    transforms = None
    if getattr(args, "plot_linear_F", False) and "logF" in fields:
    #if args.plot_linear_F and "logF" in fields:
        transforms = {"logF": np.exp}
        fields = ["F" if f=="logF" else f for f in fields]
    return fields, start_col, sort_by, transforms


def chunked_from_single(fn_one, chunk: int):
    """
    Wrap a function fn_one: (dim,) -> out_scalar so it also accepts (n, dim)
    and evaluates in chunks of size `chunk`. Returns a function that accepts
    either (dim,) or (n,dim). Output is scalar or (n,) accordingly.
    """
    vm = jax.vmap(fn_one)

    def f(x):
        
        if x.ndim == 1:
            return fn_one(x)  # ( )  scalar
        
        n, d = x.shape
        
        if chunk <= 0 or n <= chunk:
            return vm(x)  # (n,)

        # Fully sequential path (best for memory / chunk==1)
        if chunk == 1:
            return jax.lax.map(fn_one, x)          # (n,)
        
        pad = (chunk - (n % chunk)) % chunk
        x_pad = jnp.pad(x, ((0, pad), (0, 0)))
        x_groups = x_pad.reshape((-1, chunk, d))      # (g,chunk,d)
        y_groups = jax.vmap(vm)(x_groups)             # (g,chunk)
        y = y_groups.reshape((-1,))[:n]               # (n,)
        return y
    
    return f

def infer_K_and_extras(samples, fields_list, start_col=0, extras=None, *, allow_K1=True):
    """
    Infer (K, extras) for plotting.

    samples: (N, D_total)
    fields_list: list of field names for ONE component (stride)
    start_col: first column of parameters in samples (e.g. 0)
    extras:
      - None  -> infer automatically
      - int   -> treat as fixed (but we still do sanity checks)
    allow_K1: if True, accept fixed-dim K=1 case
    """
    D = samples.shape[1]
    stride = len(fields_list)
    rem = D - start_col
    if rem <= 0:
        raise ValueError(f"Bad start_col={start_col}: D={D}")

    # ---- if user didn't specify extras, infer it ----
    if extras is None:
        # Prefer exact fixed-dim match first: D == start_col + stride (+extras)
        # Common case: fixed 6D with no extras -> extras=0, K=1
        if rem >= stride:
            # choose smallest extras >=0 such that (rem - extras) is a positive multiple of stride
            # and preferably K=1 when possible.
            candidates = []
            for e in range(0, min(32, rem) + 1):  # cap search; adjust if you ever append many extras
                r = rem - e
                if r <= 0:
                    break
                if r % stride == 0:
                    K = r // stride
                    if K >= 1:
                        # score: prefer K=1, then smaller extras
                        score = (0 if K == 1 else 10, e)
                        candidates.append((score, K, e))
            if not candidates:
                raise ValueError(
                    f"Cannot infer extras/K: D={D}, start_col={start_col}, stride={stride}."
                )
            _, K, extras = sorted(candidates, key=lambda t: t[0])[0]
        else:
            raise ValueError(
                f"Not enough columns: rem={rem} < stride={stride} (D={D}, start_col={start_col})"
            )

    # ---- now validate with chosen extras ----
    if extras < 0 or extras >= rem:
        raise ValueError(f"Invalid extras={extras} for rem={rem} (D={D}, start_col={start_col})")

    r = rem - extras
    if r <= 0 or (r % stride) != 0:
        raise ValueError(
            f"Cannot infer K: D={D}, start_col={start_col}, stride={stride}, extras={extras} "
            f"(rem={rem}, rem-extras={r})"
        )
    K = r // stride

    if not allow_K1 and K == 1:
        raise ValueError(f"Inferred K=1 but allow_K1=False (D={D}, extras={extras}, stride={stride})")

    return K, extras


def _pretty_base_name(name: str, aliases: dict | None = None) -> str:
    """
    Canonicalize field names and apply aliases.
    Keeps things human-readable and stable for plotting.
    """
    s = str(name).strip()
    if aliases and s in aliases:
        s = str(aliases[s]).strip()

    # common canonicalizations for your GB params
    low = s.lower()
    if low in ("f0", "f_0", "freq", "frequency"):
        return "f0"
    if low in ("fdot", "f_dot", "dfdt", "frequencyderivative"):
        return "fdot"
    if low in ("iota",):
        return "iota"
    if low in ("psi",):
        return "psi"
    if low in ("lam", "lambda", "eclipticlongitude"):
        return "lam"
    if low in ("beta", "eclipticlatitude"):
        return "beta"
    if low in ("ln_a", "lna", "loga", "logamp", "lnamp"):
        return "lnA"
    if low in ("phi0", "phi_0", "initialphase"):
        return "phi0"
    if low in ("p", "gate", "g", "gate_p"):
        return "p"

    return s


def _label_one(base: str, k: int, style: str = "plain") -> str:
    """
    Build one safe label.
    style: 'plain' | 'math'
    NOTE: math style is valid mathtext (no double-subscript).
    """
    if style == "math":
        # valid mathtext forms
        if base == "f0":
            return fr"$f_{{0,{k}}}$"       # NOT f_0_{k}
        if base == "fdot":
            return fr"$\dot f_{{{k}}}$"
        if base == "lnA":
            return fr"$\ln A_{{{k}}}$"
        if base == "phi0":
            return fr"$\phi_{{0,{k}}}$"
        if base == "p":
            return fr"$p_{{{k}}}$"
        # generic fallback
        return fr"${base}_{{{k}}}$"

    # plain (safest for headless/HPC)
    return f"{base}_{k}"


def labels_lisa_safe(
    fields_list: list[str],
    K: int,
    aliases: dict | None = None,
    style: str = "plain",
    k_start: int = 1,
) -> list[str]:
    """
    Returns labels in source-major order:
      [f0_1, fdot_1, ..., f0_2, fdot_2, ...]
    matching X built as:
      X = samples[:, start : start + K*len(fields_list)]
    """
    out = []
    for k in range(k_start, k_start + K):
        for f in fields_list:
            base = _pretty_base_name(f, aliases=aliases)
            out.append(_label_one(base, k, style=style))
    return out


def sanitize_labels_for_matplotlib(labels: list[str]) -> list[str]:
    """
    Extra safety net: fixes common invalid mathtext patterns.
    Keeps labels unchanged unless problematic pattern is detected.
    """
    fixed = []
    for s in labels:
        t = str(s)

        # Fix known bad pattern: f_0_{1} -> f_{0,1}
        t = re.sub(r"f_0_\{(\d+)\}", r"f_{0,\1}", t)
        t = re.sub(r"phi_0_\{(\d+)\}", r"phi_{0,\1}", t)

        fixed.append(t)
    return fixed
