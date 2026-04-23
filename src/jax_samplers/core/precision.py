# precision.py
from __future__ import annotations
import os, sys, json
import numpy as np

from jax import config as _jaxcfg
#print("[precision-debug] precision.py file =", __file__)
#print("[precision-debug] argv =", sys.argv)


def _read_requested_dtype() -> str:
    requested = None

    # 1) CLI override via env (your cli.py sets this if --dtype is used)
    env_dtype = os.environ.get("JAX_SAMPLERS_DTYPE") or os.environ.get("FERMI_DTYPE")
    if env_dtype:
        return str(env_dtype).lower()

    # 2) Resolve config path from env (your cli.py sets this)
    cfg_path = os.environ.get("JAX_SAMPLERS_CONFIG") or os.environ.get("LISA_CFG")
    #print("[precision-debug] cfg_path_resolved =", cfg_path)

    if not cfg_path:
        print("[precision-debug] no cfg_path -> default float32")
        return "float32"

    if not os.path.exists(cfg_path):
        print("[precision-debug] cfg_path does not exist -> default float32")
        return "float32"

    # 3) Read and parse JSON loudly
    try:
        with open(cfg_path, "r") as f:
            txt = f.read()

        # If your "JSON" has comments/trailing commas, json.loads will fail.
        # This print helps you see what you're feeding the parser.
        print("[precision-debug] config head =", txt[:120].replace("\n", "\\n"))

        cfg = json.loads(txt)
        print("[precision-debug] config keys =", list(cfg.keys())[:40])

    except Exception as e:
        raise RuntimeError(
            f"Failed to parse config as strict JSON: {cfg_path}\n"
            f"Error: {repr(e)}\n"
            f"Tip: JSON cannot contain comments or trailing commas."
        )

    # 4) Extract dtype (top-level)
    if cfg.get("dtype") is not None:
        requested = str(cfg["dtype"]).lower()
        #print("[precision-debug] dtype from cfg['dtype'] =", requested)
        return requested

    # OPTIONAL: allow common nesting patterns too
    for path in (("precision", "dtype"), ("jax", "dtype")):
        d = cfg
        ok = True
        for k in path:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                ok = False
                break
        if ok and d is not None:
            requested = str(d).lower()
            print(f"[precision-debug] dtype from cfg{path} =", requested)
            return requested

    print("[precision-debug] dtype not found in config -> default float32")
    return "float32"


def _read_requested_dtype_old() -> str:
    requested, cfg_path = None, None
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--extras" and i + 1 < len(argv):
            try:
                extras = json.loads(argv[i + 1])
                cfg_path = extras.get("config_path", cfg_path)
                if extras.get("dtype") is not None:
                    requested = str(extras["dtype"]).lower()
            except Exception:
                pass

    if requested is None and cfg_path and os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            if cfg.get("dtype") is not None:
                requested = str(cfg["dtype"]).lower()
        except Exception:
            pass

    if requested is None:
        # better name, keep FERMI_DTYPE as legacy fallback
        requested = os.environ.get("JAX_SAMPLERS_DTYPE",
                    os.environ.get("FERMI_DTYPE", "float32")).lower()
    return requested

def _read_requested_dtype_old() -> str:
    requested, cfg_path = None, None
    argv = sys.argv

    # 1) accept --dtype (if you ever pass it)
    for i, a in enumerate(argv):
        if a == "--dtype" and i + 1 < len(argv):
            requested = str(argv[i + 1]).lower()

    # 2) accept --config
    if cfg_path is None:
        for i, a in enumerate(argv):
            if a == "--config" and i + 1 < len(argv):
                cfg_path = argv[i + 1]

    # 3) accept --extras JSON (your existing path)
    for i, a in enumerate(argv):
        if a == "--extras" and i + 1 < len(argv):
            try:
                extras = json.loads(argv[i + 1])
                cfg_path = extras.get("config_path", cfg_path)
                if extras.get("dtype") is not None:
                    requested = str(extras["dtype"]).lower()
            except Exception:
                pass

    # 4) accept env config path (THIS is what your cli.py sets)
    if cfg_path is None:
        cfg_path = os.environ.get("JAX_SAMPLERS_CONFIG") or os.environ.get("LISA_CFG")

    # 5) load dtype from config if still not set
    if requested is None and cfg_path and os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            if cfg.get("dtype") is not None:
                requested = str(cfg["dtype"]).lower()
        except Exception:
            pass

    # 6) finally env dtype override / default
    if requested is None:
        requested = os.environ.get(
            "JAX_SAMPLERS_DTYPE",
            os.environ.get("FERMI_DTYPE", "float32"),
        ).lower()

    return requested


REQUESTED_DTYPE = _read_requested_dtype()

"""
print("[precision-debug] JAX_SAMPLERS_CONFIG =", os.environ.get("JAX_SAMPLERS_CONFIG"))
print("[precision-debug] JAX_SAMPLERS_DTYPE  =", os.environ.get("JAX_SAMPLERS_DTYPE"))
print("[precision-debug] LISA_CFG            =", os.environ.get("LISA_CFG"))
print("[precision-debug] REQUESTED_DTYPE      =", REQUESTED_DTYPE)
"""

# must be set before importing jax.numpy
if REQUESTED_DTYPE in ("float64", "f64", "double"):
    _jaxcfg.update("jax_enable_x64", True)
else:
    _jaxcfg.update("jax_enable_x64", False)

#print("[precision-debug] after update: jax_enable_x64 =", _jaxcfg.read("jax_enable_x64"))

    
# NOW import jax/jnp
import jax
import jax.numpy as jnp
from jax import tree_util as tu
from jax import config as jcfg
"""
print("default dtype from ones:", jnp.ones(1).dtype)   # often float32
x = jnp.array([1.0], dtype=jnp.float64)
print("[precision-debug] test float64 dtype =", x.dtype)
"""


DTYPE_MAP = {"float32": jnp.float32, "f32": jnp.float32,
             "float64": jnp.float64, "f64": jnp.float64}

DTYPE  = jnp.float64 if REQUESTED_DTYPE.startswith("float64") else jnp.float32
CDTYPE = jnp.complex128 if DTYPE is jnp.float64 else jnp.complex64
EPS    = jnp.asarray(1e-30, DTYPE)

def to_real(x): return jnp.asarray(x, DTYPE)
def to_cplx(x): return jnp.asarray(x, CDTYPE)

def jnp_real(x): return jnp.asarray(x, jnp.float64 if DTYPE is jnp.float64 else jnp.float32)
def jnp_cplx(x): return jnp.asarray(x, jnp.complex128 if CDTYPE is jnp.complex128 else jnp.complex64)

def resolve(x):
    """Return jnp.float32 or jnp.float64 from a string, dtype object, or None."""
    if x is None:
        dt = jnp.float32
    elif isinstance(x, str):
        s = x.lower()
        if s in ("float64", "f64", "double"): dt = jnp.float64
        elif s in ("float32", "f32", "single"): dt = jnp.float32
        else: raise ValueError(f"Unsupported dtype string: {x!r}")
    elif x in (jnp.float32, jnp.float64, np.float32, np.float64):
        # direct dtype class like <class 'jax.numpy.float64'>
        dt = jnp.float64 if (x in (jnp.float64, np.float64)) else jnp.float32
    elif isinstance(x, (jnp.dtype, np.dtype)):
        dt = jnp.float64 if str(x) == "float64" else jnp.float32
    else:
        raise ValueError(f"Unsupported dtype: {x!r}")

    # Optional: guard against x64 disabled
    try:
        if dt == jnp.float64 and not jcfg.read("jax_enable_x64"):
            # choose: coerce or fail fast. Coerce here:
            # print("Warning: x64 disabled, coercing float64 -> float32")
            dt = jnp.float32
    except Exception:
        pass
    return dt

def as_dtype(x, dt: jnp.dtype):
    return jnp.asarray(x, dt)

def const(x: float, dt: jnp.dtype):
    return jnp.asarray(x, dt)

def neg_inf(dt: jnp.dtype):
    return jnp.asarray(-jnp.inf, dt)

def tiny(dt: jnp.dtype):
    return jnp.finfo(dt).tiny

def cast_tree(tree, dt: jnp.dtype):
    """Cast every array leaf in a pytree to dt."""
    return tu.tree_map(lambda a: jnp.asarray(a, dt) if hasattr(a, "dtype") else a, tree)

def ensure_branch_dtype(value_fn, bad_fn, dt: jnp.dtype):
    """Wrap two thunks so cond branches always return dt."""
    def ok():  return jnp.asarray(value_fn(), dt)
    def bad(): return jnp.asarray(bad_fn(), dt)
    return ok, bad

# RNG helpers that respect dtype
def uniform(key, shape, *, minval=0.0, maxval=1.0, dtype=jnp.float32):
    return jax.random.uniform(key, shape, minval=dtype(minval), maxval=dtype(maxval), dtype=dtype)

def normal(key, shape=(), *, dtype=jnp.float32):
    return jax.random.normal(key, shape, dtype=dtype)
