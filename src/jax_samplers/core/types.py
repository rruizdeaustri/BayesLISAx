from typing import Any
try:
    from jax.random import KeyArray as PRNGKey
except Exception:
    try:
        from jax import Array as PRNGKey # older JAX
    except Exception:
        PRNGKey = Any # last resort typing
