"""Source-tree import shim for running ``python -m jax_samplers.cli`` from checkout."""
from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

_src_pkg = Path(__file__).resolve().parents[1] / "src" / "jax_samplers"
if _src_pkg.is_dir():
    __path__ = extend_path([str(_src_pkg), *__path__], __name__)
else:
    __path__ = extend_path(__path__, __name__)

from .registry import get_sampler, list_samplers

__all__ = ["get_sampler", "list_samplers"]
