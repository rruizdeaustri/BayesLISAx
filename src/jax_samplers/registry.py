# registry.py
from __future__ import annotations
from typing import Any, Callable, Dict, Type, Union, Optional
import importlib
import importlib.util

Entry = Union[Type[Any], Callable[[], Type[Any]]]
_registry: Dict[str, Entry] = {}

def register_sampler(name: str, entry: Optional[Entry] = None):
    """
    Supports both:
      1) decorator style: @register_sampler("ns")
      2) call style:      register_sampler("ns", NSConfig)
    """
    if entry is None:
        def _decorator(cls_or_entry: Entry):
            _registry[name] = cls_or_entry
            return cls_or_entry
        return _decorator

    _registry[name] = entry
    return entry

def list_samplers():
    return list(_registry.keys())

def get_sampler(name: str) -> Type[Any]:
    try:
        entry = _registry[name]
    except KeyError:
        raise KeyError(f"Unknown sampler: {name}. Available: {list(_registry)}")

    # resolve lazy loader
    if callable(entry) and not isinstance(entry, type):
        entry = entry()
        _registry[name] = entry
    return entry


# Import modules (not classes) after register_sampler exists
from .samplers import blackjax_ns
from .samplers import smc_nuts
from .samplers import smc_hmc
from .samplers import elliptical_slice
from .samplers import pt_rw
from .samplers import rj_mh
from .samplers import rj_es
if importlib.util.find_spec("numpyro") is not None:
    from .samplers import numpyro_nuts

# Lazy JAXNS: only imported if requested
def _lazy_jaxns():
    mod = importlib.import_module("jax_samplers.samplers.jaxns_unified")
    return mod.JAXNSConfig

register_sampler("jaxns", _lazy_jaxns)




"""
from __future__ import annotations
from typing import Dict, Type

_registry: Dict[str, type] = {}

def register_sampler(name: str):
    def deco(cls: Type):
        _registry[name] = cls
        return cls
    return deco

def get_sampler(name: str):
    if name not in _registry:
        raise KeyError(f"Unknown sampler: {name}. Available: {list(_registry)}")
    return _registry[name]

def list_samplers():
    return sorted(_registry.keys())
"""
