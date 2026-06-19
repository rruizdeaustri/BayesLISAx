import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("anesthetic", types.SimpleNamespace(NestedSamples=object))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jax_samplers.samplers import blackjax_ns


def test_global_replacement_uses_blackjax_default_kwargs():
    def ctor(**kwargs):
        return kwargs

    assert blackjax_ns._nss_replacement_kwargs(ctor, "global") == {}
    assert blackjax_ns._nss_replacement_kwargs(ctor, "default") == {}
    assert blackjax_ns._normalise_replacement_strategy(None) == "global"


def test_cluster_aware_replacement_resolves_experimental_update(monkeypatch):
    def sentinel(*args, **kwargs):
        return None

    def ctor(update_strategy=None, **kwargs):
        return update_strategy

    fake_blackjax = types.SimpleNamespace(
        ns=types.SimpleNamespace(
            nss=types.SimpleNamespace(cluster_aware_update_with_mcmc_take_last=sentinel)
        )
    )
    monkeypatch.setattr(blackjax_ns, "blackjax", fake_blackjax)

    assert blackjax_ns._nss_replacement_kwargs(ctor, "cluster_aware") == {
        "update_strategy": sentinel
    }


def test_cluster_aware_replacement_requires_blackjax_support(monkeypatch):
    def ctor(update_strategy=None):
        return update_strategy

    fake_blackjax = types.SimpleNamespace(ns=types.SimpleNamespace(nss=types.SimpleNamespace()))
    monkeypatch.setattr(blackjax_ns, "blackjax", fake_blackjax)

    with pytest.raises(AttributeError, match="cluster_aware.*requires.*cluster_aware_update_with_mcmc_take_last"):
        blackjax_ns._nss_replacement_kwargs(ctor, "cluster_aware")
