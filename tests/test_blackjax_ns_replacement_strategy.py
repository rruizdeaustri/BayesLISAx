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

    kwargs = blackjax_ns._nss_replacement_kwargs(ctor, "cluster_aware")

    update_strategy = kwargs["update_strategy"]
    assert update_strategy.func is sentinel
    assert update_strategy.keywords == {
        "print_diagnostics": False,
        "auto_fallback": False,
    }


def test_cluster_aware_eager_wraps_update_with_eager_true(monkeypatch):
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

    kwargs = blackjax_ns._nss_replacement_kwargs(
        ctor, "cluster_aware", cluster_aware_eager=True
    )

    update_strategy = kwargs["update_strategy"]
    assert update_strategy.func is sentinel
    assert update_strategy.keywords == {
        "print_diagnostics": False,
        "auto_fallback": False,
        "eager": True,
    }


def test_global_diagnostics_resolves_diagnostic_update(monkeypatch):
    def diagnostic(*args, **kwargs):
        return None

    def ctor(update_strategy=None, **kwargs):
        return update_strategy

    fake_blackjax = types.SimpleNamespace(
        ns=types.SimpleNamespace(
            nss=types.SimpleNamespace(diagnostic_update_with_mcmc_take_last=diagnostic)
        )
    )
    monkeypatch.setattr(blackjax_ns, "blackjax", fake_blackjax)

    assert blackjax_ns._nss_replacement_kwargs(
        ctor, "global", replacement_diagnostics=True
    ) == {"update_strategy": diagnostic}


def test_cluster_aware_auto_fallback_options_are_forwarded(monkeypatch):
    def sentinel(
        constrained_mcmc_step_fn,
        num_mcmc_steps,
        num_delete,
        *,
        print_diagnostics=True,
        eager=False,
        auto_fallback=False,
        warmup_attempts=25,
        min_success_rate=0.5,
        max_runtime_ratio=2.0,
    ):
        return {
            "print_diagnostics": print_diagnostics,
            "eager": eager,
            "auto_fallback": auto_fallback,
            "warmup_attempts": warmup_attempts,
            "min_success_rate": min_success_rate,
            "max_runtime_ratio": max_runtime_ratio,
        }

    def ctor(update_strategy=None, **kwargs):
        return update_strategy

    fake_blackjax = types.SimpleNamespace(
        ns=types.SimpleNamespace(
            nss=types.SimpleNamespace(cluster_aware_update_with_mcmc_take_last=sentinel)
        )
    )
    monkeypatch.setattr(blackjax_ns, "blackjax", fake_blackjax)

    kwargs = blackjax_ns._nss_replacement_kwargs(
        ctor,
        "cluster_aware",
        replacement_diagnostics=True,
        cluster_aware_auto_fallback=True,
        cluster_aware_warmup_attempts=20,
        cluster_aware_min_success_rate=0.2,
        cluster_aware_max_runtime_ratio=2.0,
    )

    update_strategy = kwargs["update_strategy"]
    assert update_strategy.func is sentinel
    assert update_strategy.keywords == {
        "print_diagnostics": True,
        "auto_fallback": True,
        "warmup_attempts": 20,
        "min_success_rate": 0.2,
        "max_runtime_ratio": 2.0,
    }
    assert update_strategy(None, 16, 300) == {
        "print_diagnostics": True,
        "eager": False,
        "auto_fallback": True,
        "warmup_attempts": 20,
        "min_success_rate": 0.2,
        "max_runtime_ratio": 2.0,
    }


def test_cluster_aware_replacement_requires_blackjax_support(monkeypatch):
    def ctor(update_strategy=None):
        return update_strategy

    fake_blackjax = types.SimpleNamespace(ns=types.SimpleNamespace(nss=types.SimpleNamespace()))
    monkeypatch.setattr(blackjax_ns, "blackjax", fake_blackjax)

    with pytest.raises(AttributeError, match="cluster_aware.*requires.*cluster_aware_update_with_mcmc_take_last"):
        blackjax_ns._nss_replacement_kwargs(ctor, "cluster_aware")
