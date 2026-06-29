import importlib
import sys
import types


def load_blackjax_ns_with_fake_nss(monkeypatch):
    import blackjax

    diagnostic = object()

    def cluster_aware(*args, **kwargs):
        return None
    fake_nss = types.SimpleNamespace(
        diagnostic_update_with_mcmc_take_last=diagnostic,
        cluster_aware_update_with_mcmc_take_last=cluster_aware,
    )
    fake_ns = types.SimpleNamespace(nss=fake_nss)

    monkeypatch.setattr(blackjax, "ns", fake_ns, raising=False)
    monkeypatch.setitem(sys.modules, "blackjax.ns", fake_ns)
    monkeypatch.setitem(sys.modules, "blackjax.ns.nss", fake_nss)
    monkeypatch.setitem(sys.modules, "blackjax.ns.utils", types.SimpleNamespace(finalise=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "anesthetic", types.SimpleNamespace(NestedSamples=object))
    for name in [n for n in list(sys.modules) if n == "jax_samplers" or n.startswith("jax_samplers.")]:
        sys.modules.pop(name, None)
    sys.modules.pop("jax_samplers.samplers.blackjax_ns", None)
    module = importlib.import_module("jax_samplers.samplers.blackjax_ns")
    return module, diagnostic, cluster_aware


def test_global_diagnostics_selects_diagnostic_update(monkeypatch):
    module, diagnostic, _ = load_blackjax_ns_with_fake_nss(monkeypatch)
    cfg = module.NSConfig(replacement_strategy="global", replacement_diagnostics=True)

    kwargs = module._build_nss_kwargs(lambda x: x, lambda x: x, 2, 7, cfg)

    assert kwargs["update_strategy"] is diagnostic


def test_default_diagnostics_selects_diagnostic_update(monkeypatch):
    module, diagnostic, _ = load_blackjax_ns_with_fake_nss(monkeypatch)
    cfg = module.NSConfig(replacement_strategy="default", replacement_diagnostics=True)

    kwargs = module._build_nss_kwargs(lambda x: x, lambda x: x, 2, 7, cfg)

    assert kwargs["update_strategy"] is diagnostic


def test_cluster_aware_maps_diagnostics_and_forwards_fallback_kwargs(monkeypatch):
    module, _, cluster_aware = load_blackjax_ns_with_fake_nss(monkeypatch)
    cfg = module.NSConfig(
        replacement_strategy="cluster_aware",
        replacement_diagnostics=True,
        cluster_aware_eager=True,
        cluster_aware_auto_fallback=True,
        cluster_aware_warmup_attempts=13,
        cluster_aware_min_success_rate=0.75,
        cluster_aware_max_runtime_ratio=3.5,
    )

    kwargs = module._build_nss_kwargs(lambda x: x, lambda x: x, 2, 7, cfg)

    strategy = kwargs["update_strategy"]

    assert strategy.func is cluster_aware
    assert strategy.keywords["print_diagnostics"] is True
    assert strategy.keywords["eager"] is True
    assert strategy.keywords["auto_fallback"] is True
    assert strategy.keywords["warmup_attempts"] == 13
    assert strategy.keywords["min_success_rate"] == 0.75
    assert strategy.keywords["max_runtime_ratio"] == 3.5

    for bad_key in (
        "replacement_diagnostics",
        "print_diagnostics",
        "eager",
        "auto_fallback",
        "warmup_attempts",
        "min_success_rate",
        "max_runtime_ratio",
    ):
        assert bad_key not in kwargs
