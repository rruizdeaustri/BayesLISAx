from pathlib import Path
import csv
import json
import math
import sys

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from conditional_likelihood_selection import (  # noqa: E402
    RepresentativeCandidate,
    build_representatives,
    evaluate_conditional_significance_variable_k,
    physical_catalogue_to_theta,
    selected_candidate_rows,
)
from within_window_consensus import PosteriorBundle  # noqa: E402


class SyntheticProblem:
    use_gates = True
    marg_Aphi = True
    order_f0 = False
    f_min_cfg = 0.0
    f_max_cfg = 10.0

    def __init__(self, kmax=3):
        self.Kmax = kmax


def decode(theta, kmax=3):
    th = np.asarray(theta).reshape(kmax, 7)
    p = 1.0 / (1.0 + np.exp(-th[:, 6]))
    f0 = 10.0 / (1.0 + np.exp(-th[:, 0]))
    return f0, p


def synthetic_loglike(theta):
    f0, p = decode(theta, np.asarray(theta).size // 7)
    signal = np.sum(p * np.exp(-0.5 * ((f0 - 1.0) / 0.05) ** 2))
    signal += 0.5 * np.sum(p * np.exp(-0.5 * ((f0 - 2.0) / 0.05) ** 2))
    return signal


def test_selected_candidate_rows_ignores_bfdr_and_truth_columns_by_default():
    rows = [
        {"cluster_id": "0", "statistically_selected": "false", "catalogue_approx_snr": "999"},
        {"cluster_id": "1", "statistically_selected": "true", "catalogue_approx_snr": "0"},
        {"cluster_id": "2", "statistically_selected": "false", "catalogue_frequency_hz": "123"},
    ]
    assert [row["cluster_id"] for row in selected_candidate_rows(rows)] == ["0", "1", "2"]


def test_selected_candidate_rows_optional_prefilters():
    rows = [
        {"cluster_id": "0", "mean_inclusion": "0.9", "seed_support_fraction": "1.0", "sampling_quality": "robust"},
        {"cluster_id": "1", "mean_inclusion": "0.2", "seed_support_fraction": "1.0", "sampling_quality": "robust"},
        {"cluster_id": "2", "mean_inclusion": "0.8", "seed_support_fraction": "0.25", "sampling_quality": "confused"},
    ]
    selected = selected_candidate_rows(
        rows,
        minimum_mean_inclusion=0.5,
        minimum_seed_support=0.5,
        allowed_sampling_quality={"robust"},
    )
    assert [row["cluster_id"] for row in selected] == ["0"]


def test_pack_deactivates_inactive_slots():
    theta = physical_catalogue_to_theta(
        np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]]),
        problem=SyntheticProblem(),
        active_mask=np.array([False]),
    )
    f0, p = decode(theta)
    assert np.isclose(f0[0], 1.0)
    assert p[0] < 1e-5
    assert np.all(p[1:] < 1e-5)


def test_conditional_significance_with_synthetic_likelihood_flags_duplicate(tmp_path, monkeypatch):
    import conditional_likelihood_selection as cls

    reps = [
        RepresentativeCandidate(0, {"cluster_id": "0"}, np.array([1.0, 0, 0, 0, 0, 0, 0.9]), 10),
        RepresentativeCandidate(1, {"cluster_id": "1"}, np.array([1.01, 0, 0, 0, 0, 0, 0.9]), 10),
        RepresentativeCandidate(2, {"cluster_id": "2"}, np.array([2.0, 0, 0, 0, 0, 0, 0.9]), 10),
    ]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": {"Kmax": 3}}))
    monkeypatch.setattr(cls, "load_problem_for_kmax", lambda config, factory, kmax, tmpdir: SyntheticProblem(kmax))
    monkeypatch.setattr(SyntheticProblem, "loglikelihood", lambda self, theta: synthetic_loglike(theta), raising=False)
    rows, logl = evaluate_conditional_significance_variable_k(
        reps,
        str(config_path),
        "unused:factory",
        tmp_path,
        duplicate_bins_hz=0.02,
        exclusive_drop_tol=0.0,
    )
    assert logl > 1.4
    assert rows[0]["rho_cond_fixed"] > rows[2]["rho_cond_fixed"]
    assert {row["conditional_status"] for row in rows} >= {"kept", "duplicate"}
    assert all("delta_logl_profiled" in row and "rho_cond_profiled" in row for row in rows)
    assert rows[0]["removal_convention"] == "K-1 problem with removed source block omitted"


def test_actual_production_parameterization_decodes_packed_theta_when_dependencies_available():
    jax = pytest.importorskip("jax")
    gbmod = pytest.importorskip("jax_samplers.problems.lisa_gb_transdim_problem")

    problem = SyntheticProblem(kmax=2)
    physical = np.array([
        [1.0, 1e-18, 0.1, 0.2, 0.3, 0.4, 1.0],
        [2.0, -1e-18, -0.1, -0.2, -0.3, -0.4, 1.0],
    ])
    theta = physical_catalogue_to_theta(physical, problem=problem)
    _, f0_u, fdot_u, _, iota_u, psi_u, lam_u, beta_u, g_u = gbmod.unpack_theta_u(
        jax.numpy.asarray(theta),
        2,
        True,
        True,
    )
    f0 = gbmod.u_to_f0_unordered(f0_u, problem.f_min_cfg, problem.f_max_cfg)

    np.testing.assert_allclose(np.asarray(f0), physical[:, 0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(fdot_u), physical[:, 1], rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(iota_u), physical[:, 2], rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(psi_u), physical[:, 3], rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(lam_u), physical[:, 4], rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(beta_u), physical[:, 5], rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(jax.nn.sigmoid(g_u)), [0.999999, 0.999999], rtol=0, atol=1e-12)


def test_more_candidates_than_slots_fails_clearly():
    physical = np.array([
        [1.0, 0, 0, 0, 0, 0, 0.9],
        [2.0, 0, 0, 0, 0, 0, 0.9],
        [3.0, 0, 0, 0, 0, 0, 0.9],
        [4.0, 0, 0, 0, 0, 0, 0.9],
    ])
    try:
        physical_catalogue_to_theta(physical, problem=SyntheticProblem())
    except ValueError as exc:
        assert "exceed likelihood Kmax=3" in str(exc)
    else:
        raise AssertionError("expected too-many-candidates failure")


def test_build_representatives_from_posterior_cluster_summary():
    samples = np.array(
        [
            [1.0, 0, 0.1, 0.2, 0.3, 0.4, 0.9, 4.0, 0, 0, 0, 0, 0, 0.1],
            [1.1, 0, 0.2, 0.3, 0.4, 0.5, 0.8, 4.1, 0, 0, 0, 0, 0, 0.1],
        ]
    )
    bundle = PosteriorBundle(Path("seed0/posterior.npz"), "0", samples.reshape(2, 2, 7), np.array([0.4, 0.6]))
    reps = build_representatives(
        [{"cluster_id": "0", "f0_median_hz": "1.05", "f0_q05_hz": "0.9", "f0_q95_hz": "1.2"}],
        [bundle],
        f0_index=0,
        p_index=6,
        p_active_min=0.5,
        min_width_hz=0.0,
    )
    assert reps[0].n_support_samples == 2
    assert np.isclose(reps[0].physical[0], 1.1)


def test_integration_writes_expected_outputs(tmp_path, monkeypatch):
    import conditional_likelihood_selection as cls

    candidates = tmp_path / "candidates.csv"
    with candidates.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["window_id", "cluster_id", "statistically_selected", "f0_median_hz", "f0_q05_hz", "f0_q95_hz", "posterior_inclusion_score"])
        writer.writeheader()
        writer.writerow({"window_id": "w", "cluster_id": "0", "statistically_selected": "true", "f0_median_hz": "1.0", "f0_q05_hz": "0.95", "f0_q95_hz": "1.05", "posterior_inclusion_score": "0.9"})
        writer.writerow({"window_id": "w", "cluster_id": "1", "statistically_selected": "false", "f0_median_hz": "2.0", "f0_q05_hz": "1.95", "f0_q95_hz": "2.05", "posterior_inclusion_score": "0.2"})
    posterior = tmp_path / "posterior.npz"
    np.savez_compressed(posterior, samples=np.array([[1.0, 0, 0, 0, 0, 0, 0.9, 2.0, 0, 0, 0, 0, 0, 0.9]]), weights=np.array([1.0]), seed=np.array("0"))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": {"Kmax": 3}}))
    monkeypatch.setattr(cls, "load_problem_for_kmax", lambda config, factory, kmax, tmpdir: SyntheticProblem(kmax))
    monkeypatch.setattr(SyntheticProblem, "loglikelihood", lambda self, theta: synthetic_loglike(theta), raising=False)

    out_sig = tmp_path / "candidate_significance.csv"
    out_final = tmp_path / "final_catalogue.csv"
    out_summary = tmp_path / "summary.json"
    cls.main([
        "--candidates-csv", str(candidates),
        "--posteriors", str(posterior),
        "--config", str(config_path),
        "--window-id", "w",
        "--out-significance", str(out_sig),
        "--out-final-catalogue", str(out_final),
        "--out-summary", str(out_summary),
    ])
    assert out_sig.exists()
    assert out_final.exists()
    payload = json.loads(out_summary.read_text())
    assert payload["truth_information_used_for_selection"] is False
    assert payload["n_union_candidates"] == 2
    with out_sig.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["cluster_id"] for row in rows] == ["0", "1"]
    assert "delta_logl_fixed" in rows[0]
    assert "delta_logl_profiled" in rows[0]


def test_single_candidate_window_emits_k1_no_baseline_status(tmp_path, monkeypatch):
    """K=1 window must not raise; the sole candidate should carry k1_no_baseline."""
    import conditional_likelihood_selection as cls

    reps = [
        RepresentativeCandidate(0, {"cluster_id": "0"}, np.array([1.0, 0, 0, 0, 0, 0, 0.9]), 5),
    ]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": {"Kmax": 1}}))
    monkeypatch.setattr(cls, "load_problem_for_kmax", lambda config, factory, kmax, tmpdir: SyntheticProblem(kmax))
    monkeypatch.setattr(SyntheticProblem, "loglikelihood", lambda self, theta: synthetic_loglike(theta), raising=False)

    rows, logl_full = evaluate_conditional_significance_variable_k(
        reps,
        str(config_path),
        "unused:factory",
        tmp_path,
        duplicate_bins_hz=0.0,
        exclusive_drop_tol=0.0,
    )
    assert len(rows) == 1
    assert rows[0]["conditional_status"] == "k1_no_baseline"
    assert rows[0]["delta_logl_fixed"] == ""
    assert rows[0]["rho_cond_fixed"] == ""
    assert rows[0]["removal_convention"] == "K=0 baseline not supported; single-candidate window"
    assert rows[0]["logL_full"] == logl_full


def test_single_candidate_integration_produces_outputs(tmp_path, monkeypatch):
    """End-to-end main() must succeed and write all outputs for a K=1 window."""
    import conditional_likelihood_selection as cls

    candidates = tmp_path / "candidates.csv"
    with candidates.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["window_id", "cluster_id", "statistically_selected", "f0_median_hz", "f0_q05_hz", "f0_q95_hz", "posterior_inclusion_score"])
        writer.writeheader()
        writer.writerow({"window_id": "w", "cluster_id": "0", "statistically_selected": "true", "f0_median_hz": "1.0", "f0_q05_hz": "0.95", "f0_q95_hz": "1.05", "posterior_inclusion_score": "0.9"})
    posterior = tmp_path / "posterior.npz"
    np.savez_compressed(posterior, samples=np.array([[1.0, 0, 0, 0, 0, 0, 0.9]]), weights=np.array([1.0]), seed=np.array("0"))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": {"Kmax": 1}}))
    monkeypatch.setattr(cls, "load_problem_for_kmax", lambda config, factory, kmax, tmpdir: SyntheticProblem(kmax))
    monkeypatch.setattr(SyntheticProblem, "loglikelihood", lambda self, theta: synthetic_loglike(theta), raising=False)

    out_sig = tmp_path / "candidate_significance.csv"
    out_final = tmp_path / "final_catalogue.csv"
    out_summary = tmp_path / "summary.json"
    cls.main([
        "--candidates-csv", str(candidates),
        "--posteriors", str(posterior),
        "--config", str(config_path),
        "--window-id", "w",
        "--out-significance", str(out_sig),
        "--out-final-catalogue", str(out_final),
        "--out-summary", str(out_summary),
    ])
    assert out_sig.exists()
    assert out_final.exists()
    with out_sig.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["conditional_status"] == "k1_no_baseline"
    with out_final.open(newline="") as fh:
        final_rows = list(csv.DictReader(fh))
    assert len(final_rows) == 1, "k1_no_baseline candidate should appear in final catalogue"


def test_pack_preserves_posterior_gate_probability():
    """Active slots must use the posterior p, not a hardcoded near-1 value."""
    p_posterior = 0.72
    theta = physical_catalogue_to_theta(
        np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_posterior]]),
        problem=SyntheticProblem(kmax=1),
    )
    # Decode gate: logit space → sigmoid
    gate_u = theta[-1]  # last element of the packed vector (7 dims for kmax=1)
    p_decoded = 1.0 / (1.0 + math.exp(-gate_u))
    assert abs(p_decoded - p_posterior) < 1e-9, (
        f"packed gate decoded to {p_decoded:.6g}, expected {p_posterior}"
    )


def test_pack_inactive_slot_ignores_posterior_p():
    """Inactive slots must use inactive_p regardless of the stored posterior p."""
    theta = physical_catalogue_to_theta(
        np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95]]),
        problem=SyntheticProblem(kmax=1),
        active_mask=np.array([False]),
    )
    gate_u = theta[-1]
    p_decoded = 1.0 / (1.0 + math.exp(-gate_u))
    assert p_decoded < 1e-5, f"inactive slot decoded to {p_decoded:.6g}, expected < 1e-5"
