from pathlib import Path
import csv
import json
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from conditional_likelihood_selection import (  # noqa: E402
    RepresentativeCandidate,
    build_representatives,
    evaluate_conditional_significance,
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

    def __init__(self, Kmax=3):
        self.Kmax = Kmax

    def loglikelihood(self, theta):
        return synthetic_loglike(theta, self.Kmax)


def decode(theta, kmax=3):
    th = np.asarray(theta).reshape(kmax, 7)
    p = 1.0 / (1.0 + np.exp(-th[:, 6]))
    f0 = 10.0 / (1.0 + np.exp(-th[:, 0]))
    return f0, p


def synthetic_loglike(theta, kmax=3):
    f0, p = decode(theta, kmax)
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
        {"cluster_id": "0", "mean_inclusion": "0.9", "n_seed_support": "3", "sampling_quality": "robust"},
        {"cluster_id": "1", "mean_inclusion": "0.2", "n_seed_support": "3", "sampling_quality": "robust"},
        {"cluster_id": "2", "mean_inclusion": "0.8", "n_seed_support": "1", "sampling_quality": "confused"},
    ]
    selected = selected_candidate_rows(
        rows,
        minimum_mean_inclusion=0.5,
        minimum_seed_support=2,
        allowed_sampling_quality={"robust"},
    )
    assert [row["cluster_id"] for row in selected] == ["0"]


def test_pack_preserves_representative_p_and_pads_inactive_slots():
    theta = physical_catalogue_to_theta(
        np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]]),
        problem=SyntheticProblem(),
    )
    f0, p = decode(theta)
    assert np.isclose(f0[0], 1.0)
    assert np.isclose(p[0], 0.8)
    assert np.all(p[1:] < 1e-6)


def test_conditional_significance_with_synthetic_likelihood_flags_duplicate():
    reps = [
        RepresentativeCandidate(0, {"cluster_id": "0"}, np.array([1.0, 0, 0, 0, 0, 0, 0.9]), 10),
        RepresentativeCandidate(1, {"cluster_id": "1"}, np.array([1.01, 0, 0, 0, 0, 0, 0.9]), 10),
        RepresentativeCandidate(2, {"cluster_id": "2"}, np.array([2.0, 0, 0, 0, 0, 0, 0.9]), 10),
    ]
    problem = SyntheticProblem()
    rows, logl = evaluate_conditional_significance(
        reps,
        problem,
        lambda k: problem,
        duplicate_bins_hz=0.02,
        exclusive_drop_tol=0.0,
    )
    assert logl > 1.4
    assert rows[0]["rho_cond_fixed"] > rows[2]["rho_cond_fixed"]
    assert {row["conditional_status"] for row in rows} >= {"kept", "duplicate"}
    assert all("delta_logl_profiled" in row and "rho_cond_profiled" in row for row in rows)


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
    assert np.allclose(reps[0].physical, [1.1, 0, 0.2, 0.3, 0.4, 0.5, 0.8])
    assert reps[0].representative_seed == "0"
    assert reps[0].representative_draw_index == 1
    assert reps[0].representative_slot_index == 0



def test_minimum_seed_support_integer_threshold_retains_2_and_3():
    rows = [{"cluster_id": str(n), "n_seed_support": str(n)} for n in [1, 2, 3]]
    assert [r["cluster_id"] for r in selected_candidate_rows(rows, minimum_seed_support=2)] == ["2", "3"]


def test_one_candidate_k1_unsupported_baseline_continues():
    reps = [RepresentativeCandidate(0, {"cluster_id": "0"}, np.array([1.0, 0, 0, 0, 0, 0, 0.9]), 1)]
    rows, logl = evaluate_conditional_significance(reps, SyntheticProblem(), lambda k: None, duplicate_bins_hz=0.0, exclusive_drop_tol=0.0)
    assert np.isfinite(logl)
    assert rows[0]["conditional_status"] == "unsupported_single_candidate_baseline"


def test_all_identical_multi_candidate_likelihood_guard():
    class FlatProblem(SyntheticProblem):
        def loglikelihood(self, theta):
            return 1.0
    reps = [
        RepresentativeCandidate(0, {"cluster_id": "0"}, np.array([1.0, 0, 0, 0, 0, 0, 0.9]), 1),
        RepresentativeCandidate(1, {"cluster_id": "1"}, np.array([2.0, 0, 0, 0, 0, 0, 0.9]), 1),
    ]
    try:
        evaluate_conditional_significance(reps, FlatProblem(), lambda k: FlatProblem(), duplicate_bins_hz=0.0, exclusive_drop_tol=0.0)
    except AssertionError as exc:
        assert "all leave-one-out likelihoods" in str(exc)
    else:
        raise AssertionError("expected all-identical guard")

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
    monkeypatch.setattr(cls, "load_problem", lambda config, factory: SyntheticProblem(json.loads(Path(config).read_text()).get("model", {}).get("Kmax", 3)))
    monkeypatch.setattr(SyntheticProblem, "loglikelihood", lambda self, theta: synthetic_loglike(theta, self.Kmax), raising=False)
    (tmp_path / "config.json").write_text(json.dumps({"model": {"Kmax": 3}}))

    out_sig = tmp_path / "candidate_significance.csv"
    out_final = tmp_path / "final_catalogue.csv"
    out_summary = tmp_path / "summary.json"
    cls.main([
        "--candidates-csv", str(candidates),
        "--posteriors", str(posterior),
        "--config", str(tmp_path / "config.json"),
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


def test_conditional_snakefile_wires_pythonpath_and_diagnostic_flag():
    snakefile = (Path(__file__).resolve().parents[1] / "Snakefile.conditional_likelihood").read_text()
    assert "PYTHONPATH={params.repo_root}/src" in snakefile
    assert "run_integration_diagnostic" in snakefile
    assert "--run-integration-diagnostic" in snakefile


def test_profiled_likelihood_fields_and_mask_improve_or_preserve():
    from conditional_likelihood_selection import OptimizerSettings
    reps = [
        RepresentativeCandidate(0, {"cluster_id": "0"}, np.array([1.2, 0, 0, 0, 0, 0, 0.9]), 1),
        RepresentativeCandidate(1, {"cluster_id": "1"}, np.array([2.2, 0, 0, 0, 0, 0, 0.9]), 1),
    ]
    rows, _ = evaluate_conditional_significance(
        reps,
        SyntheticProblem(2),
        lambda k: SyntheticProblem(k),
        duplicate_bins_hz=0.0,
        exclusive_drop_tol=-99.0,
        optimizer_settings=OptimizerSettings(enabled=True, mask=("f0",), maxiter=20),
    )
    assert all(np.isfinite(float(r["delta_logl_profiled"])) for r in rows)
    assert all(r["optimized_parameter_mask"] == "f0" for r in rows)
    assert rows[0]["logL_full_profiled"] >= rows[0]["logL_full_initial"]
    assert all(float(r["logL_without_i_profiled"]) >= float(r["logL_without_i_initial"]) for r in rows)


def test_failed_optimizer_handling_with_bad_mask():
    from conditional_likelihood_selection import OptimizerSettings, profile_theta, physical_catalogue_to_theta
    theta = physical_catalogue_to_theta(np.array([[1.0, 0, 0, 0, 0, 0, 0.9]]), problem=SyntheticProblem(1))
    result = profile_theta(SyntheticProblem(1), theta, k_active=1, settings=OptimizerSettings(enabled=True, mask=("not_a_parameter",)))
    assert result.success is False
    assert "unsupported" in result.message


def test_posterior_baseline_selection():
    from conditional_likelihood_selection import evaluate_posterior_baseline
    bundle = PosteriorBundle(Path("seed9/posterior.npz"), "9", np.array([
        [[3.0, 0, 0, 0, 0, 0, 0.9]],
        [[1.0, 0, 0, 0, 0, 0, 0.9]],
    ]), np.ones(2))
    best, seed, draw = evaluate_posterior_baseline([bundle], SyntheticProblem(1), max_draws_per_seed=2)
    assert np.isfinite(best)
    assert seed == "9"
    assert draw == 1


def test_truth_fields_do_not_affect_profiled_selection():
    rows = [
        {"cluster_id": "0", "catalogue_frequency_hz": "999", "statistically_selected": "false"},
        {"cluster_id": "1", "catalogue_frequency_hz": "0", "statistically_selected": "true"},
    ]
    assert [r["cluster_id"] for r in selected_candidate_rows(rows)] == ["0", "1"]


def test_cli_profiled_options_and_posterior_baseline(tmp_path, monkeypatch):
    import conditional_likelihood_selection as cls
    candidates = tmp_path / "candidates.csv"
    with candidates.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["cluster_id", "f0_median_hz", "f0_q05_hz", "f0_q95_hz"])
        writer.writeheader(); writer.writerow({"cluster_id": "0", "f0_median_hz": "1.2", "f0_q05_hz": "1.1", "f0_q95_hz": "1.3"})
    posterior = tmp_path / "posterior.npz"
    np.savez_compressed(posterior, samples=np.array([[1.2,0,0,0,0,0,0.9],[1.0,0,0,0,0,0,0.9]]), weights=np.ones(2), seed=np.array("7"))
    monkeypatch.setattr(cls, "load_problem", lambda config, factory: SyntheticProblem(json.loads(Path(config).read_text()).get("model", {}).get("Kmax", 1)))
    cfg = tmp_path / "config.json"; cfg.write_text(json.dumps({"model": {"Kmax": 1}}))
    out_sig = tmp_path / "sig.csv"; out_final = tmp_path / "final.csv"; out_summary = tmp_path / "summary.json"
    cls.main(["--candidates-csv", str(candidates), "--posteriors", str(posterior), "--config", str(cfg), "--window-id", "w", "--profile-likelihood", "--optimized-parameter-mask", "f0", "--posterior-baseline-draws", "2", "--out-significance", str(out_sig), "--out-final-catalogue", str(out_final), "--out-summary", str(out_summary)])
    payload = json.loads(out_summary.read_text())
    assert payload["profile_likelihood_enabled"] is True
    assert payload["best_posterior_seed"] == "7"
    with out_sig.open(newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["optimized_parameter_mask"] == "f0"
    assert "rho_cond_profiled" in row
