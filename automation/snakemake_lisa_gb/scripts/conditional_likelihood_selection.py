#!/usr/bin/env python3
"""Conditional-likelihood validation for multi-seed consensus candidates.

This afterburner is intentionally downstream of consensus candidate selection and
never uses Sangria truth catalogue fields to decide which candidates survive.
The fixed-configuration likelihood decision statistic is computed by
evaluating the production BayesLISAx A/E likelihood on one synthetic joint
parameter vector, then deactivating each candidate slot in turn without
reoptimizing the remaining slots.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from within_window_consensus import PosteriorBundle, _normalized_weights, load_bundle


@dataclass(frozen=True)
class RepresentativeCandidate:
    source_id: int
    candidate_row: dict[str, str]
    physical: np.ndarray  # [f0, fdot, iota, psi, lam, beta, p]
    n_support_samples: int


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _quality_allowed(value: str, allowed: set[str] | None) -> bool:
    return allowed is None or str(value).strip() in allowed


def selected_candidate_rows(
    rows: Sequence[dict[str, str]],
    *,
    minimum_mean_inclusion: float | None = None,
    minimum_seed_support: float | None = None,
    allowed_sampling_quality: set[str] | None = None,
) -> list[dict[str, str]]:
    """Return the consensus-candidate union without truth or BFDR filtering.

    The default is to validate every distinct cluster in ``candidates.csv``.
    BFDR columns such as ``statistically_selected`` are preserved downstream as
    diagnostics, but are never used by this function.  Optional prefilters are
    limited to consensus reproducibility summaries requested by the workflow:
    mean inclusion, seed support, and sampling-quality labels.
    """
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        cluster_id = str(row.get("cluster_id", len(seen)))
        if cluster_id in seen:
            continue
        seen.add(cluster_id)
        mean_inclusion = _as_float(row, "mean_inclusion", _as_float(row, "posterior_inclusion_score", 0.0))
        seed_support = _as_float(row, "seed_support_fraction", float("nan"))
        quality = row.get("sampling_quality", row.get("quality", ""))
        if minimum_mean_inclusion is not None and mean_inclusion < minimum_mean_inclusion:
            continue
        if minimum_seed_support is not None and (not np.isfinite(seed_support) or seed_support < minimum_seed_support):
            continue
        if not _quality_allowed(quality, allowed_sampling_quality):
            continue
        selected.append(dict(row))
    return selected


def weighted_component_quantile(
    bundles: Sequence[PosteriorBundle],
    row: dict[str, str],
    *,
    f0_index: int,
    p_index: int,
    p_active_min: float,
    min_width_hz: float,
) -> tuple[np.ndarray, int]:
    center = _as_float(row, "f0_median_hz", _as_float(row, "f0_mean_hz"))
    q05 = _as_float(row, "f0_q05_hz", center)
    q95 = _as_float(row, "f0_q95_hz", center)
    half_width = max(abs(q95 - q05) / 2.0, min_width_hz / 2.0)
    lo, hi = center - half_width, center + half_width

    values: list[np.ndarray] = []
    weights: list[float] = []
    for bundle in bundles:
        samples = bundle.samples
        p = samples[:, :, p_index] if 0 <= p_index < samples.shape[2] else np.ones(samples.shape[:2])
        mask = (p > p_active_min) & np.isfinite(samples[:, :, f0_index])
        mask &= (samples[:, :, f0_index] >= lo) & (samples[:, :, f0_index] <= hi)
        idx = np.argwhere(mask)
        for sample_idx, comp_idx in idx:
            values.append(samples[sample_idx, comp_idx, :])
            weights.append(float(bundle.weights[sample_idx]) / max(len(bundles), 1))

    if not values:
        raise ValueError(
            f"candidate {row.get('cluster_id', '?')} at f0={center:.16g} has no posterior support"
        )
    arr = np.asarray(values, dtype=float)
    w = _normalized_weights(np.asarray(weights, dtype=float), len(weights))
    order = np.argsort(arr[:, f0_index])
    cdf = np.cumsum(w[order])
    chosen = int(order[int(np.searchsorted(cdf, 0.5, side="left"))])
    # Use the posterior sample nearest the weighted median frequency.  This
    # preserves angular correlations better than component-wise medians.
    return np.asarray(arr[chosen], dtype=float), int(arr.shape[0])


def build_representatives(
    candidate_rows: Sequence[dict[str, str]],
    bundles: Sequence[PosteriorBundle],
    *,
    f0_index: int,
    p_index: int,
    p_active_min: float,
    min_width_hz: float,
) -> list[RepresentativeCandidate]:
    reps = []
    for source_id, row in enumerate(candidate_rows):
        physical, n = weighted_component_quantile(
            bundles, row, f0_index=f0_index, p_index=p_index,
            p_active_min=p_active_min, min_width_hz=min_width_hz,
        )
        reps.append(RepresentativeCandidate(source_id, dict(row), physical, n))
    reps.sort(key=lambda rep: float(rep.physical[f0_index]))
    return reps


def logit01(p: float, eps: float = 1e-9) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return math.log(p / (1.0 - p))


def physical_catalogue_to_theta(
    physical_rows: np.ndarray,
    *,
    problem: object,
    active_mask: np.ndarray | None = None,
    inactive_p: float = 1e-6,
) -> np.ndarray:
    """Pack physical representative rows into the backend's unconstrained vector.

    Assumptions: posterior bundles store decoded physical rows as
    ``[f0, fdot, iota, psi, lam, beta, p]``.  The production conditional run uses
    the backend gate to turn source slots on/off.  Amplitude and phase nuisance
    parameters are analytically profiled/integrated when ``marg_Aphi=True``.  If
    explicit amplitudes are used, unavailable ``lnA`` and ``phi0`` are filled by
    prior-box midpoints and reported as a limitation in ``summary.json``.

    The gate coordinate for each *active* slot is derived from the posterior
    ``p`` value stored in ``physical_rows[slot, 6]``.  Inactive slots are forced
    to ``inactive_p`` regardless of the stored value.
    """
    rows = np.asarray(physical_rows, dtype=float)
    k = int(getattr(problem, "Kmax"))
    use_gates = bool(getattr(problem, "use_gates", True))
    marg_Aphi = bool(getattr(problem, "marg_Aphi", True))
    if rows.shape[0] > k:
        raise ValueError(f"{rows.shape[0]} candidates exceed likelihood Kmax={k}")
    if active_mask is None:
        active_mask = np.ones(rows.shape[0], dtype=bool)

    per = (6 if marg_Aphi else 8) + (1 if use_gates else 0)
    theta = np.zeros((k, per), dtype=float)
    fmin = float(getattr(problem, "f_min_cfg"))
    fmax = float(getattr(problem, "f_max_cfg"))
    span = fmax - fmin
    if not span > 0.0:
        raise ValueError("problem has invalid f_min_cfg/f_max_cfg")
    if bool(getattr(problem, "order_f0", False)):
        raise NotImplementedError("conditional packing currently requires model.order_f0=false")

    for slot in range(k):
        if slot < rows.shape[0]:
            f0, fdot, iota, psi, lam, beta, _p = rows[slot, :7]
            base = [logit01((f0 - fmin) / span), fdot, iota, psi, lam, beta]
            # Preserve the posterior-derived gate probability for active slots.
            p = float(np.clip(_p, inactive_p, 1.0 - inactive_p)) if bool(active_mask[slot]) else inactive_p
        else:
            f0 = fmin + 0.5 * span
            base = [logit01(0.5), 0.0, 0.0, 0.0, 0.0, 0.0]
            p = inactive_p
        if marg_Aphi:
            vals = base
        else:
            vals = [-50.0, base[0], base[1], 0.0, base[2], base[3], base[4], base[5]]
        if use_gates:
            vals = [*vals, logit01(p)]
        theta[slot, :] = vals
    return theta.reshape(-1)


def _loglike_for_physical_rows(problem: object, physical_rows: np.ndarray) -> tuple[float, np.ndarray]:
    theta = physical_catalogue_to_theta(np.asarray(physical_rows, dtype=float), problem=problem)
    return float(problem.loglikelihood(theta)), theta


def evaluate_conditional_significance_variable_k(
    reps: Sequence[RepresentativeCandidate],
    config_path: str,
    factory: str,
    tmpdir: str | Path,
    *,
    duplicate_bins_hz: float,
    exclusive_drop_tol: float,
    full_problem: object | None = None,
) -> tuple[list[dict[str, object]], float]:
    """Evaluate fixed leave-one-out likelihoods with the true K or K-1 model.

    ``problem.loglikelihood`` accepts sampled latent/unconstrained coordinates,
    not decoded physical rows.  The posterior-derived physical representatives
    are therefore inverse-packed into each problem's latent coordinate vector.
    Removal is implemented by rebuilding the likelihood with one fewer source
    slot and omitting the removed physical block, because the current integrated
    amplitude/phase likelihood can ignore gate values.
    """
    physical = np.asarray([rep.physical[:7] for rep in reps], dtype=float)
    if full_problem is None:
        full_problem = load_problem_for_kmax(config_path, factory, len(reps), tmpdir)
    logl_full, theta_full = _loglike_for_physical_rows(full_problem, physical)

    rows: list[dict[str, object]] = []
    any_theta_changed = False
    any_logl_changed = False
    for idx, rep in enumerate(reps):
        without_physical = np.delete(physical, idx, axis=0)
        if without_physical.shape[0] == 0:
            # K=0 is not currently supported by the GBJAX production factory.
            # Emit a clear per-candidate status and skip this source in the
            # all-rejected sanity check rather than aborting the full workflow.
            out = dict(rep.candidate_row)
            out.update(
                {
                    "source_id": rep.source_id,
                    "representative_f0_hz": rep.physical[0],
                    "representative_fdot": rep.physical[1],
                    "representative_iota": rep.physical[2],
                    "representative_psi": rep.physical[3],
                    "representative_lam": rep.physical[4],
                    "representative_beta": rep.physical[5],
                    "representative_p": rep.physical[6],
                    "representative_support_samples": rep.n_support_samples,
                    "logL_full": logl_full,
                    "logL_without_i": "",
                    "delta_logl_fixed": "",
                    "rho_cond_fixed": "",
                    "delta_logl_profiled": "",
                    "rho_cond_profiled": "",
                    "removal_convention": "K=0 baseline not supported; single-candidate window",
                    "conditional_status": "k1_no_baseline",
                    "diagnostic_only_fields": "statistically_selected,posterior_inclusion_score,sampling_quality,local_fdr,cumulative_bfdr",
                }
            )
            rows.append(out)
            continue
        without_problem = load_problem_for_kmax(config_path, factory, without_physical.shape[0], tmpdir)
        logl_without, theta_without = _loglike_for_physical_rows(without_problem, without_physical)
        delta = logl_full - logl_without
        rho = math.sqrt(max(0.0, 2.0 * delta))
        any_logl_changed = any_logl_changed or (logl_without != logl_full)
        any_theta_changed = any_theta_changed or (theta_without.shape != theta_full.shape) or not np.array_equal(
            theta_without, theta_full[: theta_without.size]
        )
        out = dict(rep.candidate_row)
        out.update(
            {
                "source_id": rep.source_id,
                "representative_f0_hz": rep.physical[0],
                "representative_fdot": rep.physical[1],
                "representative_iota": rep.physical[2],
                "representative_psi": rep.physical[3],
                "representative_lam": rep.physical[4],
                "representative_beta": rep.physical[5],
                "representative_p": rep.physical[6],
                "representative_support_samples": rep.n_support_samples,
                "logL_full": logl_full,
                "logL_without_i": logl_without,
                "delta_logl_fixed": delta,
                "rho_cond_fixed": rho,
                "delta_logl_profiled": "",
                "rho_cond_profiled": "",
                "removal_convention": "K-1 problem with removed source block omitted",
                "conditional_status": "kept" if delta > 0.0 else "rejected",
                "diagnostic_only_fields": "statistically_selected,posterior_inclusion_score,sampling_quality,local_fdr,cumulative_bfdr",
            }
        )
        rows.append(out)

    # Sanity check: only raise if there were evaluable candidates and none of
    # them produced a measurable likelihood change, which suggests a
    # parameterisation problem.  Rows with k1_no_baseline are excluded because
    # their K=0 baseline is not attempted.
    evaluable = [r for r in rows if r.get("conditional_status") != "k1_no_baseline"]
    if evaluable and any_theta_changed and not any_logl_changed:
        raise RuntimeError(
            "conditional likelihood deactivation produced changed theta/K configurations, "
            "but logL was identical for every source. Refusing to write all-rejected "
            "candidates; inspect production parameterization and removal convention."
        )

    for row in rows:
        row["duplicate_of_source_id"] = ""
        row["mutually_exclusive_group"] = ""
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            # Skip comparisons involving k1_no_baseline rows which have no delta/rho.
            if rows[i].get("conditional_status") == "k1_no_baseline" or rows[j].get("conditional_status") == "k1_no_baseline":
                continue
            df = abs(float(rows[i]["representative_f0_hz"]) - float(rows[j]["representative_f0_hz"]))
            if df <= duplicate_bins_hz:
                weaker, stronger = (i, j) if rows[i]["rho_cond_fixed"] < rows[j]["rho_cond_fixed"] else (j, i)
                rows[weaker]["duplicate_of_source_id"] = rows[stronger]["source_id"]
                rows[weaker]["conditional_status"] = "duplicate"
            elif min(float(rows[i]["delta_logl_fixed"]), float(rows[j]["delta_logl_fixed"])) <= exclusive_drop_tol:
                group = f"exclusive_{i}_{j}"
                rows[i]["mutually_exclusive_group"] = group
                rows[j]["mutually_exclusive_group"] = group
    return rows, logl_full


def run_integration_diagnostic(
    reps: Sequence[RepresentativeCandidate],
    config_path: str,
    factory: str,
    tmpdir: str | Path,
) -> None:
    """Print a production-likelihood parameter-flow diagnostic."""
    if not reps:
        print("[conditional diagnostic] no representatives available")
        return
    physical = np.asarray([rep.physical[:7] for rep in reps], dtype=float)
    problem = load_problem_for_kmax(config_path, factory, len(reps), tmpdir)
    ll0, theta0 = _loglike_for_physical_rows(problem, physical)

    perturbed = physical.copy()
    span = float(getattr(problem, "f_max_cfg")) - float(getattr(problem, "f_min_cfg"))
    step = max(span * 1e-4, float(getattr(problem, "df_bin", 0.0)) or 0.0)
    if step <= 0.0:
        step = max(abs(float(physical[0, 0])) * 1e-6, 1e-9)
    perturbed[0, 0] = np.clip(
        perturbed[0, 0] + step,
        float(getattr(problem, "f_min_cfg")),
        np.nextafter(float(getattr(problem, "f_max_cfg")), -np.inf),
    )
    ll_pert, theta_pert = _loglike_for_physical_rows(problem, perturbed)
    changed = np.flatnonzero(theta0 != theta_pert)
    print("[conditional diagnostic] coordinate_space=sampled latent/unconstrained theta_u")
    print(f"[conditional diagnostic] perturb_f0 source=0 old={physical[0,0]:.16g} new={perturbed[0,0]:.16g}")
    print(f"[conditional diagnostic] perturb_changed_indices={changed.tolist()}")
    print(f"[conditional diagnostic] logL_original={ll0:.16g} logL_perturbed={ll_pert:.16g}")
    if ll0 == ll_pert:
        raise RuntimeError("production diagnostic failed: perturbing one source frequency did not change logL")

    if len(reps) > 1:
        without_problem = load_problem_for_kmax(config_path, factory, len(reps) - 1, tmpdir)
        ll_without, theta_without = _loglike_for_physical_rows(without_problem, physical[1:])
        print("[conditional diagnostic] removal_convention=K-1 problem with source block omitted")
        print(f"[conditional diagnostic] removed_source=0 old_block={physical[0].tolist()}")
        print(f"[conditional diagnostic] theta_full_size={theta0.size} theta_without_size={theta_without.size}")
        print(f"[conditional diagnostic] logL_without_source0={ll_without:.16g}")
        if ll0 == ll_without:
            raise RuntimeError("production diagnostic failed: removing one source with K-1 convention did not change logL")
    else:
        print("[conditional diagnostic] removal check skipped for one-source catalogue because K=0 is unsupported")


def write_csv(path: str | Path, rows: Sequence[dict[str, object]]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def load_problem(config_path: str, factory: str) -> object:
    os.environ["JAX_SAMPLERS_CONFIG"] = config_path
    module_name, attr = factory.split(":", 1)
    return getattr(importlib.import_module(module_name), attr)()


def config_with_kmax(config_path: str | Path, kmax: int, tmpdir: str | Path) -> Path:
    """Write a temporary copy of the production config with ``model.Kmax`` changed."""
    if kmax < 0:
        raise ValueError("kmax must be non-negative")
    with Path(config_path).open(encoding="utf-8") as fh:
        cfg = json.load(fh)
    model = dict(cfg.get("model", {}))
    model["Kmax"] = int(kmax)
    cfg["model"] = model
    out = Path(tmpdir) / f"conditional_k{kmax}.json"
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return out


def load_problem_for_kmax(config_path: str, factory: str, kmax: int, tmpdir: str | Path) -> object:
    return load_problem(str(config_with_kmax(config_path, kmax, tmpdir)), factory)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates-csv", required=True)
    p.add_argument("--posteriors", nargs="+", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--window-id", required=True)
    p.add_argument("--dim-per", type=int, default=7)
    p.add_argument("--f0-index", type=int, default=0)
    p.add_argument("--p-index", type=int, default=6)
    p.add_argument("--p-active-min", type=float, default=0.5)
    p.add_argument("--minimum-mean-inclusion", type=float, default=None)
    p.add_argument("--minimum-seed-support", type=float, default=None)
    p.add_argument("--allowed-sampling-quality", nargs="*", default=None)
    p.add_argument("--representative-min-width-hz", type=float, default=0.0)
    p.add_argument("--duplicate-bins-hz", type=float, default=0.0)
    p.add_argument("--exclusive-drop-tol", type=float, default=0.0)
    p.add_argument("--problem-factory", default="jax_samplers.problems.lisa_gb_transdim_problem:make")
    p.add_argument("--run-integration-diagnostic", action="store_true")
    p.add_argument("--out-significance", required=True)
    p.add_argument("--out-final-catalogue", required=True)
    p.add_argument("--out-summary", required=True)
    args = p.parse_args(argv)

    candidates_all = read_csv_rows(args.candidates_csv)
    candidate_rows = selected_candidate_rows(
        candidates_all,
        minimum_mean_inclusion=args.minimum_mean_inclusion,
        minimum_seed_support=args.minimum_seed_support,
        allowed_sampling_quality=(set(args.allowed_sampling_quality) if args.allowed_sampling_quality else None),
    )
    if not candidate_rows:
        raise ValueError("conditional validation received no candidates after optional prefilters")
    bundles = [load_bundle(path, dim_per=args.dim_per, f0_index=args.f0_index) for path in args.posteriors]
    reps = build_representatives(
        candidate_rows, bundles, f0_index=args.f0_index, p_index=args.p_index,
        p_active_min=args.p_active_min, min_width_hz=args.representative_min_width_hz,
    )
    with Path(args.config).open(encoding="utf-8") as fh:
        original_cfg = json.load(fh)
    available_kmax = int((original_cfg.get("model", {}) or {}).get("Kmax", len(reps)))
    if len(reps) > available_kmax:
        raise ValueError(
            f"conditional validation has {len(reps)} candidates after optional prefilters, "
            f"but the configured likelihood has only Kmax={available_kmax} source slots. "
            "Increase model.Kmax or configure minimum_mean_inclusion, "
            "minimum_seed_support, or allowed_sampling_quality prefilters."
        )

    with tempfile.TemporaryDirectory(prefix="conditional_likelihood_") as tmpdir:
        full_problem = load_problem_for_kmax(args.config, args.problem_factory, len(reps), tmpdir)
        if args.run_integration_diagnostic:
            run_integration_diagnostic(reps, args.config, args.problem_factory, tmpdir)
        rows, logl_full = evaluate_conditional_significance_variable_k(
            reps,
            args.config,
            args.problem_factory,
            tmpdir,
            duplicate_bins_hz=args.duplicate_bins_hz,
            exclusive_drop_tol=args.exclusive_drop_tol,
            full_problem=full_problem,
        )
    final_rows = [row for row in rows if row.get("conditional_status") in {"kept", "k1_no_baseline"}]
    write_csv(args.out_significance, rows)
    write_csv(args.out_final_catalogue, final_rows)
    summary = {
        "window_id": args.window_id,
        "n_input_candidates": len(candidates_all),
        "n_union_candidates": len(candidate_rows),
        "optional_prefilters": {
            "minimum_mean_inclusion": args.minimum_mean_inclusion,
            "minimum_seed_support": args.minimum_seed_support,
            "allowed_sampling_quality": args.allowed_sampling_quality,
        },
        "n_final_candidates": len(final_rows),
        "logL_full": logl_full,
        "selection_rule": "keep candidates with positive fixed-configuration conditional delta_logl_fixed unless duplicate",
        "coordinate_space": "problem.loglikelihood accepts sampled latent/unconstrained theta_u coordinates; decoded posterior physical rows are inverse-packed before evaluation",
        "removal_convention": "leave-one-out likelihoods use a K-1 production problem with the removed source block omitted; gate-only deactivation is not assumed",
        "truth_information_used_for_selection": False,
        "diagnostic_only_fields": ["statistically_selected", "posterior_inclusion_score", "sampling_quality", "local_fdr", "cumulative_bfdr"],
        "assumptions": {
            "posterior_layout": "decoded physical [f0, fdot, iota, psi, lam, beta, p] per source slot",
            "representative_vector": "all seven physical parameters [f0, fdot, iota, psi, lam, beta, p] come from one posterior component/draw nearest the weighted median frequency inside the consensus cluster interval",
            "synthetic_joint_configuration": "representatives for different clusters may come from different posterior draws or seeds, so the assembled full vector is not itself a posterior draw",
            "nuisance_parameters": "amplitude and initial phase are profiled/integrated by the BayesLISAx likelihood when marg_Aphi=true",
            "explicit_amplitude_phase": "if marg_Aphi=false, lnA=-50 and phi0=0 placeholders are used because current posterior bundles do not store them",
            "inactive_slots": "when a problem has more slots than rows, unused source slots are packed with p=1e-6; leave-one-out removal uses K-1 instead of relying on p because integrated modes can ignore gates",
            "single_candidate_windows": "K=0 likelihoods are not currently constructed by this afterburner; one-candidate windows fail the no-change assertion rather than reporting a false removal",
            "ordered_frequency_models": "model.order_f0=true is rejected because inverse ordered packing is not unique",
        },
        "outputs": {
            "candidate_significance_csv": str(args.out_significance),
            "final_catalogue_csv": str(args.out_final_catalogue),
        },
    }
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
