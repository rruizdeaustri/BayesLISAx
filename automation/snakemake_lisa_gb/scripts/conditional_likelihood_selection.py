#!/usr/bin/env python3
"""Conditional-likelihood validation for multi-seed consensus candidates.

This afterburner is downstream of consensus candidate selection.  It validates
all consensus clusters by default, treats BFDR/sampling/truth columns as
diagnostics only, builds each representative from one complete posterior source
block, and evaluates a fixed-configuration leave-one-out statistic with the
production latent parameter flow.
"""
from __future__ import annotations

import argparse, csv, importlib, json, math, os, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from within_window_consensus import PosteriorBundle, _normalized_weights, load_bundle

DIAGNOSTIC_ONLY_FIELDS = ["statistically_selected", "posterior_inclusion_score", "sampling_quality", "local_fdr", "cumulative_bfdr", "bfdr", "catalogue_frequency_hz", "catalogue_approx_snr"]

@dataclass(frozen=True)
class RepresentativeCandidate:
    source_id: int
    candidate_row: dict[str, str]
    physical: np.ndarray  # [f0, fdot, iota, psi, lam, beta, p]
    n_support_samples: int
    representative_seed: str = ""
    representative_draw_index: int = -1
    representative_slot_index: int = -1


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try: return float(row.get(key, default))
    except (TypeError, ValueError): return default


def _as_int(row: dict[str, str], key: str, default: int = -1) -> int:
    try: return int(float(str(row.get(key, default)).strip()))
    except (TypeError, ValueError): return default


def selected_candidate_rows(rows: Sequence[dict[str, str]], *, minimum_mean_inclusion: float | None = None, minimum_seed_support: int | None = None, allowed_sampling_quality: set[str] | None = None) -> list[dict[str, str]]:
    """Return consensus clusters to validate; BFDR/truth fields are diagnostics only."""
    selected, seen = [], set()
    for row in rows:
        cluster_id = str(row.get("cluster_id", len(seen)))
        if cluster_id in seen: continue
        seen.add(cluster_id)
        mean_inclusion = _as_float(row, "mean_inclusion", _as_float(row, "posterior_inclusion_score", 0.0))
        n_seed_support = _as_int(row, "n_seed_support")
        quality = str(row.get("sampling_quality", row.get("quality", ""))).strip()
        if minimum_mean_inclusion is not None and mean_inclusion < minimum_mean_inclusion: continue
        if minimum_seed_support is not None and n_seed_support < int(minimum_seed_support): continue
        if allowed_sampling_quality is not None and quality not in allowed_sampling_quality: continue
        selected.append(dict(row))
    return selected


def weighted_component_quantile(bundles: Sequence[PosteriorBundle], row: dict[str, str], *, f0_index: int, p_index: int, p_active_min: float, min_width_hz: float) -> tuple[np.ndarray, int, str, int, int]:
    center = _as_float(row, "f0_median_hz", _as_float(row, "f0_mean_hz"))
    q05, q95 = _as_float(row, "f0_q05_hz", center), _as_float(row, "f0_q95_hz", center)
    half_width = max(abs(q95 - q05) / 2.0, min_width_hz / 2.0)
    lo, hi = center - half_width, center + half_width
    values: list[np.ndarray] = []; weights: list[float] = []; provenance: list[tuple[str,int,int]] = []
    for bundle in bundles:
        samples = bundle.samples
        p = samples[:, :, p_index] if 0 <= p_index < samples.shape[2] else np.ones(samples.shape[:2])
        mask = (p > p_active_min) & np.isfinite(samples[:, :, f0_index]) & (samples[:, :, f0_index] >= lo) & (samples[:, :, f0_index] <= hi)
        for draw_idx, slot_idx in np.argwhere(mask):
            values.append(np.asarray(samples[draw_idx, slot_idx, :7], dtype=float))
            weights.append(float(bundle.weights[draw_idx]) / max(len(bundles), 1))
            provenance.append((str(bundle.seed), int(draw_idx), int(slot_idx)))
    if not values:
        raise ValueError(f"candidate {row.get('cluster_id', '?')} at f0={center:.16g} has no posterior support")
    arr = np.asarray(values, dtype=float); w = _normalized_weights(np.asarray(weights), len(weights))
    order = np.argsort(arr[:, f0_index]); cdf = np.cumsum(w[order])
    chosen = int(order[int(np.searchsorted(cdf, 0.5, side="left"))])
    seed, draw, slot = provenance[chosen]
    return arr[chosen, :7].copy(), int(arr.shape[0]), seed, draw, slot


def build_representatives(candidate_rows: Sequence[dict[str, str]], bundles: Sequence[PosteriorBundle], *, f0_index: int, p_index: int, p_active_min: float, min_width_hz: float) -> list[RepresentativeCandidate]:
    reps = []
    for source_id, row in enumerate(candidate_rows):
        phys, n, seed, draw, slot = weighted_component_quantile(bundles, row, f0_index=f0_index, p_index=p_index, p_active_min=p_active_min, min_width_hz=min_width_hz)
        reps.append(RepresentativeCandidate(source_id, dict(row), phys, n, seed, draw, slot))
    reps.sort(key=lambda rep: float(rep.physical[0]))
    return reps


def logit01(p: float, eps: float = 1e-9) -> float:
    p = float(np.clip(p, eps, 1.0 - eps)); return math.log(p / (1.0 - p))


def physical_catalogue_to_theta(physical_rows: np.ndarray, *, problem: object) -> np.ndarray:
    """Inverse-pack decoded [f0, fdot, iota, psi, lam, beta, p] rows to theta_u.

    ``problem.loglikelihood`` receives the returned latent/unconstrained vector.
    Representative gate probabilities are preserved instead of forcing active
    sources to p≈1.  Empty slots are only used when the problem has more slots
    than representatives; source removal should use a K-1 problem.
    """
    rows = np.asarray(physical_rows, dtype=float).reshape((-1, 7))
    k = int(getattr(problem, "Kmax")); use_gates = bool(getattr(problem, "use_gates", True)); marg_Aphi = bool(getattr(problem, "marg_Aphi", True))
    if rows.shape[0] > k: raise ValueError(f"{rows.shape[0]} candidates exceed likelihood Kmax={k}")
    if bool(getattr(problem, "order_f0", False)): raise NotImplementedError("conditional packing currently requires model.order_f0=false")
    fmin, fmax = float(getattr(problem, "f_min_cfg")), float(getattr(problem, "f_max_cfg")); span = fmax - fmin
    if not span > 0: raise ValueError("problem has invalid f_min_cfg/f_max_cfg")
    per = (6 if marg_Aphi else 8) + (1 if use_gates else 0); theta = np.zeros((k, per), dtype=float)
    for slot in range(k):
        if slot < rows.shape[0]:
            f0, fdot, iota, psi, lam, beta, p = rows[slot, :7]
        else:
            f0, fdot, iota, psi, lam, beta, p = fmin + 0.5 * span, 0, 0, 0, 0, 0, 1e-9
        base = [logit01((f0 - fmin) / span), fdot, iota, psi, lam, beta]
        vals = base if marg_Aphi else [-50.0, base[0], base[1], 0.0, base[2], base[3], base[4], base[5]]
        if use_gates: vals = [*vals, logit01(p)]
        theta[slot, :] = vals
    return theta.reshape(-1)


def _same_config_with_k(config_path: str, k: int) -> str:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    cfg.setdefault("model", {})["Kmax"] = int(k)
    fh = tempfile.NamedTemporaryFile("w", suffix=f".K{k}.json", delete=False, encoding="utf-8")
    json.dump(cfg, fh); fh.close(); return fh.name


def evaluate_conditional_significance(reps: Sequence[RepresentativeCandidate], full_problem: object, problem_for_k: Callable[[int], object | None], *, duplicate_bins_hz: float, exclusive_drop_tol: float) -> tuple[list[dict[str, object]], float]:
    physical = np.asarray([rep.physical for rep in reps], dtype=float)
    if len(reps) > int(getattr(full_problem, "Kmax")): raise ValueError(f"{len(reps)} candidates exceed likelihood Kmax={getattr(full_problem, 'Kmax')}")
    full_theta = physical_catalogue_to_theta(physical, problem=full_problem); logl_full = float(full_problem.loglikelihood(full_theta))
    rows=[]; without_values=[]
    for idx, rep in enumerate(reps):
        out = dict(rep.candidate_row); status="kept"; logl_without=float("nan"); delta=float("nan"); rho=float("nan")
        if len(reps) == 1:
            k0 = problem_for_k(0)
            if k0 is None: status = "unsupported_single_candidate_baseline"
            else:
                logl_without = float(k0.loglikelihood(physical_catalogue_to_theta(np.empty((0,7)), problem=k0)))
        else:
            km1 = problem_for_k(len(reps)-1)
            if km1 is None: status = "unsupported_k_minus_1_removal"
            else:
                keep = np.delete(physical, idx, axis=0); logl_without = float(km1.loglikelihood(physical_catalogue_to_theta(keep, problem=km1)))
        if np.isfinite(logl_without):
            delta = logl_full - logl_without; rho = math.sqrt(max(0.0, 2.0 * delta)); status = "kept" if delta > 0.0 else "rejected"; without_values.append(logl_without)
        out.update({"source_id": rep.source_id,"representative_seed": rep.representative_seed,"representative_draw_index": rep.representative_draw_index,"representative_slot_index": rep.representative_slot_index,"representative_f0_hz": rep.physical[0],"representative_fdot": rep.physical[1],"representative_iota": rep.physical[2],"representative_psi": rep.physical[3],"representative_lam": rep.physical[4],"representative_beta": rep.physical[5],"representative_p": rep.physical[6],"representative_support_samples": rep.n_support_samples,"logL_full": logl_full,"logL_without_i": logl_without,"delta_logl_fixed": delta,"rho_cond_fixed": rho,"delta_logl_profiled":"","rho_cond_profiled":"","conditional_status":status,"diagnostic_only_fields":",".join(DIAGNOSTIC_ONLY_FIELDS)})
        rows.append(out)
    if len(reps) > 1 and without_values and all(v == logl_full for v in without_values):
        raise AssertionError("all leave-one-out likelihoods are exactly identical to logL_full in a multi-candidate run")
    for row in rows: row["duplicate_of_source_id"]=""; row["mutually_exclusive_group"]=""
    for i in range(len(rows)):
        for j in range(i+1, len(rows)):
            df=abs(float(rows[i]["representative_f0_hz"])-float(rows[j]["representative_f0_hz"]));
            if df <= duplicate_bins_hz:
                weaker,stronger=(i,j) if float(rows[i].get("rho_cond_fixed") or 0)<float(rows[j].get("rho_cond_fixed") or 0) else (j,i); rows[weaker]["duplicate_of_source_id"]=rows[stronger]["source_id"]; rows[weaker]["conditional_status"]="duplicate"
            elif min(float(rows[i].get("delta_logl_fixed") or 0), float(rows[j].get("delta_logl_fixed") or 0)) <= exclusive_drop_tol:
                rows[i]["mutually_exclusive_group"]=rows[j]["mutually_exclusive_group"]=f"exclusive_{i}_{j}"
    return rows, logl_full


def write_csv(path: str | Path, rows: Sequence[dict[str, object]]) -> None:
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); fields=list(dict.fromkeys(k for r in rows for k in r.keys()))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w=csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)


def load_problem(config_path: str, factory: str) -> object:
    os.environ["JAX_SAMPLERS_CONFIG"] = config_path; mod, attr = factory.split(":",1); return getattr(importlib.import_module(mod), attr)()


def main(argv: Sequence[str] | None = None) -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--candidates-csv", required=True); p.add_argument("--posteriors", nargs="+", required=True); p.add_argument("--config", required=True); p.add_argument("--window-id", required=True); p.add_argument("--dim-per", type=int, default=7); p.add_argument("--f0-index", type=int, default=0); p.add_argument("--p-index", type=int, default=6); p.add_argument("--p-active-min", type=float, default=0.5); p.add_argument("--minimum-mean-inclusion", type=float, default=None); p.add_argument("--minimum-seed-support", type=int, default=None); p.add_argument("--allowed-sampling-quality", nargs="*", default=None); p.add_argument("--representative-min-width-hz", type=float, default=0.0); p.add_argument("--duplicate-bins-hz", type=float, default=0.0); p.add_argument("--exclusive-drop-tol", type=float, default=0.0); p.add_argument("--run-integration-diagnostic", action="store_true"); p.add_argument("--problem-factory", default="jax_samplers.problems.lisa_gb_transdim_problem:make"); p.add_argument("--out-significance", required=True); p.add_argument("--out-final-catalogue", required=True); p.add_argument("--out-summary", required=True); args=p.parse_args(argv)
    all_rows=read_csv_rows(args.candidates_csv); cand=selected_candidate_rows(all_rows, minimum_mean_inclusion=args.minimum_mean_inclusion, minimum_seed_support=args.minimum_seed_support, allowed_sampling_quality=set(args.allowed_sampling_quality) if args.allowed_sampling_quality else None)
    if not cand: raise ValueError("conditional validation received no candidates after optional prefilters")
    bundles=[load_bundle(x, dim_per=args.dim_per, f0_index=args.f0_index) for x in args.posteriors]; reps=build_representatives(cand,bundles,f0_index=args.f0_index,p_index=args.p_index,p_active_min=args.p_active_min,min_width_hz=args.representative_min_width_hz)
    full_problem=load_problem(args.config,args.problem_factory); kmax=int(getattr(full_problem,"Kmax"))
    if len(reps)>kmax: raise ValueError(f"conditional validation has {len(reps)} candidates after optional prefilters, but the configured likelihood has only Kmax={kmax} source slots. Increase model.Kmax or configure minimum_mean_inclusion=0.66, minimum_seed_support, or allowed_sampling_quality prefilters.")
    made=[]
    def problem_for_k(k:int):
        if k<0: return None
        if k==kmax: return full_problem
        try:
            cfg=_same_config_with_k(args.config,k); made.append(cfg); return load_problem(cfg,args.problem_factory)
        except Exception: return None
    rows, logl_full=evaluate_conditional_significance(reps, full_problem, problem_for_k, duplicate_bins_hz=args.duplicate_bins_hz, exclusive_drop_tol=args.exclusive_drop_tol)
    physical=np.asarray([r.physical for r in reps]); synth_logl=logl_full; baseline_logl=rows[0].get("logL_without_i") if rows else None
    max_posterior_logl=None
    if args.run_integration_diagnostic and len(reps)>0:
        theta=physical_catalogue_to_theta(physical, problem=full_problem); pert=theta.copy(); pert[0]+=1e-3; lp=float(full_problem.loglikelihood(pert)); print("integration_diagnostic changed_indices", np.flatnonzero(theta!=pert).tolist(), "logL", float(logl_full), lp)
        if len(reps)>1:
            km1=problem_for_k(len(reps)-1); rem=float(km1.loglikelihood(physical_catalogue_to_theta(physical[1:], problem=km1))) if km1 else float('nan'); print("integration_diagnostic removal_logL", float(logl_full), rem)
    final=[r for r in rows if r.get("conditional_status")=="kept"]; write_csv(args.out_significance, rows); write_csv(args.out_final_catalogue, final)
    summary={"window_id":args.window_id,"n_input_candidates":len(all_rows),"n_union_candidates":len(cand),"optional_prefilters":{"minimum_mean_inclusion":args.minimum_mean_inclusion,"minimum_seed_support":args.minimum_seed_support,"allowed_sampling_quality":args.allowed_sampling_quality},"n_final_candidates":len(final),"max_posterior_logL":max_posterior_logl,"logL_selected_baseline":baseline_logl,"logL_synthetic_joint_configuration":synth_logl,"logL_full":logl_full,"truth_information_used_for_selection":False,"diagnostic_only_fields":DIAGNOSTIC_ONLY_FIELDS,"selection_rule":"keep candidates with positive fixed-configuration delta_logl_fixed; placeholders reserved for future profiled statistic","assumptions":{"posterior_layout":"decoded physical [f0, fdot, iota, psi, lam, beta, p] per source slot","latent_parameterization":"problem.loglikelihood receives theta_u; representatives are inverse-packed with production transforms","representative_vector":"all seven physical parameters come from one posterior source component/draw nearest weighted median cluster frequency","synthetic_joint_configuration":"representatives for different clusters may come from different posterior draws or seeds, so the assembled vector is synthetic","source_removal":"candidate i is removed by constructing a K-1 problem and omitting its seven-parameter block, not by editing an ignored gate coordinate","k1_handling":"K=1 uses a K=0 baseline when supported, otherwise rows are marked unsupported_single_candidate_baseline"},"outputs":{"candidate_significance_csv":str(args.out_significance),"final_catalogue_csv":str(args.out_final_catalogue)}}
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True); Path(args.out_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for f in made:
        try: os.unlink(f)
        except OSError: pass

if __name__ == "__main__": main()
