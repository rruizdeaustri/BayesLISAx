#!/usr/bin/env python3
"""Build a within-window consensus catalogue from multi-seed posterior bundles.

The input bundles are compressed ``.npz`` files written by ``SamplerResult`` when
``JAX_SAMPLERS_POSTERIOR_PATH`` is set. Each bundle contains a physical posterior
sample array with component-major layout. For the LISA Galactic-binary problem,
the default per-component fields are

    [f0, fdot, iota, psi, lam, beta, p]

The afterburner clusters component frequencies across seeds, computes per-seed
posterior inclusion probabilities, estimates between-seed stability, and builds a
co-occurrence matrix that distinguishes simultaneous candidates from alternative
posterior modes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class PosteriorBundle:
    path: Path
    seed: str
    samples: np.ndarray  # (n_samples, K, dim_per)
    weights: np.ndarray  # normalized within this seed


@dataclass
class ClusterSummary:
    original_id: int
    f0_median: float
    f0_mean: float
    f0_q05: float
    f0_q95: float
    mean_inclusion: float
    median_inclusion: float
    seed_support_fraction: float
    n_seed_support: int
    n_seeds: int
    within_seed_std_median: float
    between_seed_std: float
    quality: str
    seed_inclusion: dict[str, float]
    seed_centroid: dict[str, float | None]


def _json_scalar(npz: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in npz:
        return default
    value = npz[key]
    if np.ndim(value) == 0:
        return str(value.item())
    if np.size(value) == 1:
        return str(np.ravel(value)[0])
    return default


def _normalized_weights(raw: np.ndarray | None, n: int) -> np.ndarray:
    if raw is None or raw.size == 0 or raw.shape != (n,):
        return np.full(n, 1.0 / max(n, 1), dtype=float)
    weights = np.asarray(raw, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights >= 0.0), weights, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(n, 1.0 / max(n, 1), dtype=float)
    return weights / total


def load_bundle(path: str | Path, dim_per: int, f0_index: int) -> PosteriorBundle:
    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        if "samples" not in npz:
            raise ValueError(f"{path}: missing 'samples' array")
        flat = np.asarray(npz["samples"], dtype=float)
        raw_weights = np.asarray(npz["weights"], dtype=float) if "weights" in npz else None
        seed = _json_scalar(npz, "seed", default=path.parent.name.replace("seed", ""))

    if flat.ndim != 2:
        raise ValueError(f"{path}: expected samples with shape (N,D), got {flat.shape}")
    if dim_per <= 0 or flat.shape[1] % dim_per != 0:
        raise ValueError(
            f"{path}: sample dimension D={flat.shape[1]} is not divisible by dim_per={dim_per}"
        )
    if not (0 <= f0_index < dim_per):
        raise ValueError(f"f0_index={f0_index} is invalid for dim_per={dim_per}")

    k = flat.shape[1] // dim_per
    samples = flat.reshape(flat.shape[0], k, dim_per)
    order = np.argsort(samples[:, :, f0_index], axis=1)
    samples = np.take_along_axis(samples, order[:, :, None], axis=1)
    weights = _normalized_weights(raw_weights, samples.shape[0])
    return PosteriorBundle(path=path, seed=seed, samples=samples, weights=weights)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    total = float(weights.sum())
    if total <= 0.0:
        return float(np.quantile(values, quantile))
    cdf = np.cumsum(weights) / total
    return float(np.interp(quantile, cdf, values))


def _weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    total = float(weights.sum())
    if total <= 0.0:
        return float(np.mean(values)), float(np.std(values))
    mean = float(np.sum(weights * values) / total)
    var = float(np.sum(weights * (values - mean) ** 2) / total)
    return mean, math.sqrt(max(var, 0.0))


def _split_wide_group(
    sorted_values: np.ndarray,
    sorted_indices: np.ndarray,
    max_span: float,
    min_split_gap: float,
) -> list[np.ndarray]:
    """Recursively split a connected 1-D group when it spans too wide a range."""
    if sorted_indices.size <= 1:
        return [sorted_indices]
    vals = sorted_values[sorted_indices]
    if float(vals[-1] - vals[0]) <= max_span:
        return [sorted_indices]
    gaps = np.diff(vals)
    split_at = int(np.argmax(gaps))
    if float(gaps[split_at]) < min_split_gap:
        split_at = (sorted_indices.size // 2) - 1
    left = sorted_indices[: split_at + 1]
    right = sorted_indices[split_at + 1 :]
    return _split_wide_group(sorted_values, left, max_span, min_split_gap) + _split_wide_group(
        sorted_values, right, max_span, min_split_gap
    )


def cluster_frequencies(values: np.ndarray, eps_hz: float, max_span_factor: float) -> np.ndarray:
    """Return a cluster id for every value using deterministic 1-D connectivity."""
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if values.size == 0:
        return np.empty(0, dtype=int)
    if eps_hz <= 0.0:
        raise ValueError("eps_hz must be positive")

    order = np.argsort(values)
    sorted_values = values[order]
    breaks = np.where(np.diff(sorted_values) > eps_hz)[0] + 1
    connected = np.split(np.arange(values.size, dtype=int), breaks)

    max_span = max(float(max_span_factor) * eps_hz, eps_hz)
    groups: list[np.ndarray] = []
    for group in connected:
        groups.extend(
            _split_wide_group(
                sorted_values,
                group,
                max_span=max_span,
                min_split_gap=0.25 * eps_hz,
            )
        )

    labels_sorted = np.empty(values.size, dtype=int)
    groups.sort(key=lambda idx: float(sorted_values[idx[0]]))
    for cluster_id, group in enumerate(groups):
        labels_sorted[group] = cluster_id
    labels = np.empty_like(labels_sorted)
    labels[order] = labels_sorted
    return labels


def _active_mask(samples: np.ndarray, p_index: int, p_active_min: float) -> np.ndarray:
    if p_index < 0:
        p_index = samples.shape[2] + p_index
    if 0 <= p_index < samples.shape[2]:
        p = samples[:, :, p_index]
        return np.isfinite(p) & (p > p_active_min)
    return np.ones(samples.shape[:2], dtype=bool)


def build_consensus(
    bundles: Sequence[PosteriorBundle],
    *,
    f0_index: int,
    p_index: int,
    p_active_min: float,
    eps_hz: float,
    max_span_factor: float,
    seed_inclusion_threshold: float,
    min_mean_inclusion: float,
    robust_seed_fraction: float,
    robust_inclusion: float,
    robust_max_between_hz: float,
    plausible_seed_fraction: float,
    plausible_inclusion: float,
    exclusive_overlap_ratio: float,
) -> tuple[list[ClusterSummary], np.ndarray, dict]:
    if not bundles:
        raise ValueError("At least one posterior bundle is required")

    record_f0: list[float] = []
    record_seed: list[int] = []
    record_sample: list[int] = []
    record_weight: list[float] = []

    for seed_idx, bundle in enumerate(bundles):
        mask = _active_mask(bundle.samples, p_index=p_index, p_active_min=p_active_min)
        for sample_idx in range(bundle.samples.shape[0]):
            for component_idx in np.where(mask[sample_idx])[0]:
                f0 = float(bundle.samples[sample_idx, component_idx, f0_index])
                if not np.isfinite(f0):
                    continue
                record_f0.append(f0)
                record_seed.append(seed_idx)
                record_sample.append(sample_idx)
                record_weight.append(float(bundle.weights[sample_idx]) / len(bundles))

    values = np.asarray(record_f0, dtype=float)
    seed_ids = np.asarray(record_seed, dtype=int)
    weights_all = np.asarray(record_weight, dtype=float)
    if values.size == 0:
        raise ValueError("No active finite-frequency components were found")
    labels = cluster_frequencies(values, eps_hz=eps_hz, max_span_factor=max_span_factor)
    n_raw_clusters = int(labels.max()) + 1

    indicators: list[np.ndarray] = [
        np.zeros((bundle.samples.shape[0], n_raw_clusters), dtype=float) for bundle in bundles
    ]
    for seed_idx, sample_idx, cluster_id in zip(record_seed, record_sample, labels):
        indicators[seed_idx][sample_idx, cluster_id] = 1.0

    inclusions = np.zeros((len(bundles), n_raw_clusters), dtype=float)
    cooccurrence_raw = np.zeros((n_raw_clusters, n_raw_clusters), dtype=float)
    for seed_idx, bundle in enumerate(bundles):
        inclusions[seed_idx] = bundle.weights @ indicators[seed_idx]
        cooccurrence_raw += (indicators[seed_idx] * bundle.weights[:, None]).T @ indicators[seed_idx]
    cooccurrence_raw /= len(bundles)

    mean_inclusion = inclusions.mean(axis=0)
    keep = np.where(mean_inclusion >= min_mean_inclusion)[0]
    if keep.size == 0:
        keep = np.array([int(np.argmax(mean_inclusion))], dtype=int)

    summaries: list[ClusterSummary] = []
    for raw_id in keep:
        sel = labels == raw_id
        vals = values[sel]
        wts = weights_all[sel]
        f0_mean, _ = _weighted_mean_std(vals, wts)
        f0_median = _weighted_quantile(vals, wts, 0.5)
        f0_q05 = _weighted_quantile(vals, wts, 0.05)
        f0_q95 = _weighted_quantile(vals, wts, 0.95)

        seed_inclusion: dict[str, float] = {}
        seed_centroid: dict[str, float | None] = {}
        within_stds: list[float] = []
        centroid_values: list[float] = []
        for seed_idx, bundle in enumerate(bundles):
            inc = float(inclusions[seed_idx, raw_id])
            seed_inclusion[bundle.seed] = inc

            rec_sel = sel & (seed_ids == seed_idx)
            if np.any(rec_sel):
                seed_vals = values[rec_sel]
                seed_wts = weights_all[rec_sel] * len(bundles)
                centroid, within_std = _weighted_mean_std(seed_vals, seed_wts)
                seed_centroid[bundle.seed] = centroid
                centroid_values.append(centroid)
                within_stds.append(within_std)
            else:
                seed_centroid[bundle.seed] = None

        seed_support = inclusions[:, raw_id] >= seed_inclusion_threshold
        support_fraction = float(np.mean(seed_support))
        median_inc = float(np.median(inclusions[:, raw_id]))
        mean_inc = float(np.mean(inclusions[:, raw_id]))
        between_std = float(np.std(centroid_values)) if len(centroid_values) > 1 else 0.0
        within_median = float(np.median(within_stds)) if within_stds else float("nan")

        if (
            support_fraction >= robust_seed_fraction
            and median_inc >= robust_inclusion
            and between_std <= robust_max_between_hz
        ):
            quality = "robust"
        elif support_fraction >= plausible_seed_fraction and median_inc >= plausible_inclusion:
            quality = "plausible"
        else:
            quality = "confused"

        summaries.append(
            ClusterSummary(
                original_id=int(raw_id),
                f0_median=f0_median,
                f0_mean=f0_mean,
                f0_q05=f0_q05,
                f0_q95=f0_q95,
                mean_inclusion=mean_inc,
                median_inclusion=median_inc,
                seed_support_fraction=support_fraction,
                n_seed_support=int(np.sum(seed_support)),
                n_seeds=len(bundles),
                within_seed_std_median=within_median,
                between_seed_std=between_std,
                quality=quality,
                seed_inclusion=seed_inclusion,
                seed_centroid=seed_centroid,
            )
        )

    summaries.sort(key=lambda item: item.f0_median)
    selected_raw = [item.original_id for item in summaries]
    raw_to_new = {raw: idx for idx, raw in enumerate(selected_raw)}
    cooccurrence = cooccurrence_raw[np.ix_(selected_raw, selected_raw)]

    exclusive_pairs = []
    for i in range(len(summaries)):
        for j in range(i + 1, len(summaries)):
            denom = min(summaries[i].mean_inclusion, summaries[j].mean_inclusion)
            ratio = float(cooccurrence[i, j] / denom) if denom > 0.0 else float("nan")
            if np.isfinite(ratio) and ratio < exclusive_overlap_ratio:
                exclusive_pairs.append(
                    {
                        "cluster_i": i,
                        "cluster_j": j,
                        "cooccurrence": float(cooccurrence[i, j]),
                        "overlap_ratio": ratio,
                    }
                )

    parent = list(range(len(summaries)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for pair in exclusive_pairs:
        union(int(pair["cluster_i"]), int(pair["cluster_j"]))
    groups: dict[int, list[int]] = {}
    for idx in range(len(summaries)):
        groups.setdefault(find(idx), []).append(idx)
    mode_families = [members for members in groups.values() if len(members) > 1]

    modes = {
        "exclusive_overlap_ratio": exclusive_overlap_ratio,
        "exclusive_pairs": exclusive_pairs,
        "alternative_mode_families": mode_families,
        "raw_to_consensus_cluster": {str(raw): new for raw, new in raw_to_new.items()},
    }
    return summaries, cooccurrence, modes


def write_outputs(
    summaries: Sequence[ClusterSummary],
    cooccurrence: np.ndarray,
    modes: dict,
    *,
    window_id: str,
    input_paths: Sequence[str | Path],
    eps_hz: float,
    out_clusters: str | Path,
    out_cooccurrence: str | Path,
    out_modes: str | Path,
    out_summary: str | Path,
) -> None:
    out_clusters = Path(out_clusters)
    out_cooccurrence = Path(out_cooccurrence)
    out_modes = Path(out_modes)
    out_summary = Path(out_summary)
    for path in (out_clusters, out_cooccurrence, out_modes, out_summary):
        path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "window_id",
        "cluster_id",
        "quality",
        "f0_median_hz",
        "f0_mean_hz",
        "f0_q05_hz",
        "f0_q95_hz",
        "mean_inclusion",
        "median_inclusion",
        "seed_support_fraction",
        "n_seed_support",
        "n_seeds",
        "within_seed_std_median_hz",
        "between_seed_std_hz",
        "seed_inclusion_json",
        "seed_centroid_json",
    ]
    with out_clusters.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for cluster_id, item in enumerate(summaries):
            writer.writerow(
                {
                    "window_id": window_id,
                    "cluster_id": cluster_id,
                    "quality": item.quality,
                    "f0_median_hz": f"{item.f0_median:.16g}",
                    "f0_mean_hz": f"{item.f0_mean:.16g}",
                    "f0_q05_hz": f"{item.f0_q05:.16g}",
                    "f0_q95_hz": f"{item.f0_q95:.16g}",
                    "mean_inclusion": f"{item.mean_inclusion:.8g}",
                    "median_inclusion": f"{item.median_inclusion:.8g}",
                    "seed_support_fraction": f"{item.seed_support_fraction:.8g}",
                    "n_seed_support": item.n_seed_support,
                    "n_seeds": item.n_seeds,
                    "within_seed_std_median_hz": f"{item.within_seed_std_median:.16g}",
                    "between_seed_std_hz": f"{item.between_seed_std:.16g}",
                    "seed_inclusion_json": json.dumps(item.seed_inclusion, sort_keys=True),
                    "seed_centroid_json": json.dumps(item.seed_centroid, sort_keys=True),
                }
            )

    with out_cooccurrence.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cluster_id", *range(len(summaries))])
        for i, row in enumerate(cooccurrence):
            writer.writerow([i, *[f"{float(x):.8g}" for x in row]])

    modes_payload = dict(modes)
    modes_payload["window_id"] = window_id
    modes_payload["clusters"] = [
        {
            "cluster_id": idx,
            "quality": item.quality,
            "f0_median_hz": item.f0_median,
            "mean_inclusion": item.mean_inclusion,
        }
        for idx, item in enumerate(summaries)
    ]
    out_modes.write_text(json.dumps(modes_payload, indent=2), encoding="utf-8")

    quality_counts: dict[str, int] = {}
    for item in summaries:
        quality_counts[item.quality] = quality_counts.get(item.quality, 0) + 1
    robust_ids = [idx for idx, item in enumerate(summaries) if item.quality == "robust"]
    payload = {
        "window_id": window_id,
        "n_input_seeds": len(input_paths),
        "input_posteriors": [str(Path(path)) for path in input_paths],
        "cluster_eps_hz": eps_hz,
        "n_consensus_clusters": len(summaries),
        "quality_counts": quality_counts,
        "robust_cluster_ids": robust_ids,
        "recommended_action": (
            "subtract_or_condition_and_rescan" if robust_ids else "adaptive_rerun_or_preserve_modes"
        ),
        "outputs": {
            "clusters_csv": str(out_clusters),
            "cooccurrence_csv": str(out_cooccurrence),
            "mode_families_json": str(out_modes),
        },
    }
    out_summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Posterior .npz bundles")
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--tobs", type=float, required=True, help="Observation time in seconds")
    parser.add_argument("--cluster-eps-bins", type=float, default=5.0)
    parser.add_argument("--cluster-eps-hz", type=float, default=0.0)
    parser.add_argument("--max-span-factor", type=float, default=10.0)
    parser.add_argument("--dim-per", type=int, default=7)
    parser.add_argument("--f0-index", type=int, default=0)
    parser.add_argument("--p-index", type=int, default=6)
    parser.add_argument("--p-active-min", type=float, default=0.5)
    parser.add_argument("--seed-inclusion-threshold", type=float, default=0.1)
    parser.add_argument("--min-mean-inclusion", type=float, default=0.05)
    parser.add_argument("--robust-seed-fraction", type=float, default=0.8)
    parser.add_argument("--robust-inclusion", type=float, default=0.5)
    parser.add_argument("--robust-max-between-bins", type=float, default=5.0)
    parser.add_argument("--plausible-seed-fraction", type=float, default=0.5)
    parser.add_argument("--plausible-inclusion", type=float, default=0.1)
    parser.add_argument("--exclusive-overlap-ratio", type=float, default=0.2)
    parser.add_argument("--out-clusters", required=True)
    parser.add_argument("--out-cooccurrence", required=True)
    parser.add_argument("--out-modes", required=True)
    parser.add_argument("--out-summary", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.tobs <= 0.0:
        raise ValueError("--tobs must be positive")
    eps_hz = float(args.cluster_eps_hz)
    if eps_hz <= 0.0:
        eps_hz = float(args.cluster_eps_bins) / float(args.tobs)
    robust_max_between_hz = float(args.robust_max_between_bins) / float(args.tobs)

    bundles = [load_bundle(path, dim_per=args.dim_per, f0_index=args.f0_index) for path in args.inputs]
    summaries, cooccurrence, modes = build_consensus(
        bundles,
        f0_index=args.f0_index,
        p_index=args.p_index,
        p_active_min=args.p_active_min,
        eps_hz=eps_hz,
        max_span_factor=args.max_span_factor,
        seed_inclusion_threshold=args.seed_inclusion_threshold,
        min_mean_inclusion=args.min_mean_inclusion,
        robust_seed_fraction=args.robust_seed_fraction,
        robust_inclusion=args.robust_inclusion,
        robust_max_between_hz=robust_max_between_hz,
        plausible_seed_fraction=args.plausible_seed_fraction,
        plausible_inclusion=args.plausible_inclusion,
        exclusive_overlap_ratio=args.exclusive_overlap_ratio,
    )
    write_outputs(
        summaries,
        cooccurrence,
        modes,
        window_id=args.window_id,
        input_paths=args.inputs,
        eps_hz=eps_hz,
        out_clusters=args.out_clusters,
        out_cooccurrence=args.out_cooccurrence,
        out_modes=args.out_modes,
        out_summary=args.out_summary,
    )


if __name__ == "__main__":
    main()
