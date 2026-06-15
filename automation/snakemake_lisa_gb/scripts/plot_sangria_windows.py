#!/usr/bin/env python3
"""
Plot Sangria catalogue sources inside generated frequency windows.

This is a diagnostic/validation tool only. It uses the truth catalogue
to help choose windows and SNR thresholds before running inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sangria_catalog import lisa_gb_approx_snr

CATALOGUE_PATHS = {
    "dgb": "sky/dgb/cat",
    "igb": "sky/igb/cat",
    "vgb": "sky/vgb/cat",
}

def _to_1d(x):
    x = np.asarray(x)
    x = np.squeeze(x)

    if x.ndim == 0:
        x = x.reshape(1)

    if x.ndim > 1:
        # Last-resort fallback for columns stored as (N, 1), (N, k), etc.
        x = x.reshape(x.shape[0], -1)[:, 0]

    return x


def read_catalogue(h5_path: str, fmin: float | None = None, fmax: float | None = None) -> pd.DataFrame:
    frames = []

    wanted = {
        "Frequency",
        "Amplitude",
        "FrequencyDerivative",
        "EclipticLatitude",
        "EclipticLongitude",
        "Inclination",
        "InitialPhase",
        "Polarization",
    }

    with h5py.File(h5_path, "r") as f:
        for label, path in CATALOGUE_PATHS.items():
            if path not in f:
                print(f"[warning] missing catalogue path: {path}")
                continue

            data = f[path]

            # Compound HDF5 dataset case
            if data.dtype.names is not None:
                available = set(data.dtype.names)

                if "Frequency" not in available or "Amplitude" not in available:
                    print(f"[warning] {path} has no Frequency/Amplitude columns")
                    continue

                freq = _to_1d(data["Frequency"][()])

                mask = np.ones(len(freq), dtype=bool)
                if fmin is not None:
                    mask &= freq >= fmin
                if fmax is not None:
                    mask &= freq <= fmax

                if not np.any(mask):
                    continue

                cols = {"Frequency": freq[mask]}

                for name in sorted(wanted - {"Frequency"}):
                    if name in available:
                        arr = _to_1d(data[name][()])
                        if len(arr) == len(freq):
                            cols[name] = arr[mask]

                df = pd.DataFrame(cols)

            # Group-of-datasets case
            else:
                available = set(data.keys())

                if "Frequency" not in available or "Amplitude" not in available:
                    print(f"[warning] {path} has no Frequency/Amplitude datasets")
                    continue

                freq = _to_1d(data["Frequency"][()])

                mask = np.ones(len(freq), dtype=bool)
                if fmin is not None:
                    mask &= freq >= fmin
                if fmax is not None:
                    mask &= freq <= fmax

                if not np.any(mask):
                    continue

                cols = {"Frequency": freq[mask]}

                for name in sorted(wanted - {"Frequency"}):
                    if name in available:
                        arr = _to_1d(data[name][()])
                        if len(arr) == len(freq):
                            cols[name] = arr[mask]

                df = pd.DataFrame(cols)

            df["catalogue"] = label
            frames.append(df)

            print(f"[info] {label}: kept {len(df)} sources in frequency range")

    if not frames:
        raise RuntimeError("No Sangria catalogue sources found in the selected frequency range.")

    return pd.concat(frames, ignore_index=True)

def plot_global(
    df: pd.DataFrame,
    windows: pd.DataFrame,
    snr_threshold: float,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))

    is_bright = df["approx_snr"] >= snr_threshold

    ax.scatter(
        df.loc[~is_bright, "Frequency"],
        df.loc[~is_bright, "approx_snr"],
        s=5,
        alpha=0.25,
        label=f"SNR < {snr_threshold:g}",
    )

    ax.scatter(
        df.loc[is_bright, "Frequency"],
        df.loc[is_bright, "approx_snr"],
        s=18,
        alpha=0.9,
        label=f"SNR ≥ {snr_threshold:g}",
    )

    for _, row in windows.iterrows():
        ax.axvspan(row["analysis_f_min"], row["analysis_f_max"], alpha=0.08)
        ax.axvline(row["core_f_min"], linestyle="--", linewidth=0.8)
        ax.axvline(row["core_f_max"], linestyle="--", linewidth=0.8)

    ax.axhline(snr_threshold, linestyle=":", linewidth=1.2)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Approximate diagnostic SNR")
    ax.set_yscale("log")
    ax.set_title("Sangria catalogue sources and generated windows")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_amplitude(
    df: pd.DataFrame,
    windows: pd.DataFrame,
    snr_threshold: float,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))

    is_bright = df["approx_snr"] >= snr_threshold

    ax.scatter(
        df.loc[~is_bright, "Frequency"],
        df.loc[~is_bright, "Amplitude"],
        s=5,
        alpha=0.25,
        label=f"SNR < {snr_threshold:g}",
    )

    ax.scatter(
        df.loc[is_bright, "Frequency"],
        df.loc[is_bright, "Amplitude"],
        s=18,
        alpha=0.9,
        label=f"SNR ≥ {snr_threshold:g}",
    )

    for _, row in windows.iterrows():
        ax.axvspan(row["analysis_f_min"], row["analysis_f_max"], alpha=0.08)
        ax.axvline(row["core_f_min"], linestyle="--", linewidth=0.8)
        ax.axvline(row["core_f_max"], linestyle="--", linewidth=0.8)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude")
    ax.set_yscale("log")
    ax.set_title("Sangria catalogue amplitudes and generated windows")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_per_window(
    df: pd.DataFrame,
    windows: pd.DataFrame,
    snr_threshold: float,
    outdir: Path,
) -> None:
    for _, row in windows.iterrows():
        band_id = row["band_id"]

        mask = (
            (df["Frequency"] >= row["analysis_f_min"])
            & (df["Frequency"] <= row["analysis_f_max"])
        )
        sub = df.loc[mask].copy()

        if len(sub) == 0:
            continue

        is_bright = sub["approx_snr"] >= snr_threshold

        fig, ax = plt.subplots(figsize=(9, 4.5))

        ax.scatter(
            sub.loc[~is_bright, "Frequency"],
            sub.loc[~is_bright, "approx_snr"],
            s=7,
            alpha=0.3,
            label=f"SNR < {snr_threshold:g}",
        )

        ax.scatter(
            sub.loc[is_bright, "Frequency"],
            sub.loc[is_bright, "approx_snr"],
            s=30,
            alpha=0.9,
            label=f"SNR ≥ {snr_threshold:g}",
        )

        ax.axvspan(row["analysis_f_min"], row["analysis_f_max"], alpha=0.08, label="analysis")
        ax.axvspan(row["core_f_min"], row["core_f_max"], alpha=0.18, label="core")

        ax.axhline(snr_threshold, linestyle=":", linewidth=1.2)

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Approximate diagnostic SNR")
        ax.set_yscale("log")
        ax.set_title(
            f"{band_id}: core [{row['core_f_min']:.8f}, {row['core_f_max']:.8f}] Hz"
        )
        ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(outdir / f"{band_id}_snr.png", dpi=200)
        plt.close(fig)


def plot_bright_counts(windows: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.bar(windows["band_id"].astype(str), windows["n_bright_sources"].astype(float))
    ax.set_xlabel("Band ID")
    ax.set_ylabel("Number of bright sources")
    ax.set_title("Bright catalogue sources per window")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5", help="Sangria/LDC HDF5 file")
    parser.add_argument("--windows-csv", required=True, help="Generated Sangria windows CSV")
    parser.add_argument("--snr-threshold", type=float, default=5.0)
    parser.add_argument("--tobs", type=float, default=31557600.0)
    parser.add_argument("--outdir", default="results/sangria_window_plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(args.windows_csv)

    required = {
        "band_id",
        "core_f_min",
        "core_f_max",
        "analysis_f_min",
        "analysis_f_max",
    }
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"Missing required window columns: {sorted(missing)}")

    fmin = float(windows["analysis_f_min"].min())
    fmax = float(windows["analysis_f_max"].max())

    print(f"[info] reading catalogue: {args.h5}")
    df = read_catalogue(args.h5, fmin=fmin, fmax=fmax)

    print(f"[info] rows in plotted frequency range [{fmin}, {fmax}]: {len(df)}")

    """
    print(f"[info] total catalogue rows: {len(df)}")
    df = df[(df["Frequency"] >= fmin) & (df["Frequency"] <= fmax)].copy()
    print(f"[info] rows in plotted frequency range [{fmin}, {fmax}]: {len(df)}")
 """
    
    df["approx_snr"] = lisa_gb_approx_snr(
        np.asarray(df["Frequency"], dtype=float),
        np.asarray(df["Amplitude"], dtype=float),
        float(args.tobs),
        np.asarray(df["EclipticLatitude"], dtype=float),
        np.asarray(df["Inclination"], dtype=float),
    )

    nbright = int((df["approx_snr"] >= args.snr_threshold).sum())
    print(f"[info] sources with approx SNR >= {args.snr_threshold}: {nbright}")

    plot_global(df, windows, args.snr_threshold, outdir / "global_frequency_snr.png")
    plot_amplitude(df, windows, args.snr_threshold, outdir / "global_frequency_amplitude.png")
    plot_per_window(df, windows, args.snr_threshold, outdir)
    plot_bright_counts(windows, outdir / "bright_sources_per_window.png")

    print(f"[done] plots written to: {outdir}")


if __name__ == "__main__":
    main()
