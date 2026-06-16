#!/usr/bin/env python3
"""
Plot actual Sangria TDI data power in generated frequency windows,
with optional overlays of bright truth-catalogue source frequencies.

This is a diagnostic/validation tool only.

It plots:
  - actual TDI data power in A/E channels
  - core and padded/analysis windows
  - catalogue sources above a diagnostic approximate-SNR threshold

The sampler fits the data stream. This script helps visualize the
frequency-domain data that the sampler sees in each selected window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Allow importing helper modules from the same scripts directory.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from sangria_catalog import (  # noqa: E402
    read_catalogues,
    concatenate_catalogues,
    lisa_gb_approx_snr,
)


def read_tdi_xyz(h5_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Read Sangria TDI X/Y/Z data.

    Expected HDF5 layout:
      obs/tdi with compound dtype fields: t, X, Y, Z

    In the Sangria file inspected, obs/tdi has shape (6307200, 1).
    We squeeze the arrays to 1D.
    """
    with h5py.File(h5_path, "r") as f:
        if "obs/tdi" not in f:
            raise KeyError("Could not find 'obs/tdi' in the HDF5 file.")

        tdi = f["obs/tdi"]

        for field in ["t", "X", "Y", "Z"]:
            if field not in tdi.dtype.names:
                raise KeyError(f"Field '{field}' not found in obs/tdi dtype.")

        t = np.asarray(tdi["t"]).squeeze()
        X = np.asarray(tdi["X"]).squeeze()
        Y = np.asarray(tdi["Y"]).squeeze()
        Z = np.asarray(tdi["Z"]).squeeze()

        if "obs/config/dt_tdi" in f:
            dt = float(f["obs/config/dt_tdi"][()])
        elif "obs/config/dt" in f:
            dt = float(f["obs/config/dt"][()])
        else:
            dt = float(np.median(np.diff(t)))

    if not (len(t) == len(X) == len(Y) == len(Z)):
        raise ValueError("TDI arrays t, X, Y, Z do not have matching lengths.")

    return t, X, Y, Z, dt


def xyz_to_ae(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert X/Y/Z to approximate A/E TDI combinations.

    Normalization conventions can differ across codes. For a diagnostic
    power plot, this convention is sufficient. The sampler may use its
    own exact convention internally.
    """
    A = (2.0 * X - Y - Z) / 3.0
    E = (Z - Y) / np.sqrt(3.0)
    return A, E


def rfft_power(x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided FFT frequencies and diagnostic power |FFT(x)|^2.

    This is not a calibrated PSD. It is a simple visualization of the
    frequency-domain data power.
    """
    n = len(x)
    freq = np.fft.rfftfreq(n, dt)
    xf = np.fft.rfft(x)
    power = np.abs(xf) ** 2
    return freq, power


def validate_windows(windows: pd.DataFrame) -> None:
    required = {
        "band_id",
        "core_f_min",
        "core_f_max",
        "analysis_f_min",
        "analysis_f_max",
    }
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"Missing required columns in windows CSV: {sorted(missing)}")

    for col in ["core_f_min", "core_f_max", "analysis_f_min", "analysis_f_max"]:
        windows[col] = pd.to_numeric(windows[col], errors="raise")

    bad_core = windows["core_f_min"] >= windows["core_f_max"]
    bad_analysis = windows["analysis_f_min"] >= windows["analysis_f_max"]

    if bad_core.any():
        raise ValueError("At least one row has core_f_min >= core_f_max.")
    if bad_analysis.any():
        raise ValueError("At least one row has analysis_f_min >= analysis_f_max.")


def read_bright_catalogue_sources(
    h5_path: str,
    windows: pd.DataFrame,
    snr_threshold: float,
    tobs: float,
) -> pd.DataFrame:
    """
    Read truth catalogue sources and keep only sources above the
    approximate diagnostic SNR threshold in the plotted frequency range.

    This uses the same approximate SNR helper as make_sangria_windows.py.
    """
    fmin = float(windows["analysis_f_min"].min())
    fmax = float(windows["analysis_f_max"].max())

    print("[info] reading truth catalogues for source overlays")
    cats = read_catalogues(h5_path)
    data = concatenate_catalogues(cats)

    freq = np.asarray(data["Frequency"], dtype=float)
    amp = np.asarray(data["Amplitude"], dtype=float)
    beta = np.asarray(data["EclipticLatitude"], dtype=float)
    inc = np.asarray(data["Inclination"], dtype=float)

    snr = lisa_gb_approx_snr(freq, amp, float(tobs), beta, inc)

    mask = (freq >= fmin) & (freq <= fmax) & (snr >= snr_threshold)

    bright = pd.DataFrame(
        {
            "Frequency": freq[mask],
            "Amplitude": amp[mask],
            "approx_snr": snr[mask],
        }
    ).sort_values("Frequency").reset_index(drop=True)

    print(f"[info] bright catalogue sources with SNR >= {snr_threshold}: {len(bright)}")
    for _, r in bright.iterrows():
        print(
            f"  f = {r['Frequency']:.12f} Hz  "
            f"A = {r['Amplitude']:.3e}  "
            f"SNR = {r['approx_snr']:.2f}"
        )

    return bright


def add_window_shading(ax, row: pd.Series) -> None:
    """
    Add analysis and core-window shading to an axes.
    """
    ax.axvspan(
        row["analysis_f_min"],
        row["analysis_f_max"],
        alpha=0.08,
        label="analysis window",
    )
    ax.axvspan(
        row["core_f_min"],
        row["core_f_max"],
        alpha=0.18,
        label="core window",
    )

    ax.axvline(row["core_f_min"], linestyle="--", linewidth=0.8)
    ax.axvline(row["core_f_max"], linestyle="--", linewidth=0.8)


def add_bright_source_markers(
    ax,
    bright_sources: pd.DataFrame,
    fmin: float,
    fmax: float,
    label_prefix: str = "Catalogue source",
) -> int:
    """
    Overlay bright catalogue source frequencies as vertical lines.

    Returns the number of sources plotted.
    """
    if bright_sources is None or len(bright_sources) == 0:
        return 0

    mask = (
        (bright_sources["Frequency"] >= fmin)
        & (bright_sources["Frequency"] <= fmax)
    )
    band_bright = bright_sources.loc[mask].copy()

    if len(band_bright) == 0:
        return 0

    ymin, ymax = ax.get_ylim()
    y_text = ymax / 3.0

    first = True
    for _, src in band_bright.iterrows():
        ax.axvline(
            src["Frequency"],
            linestyle="-",
            linewidth=1.3,
            alpha=0.95,
            label=f"{label_prefix} SNR ≥ threshold" if first else None,
        )
        ax.text(
            src["Frequency"],
            y_text,
            f"SNR {src['approx_snr']:.1f}",
            rotation=90,
            va="center",
            ha="right",
            fontsize=8,
        )
        first = False

    return len(band_bright)


def plot_global(
    freq: np.ndarray,
    power_a: np.ndarray,
    power_e: np.ndarray,
    windows: pd.DataFrame,
    bright_sources: pd.DataFrame,
    outpath: Path,
) -> None:
    fmin = float(windows["analysis_f_min"].min())
    fmax = float(windows["analysis_f_max"].max())

    mask = (freq >= fmin) & (freq <= fmax)

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(freq[mask], power_a[mask], linewidth=0.8, label="A data power")
    ax.plot(freq[mask], power_e[mask], linewidth=0.8, alpha=0.8, label="E data power")

    # Global plot: shade all analysis windows lightly and mark core edges.
    first_analysis = True
    first_core = True
    for _, row in windows.iterrows():
        ax.axvspan(
            row["analysis_f_min"],
            row["analysis_f_max"],
            alpha=0.06,
            label="analysis windows" if first_analysis else None,
        )
        ax.axvline(
            row["core_f_min"],
            linestyle="--",
            linewidth=0.8,
            label="core boundaries" if first_core else None,
        )
        ax.axvline(row["core_f_max"], linestyle="--", linewidth=0.8)
        first_analysis = False
        first_core = False

    ax.set_yscale("log")
    nsrc = add_bright_source_markers(ax, bright_sources, fmin, fmax)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$|\tilde d(f)|^2$ diagnostic power")
    ax.set_title(f"Sangria TDI data power in selected windows ({nsrc} bright source markers)")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_per_window(
    freq: np.ndarray,
    power_a: np.ndarray,
    power_e: np.ndarray,
    windows: pd.DataFrame,
    bright_sources: pd.DataFrame,
    outdir: Path,
) -> None:
    for _, row in windows.iterrows():
        band_id = str(row["band_id"])

        fmin = float(row["analysis_f_min"])
        fmax = float(row["analysis_f_max"])

        mask = (freq >= fmin) & (freq <= fmax)

        if not np.any(mask):
            print(f"[warning] no FFT bins found for {band_id} in [{fmin}, {fmax}]")
            continue

        fig, ax = plt.subplots(figsize=(9, 4.5))

        ax.plot(freq[mask], power_a[mask], linewidth=0.8, label="A data power")
        ax.plot(freq[mask], power_e[mask], linewidth=0.8, alpha=0.8, label="E data power")

        add_window_shading(ax, row)

        ax.set_yscale("log")
        nsrc = add_bright_source_markers(ax, bright_sources, fmin, fmax)

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(r"$|\tilde d(f)|^2$ diagnostic power")
        ax.set_title(
            f"{band_id}: Sangria TDI data power "
            f"[{fmin:.8f}, {fmax:.8f}] Hz "
            f"({nsrc} bright source markers)"
        )
        ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(outdir / f"{band_id}_data_power.png", dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5", help="Sangria/LDC HDF5 file")
    parser.add_argument("--windows-csv", required=True, help="Generated Sangria windows CSV")
    parser.add_argument("--snr-threshold", type=float, default=5.0)
    parser.add_argument("--tobs", type=float, default=31557600.0)
    parser.add_argument("--outdir", default="results/sangria_data_window_plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(args.windows_csv)
    validate_windows(windows)

    print(f"[info] reading TDI data from {args.h5}")
    t, X, Y, Z, dt = read_tdi_xyz(args.h5)

    duration = len(t) * dt
    df = 1.0 / duration

    print(f"[info] N = {len(t)}")
    print(f"[info] dt = {dt}")
    print(f"[info] duration = {duration} s")
    print(f"[info] frequency resolution df = {df} Hz")
    print(
        "[info] plotted frequency range = "
        f"[{windows['analysis_f_min'].min()}, {windows['analysis_f_max'].max()}] Hz"
    )

    bright_sources = read_bright_catalogue_sources(
        args.h5,
        windows,
        snr_threshold=args.snr_threshold,
        tobs=args.tobs,
    )

    print("[info] converting X/Y/Z to approximate A/E")
    A, E = xyz_to_ae(X, Y, Z)

    print("[info] computing FFTs")
    freq, power_a = rfft_power(A, dt)
    _, power_e = rfft_power(E, dt)

    print("[info] plotting")
    plot_global(
        freq,
        power_a,
        power_e,
        windows,
        bright_sources,
        outdir / "global_data_AE_power.png",
    )
    plot_per_window(freq, power_a, power_e, windows, bright_sources, outdir)

    print(f"[done] plots written to: {outdir}")


if __name__ == "__main__":
    main()
