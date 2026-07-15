from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from consensus_candidate_selection import build_candidates  # noqa: E402


def test_bfdr_selection_uses_posterior_inclusion_not_sampling_label():
    clusters = [
        {
            "window_id": "w",
            "cluster_id": "0",
            "quality": "confused",
            "mean_inclusion": "0.98",
            "median_inclusion": "0.0",
            "f0_median_hz": "0.1",
        },
        {
            "window_id": "w",
            "cluster_id": "1",
            "quality": "robust",
            "mean_inclusion": "0.85",
            "median_inclusion": "1.0",
            "f0_median_hz": "0.2",
        },
        {
            "window_id": "w",
            "cluster_id": "2",
            "quality": "plausible",
            "mean_inclusion": "0.20",
            "median_inclusion": "0.0",
            "f0_median_hz": "0.3",
        },
    ]

    rows = build_candidates(
        clusters,
        catalogue=None,
        tobs=10.0,
        bfdr_q=0.10,
        probability_field="mean_inclusion",
        max_match_bins=5.0,
    )

    assert [row["statistically_selected"] for row in rows] == [True, True, False]
    assert rows[0]["sampling_quality"] == "confused"
    assert rows[0]["truth_match_quality"] == "not_requested"


def test_full_catalogue_nearest_match_reports_fourier_bins():
    clusters = [
        {
            "window_id": "w",
            "cluster_id": "3",
            "quality": "robust",
            "mean_inclusion": "1.0",
            "median_inclusion": "1.0",
            "f0_median_hz": "0.1001",
        }
    ]
    catalogue = {
        "Frequency": np.array([0.1, 0.2]),
        "Amplitude": np.array([1.0, 2.0]),
        "FrequencyDerivative": np.array([3.0, 4.0]),
        "EclipticLatitude": np.array([0.1, 0.2]),
        "EclipticLongitude": np.array([0.3, 0.4]),
        "Inclination": np.array([0.5, 0.6]),
        "InitialPhase": np.array([0.7, 0.8]),
        "Polarization": np.array([0.9, 1.0]),
        "source_type": np.array(["dgb", "igb"], dtype=object),
        "approx_snr": np.array([5.0, 1.0]),
    }

    rows = build_candidates(
        clusters,
        catalogue=catalogue,
        tobs=1000.0,
        bfdr_q=0.10,
        probability_field="mean_inclusion",
        max_match_bins=5.0,
    )

    assert np.isclose(rows[0]["delta_f_bins"], 0.1)
    assert rows[0]["truth_match_quality"] == "within_one_bin"
    assert rows[0]["catalogue_source_type"] == "dgb"
