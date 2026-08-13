#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path


MODULE = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "model_order_controller.py"
)

spec = importlib.util.spec_from_file_location(
    "controller",
    MODULE,
)

controller = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = controller

assert spec.loader is not None
spec.loader.exec_module(controller)


def rr(k, seed, logz):
    return controller.RunResult(
        k=k,
        seed=seed,
        logz=logz,
        logz_err=0.5,
        summary_path=f"/tmp/k{k}/seed{seed}/summary.json",
        fidelity="search",
    )


def test_summarize_evidence_selects_expected_pair():
    point = controller.summarize_evidence(
        [
            rr(9, 0, -76065.697),
            rr(9, 11, -76075.720),
            rr(9, 22, -78398.859),
        ],
        150.0,
    )

    assert point.agreeing_seeds == [0, 11]
    assert abs(
        point.robust_logz + 76070.7085
    ) < 1e-6
    assert point.confidence == "moderate"


def test_summarize_evidence_selects_second_pair():
    point = controller.summarize_evidence(
        [
            rr(4, 0, -1000.0),
            rr(4, 11, -1300.0),
            rr(4, 22, -1310.0),
        ],
        50.0,
    )

    assert point.agreeing_seeds == [11, 22]
    assert abs(
        point.robust_logz + 1305.0
    ) < 1e-6
    assert point.confidence == "moderate"


def test_find_bracketed_peak():
    raw = {
        7: [
            -76703.0,
            -76653.0,
            -78831.0,
        ],
        8: [
            -75926.0,
            -75971.0,
            -75692.0,
        ],
        9: [
            -76065.0,
            -76075.0,
            -78398.0,
        ],
        10: [
            -76300.0,
            -76320.0,
            -78000.0,
        ],
    }

    points = {
        k: controller.summarize_evidence(
            [
                rr(k, seed, value)
                for seed, value in zip(
                    [0, 11, 22],
                    values,
                )
            ],
            150.0,
        )
        for k, values in raw.items()
    }

    peak = controller.find_bracketed_peak(
        points,
        confirm_lower_points=2,
        min_drop=20.0,
    )

    assert peak == 8


def test_per_seed_model_order_diagnostic():
    runs = [
        rr(8, 0, -100.0),
        rr(9, 0, -10.0),
        rr(10, 0, -30.0),

        rr(8, 11, -120.0),
        rr(9, 11, -20.0),
        rr(10, 11, -40.0),

        rr(8, 22, -50.0),
        rr(9, 22, -70.0),
        rr(10, 22, -10.0),
    ]

    ranking = (
        controller.per_seed_model_order_diagnostic(
            runs,
            [8, 9, 10],
        )
    )

    assert ranking["seed_best_k"] == {
        0: 9,
        11: 9,
        22: 10,
    }

    assert ranking["majority_best_k"] == 9
    assert ranking["n_support"] == 2
    assert ranking["n_seeds"] == 3

    assert abs(
        ranking["support_fraction"]
        - 2.0 / 3.0
    ) < 1e-12


def test_select_best_from_seed_deltas():
    ranking = {
        "seed_delta_logz": {
            0: {
                8: -3314.019,
                9: 0.0,
                10: -7180.193,
            },
            11: {
                8: -830.966,
                9: 0.0,
                10: -25.012,
            },
            22: {
                8: -483.124,
                9: -992.464,
                10: 0.0,
            },
        }
    }

    best_k, median_delta = (
        controller.select_best_from_seed_deltas(
            ranking
        )
    )

    assert best_k == 9

    assert abs(
        median_delta[8]
        + 830.966
    ) < 1e-9

    assert median_delta[9] == 0.0

    assert abs(
        median_delta[10]
        + 25.012
    ) < 1e-9
