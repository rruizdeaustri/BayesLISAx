#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path


MODULE = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "stability_consensus.py"
)

spec = importlib.util.spec_from_file_location(
    "stability_consensus",
    MODULE,
)

module = importlib.util.module_from_spec(
    spec
)

sys.modules[
    spec.name
] = module

assert spec.loader is not None
spec.loader.exec_module(
    module
)


def rr(
    k,
    seed,
    logz,
    origin="stability_consensus",
):
    return module.RunResult(
        k=k,
        seed=seed,
        logz=logz,
        logz_err=0.5,
        summary_path=(
            f"/tmp/k{k}/"
            f"seed{seed}/summary.json"
        ),
        fidelity="stability",
        origin=origin,
    )


def test_ranking_selects_k9():
    runs = [
        rr(8, 33, -100.0),
        rr(9, 33, -10.0),
        rr(10, 33, -50.0),

        rr(8, 44, -120.0),
        rr(9, 44, -20.0),
        rr(10, 44, -60.0),

        rr(8, 55, -80.0),
        rr(9, 55, -30.0),
        rr(10, 55, -10.0),
    ]

    ranking = (
        module.build_ranking_diagnostic(
            runs,
            [8, 9, 10],
            selected_k=9,
        )
    )

    assert ranking.best_k == 9
    assert ranking.selected_k_support == 2
    assert ranking.n_seeds == 3

    assert abs(
        ranking.selected_k_support_fraction
        - 2.0 / 3.0
    ) < 1e-12


def test_stable_consensus():
    new_ranking = module.RankingDiagnostic(
        ks=[8, 9, 10],
        seeds=[33, 44, 55],
        seed_best_k={
            33: 9,
            44: 9,
            55: 10,
        },
        seed_delta_logz={},
        median_delta_logz_by_k={
            8: -500.0,
            9: 0.0,
            10: -50.0,
        },
        best_k=9,
        runner_up_k=10,
        margin_to_runner_up=50.0,
        best_k_counts={
            8: 0,
            9: 2,
            10: 1,
        },
        selected_k_support=2,
        selected_k_support_fraction=(
            2.0 / 3.0
        ),
        n_seeds=3,
    )

    combined_ranking = module.RankingDiagnostic(
        ks=[8, 9, 10],
        seeds=[0, 11, 22, 33, 44, 55],
        seed_best_k={
            0: 9,
            11: 9,
            22: 10,
            33: 9,
            44: 9,
            55: 10,
        },
        seed_delta_logz={},
        median_delta_logz_by_k={
            8: -700.0,
            9: 0.0,
            10: -40.0,
        },
        best_k=9,
        runner_up_k=10,
        margin_to_runner_up=40.0,
        best_k_counts={
            8: 0,
            9: 4,
            10: 2,
        },
        selected_k_support=4,
        selected_k_support_fraction=(
            4.0 / 6.0
        ),
        n_seeds=6,
    )

    decision = module.decide_consensus(
        selected_k=9,
        new_ranking=new_ranking,
        combined_ranking=combined_ranking,
        min_support_fraction=(
            2.0 / 3.0
        ),
        min_margin=20.0,
    )

    assert (
        decision.status
        == "stable_consensus"
    )

    assert (
        decision.next_action
        == "accept_model_order"
    )


def test_model_order_shift():
    new_ranking = module.RankingDiagnostic(
        ks=[8, 9, 10],
        seeds=[33, 44, 55],
        seed_best_k={
            33: 10,
            44: 10,
            55: 9,
        },
        seed_delta_logz={},
        median_delta_logz_by_k={
            8: -500.0,
            9: -100.0,
            10: 0.0,
        },
        best_k=10,
        runner_up_k=9,
        margin_to_runner_up=100.0,
        best_k_counts={
            8: 0,
            9: 1,
            10: 2,
        },
        selected_k_support=1,
        selected_k_support_fraction=(
            1.0 / 3.0
        ),
        n_seeds=3,
    )

    combined_ranking = module.RankingDiagnostic(
        ks=[8, 9, 10],
        seeds=[0, 11, 22, 33, 44, 55],
        seed_best_k={
            0: 9,
            11: 9,
            22: 10,
            33: 10,
            44: 10,
            55: 9,
        },
        seed_delta_logz={},
        median_delta_logz_by_k={
            8: -500.0,
            9: -25.0,
            10: 0.0,
        },
        best_k=10,
        runner_up_k=9,
        margin_to_runner_up=25.0,
        best_k_counts={
            8: 0,
            9: 3,
            10: 3,
        },
        selected_k_support=3,
        selected_k_support_fraction=0.5,
        n_seeds=6,
    )

    decision = module.decide_consensus(
        selected_k=9,
        new_ranking=new_ranking,
        combined_ranking=combined_ranking,
        min_support_fraction=(
            2.0 / 3.0
        ),
        min_margin=20.0,
    )

    assert (
        decision.status
        == "model_order_shifted"
    )
