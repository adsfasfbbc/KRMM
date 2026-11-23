"""Verify paired cluster bootstrap CI invariants, clustering, and metadata fields."""
from __future__ import annotations

from krmm.v3.metrics import paired_cluster_bootstrap_ci


def test_paired_bootstrap_fields_and_clustering():
    a = [1, 1, 1, 1, 0, 0]
    b = [0, 0, 0, 0, 0, 0]
    clusters = ["g1", "g1", "g2", "g3", "g4", "g5"]
    res = paired_cluster_bootstrap_ci(
        a, b, clusters, n_boot=1000, seed=107, exploratory_or_confirmatory="exploratory"
    )
    assert res["clusters"] == 5
    assert res["n_pairs"] == 6
    assert res["has_duplicate_clusters"] is True
    assert res["statistical_unit"] == "source_group"
    assert res["exploratory_or_confirmatory"] == "exploratory"
    assert res["effect_size"] == res["point_estimate"] == res["mean_diff"]
    assert res["confidence_interval"] == res["ci95"]
    assert res["ci95"][0] <= res["point_estimate"] <= res["ci95"][1]


def test_paired_bootstrap_degenerate_zero_variance():
    a = [1, 1, 1, 1]
    b = [1, 1, 1, 1]
    res = paired_cluster_bootstrap_ci(a, b, ["c1", "c2", "c3", "c4"], n_boot=500)
    assert res["degenerate_zero_variance"] is True
    assert res["ci95"] == [0.0, 0.0]
