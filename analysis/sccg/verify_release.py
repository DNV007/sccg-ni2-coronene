#!/usr/bin/env python3
"""Read-only validation of the archived benchmark and published headline values.

Run from the release root with ``python analysis/sccg/verify_release.py``.
No electronic-structure calculation or network access is required.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sccg.descriptors import all_descriptors
from sccg.graph import SCCG
from sccg.io import load_manifold
from sccg.retained_state_audit import B1, GRID, TIERS, build, load_inputs, pair_correlations, sweep


def verify():
    rows = list(csv.DictReader((B1 / "descriptor_sweeps.csv").open()))
    prefixes = {"production": "production", "bridge": "bridge", "anchor": "compact CASCI"}
    expected_cases = {f"omit_{tag}{n}" for tag in "QTS"
                      for n in range(1 if tag == "T" else 2, 6)}
    archived = json.loads((B1 / "retained_state_audit.json").read_text())
    assert expected_cases <= archived["cross_tier_spearman"].keys()
    curves_by_case = {}
    for tier, (label, relpath) in TIERS.items():
        manifold = load_manifold(B1 / relpath)
        tier_rows = [row for row in rows if row["tier"].startswith(prefixes[tier])]
        assert len(tier_rows) == len(GRID) == 14
        errors = {key: 0.0 for key in ("D1", "D2", "D3", "D4", "D5")}
        for row in tier_rows:
            graph = SCCG(manifold, float(row["photon_energy_eV"]))
            assert np.all(np.diag(graph.weights) == 0)
            actual = all_descriptors(graph)
            for key in errors:
                errors[key] = max(errors[key], abs(actual[key] - float(row[key])))
        assert max(errors.values()) < 1e-12, (tier, errors)
        data = load_inputs(label)
        masks = {"N5": np.ones(15, dtype=bool)}
        for tag, mult in (("Q", 5), ("T", 3), ("S", 1)):
            for n in range(1 if tag == "T" else 2, 6):
                masks[f"omit_{tag}{n}"] = ~((data["multiplicities"] == mult) &
                                           (data["local_roots"] == n))
        assert set(masks) - {"N5"} == expected_cases
        for case, mask in masks.items():
            model = build(data, mask)
            np.testing.assert_allclose(model["endpoint"].sum(axis=0), [5, 1], atol=1e-12)
            curve = sweep(model)
            curves_by_case.setdefault(case, {})[tier] = curve
            recorded = archived["tiers"][tier]["cases"][case]["curves"]
            for key in ("D2", "PR", "left_total", "right_total"):
                np.testing.assert_allclose(curve[key], recorded[key], rtol=1e-10, atol=1e-12)
        print(tier, "14-row descriptor errors", errors)
    correlations = {case: pair_correlations(curves) for case, curves in curves_by_case.items()}
    for case, values in correlations.items():
        for leg, value in values.items():
            assert abs(value - archived["cross_tier_spearman"][case][leg]) < 1e-12
    reversals = {case for case in expected_cases if correlations[case]["two_MS_tiers"] < 0}
    assert reversals == {"omit_Q5", "omit_S5"}, reversals
    full = curves_by_case["N5"]["production"]
    omit = curves_by_case["omit_Q5"]["production"]
    ratios = {key: float(omit[key][-1] / full[key][-1])
              for key in ("D2", "left_total", "right_total")}
    g_ratio = float(np.prod(list(ratios.values())))
    np.testing.assert_allclose([ratios["D2"], ratios["left_total"], g_ratio],
                               [111.36201766245, 0.001441671543089, 0.156419339915], rtol=1e-9)
    print("13 single-root cases; reversals:", sorted(reversals))
    print("Baseline correlations:", correlations["N5"])
    print("T1 correlations:", correlations["omit_T1"])
    print("Production Q5 deletion at 0.65 eV:", ratios, "G ratio", g_ratio)
    from sccg.tools.verify_support_records import verify as verify_support
    verify_support()
    print("Release numerical validation passed.")


if __name__ == "__main__":
    verify()
