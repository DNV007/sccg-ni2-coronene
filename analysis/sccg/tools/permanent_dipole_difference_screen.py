#!/usr/bin/env python3
"""Fixed-upstream sensitivity to spin-free permanent-dipole differences.

Run from the repository root with its locked Python environment. The output is
the archived sensitivity record; this script does not perform an
electronic-structure recalculation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))
from sccg import retained_state_audit as audit  # noqa: E402

OUTPUT = ROOT / "calculations" / "B1" / "permanent_dipole_difference_screen.json"
CASES = ("N5", "N4", "omit_Q5", "omit_S5", "omit_Q5_S5")


def masks(data: dict) -> dict[str, np.ndarray]:
    roots = data["local_roots"]
    mult = data["multiplicities"]
    return {
        "N5": np.ones(len(roots), dtype=bool),
        "N4": roots <= 4,
        "omit_Q5": ~((mult == 5) & (roots == 5)),
        "omit_S5": ~((mult == 1) & (roots == 5)),
        "omit_Q5_S5": ~((roots == 5) & np.isin(mult, (5, 1))),
    }


def replace_diagonal(data: dict, common: np.ndarray) -> dict:
    updated = dict(data)
    updated["basis_mu"] = data["basis_mu"].copy()
    indices = np.arange(updated["basis_mu"].shape[1])
    updated["basis_mu"][:, indices, indices] = common[:, None]
    return updated


def record(model: dict) -> dict:
    curve = audit.sweep(model)
    leading = np.argmax(curve["p"], axis=1)
    parents = np.argmax(model["parent"][leading], axis=1)
    left = curve["left_total"]
    right = curve["right_total"]
    d2 = curve["D2"]
    return {
        "n_so": int(len(model["energy"])),
        "D2": d2.tolist(),
        "S_L": left.tolist(),
        "S_R": right.tolist(),
        "G": (d2 * left * right).tolist(),
        "PR": curve["PR"].tolist(),
        "leading_so_state_one_based": (leading + 1).tolist(),
        "leading_spin_free_root_one_based": (parents + 1).tolist(),
    }


def corr(rows: dict, case: str, key: str) -> float:
    p = rows["production"][case][key]
    b = rows["bridge"][case][key]
    return float(spearmanr(p, b).statistic)


def log_terms(rows: dict, tier: str, case: str, i: int) -> dict[str, float]:
    full = rows[tier]["N5"]
    changed = rows[tier][case]
    if any(full[key][i] <= 0 or changed[key][i] <= 0 for key in ("D2", "S_L", "S_R", "G")):
        return {"status": "zero_support_or_coincidence"}
    return {
        "delta_log10_G": float(np.log10(changed["G"][i] / full["G"][i])),
        "minus_delta_log10_S_L": float(-np.log10(changed["S_L"][i] / full["S_L"][i])),
        "minus_delta_log10_S_R": float(-np.log10(changed["S_R"][i] / full["S_R"][i])),
        "delta_log10_D2": float(np.log10(changed["D2"][i] / full["D2"][i])),
    }


def main() -> None:
    rows: dict = {"original": {}, "common_diagonal": {}}
    invariance: dict = {}
    diagonal_vectors: dict = {}
    for tier, (label, _) in audit.TIERS.items():
        data = audit.load_inputs(label)
        diagonals = np.diagonal(data["sf_dipoles"], axis1=1, axis2=2)
        common = diagonals.mean(axis=1)
        diagonal_vectors[tier] = {
            "componentwise_mean_au": common.tolist(),
            "componentwise_range_au": np.ptp(diagonals, axis=1).tolist(),
        }
        mean_data = replace_diagonal(data, common)
        zero_data = replace_diagonal(data, np.zeros(3))
        invariance[tier] = {}
        rows["original"][tier] = {}
        rows["common_diagonal"][tier] = {}
        for case, mask in masks(data).items():
            original = audit.build(data, mask)
            mean_model = audit.build(mean_data, mask)
            zero_model = audit.build(zero_data, mask)
            offdiag = ~np.eye(len(mean_model["energy"]), dtype=bool)
            invariance[tier][case] = float(np.max(np.abs(
                mean_model["mu"][:, offdiag] - zero_model["mu"][:, offdiag]
            )))
            rows["original"][tier][case] = record(original)
            rows["common_diagonal"][tier][case] = record(mean_model)

    summary: dict = {}
    for condition in ("original", "common_diagonal"):
        subset = rows[condition]
        summary[condition] = {}
        for case in CASES:
            summary[condition][case] = {
                "production_bridge_rho": {
                    key: corr(subset, case, key) for key in ("D2", "S_L", "S_R", "G")
                },
                "production_log_terms_0p65_eV": log_terms(subset, "production", case, -1),
                "low_energy_0p02_eV": {
                    tier: {
                        "PR": subset[tier][case]["PR"][0],
                        "D2": subset[tier][case]["D2"][0],
                        "leading_spin_free_root_one_based": subset[tier][case]["leading_spin_free_root_one_based"][0],
                        "leading_so_state_one_based": subset[tier][case]["leading_so_state_one_based"][0],
                    }
                    for tier in ("production", "bridge")
                },
            }
    archived = json.loads((ROOT / "calculations" / "B1" / "retained_state_audit.json").read_text())
    validation: dict = {}
    for tier in audit.TIERS:
        validation[tier] = {}
        for case in ("N5", "N4", "omit_Q5", "omit_S5"):
            old = archived["tiers"][tier]["cases"][case]["curves"]
            new = rows["original"][tier][case]
            errors = {
                key: float(np.max(np.abs(np.asarray(new[key]) - np.asarray(old[old_key]))))
                for key, old_key in (("D2", "D2"), ("S_L", "left_total"), ("S_R", "right_total"))
            }
            assert max(errors.values()) < 1e-9, (tier, case, errors)
            validation[tier][case] = errors
    output = {
        "scope": "Post-hoc fixed-upstream spin-free diagonal-dipole sensitivity; no new electronic-structure calculation.",
        "criteria": "SCREEN_CRITERIA_2026-09-30.md, recorded before this run",
        "grid_eV": audit.GRID.tolist(),
        "mean_vs_zero_common_diagonal_max_offdiagonal_mu_error_au": invariance,
        "spin_free_diagonal_dipoles": diagonal_vectors,
        "original_vs_archived_max_abs_error": validation,
        "summary": summary,
        "curves": rows,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")
    for case in CASES:
        print(case, "rho_D2", summary["original"][case]["production_bridge_rho"]["D2"],
              "rho_D2_common", summary["common_diagonal"][case]["production_bridge_rho"]["D2"])


if __name__ == "__main__":
    main()
