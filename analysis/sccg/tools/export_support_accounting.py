#!/usr/bin/env python3
"""Regenerate the full-grid support-accounting JSON and CSV from the dipole screen."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "calculations" / "B1" / "permanent_dipole_difference_screen.json"
OUT_JSON = ROOT / "calculations" / "B1" / "support_accounting_record.json"
OUT_CSV = ROOT / "calculations" / "B1" / "support_accounting_curves.csv"
KEYS = ("D2", "S_L", "S_R", "G")
CASES = ("N5", "N4", "omit_Q5", "omit_S5")
PAIRS = {
    "Production--bridge": ("production", "bridge"),
    "Bridge--anchor": ("bridge", "anchor"),
    "Production--anchor": ("production", "anchor"),
}


def rho(a: list[float], b: list[float]) -> float:
    return float(spearmanr(a, b).statistic)


def stats(a: list[float], b: list[float], x: np.ndarray) -> list[float]:
    a, b = np.asarray(a), np.asarray(b)
    return [rho(a, b), rho(np.diff(a), np.diff(b)), rho(np.diff(a) / np.diff(x), np.diff(b) / np.diff(x))]


def main() -> None:
    data = json.loads(SOURCE.read_text())
    curves = data["curves"]["original"]
    x = np.asarray(data["grid_eV"])
    record = {
        "source": SOURCE.name,
        "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "grid_eV": x.tolist(),
        "statistics": {},
        "mixed_N4": {},
        "log_curves": {},
    }
    for case in CASES:
        record["statistics"][case] = {
            pair: {key: stats(curves[a][case][key], curves[b][case][key], x) for key in KEYS}
            for pair, (a, b) in PAIRS.items()
        }
    for production_case, bridge_case in (("N5", "N5"), ("N4", "N5"), ("N5", "N4"), ("N4", "N4")):
        record["mixed_N4"][production_case + "/" + bridge_case] = {
            key: rho(curves["production"][production_case][key], curves["bridge"][bridge_case][key])
            for key in KEYS
        }
    residual = 0.0
    for tier in curves:
        record["log_curves"][tier] = {}
        for case in CASES[1:]:
            terms = {
                key: np.log10(np.asarray(curves[tier][case][key]) / np.asarray(curves[tier]["N5"][key]))
                for key in KEYS
            }
            residual = max(residual, float(np.max(np.abs(
                terms["D2"] - terms["G"] + terms["S_L"] + terms["S_R"]
            ))))
            record["log_curves"][tier][case] = {key: value.tolist() for key, value in terms.items()}
    record["maximum_log_identity_residual_decades"] = residual
    record["identical_N4_Q5_left_support_ranks"] = {
        tier: bool(np.array_equal(rankdata(curves[tier]["N4"]["S_L"]), rankdata(curves[tier]["omit_Q5"]["S_L"])))
        for tier in ("production", "bridge")
    }
    record["Q5_S5_vs_N4_max_abs_D2_difference"] = {
        tier: float(np.max(np.abs(
            np.asarray(curves[tier]["omit_Q5_S5"]["D2"]) - np.asarray(curves[tier]["N4"]["D2"])
        )))
        for tier in curves
    }
    OUT_JSON.write_text(json.dumps(record, indent=2) + "\n")
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "tier", "retention", "photon_energy_eV", *KEYS, "delta_log10_D2", "delta_log10_G",
            "minus_delta_log10_S_L", "minus_delta_log10_S_R",
        ])
        for tier in curves:
            for case in CASES:
                for i, energy in enumerate(x):
                    terms = {
                        key: np.log10(curves[tier][case][key][i] / curves[tier]["N5"][key][i])
                        for key in KEYS
                    }
                    writer.writerow([
                        tier, case, energy, *[curves[tier][case][key][i] for key in KEYS],
                        terms["D2"], terms["G"], -terms["S_L"], -terms["S_R"],
                    ])
    print(f"Wrote {OUT_JSON} and {OUT_CSV}; maximum identity residual {residual:.3e} decades")


if __name__ == "__main__":
    main()
