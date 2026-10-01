#!/usr/bin/env python3
"""Cross-tier sensitivity of D4 to its spin-mixing threshold."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))

from sccg.descriptors import D4_spin_mixing  # noqa: E402
from sccg.graph import SCCG  # noqa: E402
from sccg.io import load_manifold  # noqa: E402

B1 = ROOT / "calculations" / "B1"
OUT = B1 / "d4_threshold_sensitivity.json"
GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
THRESHOLDS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
PATHS = {
    "production": B1 / "02_cheap" / "manifold_cheap.json",
    "bridge": B1 / "03c_caspt2_compact" / "manifold_caspt2_compact.json",
    "compact_model": B1 / "03b_exact" / "manifold_exact.json",
}


def rho(x, y):
    if max(x) < 1e-12 or max(y) < 1e-12 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    return float(spearmanr(x, y).statistic)


def main() -> None:
    manifolds = {key: load_manifold(path) for key, path in PATHS.items()}
    rows = []
    for threshold in THRESHOLDS:
        sweeps = {
            key: [
                D4_spin_mixing(SCCG(manifold, omega=w, sigma=0.10,
                                    alpha=1.0, beta=0.0), threshold)
                for w in GRID
            ]
            for key, manifold in manifolds.items()
        }
        rows.append({
            "threshold": threshold,
            "sweeps": sweeps,
            "production_bridge_spearman": rho(sweeps["production"], sweeps["bridge"]),
            "bridge_model_spearman": rho(sweeps["bridge"], sweeps["compact_model"]),
            "production_model_spearman": rho(sweeps["production"], sweeps["compact_model"]),
            "ranges": {key: [float(min(values)), float(max(values))]
                       for key, values in sweeps.items()},
        })
    OUT.write_text(json.dumps({"photon_energy_grid_eV": GRID, "rows": rows}, indent=2) + "\n")
    print(f"wrote {OUT}")
    for row in rows:
        print(row["threshold"], row["production_bridge_spearman"], row["ranges"])


if __name__ == "__main__":
    main()
