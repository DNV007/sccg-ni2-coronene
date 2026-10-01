#!/usr/bin/env python3
"""Scan the D5 nonnegligible-pathway cutoff on the three B1 tiers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))

from sccg.descriptors import D5_pathway_evenness  # noqa: E402
from sccg.graph import SCCG  # noqa: E402
from sccg.io import load_manifold  # noqa: E402

B1 = ROOT / "calculations" / "B1"
OUT = B1 / "d5_cutoff_sensitivity.json"
CUTOFFS = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4)


def _omegas() -> list[float]:
    source = B1 / "02_cheap" / "descriptors_cheap_dipole_only.json"
    return [float(row["omega_eV"]) for row in json.loads(source.read_text())["sweep"]]


def _rho(x: list[float], y: list[float]) -> float:
    return float(spearmanr(x, y).statistic)


def main() -> None:
    paths = {
        "low_cost": B1 / "02_cheap" / "manifold_cheap.json",
        "bridge": B1 / "03c_caspt2_compact" / "manifold_caspt2_compact.json",
        "reference": B1 / "03b_exact" / "manifold_exact.json",
    }
    manifolds = {name: load_manifold(path) for name, path in paths.items()}
    omegas = _omegas()
    rows = []
    for cutoff in CUTOFFS:
        sweeps = {
            name: [
                D5_pathway_evenness(
                    SCCG(manifold, omega=omega, sigma=0.10, alpha=1.0, beta=0.0),
                    pathway_tol=cutoff,
                )
                for omega in omegas
            ]
            for name, manifold in manifolds.items()
        }
        rows.append(
            {
                "cutoff": cutoff,
                "rho_low_cost_reference": _rho(sweeps["low_cost"], sweeps["reference"]),
                "rho_bridge_reference": _rho(sweeps["bridge"], sweeps["reference"]),
                "rho_low_cost_bridge": _rho(sweeps["low_cost"], sweeps["bridge"]),
            }
        )

    result = {
        "omega_eV": omegas,
        "sigma_eV": 0.10,
        "production_cutoff": 1e-6,
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("cutoff       low--reference   bridge--reference   low--bridge")
    for row in rows:
        print(
            f"{row['cutoff']:.0e}"
            f"          {row['rho_low_cost_reference']:+.3f}"
            f"              {row['rho_bridge_reference']:+.3f}"
            f"             {row['rho_low_cost_bridge']:+.3f}"
        )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
