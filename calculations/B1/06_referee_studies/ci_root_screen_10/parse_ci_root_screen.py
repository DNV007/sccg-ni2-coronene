#!/usr/bin/env python3
"""Summarize fixed-orbital Q/T/S CI roots after all three jobs complete."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py

HERE = Path(__file__).resolve().parent
EV_PER_HARTREE = 27.211386245988
LABELS = {"Q": "quintet", "T": "triplet", "S": "singlet"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    values: dict[str, list[float]] = {}
    files: dict[str, dict[str, str]] = {}
    for tag in LABELS:
        path = HERE / f"ci_{tag}.rasscf.h5"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed CI output: {path}")
        with h5py.File(path) as handle:
            if "ROOT_ENERGIES" not in handle:
                raise ValueError(f"ROOT_ENERGIES absent from {path}")
            roots = [float(x) for x in handle["ROOT_ENERGIES"][:]]
        if len(roots) < 10:
            raise ValueError(f"Expected at least ten {LABELS[tag]} roots, got {len(roots)}")
        values[tag] = roots[:10]
        files[tag] = {"path": path.name, "sha256": sha256(path)}

    zero = min(x for block in values.values() for x in block)
    rows = [
        {
            "multiplicity": LABELS[tag],
            "root": rank,
            "energy_hartree": energy,
            "relative_energy_eV": (energy - zero) * EV_PER_HARTREE,
            "retained_in_five_root_rule": rank <= 5,
        }
        for tag, block in values.items()
        for rank, energy in enumerate(block, start=1)
    ]
    extras = [row for row in rows if row["root"] >= 6]
    in_window = [row for row in extras if row["relative_energy_eV"] < 0.65]
    interleaved = [
        row for row in extras if any(
            row["energy_hartree"] < kept["energy_hartree"]
            for kept in rows
            if kept["retained_in_five_root_rule"] and kept["multiplicity"] != row["multiplicity"]
        )
    ]
    if in_window:
        classification = "additional_CI_roots_inside_0p65_eV"
    elif interleaved:
        classification = "additional_CI_roots_interleaved_above_window"
    else:
        classification = "no_additional_CI_roots_inside_or_interleaved"

    output = {
        "scope": "Fixed-orbital production CAS(26,15) CI-only screen; no orbital reoptimization or CASPT2.",
        "criteria": "calculations/B1/06_referee_studies/ci_root_screen_10/SCREEN_CRITERIA_2026-09-30.md",
        "files": files,
        "lowest_CI_energy_hartree": zero,
        "classification": classification,
        "additional_roots_below_0p65_eV": in_window,
        "additional_roots_interleaved_below_retained_other_multiplicity": interleaved,
        "roots": sorted(rows, key=lambda row: row["relative_energy_eV"]),
        "caution": "CI-only positions do not determine MS-CASPT2 positions or SOC state-count convergence.",
    }
    target = HERE / "ci_root_screen_result.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {target}: {classification}; {len(in_window)} extra CI roots below 0.65 eV")


if __name__ == "__main__":
    main()
