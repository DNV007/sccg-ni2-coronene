"""
Load energies, dipoles, and endpoint-projector weights from archived JSON.
Optional SOC matrices are retained as source data, not added to graph edges.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sccg.graph import Manifold


def load_manifold(path: str | Path) -> Manifold:
    """Load a manifold from a JSON file in the canonical schema.

    Schema (illustrative):
    {
      "label": "B1_cheap",
      "energies_eV": [...],                  # length N
      "dipoles_au": [[[x,y,z], ...], ...],   # NxNx3
      "soc_cm-1_real": [[...], ...],         # NxN, optional
      "soc_cm-1_imag": [[...], ...],         # NxN, optional
      "endpoint_overlaps": [[pL_i, pR_i], ...]  # Nx2
    }
    """
    p = Path(path)
    with p.open() as f:
        d = json.load(f)
    energies = np.array(d["energies_eV"], dtype=float)
    dipoles = np.array(d["dipoles_au"], dtype=float)
    overlaps = np.array(d["endpoint_overlaps"], dtype=float)
    soc = None
    if "soc_cm-1_real" in d:
        soc_r = np.array(d["soc_cm-1_real"], dtype=float)
        soc_i = np.array(d.get("soc_cm-1_imag", np.zeros_like(soc_r)), dtype=float)
        soc = soc_r + 1j * soc_i
    dipoles_complex = None
    if "dipoles_real_au" in d:
        dre = np.array(d["dipoles_real_au"], dtype=float)
        dim = np.array(d.get("dipoles_imag_au", np.zeros_like(dre)), dtype=float)
        dipoles_complex = dre + 1j * dim
    spin_mixing = np.array(d["spin_mixing"], dtype=float) if "spin_mixing" in d else None
    s2 = np.array(d["s2_expectation"], dtype=float) if "s2_expectation" in d else None
    return Manifold(
        energies=energies,
        dipoles=dipoles,
        soc=soc,
        endpoint_overlaps=overlaps,
        label=d.get("label", p.stem),
        dipoles_complex=dipoles_complex,
        spin_mixing=spin_mixing,
        s2=s2,
    )
