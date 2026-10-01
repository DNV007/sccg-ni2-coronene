#!/usr/bin/env python3
"""Stress-test SCCG results by retaining 3, 4, or 5 roots per multiplicity.

This reconstructs a smaller RASSI state interaction from the archived
spin-orbit Hamiltonian and spin-free dipole matrices.  It is a downward
truncation test of the available five-root calculation, not evidence of
convergence with respect to additional roots.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))

from sccg.descriptors import D2_mediator_bridge, D5_pathway_evenness  # noqa: E402
from sccg.graph import Manifold, SCCG  # noqa: E402
from sccg.io import load_manifold  # noqa: E402
from sccg.rassi_reconstruction import (  # noqa: E402
    load_rassi_reconstruction,
    preferred_reconstruction_source,
)

HARTREE_TO_EV = 27.211386245988
B1 = ROOT / "calculations" / "B1"
OUT = B1 / "root_truncation_sensitivity.json"
GRID = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65])

TIERS = {
    "production_MS_CASPT2_CAS26_15": (
        B1 / "02_cheap" / "rassi_pt2.rassi.h5",
        B1 / "reconstruction_inputs" / "production_MS_CASPT2_CAS26_15.npz",
        B1 / "02_cheap" / "manifold_cheap.json",
    ),
    "bridge_MS_CASPT2_CAS12_12": (
        B1 / "03c_caspt2_compact" / "rassi_caspt2.rassi.h5",
        B1 / "reconstruction_inputs" / "bridge_MS_CASPT2_CAS12_12.npz",
        B1 / "03c_caspt2_compact" / "manifold_caspt2_compact.json",
    ),
    "compact_CASCI_model_CAS12_12": (
        B1 / "03b_exact" / "cas_ci.rassi.h5",
        B1 / "reconstruction_inputs" / "compact_CASCI_model_CAS12_12.npz",
        B1 / "03b_exact" / "manifold_exact.json",
    ),
}


def _spin_basis_dipoles(sf_dipoles: np.ndarray, multiplicities: np.ndarray) -> np.ndarray:
    """Expand spin-free dipoles into the spin-projection basis used by H_SO."""
    offsets = np.concatenate(([0], np.cumsum(multiplicities)))
    n_basis = int(offsets[-1])
    out = np.zeros((3, n_basis, n_basis), dtype=complex)
    for first, first_mult in enumerate(multiplicities):
        for second, second_mult in enumerate(multiplicities):
            if first_mult != second_mult:
                continue
            for component in range(int(first_mult)):
                out[:, offsets[first] + component, offsets[second] + component] = (
                    sf_dipoles[:, first, second]
                )
    return out


def reconstruct(path: Path, roots_per_multiplicity: int) -> Manifold:
    """Diagonalize the archived H_SO after dropping higher spin-free roots."""
    data = load_rassi_reconstruction(path)
    multiplicities = np.asarray(data["multiplicities"], dtype=int)
    local_roots = np.asarray(data["local_roots"], dtype=int)
    h_so = np.asarray(data["h_so"])
    basis_dipoles = _spin_basis_dipoles(
        np.asarray(data["sf_dipoles"]), multiplicities
    )

    keep_sf = local_roots <= roots_per_multiplicity
    basis_to_sf = np.repeat(np.arange(len(multiplicities)), multiplicities)
    keep_basis = keep_sf[basis_to_sf]
    kept_sf_indices = np.flatnonzero(keep_sf)
    reduced_basis_to_original_sf = basis_to_sf[keep_basis]

    h_reduced = h_so[np.ix_(keep_basis, keep_basis)]
    energies_hartree, eigenvectors = np.linalg.eigh(h_reduced)
    coefficients = eigenvectors.conj().T  # rows are SO eigenstates

    dipoles_reduced = basis_dipoles[:, keep_basis][:, :, keep_basis]
    dipoles_so = np.array(
        [coefficients @ operator @ coefficients.conj().T for operator in dipoles_reduced]
    )

    weights = np.zeros((len(kept_sf_indices), len(energies_hartree)))
    for position, original_sf in enumerate(kept_sf_indices):
        columns = reduced_basis_to_original_sf == original_sf
        weights[position] = (np.abs(coefficients[:, columns]) ** 2).sum(axis=1)

    kept_mults = multiplicities[keep_sf]
    q1 = int(np.flatnonzero(kept_mults == 5)[0])
    s1 = int(np.flatnonzero(kept_mults == 1)[0])
    endpoint_overlaps = np.column_stack((weights[q1], weights[s1]))

    mult_weights = np.array(
        [weights[kept_mults == mult].sum(axis=0) for mult in np.unique(kept_mults)]
    )
    spin_mixing = 1.0 - mult_weights.max(axis=0)
    spins = (kept_mults - 1.0) / 2.0
    s2 = (weights * (spins * (spins + 1.0))[:, None]).sum(axis=0)

    return Manifold(
        energies=(energies_hartree - energies_hartree.min()) * HARTREE_TO_EV,
        dipoles=np.abs(dipoles_so).transpose(1, 2, 0),
        soc=None,
        endpoint_overlaps=endpoint_overlaps,
        label=f"{path.parent.name}_{roots_per_multiplicity}_roots_per_multiplicity",
        dipoles_complex=dipoles_so.transpose(1, 2, 0),
        spin_mixing=spin_mixing,
        s2=s2,
    )


def sweep(manifold: Manifold) -> dict[str, object]:
    d2: list[float] = []
    d5: list[float] = []
    pr: list[float] = []
    largest: list[float] = []
    for photon_energy in GRID:
        graph = SCCG(manifold, float(photon_energy), sigma=0.10, alpha=1.0, beta=0.0)
        d2.append(D2_mediator_bridge(graph))
        d5.append(D5_pathway_evenness(graph))
        left, right = graph.endpoint_indices()
        raw = (graph.weights @ left) * (graph.weights @ right)
        normalized = raw / raw.sum() if raw.sum() else np.zeros_like(raw)
        pr.append(float(1.0 / np.sum(normalized ** 2)) if raw.sum() else 0.0)
        largest.append(float(normalized.max()) if raw.sum() else 0.0)
    return {
        "n_so_states": manifold.n_states,
        "D2": d2,
        "D5": d5,
        "participation_ratio": pr,
        "largest_shared_mediator_weight": largest,
        "D2_argmax_eV": float(GRID[int(np.argmax(d2))]),
    }


def rho(first: list[float], second: list[float]) -> float:
    return float(spearmanr(first, second).statistic)


def main() -> None:
    results: dict[str, dict[str, object]] = {}
    validations: dict[str, dict[str, float]] = {}
    for tier, (h5_path, compact_path, json_path) in TIERS.items():
        source_path = preferred_reconstruction_source(h5_path, compact_path)
        results[tier] = {}
        for root_count in (3, 4, 5):
            results[tier][str(root_count)] = sweep(reconstruct(source_path, root_count))

        rebuilt = reconstruct(source_path, 5)
        archived = load_manifold(json_path)
        validations[tier] = {
            "max_abs_energy_error_eV": float(np.max(np.abs(rebuilt.energies - archived.energies))),
            "max_abs_dipole_error_au": float(np.max(np.abs(rebuilt.dipoles - archived.dipoles))),
            "max_abs_endpoint_weight_error": float(
                np.max(np.abs(rebuilt.endpoint_overlaps - archived.endpoint_overlaps))
            ),
        }

    correlations: dict[str, dict[str, dict[str, float]]] = {}
    for root_count in (3, 4, 5):
        key = str(root_count)
        production = results["production_MS_CASPT2_CAS26_15"][key]
        bridge = results["bridge_MS_CASPT2_CAS12_12"][key]
        model = results["compact_CASCI_model_CAS12_12"][key]
        correlations[key] = {}
        for descriptor in ("D2", "D5", "participation_ratio"):
            correlations[key][descriptor] = {
                "model_vs_bridge": rho(model[descriptor], bridge[descriptor]),
                "bridge_vs_production": rho(bridge[descriptor], production[descriptor]),
                "model_vs_production": rho(model[descriptor], production[descriptor]),
            }

    record = {
        "scope": (
            "Downward truncation of the archived five-root RASSI interaction to the first "
            "three or four roots per multiplicity. This is a boundary stress test, not "
            "convergence with respect to omitted higher roots."
        ),
        "photon_energy_grid_eV": GRID.tolist(),
        "reconstruction_validation_at_five_roots": validations,
        "tiers": results,
        "cross_tier_spearman": correlations,
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {OUT}")
    for root_count, table in correlations.items():
        print(root_count, table)
    print("validation", validations)


if __name__ == "__main__":
    main()
