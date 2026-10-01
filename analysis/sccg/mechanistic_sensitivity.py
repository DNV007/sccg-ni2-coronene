#!/usr/bin/env python3
"""Mechanistic and analysis-domain controls for the B1 SCCGs."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))

from sccg.descriptors import D2_decomposition, D5_pathway_evenness  # noqa: E402
from sccg.graph import Manifold, SCCG  # noqa: E402
from sccg.io import load_manifold  # noqa: E402
from sccg.rassi_reconstruction import (  # noqa: E402
    load_rassi_reconstruction,
    preferred_reconstruction_source,
)

B1 = ROOT / "calculations" / "B1"
OUT = B1 / "mechanistic_sensitivity.json"
GRID = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65])

TIERS = {
    "production_MS_CASPT2_CAS26_15": (
        B1 / "02_cheap" / "manifold_cheap.json",
        B1 / "02_cheap" / "rassi_pt2.rassi.h5",
        B1 / "reconstruction_inputs" / "production_MS_CASPT2_CAS26_15.npz",
    ),
    "bridge_MS_CASPT2_CAS12_12": (
        B1 / "03c_caspt2_compact" / "manifold_caspt2_compact.json",
        B1 / "03c_caspt2_compact" / "rassi_caspt2.rassi.h5",
        B1 / "reconstruction_inputs" / "bridge_MS_CASPT2_CAS12_12.npz",
    ),
    "compact_CASCI_model_CAS12_12": (
        B1 / "03b_exact" / "manifold_exact.json",
        B1 / "03b_exact" / "cas_ci.rassi.h5",
        B1 / "reconstruction_inputs" / "compact_CASCI_model_CAS12_12.npz",
    ),
}


def rho(first: list[float], second: list[float]) -> float | None:
    # Constant curves have no rank correlation; JSON null preserves that domain.
    if np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return None
    return float(spearmanr(first, second).statistic)


def shared_weights(graph: SCCG) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left, right = graph.endpoint_indices()
    left_edges = graph.weights @ left
    right_edges = graph.weights @ right
    raw = left_edges * right_edges
    normalized = raw / raw.sum() if raw.sum() else np.zeros_like(raw)
    return left_edges, right_edges, raw, normalized


def pathway_statistics(graph: SCCG, keep: np.ndarray | None = None) -> dict[str, float]:
    left_edges, right_edges, raw, normalized = shared_weights(graph)
    if keep is not None:
        left_edges = left_edges[keep]
        right_edges = right_edges[keep]
        raw = raw[keep]
        normalized = raw / raw.sum() if raw.sum() else np.zeros_like(raw)
    left_total = left_edges.sum()
    right_total = right_edges.sum()
    d2 = (float(left_edges @ right_edges) / float(left_total * right_total)
          if left_total and right_total else 0.0)
    return {
        "D2": d2,
        "participation_ratio": (
            float(1.0 / np.sum(normalized ** 2)) if normalized.sum() else 0.0
        ),
        "largest_shared_mediator_weight": float(normalized.max()) if normalized.size else 0.0,
    }


def sweep(manifold: Manifold, grid: np.ndarray = GRID, sigma: float = 0.10) -> dict[str, list[float]]:
    result = {key: [] for key in (
        "D2", "cosine_alignment", "concentration", "N_q", "N_r",
        "D5", "participation_ratio", "largest_shared_mediator_weight",
    )}
    for photon_energy in grid:
        graph = SCCG(manifold, float(photon_energy), sigma=sigma, alpha=1.0, beta=0.0)
        decomposition = D2_decomposition(graph)
        statistics = pathway_statistics(graph)
        for key in ("D2", "cosine_alignment", "concentration", "N_q", "N_r"):
            result[key].append(float(decomposition[key]))
        result["D5"].append(float(D5_pathway_evenness(graph)))
        result["participation_ratio"].append(statistics["participation_ratio"])
        result["largest_shared_mediator_weight"].append(
            statistics["largest_shared_mediator_weight"]
        )
    return result


def cross_tier_correlations(tier_sweeps: dict[str, dict[str, list[float]]]) -> dict[str, object]:
    production = tier_sweeps["production_MS_CASPT2_CAS26_15"]
    bridge = tier_sweeps["bridge_MS_CASPT2_CAS12_12"]
    model = tier_sweeps["compact_CASCI_model_CAS12_12"]
    output = {}
    for key in tier_sweeps[next(iter(tier_sweeps))]:
        output[key] = {
            "model_vs_bridge": rho(model[key], bridge[key]),
            "bridge_vs_production": rho(bridge[key], production[key]),
            "model_vs_production": rho(model[key], production[key]),
        }
    return output


def parentage(path: Path) -> dict[str, np.ndarray]:
    data = load_rassi_reconstruction(path)
    return {
        "weights": np.asarray(data["sf_to_so_weights"]),
        "multiplicities": np.asarray(data["multiplicities"], dtype=int),
        "local_roots": np.asarray(data["local_roots"], dtype=int),
    }


def dominant_edge_contribution(graph: SCCG, mediator: int, side: str) -> dict[str, float | int]:
    left, right = graph.endpoint_indices()
    endpoint = left if side == "left" else right
    contributions = graph.weights[:, mediator] * endpoint
    state = int(np.argmax(contributions))
    dipole_strength = float(np.sum(graph.manifold.dipoles[state, mediator] ** 2))
    return {
        "so_state": state + 1,
        "contribution": float(contributions[state]),
        "energy_gap_eV": float(abs(graph.manifold.energies[state] - graph.manifold.energies[mediator])),
        "dipole_strength_au2": dipole_strength,
        "endpoint_projector_weight": float(endpoint[state]),
    }


def mediator_anatomy(manifold: Manifold, parents: dict[str, np.ndarray],
                     photon_energy: float) -> dict[str, object]:
    graph = SCCG(manifold, photon_energy, sigma=0.10, alpha=1.0, beta=0.0)
    left_edges, right_edges, _, normalized = shared_weights(graph)
    left, right = graph.endpoint_indices()
    order = np.argsort(normalized)[::-1][:5]
    records = []
    for index in order:
        sf_index = int(np.argmax(parents["weights"][:, index]))
        records.append({
            "so_state": int(index + 1),
            "energy_eV": float(manifold.energies[index]),
            "shared_mediator_weight": float(normalized[index]),
            "W_Li": float(left_edges[index]),
            "W_iR": float(right_edges[index]),
            "p_L": float(left[index]),
            "p_R": float(right[index]),
            "spin_mixing": float(manifold.spin_mixing[index]),
            "dominant_spin_free_root_global": sf_index + 1,
            "dominant_spin_free_root_within_multiplicity": int(parents["local_roots"][sf_index]),
            "dominant_multiplicity": int(parents["multiplicities"][sf_index]),
            "dominant_parent_weight": float(parents["weights"][sf_index, index]),
            "largest_left_edge_contribution": dominant_edge_contribution(graph, int(index), "left"),
            "largest_right_edge_contribution": dominant_edge_contribution(graph, int(index), "right"),
        })

    endpoint_representatives = sorted({int(np.argmax(left)), int(np.argmax(right))})
    keep_representatives = np.ones(manifold.n_states, dtype=bool)
    keep_representatives[endpoint_representatives] = False
    keep_character = (left + right) <= 0.5
    return {
        "photon_energy_eV": photon_energy,
        "baseline": pathway_statistics(graph),
        "endpoint_representative_SO_states": [value + 1 for value in endpoint_representatives],
        "excluding_endpoint_representatives": pathway_statistics(graph, keep_representatives),
        "excluding_states_with_pL_plus_pR_above_0_5": pathway_statistics(graph, keep_character),
        "top_shared_mediators": records,
    }


def perturbed_mediator(manifold: Manifold, photon_energy: float) -> dict[str, object]:
    baseline_graph = SCCG(manifold, photon_energy, sigma=0.10, alpha=1.0, beta=0.0)
    _, _, _, weights = shared_weights(baseline_graph)
    top = int(np.argmax(weights))
    scans = {}
    for shift in (-0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.10):
        energies = manifold.energies.copy()
        energies[top] += shift
        shifted = replace(manifold, energies=energies)
        scans[f"{shift:+.3f}"] = pathway_statistics(
            SCCG(shifted, photon_energy, sigma=0.10, alpha=1.0, beta=0.0)
        )
    keep = np.ones(manifold.n_states, dtype=bool)
    keep[top] = False
    return {
        "baseline_top_SO_state": top + 1,
        "energy_shift_eV": scans,
        "top_mediator_deleted": pathway_statistics(baseline_graph, keep),
    }


def subset_by_energy(manifold: Manifold, cutoff: float) -> Manifold:
    keep = manifold.energies <= cutoff
    indices = np.flatnonzero(keep)
    return Manifold(
        energies=manifold.energies[keep],
        dipoles=manifold.dipoles[np.ix_(indices, indices, np.arange(3))],
        soc=None,
        endpoint_overlaps=manifold.endpoint_overlaps[keep],
        label=f"{manifold.label}_E_le_{cutoff}",
        dipoles_complex=(
            manifold.dipoles_complex[np.ix_(indices, indices, np.arange(3))]
            if manifold.dipoles_complex is not None else None
        ),
        spin_mixing=manifold.spin_mixing[keep] if manifold.spin_mixing is not None else None,
        s2=manifold.s2[keep] if manifold.s2 is not None else None,
    )


def main() -> None:
    manifolds = {tier: load_manifold(paths[0]) for tier, paths in TIERS.items()}
    parents = {
        tier: parentage(preferred_reconstruction_source(paths[1], paths[2]))
        for tier, paths in TIERS.items()
    }

    base_sweeps = {tier: sweep(manifold) for tier, manifold in manifolds.items()}

    upper_limit_results = {}
    for upper in (0.50, 0.65, 0.80, 1.00):
        grid = np.concatenate(([0.02], np.arange(0.05, upper + 0.001, 0.05)))
        tier_sweeps = {tier: sweep(manifold, grid) for tier, manifold in manifolds.items()}
        upper_limit_results[f"{upper:.2f}"] = {
            "grid_eV": grid.tolist(),
            "correlations": cross_tier_correlations(tier_sweeps),
        }

    production_span = float(np.ptp(manifolds["production_MS_CASPT2_CAS26_15"].energies))
    dimensionless_sigma = 0.10 / production_span
    dimensionless_grid = np.concatenate(([0.02], np.arange(0.05, 1.001, 0.05)))
    normalized_manifolds = {
        tier: replace(manifold, energies=(manifold.energies - manifold.energies.min()) /
                      np.ptp(manifold.energies))
        for tier, manifold in manifolds.items()
    }
    normalized_sweeps = {
        tier: sweep(manifold, dimensionless_grid, sigma=dimensionless_sigma)
        for tier, manifold in normalized_manifolds.items()
    }

    matched_windows = {}
    for cutoff in (0.25, 0.35, 0.50, 0.58):
        subsets = {tier: subset_by_energy(manifold, cutoff)
                   for tier, manifold in manifolds.items()}
        tier_sweeps = {tier: sweep(manifold) for tier, manifold in subsets.items()}
        matched_windows[f"{cutoff:.2f}"] = {
            "n_states": {tier: manifold.n_states for tier, manifold in subsets.items()},
            "endpoint_projector_weight_retained": {
                tier: [float(manifold.endpoint_overlaps[:, 0].sum()),
                       float(manifold.endpoint_overlaps[:, 1].sum())]
                for tier, manifold in subsets.items()
            },
            "correlations": cross_tier_correlations(tier_sweeps),
        }

    anatomy = {
        tier: {
            f"{energy:.2f}": mediator_anatomy(manifolds[tier], parents[tier], energy)
            for energy in (0.02, 0.25, 0.50, 0.65)
        }
        for tier in manifolds
    }
    detuning = {
        tier: {
            f"{energy:.2f}": perturbed_mediator(manifolds[tier], energy)
            for energy in (0.02, 0.25, 0.50)
        }
        for tier in manifolds
    }

    record = {
        "projector_parser_validation": (
            "OpenMolcas SO coefficients are C[so,basis]; C @ H_SO @ C.conj().T is diagonal. "
            "The whole-multiplet projectors reproduce the printed RASSI parent weights."
        ),
        "photon_energy_grid_eV": GRID.tolist(),
        "D2_exact_decomposition": base_sweeps,
        "base_cross_tier_spearman": cross_tier_correlations(base_sweeps),
        "photon_energy_upper_limit_sensitivity": upper_limit_results,
        "spectrally_normalized_coordinate": {
            "coordinate_grid": dimensionless_grid.tolist(),
            "dimensionless_sigma": dimensionless_sigma,
            "spectral_widths_eV": {
                tier: float(np.ptp(manifold.energies)) for tier, manifold in manifolds.items()
            },
            "definition": "E_tilde=(E-E_min)/B_t; photon_energy_t=x*B_t; "
                          "sigma_t=B_t*(0.10 eV/B_production); kernel recomputed, no interpolation",
            "raw_sweeps": normalized_sweeps,
            "correlations": cross_tier_correlations(normalized_sweeps),
        },
        "matched_absolute_state_energy_windows": matched_windows,
        "mediator_anatomy_and_endpoint_exclusion": anatomy,
        "graph_level_top_mediator_detuning": detuning,
    }
    OUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT}")
    print("base", record["base_cross_tier_spearman"])
    print("upper limits", {
        key: value["correlations"]["D2"]
        for key, value in upper_limit_results.items()
    })
    print("normalized", record["spectrally_normalized_coordinate"]["correlations"]["D2"])
    print("matched windows", {
        key: value["correlations"]["D2"] for key, value in matched_windows.items()
    })


if __name__ == "__main__":
    main()
