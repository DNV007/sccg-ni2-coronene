#!/usr/bin/env python3
"""Independent matrix audit and retained-state controls for the benchmark.

No production SCCG modules are imported. Inputs are the lossless RASSI NPZ
exports and archived JSON manifolds. All truncations keep the original five-root
orbitals, spin-free solutions, energies and operators; only state interaction
is changed. Output includes full curves, root-by-root influence, a conditional
decoupling/deletion path, matching diagnostics, and zero-coupling-node checks.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
B1 = ROOT / "calculations" / "B1"
EV = 27.211386245988
GRID = np.r_[0.02, np.arange(1, 14) * 0.05]
LOW_GRID = np.round(np.arange(0.02, 0.10001, 0.005), 6)
TIERS = {
    "production": ("production_MS_CASPT2_CAS26_15", "02_cheap/manifold_cheap.json"),
    "bridge": ("bridge_MS_CASPT2_CAS12_12", "03c_caspt2_compact/manifold_caspt2_compact.json"),
    "anchor": ("compact_CASCI_model_CAS12_12", "03b_exact/manifold_exact.json"),
}


def load_inputs(label):
    with np.load(B1 / "reconstruction_inputs" / f"{label}.npz") as archive:
        data = {k: archive[k] for k in archive.files}
    mult = data["multiplicities"]
    root = np.repeat(np.arange(len(mult)), mult)
    ms = np.concatenate([np.arange(m) - (m - 1) / 2 for m in mult])
    spin_delta = (mult[root, None] == mult[root][None, :]) & (ms[:, None] == ms)
    data["basis_root"] = root
    data["basis_mu"] = data["sf_dipoles"][:, root[:, None], root] * spin_delta
    return data


def build(data, keep_sf=None, h_override=None, archived=False):
    if keep_sf is None:
        keep_sf = np.ones(len(data["multiplicities"]), dtype=bool)
    keep = np.flatnonzero(keep_sf[data["basis_root"]])
    h = data["h_so"] if h_override is None else h_override
    if archived:
        e, c = data["so_energies"], data["so_coefficients"]
    else:
        e, u = np.linalg.eigh(h[np.ix_(keep, keep)])
        c = u.conj().T
    c_full = np.zeros((len(e), len(data["basis_root"])), complex)
    c_full[:, keep] = c
    mu = np.array([c_full @ op @ c_full.conj().T for op in data["basis_mu"]])
    parent = np.column_stack([
        np.sum(np.abs(c_full[:, data["basis_root"] == r]) ** 2, axis=1)
        for r in range(len(keep_sf))
    ])
    endpoint_roots = [np.flatnonzero((data["multiplicities"] == m) &
                                    (data["local_roots"] == 1))[0] for m in (5, 1)]
    return {"energy": (e - np.min(e)) * EV, "absolute": e * EV, "hartree": e,
            "mu": mu, "endpoint": parent[:, endpoint_roots],
            "parent": parent, "coeff": c_full}


def measure(model, omega, loop=False):
    e, mu, endpoints = model["energy"], model["mu"], model["endpoint"]
    n = len(e)
    if loop:
        w = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    strength = sum(abs(mu[c, i, j]) ** 2 for c in range(3))
                    w[i, j] = strength * np.exp(-0.5 * ((abs(e[i] - e[j]) - omega) / 0.1) ** 2)
        left = np.array([sum(w[j, i] * endpoints[j, 0] for j in range(n)) for i in range(n)])
        right = np.array([sum(w[i, j] * endpoints[j, 1] for j in range(n)) for i in range(n)])
    else:
        w = np.sum(np.abs(mu) ** 2, axis=0) * np.exp(
            -0.5 * ((np.abs(e[:, None] - e[None, :]) - omega) / 0.1) ** 2)
        np.fill_diagonal(w, 0.0)
        left, right = w.T @ endpoints[:, 0], w @ endpoints[:, 1]
    assert left.sum() > 0 and right.sum() > 0
    q, r = left / left.sum(), right / right.sum()
    d2 = float(q @ r)
    p = q * r / d2
    active = p[p > 1e-6]
    active /= active.sum()
    d5 = 0.0 if len(active) <= 1 else -np.log(np.sum(active**2) + 1e-12) / np.log(len(active))
    return {"D2": d2, "PR": float(1 / np.sum(p**2)), "max_p": float(p.max()),
            "D5": float(d5), "q": q, "r": r, "p": p,
            "left_total": float(left.sum()), "right_total": float(right.sum())}


def sweep(model, grid=GRID):
    rows = [measure(model, w) for w in grid]
    return {key: np.array([row[key] for row in rows]) for key in rows[0]}


def ranked_tv(p, q):
    n = max(len(p), len(q))
    a = np.pad(np.sort(p)[::-1], (0, n - len(p)))
    b = np.pad(np.sort(q)[::-1], (0, n - len(q)))
    return float(0.5 * np.abs(a - b).sum())


def compare(curve, baseline):
    delta = curve["D2"] - baseline["D2"]
    tv = [ranked_tv(p, q) for p, q in zip(curve["p"], baseline["p"])]
    return {"D2_range": [float(curve["D2"].min()), float(curve["D2"].max())],
            "max_abs_D2_change": float(np.max(np.abs(delta))),
            "relative_L2_D2": float(np.linalg.norm(delta) / np.linalg.norm(baseline["D2"])),
            "max_ranked_TV": max(tv), "ranked_TV": tv,
            "omega_of_max_abs_D2_change_eV": float(GRID[np.argmax(np.abs(delta))])}


def anatomy(model, omega):
    stats = measure(model, omega)
    top = int(np.argmax(stats["p"]))
    edges = {}
    for side, endpoint in enumerate(("left", "right")):
        gap = np.abs(model["energy"] - model["energy"][top])
        strength = np.sum(np.abs(model["mu"][:, top, :]) ** 2, axis=0)
        contribution = strength * np.exp(-0.5 * ((gap - omega) / 0.1) ** 2) * model["endpoint"][:, side]
        contribution[top] = 0
        j = int(np.argmax(contribution))
        edges[endpoint] = {"so_state": j + 1, "gap_eV": float(gap[j]),
                           "endpoint_weight": float(model["endpoint"][j, side]),
                           "dipole_strength_au2": float(strength[j]),
                           "fraction_of_contracted_edge": float(contribution[j] / contribution.sum())}
    return {"so_state": top + 1, "energy_eV": float(model["energy"][top]),
            "parent_root_global": int(model["parent"][top].argmax()) + 1,
            "parent_weight": float(model["parent"][top].max()),
            "p_L": float(model["endpoint"][top, 0]), "p_R": float(model["endpoint"][top, 1]),
            "q": float(stats["q"][top]), "r": float(stats["r"][top]),
            "shared_weight": float(stats["p"][top]), "dominant_edges": edges}


def pair_correlations(curves):
    return {name: float(spearmanr(curves[a]["D2"], curves[b]["D2"]).statistic)
            for name, a, b in (("compact_leg", "anchor", "bridge"),
                               ("two_MS_tiers", "bridge", "production"),
                               ("end_to_end", "production", "anchor"))}


def reduced_model(full, keep):
    return {"energy": full["energy"][keep],
            "mu": full["mu"][:, keep][:, :, keep], "endpoint": full["endpoint"][keep]}


def omission_control(data, full, baseline, keep_sf, reduced):
    omitted_basis = ~keep_sf[data["basis_root"]]
    omitted_parentage = np.sum(np.abs(full["coeff"][:, omitted_basis]) ** 2, axis=1)
    h_cut = data["h_so"].copy()
    cross = omitted_basis[:, None] != omitted_basis[None, :]
    coupling = np.max(np.abs(h_cut[cross])) * EV
    h_cut[cross] = 0
    # Diagonalize each sector separately to use exactly the same retained-sector
    # eigenbasis as the truncated model, including near/exact degeneracies.
    # Diagonalizing the whole block-diagonal matrix could pick a different gauge.
    discarded = build(data, ~keep_sf)
    energies = np.r_[reduced["hartree"], discarded["hartree"]]
    order = np.argsort(energies, kind="stable")
    coefficients = np.vstack((reduced["coeff"], discarded["coeff"]))[order]
    decoupled = {"energy": (energies[order] - energies.min()) * EV,
                 "mu": np.array([coefficients @ op @ coefficients.conj().T
                                 for op in data["basis_mu"]]),
                 "endpoint": np.vstack((reduced["endpoint"], discarded["endpoint"]))[order]}
    assert np.max(np.abs(coefficients @ h_cut @ coefficients.conj().T -
                         np.diag(energies[order]))) < 1e-10
    decoupled_sweep = sweep(decoupled)
    # A graph-node deletion comparison needs a stated association rule. It is
    # not an exact identification of deleted SO states with spin-free roots.
    n_remove = int(omitted_basis.sum())
    removed_nodes = np.argsort(omitted_parentage, kind="stable")[-n_remove:]
    keep_nodes = np.ones(len(omitted_parentage), bool)
    keep_nodes[removed_nodes] = False
    frozen_sweep = sweep(reduced_model(full, keep_nodes))
    mask_p = baseline["p"][:, keep_nodes]
    mediator_only_p = mask_p / mask_p.sum(axis=1, keepdims=True)

    reduced_sweep = sweep(reduced)
    retained_in_decoupled = order < len(reduced["energy"])
    same_reduced = sweep(reduced_model(decoupled, retained_in_decoupled))
    assert np.max(np.abs(same_reduced["D2"] - reduced_sweep["D2"])) < 1e-10
    overlaps = np.abs(full["coeff"] @ reduced["coeff"].conj().T) ** 2
    matched_full, matched_reduced = linear_sum_assignment(-overlaps)
    order = np.argsort(matched_reduced)
    mapped = matched_full[order]
    n = len(mapped)
    old_mu2 = np.sum(np.abs(full["mu"][:, mapped][:, :, mapped]) ** 2, axis=0)
    new_mu2 = np.sum(np.abs(reduced["mu"]) ** 2, axis=0)
    np.fill_diagonal(old_mu2, 0)
    np.fill_diagonal(new_mu2, 0)
    omega = GRID[np.argmax(np.abs(reduced_sweep["D2"] - baseline["D2"]))]
    old_stats, new_stats = measure(full, omega), measure(reduced, omega)
    j = int(np.argmax(new_stats["p"]))
    i = int(mapped[j])
    # Eigenstate matching is a diagnostic, reported together with its overlap.
    matching = {
        "minimum_squared_overlap": float(overlaps[mapped, np.arange(n)].min()),
        "median_squared_overlap": float(np.median(overlaps[mapped, np.arange(n)])),
        "max_absolute_eigenvalue_shift_eV": float(np.max(np.abs(full["absolute"][mapped] - reduced["absolute"]))),
        "max_relative_energy_shift_eV": float(np.max(np.abs(full["energy"][mapped] - reduced["energy"]))),
        "max_endpoint_weight_change": float(np.max(np.abs(full["endpoint"][mapped] - reduced["endpoint"]))),
        "transition_strength_relative_Frobenius_change": float(np.linalg.norm(new_mu2 - old_mu2) / np.linalg.norm(old_mu2)),
        "example_omega_eV": float(omega), "example_reduced_top_mediator": anatomy(reduced, omega),
        "matched_full_so_state": i + 1, "squared_overlap": float(overlaps[i, j]),
        "matched_full_energy_eV": float(full["energy"][i]),
        "matched_full_endpoint": full["endpoint"][i],
        "matched_full_q_r_p": [float(old_stats[k][i]) for k in ("q", "r", "p")],
    }
    return {
        "n_omitted_spin_projection_components": n_remove,
        "omitted_parentage_per_full_SO_state": omitted_parentage,
        "parentage_weighted_q_fraction": baseline["q"] @ omitted_parentage,
        "parentage_weighted_r_fraction": baseline["r"] @ omitted_parentage,
        "parentage_weighted_shared_fraction": baseline["p"] @ omitted_parentage,
        "max_retained_omitted_H_element_eV": float(coupling),
        "decoupled_full_dimension": decoupled_sweep,
        "decoupling_change_vs_full": compare(decoupled_sweep, baseline),
        "deletion_change_vs_decoupled": compare(reduced_sweep, decoupled_sweep),
        "frozen_SO_node_deletion": {"removed_SO_states": removed_nodes + 1,
            "remaining_endpoint_traces": full["endpoint"][keep_nodes].sum(axis=0),
            "removed_shared_fraction": baseline["p"][:, ~keep_nodes].sum(axis=1),
            "removed_q_fraction": baseline["q"][:, ~keep_nodes].sum(axis=1),
            "removed_r_fraction": baseline["r"][:, ~keep_nodes].sum(axis=1),
            "mediator_only_PR": 1 / np.sum(mediator_only_p**2, axis=1),
            "mediator_only_max_p": mediator_only_p.max(axis=1),
            "full_node_deletion_curves": frozen_sweep,
            "full_node_deletion_change": compare(frozen_sweep, baseline)},
        "matching_diagnostic": matching,
    }


def jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def main():
    results, curves_by_case, low_curves, full_models = {}, {}, {}, {}
    existing = json.loads((B1 / "root_truncation_sensitivity.json").read_text())
    for tier, (label, manifold_path) in TIERS.items():
        data = load_inputs(label)
        original = json.loads((B1 / manifold_path).read_text())
        archived = {"energy": np.array(original["energies_eV"]),
                    "mu": np.array(original["dipoles_au"]).transpose(2, 0, 1),
                    "endpoint": np.array(original["endpoint_overlaps"])}
        full = build(data)
        full_models[tier] = full
        from_coefficients = build(data, archived=True)
        full_curve = sweep(full)
        independent_curve = sweep(from_coefficients)
        # Explicit scalar summation provides a second contraction implementation.
        loop_rows = [measure(archived, w, loop=True) for w in GRID]
        archived_sweep = sweep(archived)
        reference_sweep = existing["tiers"][label]["5"]
        validation = {
            "endpoint_projector_max_error_vs_archived": float(np.max(np.abs(from_coefficients["endpoint"] - archived["endpoint"]))),
            "D2_scalar_loop_vs_array_max_error": float(max(abs(row["D2"] - archived_sweep["D2"][i]) for i, row in enumerate(loop_rows))),
            "D2_reconstruction_vs_published_max_error": float(np.max(np.abs(full_curve["D2"] - reference_sweep["D2"]))),
            "PR_reconstruction_vs_published_max_error": float(np.max(np.abs(full_curve["PR"] - reference_sweep["participation_ratio"]))),
            "D2_independent_SO_transform_vs_archived_max_error": float(np.max(np.abs(independent_curve["D2"] - archived_sweep["D2"]))),
            "PR_independent_SO_transform_vs_archived_max_error": float(np.max(np.abs(independent_curve["PR"] - archived_sweep["PR"]))),
        }
        assert validation["endpoint_projector_max_error_vs_archived"] < 1e-9
        assert validation["D2_scalar_loop_vs_array_max_error"] < 1e-12
        assert validation["D2_reconstruction_vs_published_max_error"] < 1e-9
        assert validation["PR_reconstruction_vs_published_max_error"] < 1e-6
        masks = {f"N{n}": data["local_roots"] <= n for n in (3, 4, 5)}
        for m, tag in ((5, "Q"), (3, "T"), (1, "S")):
            for n in range(1 if tag == "T" else 2, 6):
                masks[f"omit_{tag}{n}"] = ~((data["multiplicities"] == m) & (data["local_roots"] == n))
        cases, controls = {}, {}
        for case, keep in masks.items():
            model = build(data, keep)
            curve = sweep(model)
            traces = model["endpoint"].sum(axis=0)
            assert np.max(np.abs(traces - [5, 1])) < 1e-10
            cases[case] = {"curves": curve, "n_SO": len(model["energy"]),
                           "endpoint_projector_traces": traces,
                           "difference_from_N5": compare(curve, full_curve)}
            curves_by_case.setdefault(case, {})[tier] = curve
            if case in ("N3", "N4", "omit_Q5", "omit_T5", "omit_S5"):
                controls[case] = omission_control(data, full, full_curve, keep, model)
        padded = {"energy": np.r_[full["energy"], np.linspace(2, 3, 9)],
                  "mu": np.pad(full["mu"], ((0, 0), (0, 9), (0, 9))),
                  "endpoint": np.pad(full["endpoint"], ((0, 9), (0, 0)))}
        padded_curve = sweep(padded)
        null = {key: float(np.max(np.abs(padded_curve[key] - full_curve[key])))
                for key in ("D2", "PR", "max_p", "D5")}
        assert max(null.values()) < 1e-10
        low = sweep(full, LOW_GRID)
        low_curves[tier] = low
        results[tier] = {"independent_validation": validation,
                         "zero_coupling_node_test_max_errors": null,
                         "cases": cases, "omission_controls": controls,
                         "low_energy_curves": low,
                         "low_energy_summary": {key: {"min": float(low[key].min()),
                            "median": float(np.median(low[key])), "max": float(low[key].max())}
                            for key in ("D2", "PR", "max_p")},
                         "mediator_edges": {f"{w:.2f}": anatomy(full, w) for w in (0.02, 0.05, 0.10)}}
    # A second retention rule on already diagonalized SO eigenstates. Select the
    # common absolute-energy cutoff before computing cross-tier correlations.
    completeness = 0.999
    required_cutoffs = {}
    for tier, model in full_models.items():
        cumulative = np.cumsum(model["endpoint"], axis=0) / np.array([5, 1])
        first = np.flatnonzero(np.all(cumulative >= completeness, axis=1))[0]
        required_cutoffs[tier] = float(model["energy"][first])
    common_cutoff = max(required_cutoffs.values())
    selected_curves, selected_records = {}, {}
    for tier, model in full_models.items():
        keep = model["energy"] <= common_cutoff + 1e-12
        selected = reduced_model(model, keep)
        selected_curves[tier] = sweep(selected)
        selected_records[tier] = {"n_SO": int(keep.sum()),
            "endpoint_fractions_retained": selected["endpoint"].sum(axis=0) / [5, 1],
            "curves": selected_curves[tier]}
    energy_rule = {"scope": "Select already diagonalized SO eigenstates at a common absolute energy; no rediagonalization",
        "required_endpoint_fraction": completeness, "minimum_cutoff_by_tier_eV": required_cutoffs,
        "common_cutoff_eV": common_cutoff, "tiers": selected_records,
        "cross_tier_spearman": pair_correlations(selected_curves)}
    output = {"scope": __doc__, "photon_energy_grid_eV": GRID,
              "low_energy_grid_eV": LOW_GRID,
              "definitions": {
                  "relative_L2": "||D2_case-D2_N5||_2 / ||D2_N5||_2 on the declared 14-point grid",
                  "ranked_TV": "Half L1 distance between descending-sorted normalized p vectors, padded with zeros; compares concentration profiles, not state identity",
                  "omitted_parentage": "sum of full-manifold SO squared coefficients on omitted spin-free/Ms basis; weighted q/r/p fractions are soft parentage measures, not unique deleted nodes",
                  "decoupling_path": "Zero retained-omitted H_SO blocks at full dimension, rediagonalize and transform dipoles; then remove omitted sector. Conditional nonlinear changes, not additive attribution",
                  "frozen_node_rule": "Delete exactly as many full SO nodes as omitted spin-projection components, choosing greatest omitted-root parentage; freeze energies/dipoles/projectors, then recontract and normalize",
              },
              "tiers": results,
              "energy_projector_retention_rule": energy_rule,
              "decoupled_full_dimension_cross_tier_spearman": {
                  case: pair_correlations({tier: result["omission_controls"][case]["decoupled_full_dimension"]
                                           for tier, result in results.items()})
                  for case in ("N3", "N4", "omit_Q5", "omit_T5", "omit_S5")},
              "cross_tier_spearman": {case: pair_correlations(curves) for case, curves in curves_by_case.items()},
              "production_bridge_ranked_TV": [ranked_tv(p, q) for p, q in zip(curves_by_case["N5"]["production"]["p"], curves_by_case["N5"]["bridge"]["p"])],
              "low_energy_production_bridge_ranked_TV": [ranked_tv(p, q) for p, q in zip(low_curves["production"]["p"], low_curves["bridge"]["p"])],
              }
    destination = B1 / "retained_state_audit.json"
    destination.write_text(json.dumps(output, indent=2, default=jsonable, allow_nan=False) + "\n")
    columns = ["tier", "retained_state_case", "photon_energy_eV", "D2", "PR", "max_p",
               "left_total", "right_total", "trace_L", "trace_R"]
    with (B1 / "retained_state_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for tier, result in results.items():
            for case, record in result["cases"].items():
                for index, energy in enumerate(GRID):
                    row = {"tier": tier, "retained_state_case": case,
                           "photon_energy_eV": float(energy),
                           "trace_L": float(record["endpoint_projector_traces"][0]),
                           "trace_R": float(record["endpoint_projector_traces"][1])}
                    row.update({key: float(record["curves"][key][index])
                                for key in columns[3:8]})
                    writer.writerow(row)
    print("wrote", destination)
    for tier, result in results.items():
        print(tier, "validation", result["independent_validation"])
        print(tier, "low energy", result["low_energy_summary"])
        for case in ("N3", "N4", "omit_Q5", "omit_T5", "omit_S5"):
            print(tier, case, {k: v for k, v in result["cases"][case]["difference_from_N5"].items() if k != "ranked_TV"})
    print("correlations", output["cross_tier_spearman"])
    print("energy/projector rule", common_cutoff, energy_rule["cross_tier_spearman"],
          {k: (v["n_SO"], v["endpoint_fractions_retained"].tolist()) for k, v in selected_records.items()})


if __name__ == "__main__":
    main()
