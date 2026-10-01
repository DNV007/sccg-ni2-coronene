#!/usr/bin/env python3
"""Reproduce cross-tier and spin-free ordering statistics for Ni2/coronene.

The script compares the production multireference tier (SA-CASSCF -> MS-CASPT2
-> RASSI-SO), the compact CASCI model-space anchor, and the TDDFT comparison.

Rank statistics use NumPy; Spearman rho is Pearson r on average ranks.

Sources
-------
- 45 spin-orbit energies : archived manifolds (manifold_{cheap,exact}.json).
- D1--D5 descriptors      : archived descriptor sweeps.
- 15 spin-free energies   : archived sf_energies.json and selected RASSI outputs.
  The TDDFT comparison uses the archived energies; full TDDFT source outputs
  are outside this release and that comparison is not evidence for the
  retained-state result.

Usage:  python analysis/sccg/correlate_tiers.py
Writes: calculations/B1/sf_energies.json, calculations/B1/tier_correlations.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
B1 = ROOT / "calculations" / "B1"

# Historical reporting threshold, not a physical zero or numerical accuracy bound.
# It does not alter descriptor values or correlations. No reported sweep here
# triggers this auxiliary flag; it must not be used to infer physical signal loss.
FLOOR = 1e-2


# --- statistics (numpy only) -------------------------------------------------
def _avg_ranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    plain = np.empty(len(x), float)
    plain[order] = np.arange(1.0, len(x) + 1.0)
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, plain)
    return (sums / counts)[inv]


def pearson(a, b) -> float:
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom else float("nan")


def spearman(a, b) -> float:
    return pearson(_avg_ranks(a), _avg_ranks(b))


def spearman_boot_ci(a, b, n_boot: int = 20000, seed: int = 0):
    """Paired grid-point resampling percentiles, not a validated confidence interval.

    The deterministic photon-energy points are correlated. Independent paired
    resampling does not account for that dependence or establish sampling
    uncertainty. Retained for reproducibility of the archived boot95 fields;
    these percentiles are not used as uncertainty estimates in the study.
    Resamples that collapse to a constant are skipped.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    # A degenerate (constant) sweep has no rank information; report no CI rather
    # than crash. No particular tier is assumed to have a constant descriptor.
    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        if len(np.unique(a[idx])) > 1 and len(np.unique(b[idx])) > 1:
            vals.append(spearman(a[idx], b[idx]))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _peak_agreement(ks, a, b, top: int = 3):
    """Driving-frequency at the descriptor maximum in each tier and the overlap
    of their top-`top` windows -- the screening-relevant question ('does the
    cheap tier pick the same frequency window?') behind the bare rho."""
    ks = np.asarray(ks, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ta = set(np.round(ks[np.argsort(a)[-top:]], 6))
    tb = set(np.round(ks[np.argsort(b)[-top:]], 6))
    return {
        "peak_omega_cheap": float(ks[int(np.argmax(a))]),
        "peak_omega_exact": float(ks[int(np.argmax(b))]),
        f"top{top}_window_overlap": int(len(ta & tb)),
        f"top{top}_n": top,
    }


# --- committed-artifact readers ---------------------------------------------
def _energies(path: Path) -> np.ndarray:
    return np.array(json.loads(path.read_text())["energies_eV"], float)


def _descriptor_map(path: Path, key: str) -> dict[float, float]:
    sweep = json.loads(path.read_text())["sweep"]
    return {round(r["omega_eV"], 6): r[key] for r in sweep if r.get(key) is not None}


def _aligned(ma: dict, mb: dict):
    ks = sorted(set(ma) & set(mb))
    return np.array([ma[k] for k in ks]), np.array([mb[k] for k in ks])


# --- spin-free energy extraction (one-time, from gitignored RASSI outputs) ---
# Anchor on ":: RASSI" so we do NOT also match ":: SO-RASSI State ..." (the 45
# spin-orbit states), which would silently overwrite the 15 spin-free values
# with the energy-sorted SO list and fake a perfect correlation.
_RASSI = re.compile(r"::\s+RASSI State\s+(\d+)\s+Total energy:\s+(-?\d+\.\d+)")


def _parse_sf(out_path: Path, n: int = 15) -> list[float]:
    e = {int(i): float(v) for i, v in _RASSI.findall(out_path.read_text())}
    return [e[k] for k in range(1, n + 1)]


def _tddft_sf(manifold_tddft: Path) -> list[float]:
    states = json.loads(manifold_tddft.read_text())["states"]
    out: list[float] = []
    for mult in (5, 3, 1):  # quintet, triplet, singlet -- matches RASSI block order
        grp = [s for s in states if s["mult"] == mult]
        gs = [s for s in grp if s["type"] == "GS"][:1]
        ex = sorted((s for s in grp if s["type"] != "GS"), key=lambda s: s["E_eV"])
        out += [s["E_eV"] for s in (gs + ex)[:5]]
    return out


def _sf_energies() -> dict:
    path = B1 / "sf_energies.json"
    if path.exists():
        return json.loads(path.read_text())
    data = {
        "_provenance": {
            "cheap_Eh": "02_cheap/rassi_pt2.out, RASSI State 1..15 (spin-free, Hartree)",
            "exact_Eh": "03b_exact/cas_ci.out, RASSI State 1..15 (spin-free, Hartree)",
            "tddft_eV": "03_reference/manifold_tddft.json, per multiplicity GS + 4 lowest EX",
            "state_order": "1-5 quintet (S=2), 6-10 triplet (S=1), 11-15 singlet (S=0)",
        },
        "cheap_Eh": _parse_sf(B1 / "02_cheap" / "rassi_pt2.out"),
        "exact_Eh": _parse_sf(B1 / "03b_exact" / "cas_ci.out"),
        "tddft_eV": _tddft_sf(B1 / "03_reference" / "manifold_tddft.json"),
    }
    path.write_text(json.dumps(data, indent=2))
    return data


# --- active-space sensitivity (03c compact-CASPT2 tier; auto-detected) -------
def _active_space_decomposition() -> dict | None:
    """Bridge-ladder sensitivities, computed only if the compact-CASPT2
    tier (03c) has been run and parsed. Returns None otherwise so the standard
    calibration run is unaffected.

    compact = CAS(12,12) MS-CASPT2 (03c) -- the bridge tier:
      correlation leg : compact CASPT2 vs compact CASCI (active space fixed)
      active-space leg: compact CASPT2 vs low-cost CASPT2 (method fixed)

    These are conditional, pathwise comparisons from three corners of a
    method-by-active-space design, not interaction-free factorial main effects.
    """
    cdir = B1 / "03c_caspt2_compact"
    mani = cdir / "manifold_caspt2_compact.json"
    desc = cdir / "descriptors_caspt2_compact_dipole_only.json"
    if not (mani.exists() and desc.exists()):
        return None

    out: dict[str, dict] = {}
    so_compact = _energies(mani)
    so_cheap = _energies(B1 / "02_cheap" / "manifold_cheap.json")
    so_exact = _energies(B1 / "03b_exact" / "manifold_exact.json")
    out["SO_45_method_compactCASPT2_vs_exactCASCI"] = {
        "pearson": pearson(so_compact, so_exact), "n": int(len(so_compact)),
        "_held_fixed": "active space CAS(12,12)",
    }
    out["SO_45_activespace_compactCASPT2_vs_cheapCASPT2"] = {
        "pearson": pearson(so_compact, so_cheap), "n": int(len(so_compact)),
        "_held_fixed": "method MS-CASPT2",
    }

    de = B1 / "03b_exact" / "descriptors_exact_dipole_only.json"
    dch = B1 / "02_cheap" / "descriptors_cheap_dipole_only.json"
    for key in ("D2", "D3", "D4", "D5"):
        mco = _descriptor_map(desc, key)
        for tag, other in (("method_vs_exactCASCI", de),
                           ("activespace_vs_cheapCASPT2", dch)):
            mo = _descriptor_map(other, key)
            ks = sorted(set(mco) & set(mo))
            if not ks:
                continue
            a, b = np.array([mco[k] for k in ks]), np.array([mo[k] for k in ks])
            lo, hi = spearman_boot_ci(a, b)
            entry = {
                "spearman": spearman(a, b), "n": int(len(a)),
                "boot95_lo": lo, "boot95_hi": hi,
                "compact_range": [float(a.min()), float(a.max())],
                "other_range": [float(b.min()), float(b.max())],
                **_peak_agreement(ks, a, b),
            }
            if a.max() < FLOOR or b.max() < FLOOR:
                entry["_meaningful"] = False
                entry["_note"] = (f"one tier at numerical floor (<{FLOOR:g}); "
                                  "descriptor collapses rather than transfers")
            out[f"{key}_{tag}"] = entry

    # spin-free ordering of the compact tier, if its RASSI .out is present
    rout = cdir / "rassi_caspt2.out"
    if rout.exists():
        try:
            sf_compact = _parse_sf(rout)
            sf = _sf_energies()
            out["SF_15_method_compactCASPT2_vs_exactCASCI"] = {
                "spearman": spearman(sf_compact, sf["exact_Eh"]), "n": 15}
            out["SF_15_activespace_compactCASPT2_vs_cheapCASPT2"] = {
                "spearman": spearman(sf_compact, sf["cheap_Eh"]), "n": 15}
        except (KeyError, ValueError):
            pass
    return out


# --- main --------------------------------------------------------------------
def main() -> None:
    res = {"_resampling_note": "boot95_lo and boot95_hi are paired grid-point resampling percentiles, not validated confidence intervals for correlated deterministic sweeps. They are not used as uncertainty estimates in the associated study."}

    so_c = _energies(B1 / "02_cheap" / "manifold_cheap.json")
    so_e = _energies(B1 / "03b_exact" / "manifold_exact.json")
    # Both SO spectra are exported in ascending energy order, so their Spearman
    # rho is structurally 1.0 and is NOT an independent test of state ordering.
    # Pearson r summarizes linear correspondence between the two energy-ordered
    # envelopes; it does not measure absolute spectral width or state identity.
    # Spin-free Spearman below uses states matched by multiplicity block.
    res["SO_45_energies"] = {
        "spearman": spearman(so_c, so_e), "pearson": pearson(so_c, so_e),
        "n": int(len(so_c)),
        "both_energy_sorted": bool(np.all(np.diff(so_c) >= 0) and np.all(np.diff(so_e) >= 0)),
        "scale_ratio_exact_over_cheap": float((so_e.max() - so_e.min()) / (so_c.max() - so_c.min())),
        "_note": "spearman is trivial here (both spectra energy-sorted); pearson measures linear envelope correspondence only",
    }

    dc = B1 / "02_cheap" / "descriptors_cheap_dipole_only.json"
    de = B1 / "03b_exact" / "descriptors_exact_dipole_only.json"
    for key in ("D2", "D3", "D4", "D5"):
        ma, mb = _descriptor_map(dc, key), _descriptor_map(de, key)
        ks = sorted(set(ma) & set(mb))
        if not ks:
            continue
        a, b = np.array([ma[k] for k in ks]), np.array([mb[k] for k in ks])
        lo, hi = spearman_boot_ci(a, b)
        entry = {
            "spearman": spearman(a, b), "n": int(len(a)),
            "boot95_lo": lo, "boot95_hi": hi,
            "cheap_range": [float(a.min()), float(a.max())],
            "exact_range": [float(b.min()), float(b.max())],
            **_peak_agreement(ks, a, b),
        }
        # Retain the auxiliary reporting flag without interpreting small magnitude
        # as physical signal loss or numerical noise.
        if a.max() < FLOOR or b.max() < FLOOR:
            entry["_meaningful"] = False
            entry["_note"] = (f"one tier is below the auxiliary reporting threshold {FLOOR:g}; "
                              "the _meaningful flag is not a physical or numerical-validity criterion")
        res[f"{key}_omega_sweep"] = entry
    res["D1_value"] = {
        "cheap": _descriptor_map(dc, "D1")[min(_descriptor_map(dc, "D1"))],
        "exact": _descriptor_map(de, "D1")[min(_descriptor_map(de, "D1"))],
    }

    sf = _sf_energies()
    res["SF_15_cheap_vs_exact"] = {
        "spearman": spearman(sf["cheap_Eh"], sf["exact_Eh"]), "n": 15,
    }
    res["SF_15_cheap_vs_TDDFT"] = {
        "spearman": spearman(sf["cheap_Eh"], sf["tddft_eV"]), "n": 15,
    }

    # Bridge-ladder analysis (auto-runs once the 03c compact-CASPT2 tier
    # exists). The compact CAS(12,12) CASPT2 manifold is the bridge:
    #   correlation leg = compact CASPT2 vs compact CASCI (active space fixed)
    #   active-space leg = compact CASPT2 vs low-cost CASPT2 (method fixed)
    # Existing JSON key names are retained for compatibility with plotting scripts.
    decomp = _active_space_decomposition()
    if decomp is not None:
        res["active_space_decomposition"] = decomp

    (B1 / "tier_correlations.json").write_text(json.dumps(res, indent=2))

    print(f"{'quantity':<26}{'rho':>9}{'pearson':>10}{'n':>5}")
    print("-" * 50)
    for k, v in res.items():
        if isinstance(v, dict) and "spearman" in v:
            r = v.get("pearson")
            print(f"{k:<26}{v['spearman']:>9.4f}{('' if r is None else f'{r:>10.4f}')}{v['n']:>5}")
    print(f"\nD1 (omega-independent):  low-cost = {res['D1_value']['cheap']:.5f}   "
          f"compact CASCI = {res['D1_value']['exact']:.5f}")
    print("\nwrote calculations/B1/sf_energies.json and calculations/B1/tier_correlations.json")


if __name__ == "__main__":
    main()
