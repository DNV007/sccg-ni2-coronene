#!/usr/bin/env python3
"""Analysis-choice robustness of the single-root deletion effect.

Companion to ``retained_state_audit.py`` (whose matrix loaders it reuses; no
production SCCG modules are imported). For the full five-root manifolds and the
Q5/T5/S5 single-root RASSI deletions it records:

* cross-tier D2 Spearman correlations for photon windows ending at 0.50, 0.65,
  0.80 and 1.00 eV and for Gaussian widths 0.050--0.200 eV;
* the same correlations for the four Q1/Q2 -> S1/S2 endpoint operations;
* absolute endpoint supports S_L=sum_i W_Li, S_R=sum_i W_iR and the shared
  numerator G=sum_i W_Li W_iR=D2*S_L*S_R;
* the share of S_L and S_R carried by mediators of quintet, triplet and singlet
  spin-free parentage (dipoles are spin-conserving, so an endpoint's own
  multiplicity block is its spin-allowed partner set);
* the proposed stability-record quantities for the five-to-four deletion.

As in the audit, orbitals, spin-free solutions, energies and operators are the
archived five-root ones; only the state interaction changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

try:
    from sccg.retained_state_audit import B1, TIERS, build, jsonable, load_inputs, ranked_tv
except ImportError:  # run as a script from analysis/sccg
    from retained_state_audit import B1, TIERS, build, jsonable, load_inputs, ranked_tv

GRID = np.round(np.r_[0.02, np.arange(1, 21) * 0.05], 6)   # 0.02, 0.05 ... 1.00 eV
DEFAULT_MAX, SIGMA0 = 0.65, 0.10
WINDOWS = (0.50, 0.65, 0.80, 1.00)
SIGMAS = (0.050, 0.075, 0.100, 0.125, 0.150, 0.200)
ENDPOINTS = ((1, 1), (1, 2), (2, 1), (2, 2))
SUPPORT_FLOOR = 0.0  # application-defined; no universal value is implied
CASES = ("full", "omit_Q5", "omit_T5", "omit_S5")
PAIRS = (("compact_leg", "anchor", "bridge"),
         ("two_MS_tiers", "bridge", "production"),
         ("end_to_end", "production", "anchor"))


def keep_mask(data, case):
    if case == "full":
        return np.ones(len(data["multiplicities"]), bool)
    if case.startswith("N"):
        return data["local_roots"] <= int(case[1:])
    mult = {"Q": 5, "T": 3, "S": 1}[case[5]]
    return ~((data["multiplicities"] == mult) & (data["local_roots"] == int(case[6])))


def endpoint_columns(data, qa, sb):
    m, r = data["multiplicities"], data["local_roots"]
    return [int(np.flatnonzero((m == 5) & (r == qa))[0]), int(np.flatnonzero((m == 1) & (r == sb))[0])]


def profiles(model, cols, omega, sigma):
    e = model["energy"]
    w = np.sum(np.abs(model["mu"]) ** 2, axis=0) * np.exp(
        -0.5 * ((np.abs(e[:, None] - e[None, :]) - omega) / sigma) ** 2)
    np.fill_diagonal(w, 0.0)
    pl, pr = model["parent"][:, cols[0]], model["parent"][:, cols[1]]
    left, right = w.T @ pl, w @ pr
    s_l, s_r = left.sum(), right.sum()
    q, r = left / s_l, right / s_r
    d2 = float(q @ r)
    return {"D2": d2, "S_L": float(s_l), "S_R": float(s_r), "G": float(left @ right),
            "q": q, "r": r, "p": q * r / d2, "left": left, "right": right}


def curve(model, cols, sigma=SIGMA0, grid=GRID):
    rows = [profiles(model, cols, w, sigma) for w in grid]
    return {k: np.array([row[k] for row in rows]) for k in ("D2", "S_L", "S_R", "G")}, rows


def window_idx(wmax):
    return GRID <= wmax + 1e-9


def correlations(curves, idx):
    return {name: float(spearmanr(curves[a]["D2"][idx], curves[b]["D2"][idx]).statistic)
            for name, a, b in PAIRS}


def lower_boundary_scan(models, data_by_tier):
    """Dense-grid lower-bound control; preserve raw curves for regeneration.

    A zero kernel center is a mathematical sensitivity test, not an optical pulse.
    No upstream electronic-structure calculation is repeated.
    """
    result = {}
    for lower in (0.0, 0.01, 0.02, 0.05):
        grid = np.round(np.arange(lower, DEFAULT_MAX + 0.000001, 0.005), 6)
        cases = {}
        for case in ("full", "omit_Q5"):
            curves = {t: curve(models[t][case], endpoint_columns(data_by_tier[t], 1, 1),
                               sigma=SIGMA0, grid=grid)[0] for t in TIERS}
            cases[case] = {"curves": curves, "correlations": correlations(curves, slice(None))}
        result[f"{lower:.2f}"] = {"grid_eV": grid, "sigma_eV": SIGMA0, "cases": cases}
    return result


def block_shares(data, model, rows):
    """Share of S_L and S_R carried by mediators of each spin-free multiplicity."""
    blocks = {"quintet": 5, "triplet": 3, "singlet": 1}
    out = {}
    for name, m in blocks.items():
        weight = model["parent"][:, data["multiplicities"] == m].sum(axis=1)
        out[name] = {"S_L_share": [float(row["left"] @ weight / row["S_L"]) for row in rows],
                     "S_R_share": [float(row["right"] @ weight / row["S_R"]) for row in rows]}
    return out


def main():
    data_by_tier, models = {}, {}
    for tier, (label, _) in TIERS.items():
        data = load_inputs(label)
        data_by_tier[tier] = data
        models[tier] = {case: build(data, keep_mask(data, case)) for case in CASES + ("N4",)}

    # Consistency with the audit on the declared 14-point grid.
    audit = json.loads((B1 / "retained_state_audit.json").read_text())
    idx0 = window_idx(DEFAULT_MAX)
    for tier in TIERS:
        cols = endpoint_columns(data_by_tier[tier], 1, 1)
        for case, key in (("full", "N5"), ("omit_Q5", "omit_Q5"), ("N4", "N4")):
            c, _ = curve(models[tier][case], cols)
            ref = np.array(audit["tiers"][tier]["cases"][key]["curves"]["D2"])
            assert np.max(np.abs(c["D2"][idx0] - ref)) < 1e-12, (tier, case)

    record = {"scope": __doc__, "grid_eV": GRID, "default_window_max_eV": DEFAULT_MAX,
              "default_sigma_eV": SIGMA0}

    # 1. Window and width robustness for Q1 -> S1.
    window, width = {}, {}
    for case in CASES:
        curves = {}
        for tier in TIERS:
            cols = endpoint_columns(data_by_tier[tier], 1, 1)
            curves[tier], _ = curve(models[tier][case], cols)
        window[case] = {f"{w:.2f}": correlations(curves, window_idx(w)) for w in WINDOWS}
        width[case] = {}
        for s in SIGMAS:
            cs = {t: curve(models[t][case], endpoint_columns(data_by_tier[t], 1, 1), sigma=s)[0]
                  for t in TIERS}
            width[case][f"{s:.3f}"] = correlations(cs, idx0)
    record["window_sensitivity"] = window
    record["width_sensitivity"] = width
    record["lower_boundary_sensitivity"] = lower_boundary_scan(models, data_by_tier)

    # 2. Endpoint operations.
    endpoint = {}
    for qa, sb in ENDPOINTS:
        tag = f"Q{qa}_S{sb}"
        endpoint[tag] = {}
        for case in CASES:
            curves = {t: curve(models[t][case], endpoint_columns(data_by_tier[t], qa, sb))[0]
                      for t in TIERS}
            endpoint[tag][case] = correlations(curves, idx0)
    record["endpoint_sensitivity"] = endpoint

    # 3. Supports, shared numerator, spin-block shares, and monotone trend.
    supports = {}
    for tier in TIERS:
        data = data_by_tier[tier]
        cols = endpoint_columns(data, 1, 1)
        full_c, full_rows = curve(models[tier]["full"], cols)
        entry = {"full": {k: v[idx0] for k, v in full_c.items()},
                 "full_block_shares": block_shares(data, models[tier]["full"],
                                                   [r for r, k in zip(full_rows, idx0) if k]),
                 "spearman_D2_vs_photon_energy": float(
                     spearmanr(full_c["D2"][idx0], GRID[idx0]).statistic)}
        for case in ("omit_Q5", "omit_T5", "omit_S5", "N4"):
            c, _ = curve(models[tier][case], cols)
            entry[case] = {k: v[idx0] for k, v in c.items()}
            entry[case]["S_L_ratio_to_full"] = c["S_L"][idx0] / full_c["S_L"][idx0]
            entry[case]["S_R_ratio_to_full"] = c["S_R"][idx0] / full_c["S_R"][idx0]
            entry[case]["G_ratio_to_full"] = c["G"][idx0] / full_c["G"][idx0]
        sf_energy = np.real(np.diag(data["h_so"]))[np.r_[0, np.cumsum(data["multiplicities"])[:-1]]]
        sf_energy = (sf_energy - sf_energy.min()) * 27.211386245988
        entry["spin_free_energies_eV"] = {
            f"{'QTS'[[5, 3, 1].index(m)]}{r}": float(e)
            for m, r, e in zip(data["multiplicities"], data["local_roots"], sf_energy)}
        supports[tier] = entry
    record["supports"] = supports

    # 4. Stability-record quantities for the five-to-four deletion.
    stability = {}
    for tier in TIERS:
        data = data_by_tier[tier]
        cols = endpoint_columns(data, 1, 1)
        rows5 = [profiles(models[tier]["full"], cols, w, SIGMA0) for w in GRID[idx0]]
        rows4 = [profiles(models[tier]["N4"], cols, w, SIGMA0) for w in GRID[idx0]]
        # Lambda_S with an explicit domain rule: a grid point enters only if the larger of the
        # two compared supports reaches SUPPORT_FLOOR (0 here, i.e. every point; no support
        # vanishes). An exact zero opposite a support above the floor is complete loss (inf).
        # The attainment point and its relative support level are recorded so that a ratio of
        # two Gaussian-tail supports cannot pass unnoticed.
        attain = None
        for side in ("S_L", "S_R"):
            ref = np.array([b[side] for b in rows5])
            new = np.array([a[side] for a in rows4])
            ok = (np.maximum(ref, new) >= SUPPORT_FLOOR) & (np.maximum(ref, new) > 0)
            with np.errstate(divide="ignore"):
                lam = np.where(ok, np.abs(np.log10(new / ref)), -np.inf)
            i = int(np.argmax(lam))
            if attain is None or lam[i] > attain["Lambda_S"]:
                attain = {"Lambda_S": float(lam[i]), "side": side, "omega_eV": float(GRID[idx0][i]),
                          "support_N5": float(ref[i]), "support_N4": float(new[i]),
                          "relative_level_N5": float(ref[i] / ref.max())}
        log_support = attain["Lambda_S"]
        endpoint_tv = []
        for k, dim in ((0, 5), (1, 1)):
            a = models[tier]["full"]["parent"][:, cols[k]] / dim
            b = models[tier]["N4"]["parent"][:, cols[k]] / dim
            endpoint_tv.append(ranked_tv(b, a))
        stability[tier] = {
            "Delta_D": float(max(abs(a["D2"] - b["D2"]) for a, b in zip(rows4, rows5))),
            "T_p": float(max(ranked_tv(a["p"], b["p"]) for a, b in zip(rows4, rows5))),
            "Lambda_S_max_abs_log10_support_ratio": float(log_support),
            "Lambda_S_attainment": attain,
            "support_floor": SUPPORT_FLOOR,
            "T_endpoint_ranked_TV_L_R": endpoint_tv,
            "endpoint_trace_error": float(np.max(np.abs(
                models[tier]["N4"]["parent"][:, cols].sum(axis=0) - [5, 1]))),
        }
    record["five_to_four_stability_record"] = stability

    out = B1 / "retained_state_robustness.json"
    out.write_text(json.dumps(record, indent=2, default=jsonable, allow_nan=False) + "\n")
    print("wrote", out)
    for case in CASES:
        print(case, "window", {w: round(v["two_MS_tiers"], 3) for w, v in window[case].items()})
        print(case, "sigma ", {s: round(v["two_MS_tiers"], 3) for s, v in width[case].items()})
    for tag, v in endpoint.items():
        print(tag, {c: round(x["two_MS_tiers"], 3) for c, x in v.items()})
    for tier, v in stability.items():
        print(tier, {k: (np.round(x, 4).tolist() if isinstance(x, list)
                         else round(x, 4) if isinstance(x, (int, float)) else x)
                     for k, x in v.items()})
    for tier, v in supports.items():
        print(tier, "rho(D2,omega)=%.3f" % v["spearman_D2_vs_photon_energy"],
              "Q5 min S_L ratio %.2e" % v["omit_Q5"]["S_L_ratio_to_full"].min(),
              "G ratio @0.65 %.2e" % v["omit_Q5"]["G_ratio_to_full"][-1])
    return record


if __name__ == "__main__":
    main()
