"""Endpoint, connectivity, phase, spin-mixing, and pathway descriptors."""

from __future__ import annotations

import numpy as np

from sccg.graph import SCCG

EPS = 1e-12
# A mediator counts as spin-mixed for D4 when more than this
# fraction of its spin-free parentage sits off its dominant multiplicity block.
# Spin-orbit coupling mixes every state a little, so a small floor keeps D4 a
# filter rather than a tautology; the choice is reported and its sensitivity
# tabulated in the SI.
SPIN_MIX_TOL = 1e-3
PATHWAY_TOL = 1e-6


def normalized_endpoint_profiles(g: SCCG) -> tuple[np.ndarray, np.ndarray] | None:
    """Return normalized projection-weighted endpoint edge profiles ``q`` and ``r``.

    ``None`` denotes the explicit zero-throughput boundary at which at least
    one profile cannot be normalized.
    """
    pL, pR = g.endpoint_indices()
    W_Li = g.weights @ pL
    W_iR = g.weights @ pR
    total_L = float(W_Li.sum())
    total_R = float(W_iR.sum())
    if total_L == 0.0 or total_R == 0.0:
        return None
    return W_Li / total_L, W_iR / total_R


def D2_decomposition(g: SCCG) -> dict[str, float]:
    """Exact alignment/concentration decomposition of endpoint coincidence.

    ``D2 = cosine_alignment * concentration``, where
    ``concentration = ||q||_2 ||r||_2 = 1/sqrt(N_q N_r)`` and ``N_q`` and
    ``N_r`` are inverse participation ratios of the two endpoint profiles.
    """
    profiles = normalized_endpoint_profiles(g)
    if profiles is None:
        return {"D2": 0.0, "cosine_alignment": 0.0, "concentration": 0.0,
                "N_q": 0.0, "N_r": 0.0}
    q, r = profiles
    norm_q = float(np.linalg.norm(q))
    norm_r = float(np.linalg.norm(r))
    concentration = norm_q * norm_r
    dot = float(q @ r)
    cosine = 0.0 if concentration == 0.0 else dot / concentration
    return {
        "D2": dot,
        "cosine_alignment": cosine,
        "concentration": concentration,
        "N_q": 1.0 / (norm_q ** 2),
        "N_r": 1.0 / (norm_r ** 2),
    }


def D1_endpoint_localization(g: SCCG) -> float:
    """Endpoint localization score.

    Penalizes both poor localization of the endpoint reference states
    and their mutual contamination on each other's reference state.

    L and R denote spin-free root multiplets, not graph nodes.
    This deliberately rigid diagnostic uses energy-ordered indices 0 and 1 as
    the L and R representatives; it does not reselect the dominant endpoint
    states at each tier.
    """
    pL, pR = g.endpoint_indices()
    pL_on_L, pR_on_R = pL[0], pR[1]
    pL_on_R, pR_on_L = pL[1], pR[0]
    contamination = max(0.0, 1.0 - pL_on_R - pR_on_L)
    return 0.5 * (pL_on_L + pR_on_R) * contamination


def D2_mediator_bridge(g: SCCG) -> float:
    """Normalized endpoint-profile coincidence.

    Phase-blind two-step connectivity L -> i -> R, normalized against
    total reachable weight from each endpoint. Because L and R are root
    multiplets rather than graph nodes, the effective endpoint couplings
    are projection-weighted contractions: W_Li = sum_j p_L(j) W_ji and
    W_iR = sum_j W_ij p_R(j).
    """
    return D2_decomposition(g)["D2"]


def D3_phase_alignment(g: SCCG) -> float:
    """Phase-alignment score — coherent vs incoherent two-step amplitude.

    D3 = |Σ_i A_i g_i| / (Σ_i |A_i| g_i + EPS).

    Endpoint convention: |L⟩,|R⟩ are represented by their dominant spin-orbit
    eigenstate (the state carrying the most endpoint weight).  This differs from
    D1's fixed energy-order convention; the D2/D5 distribution treatment is
    phase-blind and cannot carry the dipole phase D3 needs. The complex two-step
    amplitude through mediator i is the Cartesian dot product
    A_i = Σ_c μ_c[i_L, i] μ_c[i, i_R], the phase-resolved analogue of D2's
    Σ_c |μ_c|² trace. The one-photon window gates the L→i leg on |E_i − E_{i_L}|,
    consistent with Eq. (edge weight). D3→1 when the pathways add in phase,
    D3→0 under destructive interference. Requires phase-resolved dipoles.
    """
    muC = g.manifold.dipoles_complex
    if muC is None:
        raise NotImplementedError("D3 requires phase-resolved dipoles (dipoles_complex)")
    pL, pR = g.endpoint_indices()
    iL, iR = int(np.argmax(pL)), int(np.argmax(pR))
    E = g.manifold.energies
    # complex two-step amplitude per mediator (Cartesian dot of the two legs)
    A = np.einsum("ic,ic->i", muC[iL], muC[:, iR])       # (N,) complex
    dE = np.abs(E - E[iL])
    gwin = np.exp(-((dE - g.omega) ** 2) / (2 * g.sigma ** 2))
    keep = np.ones(len(E), dtype=bool)
    keep[[iL, iR]] = False                                # exclude the endpoints
    coh = np.abs(np.sum(A[keep] * gwin[keep]))
    inc = np.sum(np.abs(A[keep]) * gwin[keep])
    return float(coh / (inc + EPS))


def D4_spin_mixing(g: SCCG, spin_mix_tol: float = SPIN_MIX_TOL) -> float:
    """Spin-mixing accessibility — fraction of D2's connectivity via spin-mixed mediators.

    D4 = ( Σ_{i: spin-mixed} W_{Li} W_{iR} ) / ( Σ_i W_{Li} W_{iR} )  ∈ [0,1].

    Defined on the same dipole graph as D2 (SI form, no incommensurate SOC channel
    and no fit constant β): it is the share of the two-step connectivity that runs
    through mediators carrying spin-orbit-induced spin mixing. It is a mechanistic
    diagnostic, not a validated predictor of spin-flip or relocation fidelity.
    Requires per-state spin-mixing tags.
    """
    sm = g.manifold.spin_mixing
    if sm is None:
        raise NotImplementedError("D4 requires per-state spin_mixing tags")
    pL, pR = g.endpoint_indices()
    W = g.weights
    W_Li = W @ pL
    W_iR = W @ pR
    p = W_Li * W_iR
    mixed = sm > spin_mix_tol
    total = float(p.sum())
    return 0.0 if total == 0.0 else float(p[mixed].sum() / total)


def D5_pathway_evenness(g: SCCG, pathway_tol: float = PATHWAY_TOL) -> float:
    """Pathway evenness — normalized Renyi-2 entropy of pathway weights.

    ``pathway_tol`` sets the nonnegligible-weight cutoff used to count N_eff.
    The production value is 1e-6; the SI reports a cutoff-sensitivity scan.

    Because H2 is normalized by log(N_eff), this descriptor measures evenness
    on the retained support; it does not by itself measure the number of active
    pathways. Use the participation ratio or explicit weights for that purpose.
    """
    pL, pR = g.endpoint_indices()
    W = g.weights
    W_Li = W @ pL
    W_iR = W @ pR
    p = W_Li * W_iR
    p_total = p.sum()
    if p_total < EPS:
        return 0.0
    p = p / p_total
    keep = p > pathway_tol
    n_eff = float(np.sum(keep))
    if n_eff <= 1:
        return 0.0
    # Compute entropy on the retained support so H2 <= log(N_eff) exactly.
    p_active = p[keep]
    p_active = p_active / p_active.sum()
    H2 = -np.log(np.sum(p_active ** 2) + EPS)
    return float(H2 / np.log(n_eff))


# Compatibility alias for pathway-evenness terminology.
D5_pathway_redundancy = D5_pathway_evenness


def all_descriptors(g: SCCG) -> dict[str, float | None]:
    """Compute the five descriptors used in the archived analysis."""
    out: dict[str, float | None] = {}
    for name, fn in [
        ("D1", D1_endpoint_localization),
        ("D2", D2_mediator_bridge),
        ("D3", D3_phase_alignment),
        ("D4", D4_spin_mixing),
        ("D5", D5_pathway_evenness),
    ]:
        try:
            out[name] = fn(g)
        except NotImplementedError:
            out[name] = None
    return out


def _main() -> None:
    import argparse
    import json
    from pathlib import Path
    from sccg.io import load_manifold

    ap = argparse.ArgumentParser(description="Compute SCCG descriptors D1–D5.")
    ap.add_argument("manifold", type=Path, help="manifold JSON (sccg.io schema)")
    ap.add_argument("--out", type=Path, required=True, help="output descriptors JSON")
    ap.add_argument("--omega", type=float, default=None,
                    help="single photon energy hbar*omega (eV); if omitted, sweep")
    ap.add_argument("--omega-grid", type=str,
                    default="0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65",
                    help="photon-energy grid in eV; defaults to the reported 14-point grid")
    ap.add_argument("--sigma", type=float, default=0.10, help="energy-window sigma (eV)")
    ap.add_argument("--alpha", type=float, default=1.0, help="dipole-channel weight")
    ap.add_argument("--beta", type=float, choices=[0.0], default=0.0,
                    help="dipole-only benchmark: must be zero")
    a = ap.parse_args()

    m = load_manifold(a.manifold)
    omegas = [a.omega] if a.omega is not None else [float(x) for x in a.omega_grid.split(",")]
    sweep = []
    for w in omegas:
        g = SCCG(manifold=m, omega=w, sigma=a.sigma, alpha=a.alpha, beta=a.beta)
        sweep.append({"omega_eV": w, **all_descriptors(g)})

    out = {
        "manifold": str(a.manifold),
        "label": m.label,
        "n_states": int(m.n_states),
        "params": {"sigma": a.sigma, "alpha": a.alpha, "beta": a.beta},
        "sweep": sweep,
    }
    a.out.write_text(json.dumps(out, indent=2))
    # console summary
    keys = ["D1", "D2", "D3", "D4", "D5"]
    print(f"manifold = {m.label}  N = {m.n_states}")
    print("ω(eV)   " + "  ".join(f"{k:>7s}" for k in keys))
    for row in sweep:
        line = f"{row['omega_eV']:>5.3f}   "
        for k in keys:
            v = row[k]
            line += f"{'  -    ' if v is None else f'{v:>7.4f}'}  "
        print(line)


if __name__ == "__main__":
    _main()
