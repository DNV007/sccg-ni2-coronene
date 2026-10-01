"""Parse OpenMolcas RASSI .rassi.h5 (+ rasscf.h5) into manifold JSON.

Three endpoint conventions are supported:
  - ni-coronene : Mulliken population on Ni atoms vs the rest, projected via
                  the spin-free → SO coefficients.  Requires the per-multiplicity
                  rasscf.rasscf.h5 files (Q/T/S) for the per-root density
                  matrices.
  - energetic   : whole-multiplet projector weight for the lowest-energy
                  spin-free root of the highest / lowest multiplicity (here:
                  lowest quintet vs lowest singlet).
                  No extra files needed.
  - uniform     : synthetic 0.5/0.5 weights for parser diagnostics, not physical
                  endpoint projectors. Not used for the reported benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_CM = 219474.6313708
NI_Z = 28


def _ao_overlap(f: h5py.File) -> np.ndarray:
    nbas = int(f.attrs["NBAS"][0])
    return f["AO_OVERLAP_MATRIX"][:].reshape(nbas, nbas)


def _ao_atom_index(f: h5py.File) -> np.ndarray:
    """0-indexed atom number for each AO."""
    return f["BASIS_FUNCTION_IDS"][:, 0] - 1


def _coefficients_to_sf_weights(coefficients: np.ndarray,
                                mults: np.ndarray) -> np.ndarray:
    """Collapse OpenMolcas SO eigenvectors onto spin-free roots.

    OpenMolcas stores ``SOS_COEFFICIENTS`` as ``C[so, basis]``: each row is an
    SO eigenstate and the columns are the spin-projection components of the
    spin-free roots. This orientation is independently fixed by
    ``C @ H_SO @ C.conj().T`` being diagonal. The returned array is
    ``w[sf, so]``.
    """
    abs2 = np.abs(coefficients) ** 2  # (so, spin-projection basis)
    bidx_to_sf = np.repeat(np.arange(len(mults)), mults)
    n_sf = len(mults)
    w = np.zeros((n_sf, abs2.shape[0]))
    for sf in range(n_sf):
        w[sf] = abs2[:, bidx_to_sf == sf].sum(axis=1)
    return w


def _sf_to_so_weights(rassi: h5py.File) -> np.ndarray:
    """Return w[sf, so] = whole-multiplet weight of SF root sf in SO state so."""
    coefficients = (rassi["SOS_COEFFICIENTS_REAL"][:]
                    + 1j * rassi["SOS_COEFFICIENTS_IMAG"][:])
    mults = np.array(rassi.attrs["STATE_SPINMULT"], dtype=int)
    return _coefficients_to_sf_weights(coefficients, mults)


def _spin_composition(rassi: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    """Per-SO-state spin descriptors, used by D4 (spin-mixing accessibility).

    From the spin-free -> spin-orbit weight matrix w[sf, so] (each SO column
    normalized to 1 over the spin-free basis) and the spin-free multiplicities,
    return two length-N_SO arrays:

      s2[so]           : <S^2> = sum_sf w[sf,so] * S_sf(S_sf+1), S_sf=(mult-1)/2.
      spin_mixing[so]  : 1 - max_m (weight on multiplicity m).  Zero for an SO
                         state of pure spin-free parentage (single multiplicity),
                         positive when spin-orbit coupling has mixed multiplicities
                         into the node. D4 thresholds this multiplicity impurity,
                         not the change in the expectation value of S^2.
    """
    w = _sf_to_so_weights(rassi)                       # (n_sf, n_so)
    mults = np.array(rassi.attrs["STATE_SPINMULT"], dtype=int)
    S = (mults - 1) / 2.0
    col = w.sum(axis=0)                                 # ~1 per SO state
    col = np.where(col > 0, col, 1.0)
    s2 = (w * (S * (S + 1.0))[:, None]).sum(axis=0) / col
    # collapse spin-free weight onto distinct multiplicity blocks, then purity
    uniq = np.unique(mults)
    mult_weight = np.array([w[mults == m].sum(axis=0) for m in uniq])  # (n_mult, n_so)
    spin_mixing = 1.0 - mult_weight.max(axis=0) / col
    return s2, spin_mixing


def _ao_density_per_root(rasscf_path: Path, active_only: bool) -> np.ndarray:
    """Return per-root AO-basis state densities (n_root, nbas, nbas).

    If active_only, the inactive contribution (which is root-independent and
    drowns out per-state variation) is omitted, so the partitioning reflects
    only the 26 active electrons in the CAS.
    """
    with h5py.File(rasscf_path, "r") as f:
        D_act = f["DENSITY_MATRIX"][:]
        nbas = int(f.attrs["NBAS"][0])
        # MO_VECTORS is column-major flat: reshape→transpose gives C[ao, mo]
        C = f["MO_VECTORS"][:].reshape(nbas, nbas).T
        types = f["MO_TYPEINDICES"][:]
    inact = np.where(types == b"I")[0]
    act = np.where(types == b"2")[0]
    n_root = D_act.shape[0]
    D_inact = 0.0 if active_only else 2.0 * C[:, inact] @ C[:, inact].T
    out = np.empty((n_root, nbas, nbas))
    for r in range(n_root):
        out[r] = D_inact + C[:, act] @ D_act[r] @ C[:, act].T
    return out


def _ni_fractions_per_sf(
    rassi: h5py.File, rasscf_paths: dict[int, Path], active_only: bool
) -> np.ndarray:
    """For each spin-free state, return Ni-Mulliken-population fraction."""
    S = _ao_overlap(rassi)
    atom_idx = _ao_atom_index(rassi)
    atnums = rassi["CENTER_ATNUMS"][:]
    ni_atoms = np.where(atnums == NI_Z)[0]
    on_ni = np.isin(atom_idx, ni_atoms)
    mults = np.array(rassi.attrs["STATE_SPINMULT"], dtype=int)

    fracs = np.empty(len(mults))
    for mult in (5, 3, 1):
        D = _ao_density_per_root(rasscf_paths[mult], active_only=active_only)
        for k, sf_i in enumerate(np.where(mults == mult)[0]):
            DS_diag = (D[k] @ S).diagonal()
            tot = DS_diag.sum()
            ni = DS_diag[on_ni].sum()
            fracs[sf_i] = ni / tot
    return fracs


def parse(
    rassi_path: Path,
    mode: str,
    rasscf_paths: dict[int, Path] | None,
    label: str,
) -> dict:
    with h5py.File(rassi_path, "r") as f:
        E_ha = f["SOS_ENERGIES"][:]
        E_eV = (E_ha - E_ha.min()) * HARTREE_TO_EV
        n_so = E_ha.size

        dipR = f["SOS_EDIPMOM_REAL"][:]   # (3, N, N)
        dipI = f["SOS_EDIPMOM_IMAG"][:]
        dip = np.sqrt(dipR ** 2 + dipI ** 2)
        dip = np.transpose(dip, (1, 2, 0))   # (N, N, 3)
        # Phase-resolved dipoles (needed by D3); magnitude `dip` above is kept
        # unchanged so D2/D5 are byte-for-byte identical.
        dip_re = np.transpose(dipR, (1, 2, 0))   # (N, N, 3)
        dip_im = np.transpose(dipI, (1, 2, 0))

        hsoR = f["HSO_MATRIX_REAL"][:] * HARTREE_TO_CM
        hsoI = f["HSO_MATRIX_IMAG"][:] * HARTREE_TO_CM

        # Per-SO-state spin composition (needed by D4).
        s2, spin_mixing = _spin_composition(f)

        if mode == "uniform":
            overlaps = np.full((n_so, 2), 0.5)
            meta_L = "uniform_0.5"
            meta_R = "uniform_0.5"
        elif mode == "energetic":
            mults = np.array(f.attrs["STATE_SPINMULT"], dtype=int)
            sf_E = f["SFS_ENERGIES"][:]
            q_idx = np.where(mults == 5)[0]
            s_idx = np.where(mults == 1)[0]
            L_sf = q_idx[np.argmin(sf_E[q_idx])]
            R_sf = s_idx[np.argmin(sf_E[s_idx])]
            w = _sf_to_so_weights(f)
            overlaps = np.column_stack([w[L_sf], w[R_sf]])
            meta_L = f"lowest_quintet_SF (sf_idx={L_sf})"
            meta_R = f"lowest_singlet_SF (sf_idx={R_sf})"
        elif mode in ("ni-coronene", "ni-coronene-active"):
            assert rasscf_paths is not None, "ni-coronene[-active] requires --rasscf-q/-t/-s"
            active_only = mode == "ni-coronene-active"
            ni_frac = _ni_fractions_per_sf(f, rasscf_paths, active_only=active_only)
            cor_frac = 1.0 - ni_frac
            w = _sf_to_so_weights(f)
            ni_so = ni_frac @ w
            cor_so = cor_frac @ w
            overlaps = np.column_stack([ni_so, cor_so])
            scope = "active-only" if active_only else "full"
            meta_L = f"Ni_Mulliken_fraction ({scope})"
            meta_R = f"non-Ni_remainder ({scope})"
        else:
            raise ValueError(f"unknown mode: {mode}")

    return {
        "label": f"{label}_{mode}",
        "energies_eV": E_eV.tolist(),
        "dipoles_au": dip.tolist(),
        "dipoles_real_au": dip_re.tolist(),
        "dipoles_imag_au": dip_im.tolist(),
        "soc_cm-1_real": hsoR.tolist(),
        "soc_cm-1_imag": hsoI.tolist(),
        "endpoint_overlaps": overlaps.tolist(),
        "s2_expectation": s2.tolist(),
        "spin_mixing": spin_mixing.tolist(),
        "_meta": {
            "mode": mode,
            "n_states": int(n_so),
            "source_h5": str(rassi_path),
            "endpoint_L": meta_L,
            "endpoint_R": meta_R,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("h5", type=Path, help="RASSI .rassi.h5 file")
    ap.add_argument(
        "--mode",
        choices=["ni-coronene", "ni-coronene-active", "energetic", "uniform"],
        default="ni-coronene-active",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="B1_cheap")
    ap.add_argument("--rasscf-q", type=Path, help="rasscf.rasscf.h5 (quintet)")
    ap.add_argument("--rasscf-t", type=Path, help="rasscf_T.rasscf.h5 (triplet)")
    ap.add_argument("--rasscf-s", type=Path, help="rasscf_S.rasscf.h5 (singlet)")
    a = ap.parse_args()
    rasscf_paths = None
    if a.mode in ("ni-coronene", "ni-coronene-active"):
        if not (a.rasscf_q and a.rasscf_t and a.rasscf_s):
            ap.error(f"--rasscf-q, --rasscf-t, --rasscf-s are required for mode {a.mode}")
        rasscf_paths = {5: a.rasscf_q, 3: a.rasscf_t, 1: a.rasscf_s}
    d = parse(a.h5, a.mode, rasscf_paths, a.label)
    a.out.write_text(json.dumps(d, indent=2))
    m = d["_meta"]
    print(f"Wrote {a.out}: mode={m['mode']}, n_states={m['n_states']}, L={m['endpoint_L']}, R={m['endpoint_R']}")


if __name__ == "__main__":
    main()
