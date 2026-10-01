"""Load the small RASSI arrays needed for manifold reconstruction.

The full OpenMolcas HDF5 files also contain large AO and density datasets that
are irrelevant to the SCCG root-boundary test.  For long-term
archiving, the required arrays can therefore be stored losslessly in a compact
``.npz`` file.  This module presents the same dictionary interface for either
source.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def _weights_from_coefficients(
    coefficients: np.ndarray, multiplicities: np.ndarray
) -> np.ndarray:
    """Return spin-free-root weights with rows=root and columns=SO state."""
    basis_to_root = np.repeat(np.arange(len(multiplicities)), multiplicities)
    weights = np.zeros((len(multiplicities), coefficients.shape[0]))
    for root in range(len(multiplicities)):
        weights[root] = np.sum(
            np.abs(coefficients[:, basis_to_root == root]) ** 2, axis=1
        )
    return weights


def load_rassi_reconstruction(path: Path) -> dict[str, np.ndarray]:
    """Load reconstruction arrays from a full OpenMolcas HDF5 or compact NPZ."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as archive:
            return {name: archive[name] for name in archive.files}

    with h5py.File(path, "r") as handle:
        multiplicities = np.asarray(handle.attrs["STATE_SPINMULT"], dtype=int)
        local_roots = np.asarray(handle.attrs["STATE_LROOT"], dtype=int)
        coefficients = np.asarray(handle["SOS_COEFFICIENTS_REAL"]) + 1j * np.asarray(
            handle["SOS_COEFFICIENTS_IMAG"]
        )
        return {
            "multiplicities": multiplicities,
            "local_roots": local_roots,
            "h_so": np.asarray(handle["HSO_MATRIX_REAL"]) + 1j * np.asarray(
                handle["HSO_MATRIX_IMAG"]
            ),
            "sf_dipoles": np.asarray(handle["SFS_EDIPMOM"]),
            "so_coefficients": coefficients,
            "so_energies": np.asarray(handle["SOS_ENERGIES"]),
            "sf_to_so_weights": _weights_from_coefficients(
                coefficients, multiplicities
            ),
        }


def preferred_reconstruction_source(h5_path: Path, compact_path: Path) -> Path:
    """Prefer the original HDF5, with the compact matrix export as fallback."""
    return h5_path if h5_path.exists() else compact_path
