"""
Spin-Control Connectivity Graph object.

A manifold of correlated SOC-coupled excited states is represented as a graph
whose nodes are eigenstates and whose edges are resonance-windowed dipole
strengths. Spin-orbit coupling enters through the eigenstates and dipoles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Manifold:
    """Container for the inputs to an SCCG.

    Attributes
    ----------
    energies : (N,) array of state energies (eV).
    dipoles : (N, N, 3) array of transition dipole magnitudes (a.u.).
    soc : (N, N) array of complex SOC matrix elements (cm^-1) or None.
    endpoint_overlaps : (N, 2) array of <i|P_L|i> and <i|P_R|i>
                        (dimensionless whole-multiplet projector weights).
    label : human-readable identifier of the manifold.
    dipoles_complex : (N, N, 3) complex transition dipoles (a.u.) or None.
                      Phase-resolved; only needed by D3. `dipoles` (magnitude)
                      is used by every phase-blind descriptor.
    spin_mixing : (N,) array in [0,1); per-state spin-multiplicity impurity
                  (0 = pure spin-free parentage). Needed by D4. None if absent.
    s2 : (N,) array of <S^2> per state, or None. Diagnostic companion to
         spin_mixing.
    """

    energies: np.ndarray
    dipoles: np.ndarray
    soc: np.ndarray | None
    endpoint_overlaps: np.ndarray
    label: str = ""
    dipoles_complex: np.ndarray | None = None
    spin_mixing: np.ndarray | None = None
    s2: np.ndarray | None = None

    @property
    def n_states(self) -> int:
        return len(self.energies)


@dataclass
class SCCG:
    """Spin-Control Connectivity Graph built from a Manifold and a driving field."""

    manifold: Manifold
    omega: float                       # photon energy hbar*omega (eV)
    sigma: float = 0.10                # energy-window kernel width (eV)
    sigma_soc: float = 0.05            # retained API parameter; unused
    alpha: float = 1.0                 # dipole-channel weight
    beta: float = 0.0                  # must be zero for this dipole-only graph
    gamma: float = 0.0                 # unsupported regularizer; must be zero

    weights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.beta != 0.0 or self.gamma != 0.0:
            raise ValueError("This benchmark defines dipole-only edges: beta and gamma must be zero.")
        self.weights = self._build_weights()

    def _build_weights(self) -> np.ndarray:
        """Compute the symmetric edge-weight matrix W_ij.

        Dipole channel: edge (i, j) is on-resonance when |E_i - E_j| approx omega
        (transition gap matched to photon energy). SOC is not added as a
        second edge channel; its archived matrix uses different physical units.
        """
        E = self.manifold.energies
        mu = self.manifold.dipoles

        # |mu_ij|^2 summed over Cartesian components
        mu2 = np.einsum("ija,ija->ij", mu.conj(), mu).real

        # One-photon resonance kernel
        dE = np.abs(E[:, None] - E[None, :])
        g_drive = np.exp(-((dE - self.omega) ** 2) / (2 * self.sigma ** 2))

        W = self.alpha * mu2 * g_drive

        np.fill_diagonal(W, 0.0)
        return W

    def endpoint_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return arrays of overlap weights p_L, p_R for every node."""
        return self.manifold.endpoint_overlaps[:, 0], self.manifold.endpoint_overlaps[:, 1]
