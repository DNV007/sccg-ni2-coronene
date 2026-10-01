"""
Synthetic manifolds for descriptor testing without a real QC calculation.

Convention: state 0 = |L>, state 1 = |R>, near-degenerate at E ~ 0
(spin-localized endpoints). Mediator states at E ~ omega.

Used by the test suite and by exploratory notebooks during pipeline build-out.
"""

from __future__ import annotations

import numpy as np

from sccg.graph import Manifold


def sparse_bridge(n_states: int = 10, omega_center: float = 1.5,
                  seed: int = 0) -> Manifold:
    """Toy manifold with one dominant mediator bridging |L> and |R>.

    Endpoints |L>, |R> sit at E approx 0 (near-degenerate). One bridge
    mediator at E = omega_center has strong dipole couplings to both.
    Other states have only background-level dipoles.
    """
    rng = np.random.default_rng(seed)
    E = np.zeros(n_states)
    E[0] = 0.0
    E[1] = 0.0
    E[2:] = omega_center + rng.normal(0, 0.4, size=n_states - 2)
    bridge = 4
    E[bridge] = omega_center

    mu = rng.normal(0, 0.02, size=(n_states, n_states, 3))
    mu[0, bridge] = mu[bridge, 0] = np.array([1.0, 0, 0])
    mu[1, bridge] = mu[bridge, 1] = np.array([0.9, 0, 0])

    overlaps = np.zeros((n_states, 2))
    overlaps[0, 0] = 1.0
    overlaps[1, 1] = 1.0
    return Manifold(
        energies=E, dipoles=mu, soc=None,
        endpoint_overlaps=overlaps, label="synthetic_sparse_bridge",
    )


def dense_redundant(n_states: int = 30, omega_center: float = 2.5,
                    seed: int = 0) -> Manifold:
    """Toy manifold with many comparable mediator bridges (high D5).

    Endpoints |L>, |R> at E approx 0. Twenty bridge mediators packed
    tightly around E = omega_center, all with comparable dipole couplings.
    """
    rng = np.random.default_rng(seed)
    n_bridges = 20
    bridge_states = list(range(2, 2 + n_bridges))

    E = np.zeros(n_states)
    E[0] = 0.0
    E[1] = 0.0
    E[2:2 + n_bridges] = omega_center + np.linspace(-0.05, 0.05, n_bridges)
    E[2 + n_bridges:] = omega_center + 1.0 + np.linspace(0, 1.5,
                                                         n_states - 2 - n_bridges)

    mu = rng.normal(0, 0.02, size=(n_states, n_states, 3))
    for k in bridge_states:
        amp = 0.6 + rng.normal(0, 0.02)
        mu[0, k] = mu[k, 0] = np.array([amp, 0, 0])
        mu[1, k] = mu[k, 1] = np.array([amp * 0.97, 0, 0])

    overlaps = np.zeros((n_states, 2))
    overlaps[0, 0] = 1.0
    overlaps[1, 1] = 1.0
    return Manifold(
        energies=E, dipoles=mu, soc=None,
        endpoint_overlaps=overlaps, label="synthetic_dense_redundant",
    )
