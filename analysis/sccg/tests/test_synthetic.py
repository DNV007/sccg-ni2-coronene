"""Descriptor tests on synthetic manifolds."""

import numpy as np
import pytest

from sccg.descriptors import (
    D1_endpoint_localization,
    D2_decomposition,
    D2_mediator_bridge,
    D3_phase_alignment,
    D4_spin_mixing,
    D5_pathway_evenness,
    all_descriptors,
)
from sccg.graph import SCCG, Manifold
from sccg.parse_rassi import _coefficients_to_sf_weights
from sccg.synthetic import dense_redundant, sparse_bridge


def _two_mediator_manifold(A2, A3, spin_mixing, omega=1.5):
    """Minimal 4-state manifold: endpoints 0,1 at E=0; mediators 2,3 at E=omega.

    Each mediator k carries a single-Cartesian complex two-step amplitude
    A_k = muC[0,k]*muC[k,1] (set via the L-leg, R-leg = 1). Magnitudes drive the
    phase-blind weight graph; `spin_mixing` tags the mediators for D4.
    """
    E = np.array([0.0, 0.0, omega, omega])
    muC = np.zeros((4, 4, 3), dtype=complex)
    for k, Ak in zip((2, 3), (A2, A3)):
        muC[0, k, 0] = Ak                     # L-leg carries the amplitude/phase
        muC[k, 0, 0] = np.conj(Ak)            # Hermitian dipole operator
        muC[k, 1, 0] = muC[1, k, 0] = 1.0     # R-leg = 1
    overlaps = np.zeros((4, 2))
    overlaps[0, 0] = 1.0
    overlaps[1, 1] = 1.0
    return Manifold(
        energies=E, dipoles=np.abs(muC), soc=None, endpoint_overlaps=overlaps,
        label="synthetic_two_mediator", dipoles_complex=muC,
        spin_mixing=np.asarray(spin_mixing, float),
    )


def test_d3_in_phase_is_one():
    """Two mediators with equal real amplitudes add constructively: D3 = 1."""
    m = _two_mediator_manifold(A2=1.0, A3=1.0, spin_mixing=[0, 0, 0, 0])
    g = SCCG(manifold=m, omega=1.5)
    assert D3_phase_alignment(g) == pytest.approx(1.0, abs=1e-9)


def test_d3_destructive_is_zero():
    """Equal-and-opposite amplitudes cancel coherently: D3 -> 0."""
    m = _two_mediator_manifold(A2=1.0, A3=-1.0, spin_mixing=[0, 0, 0, 0])
    g = SCCG(manifold=m, omega=1.5)
    assert D3_phase_alignment(g) == pytest.approx(0.0, abs=1e-9)


def test_d3_quarter_turn_partial():
    """A 90-degree phase offset gives partial coherence |1+i|/2 = 1/sqrt(2)."""
    m = _two_mediator_manifold(A2=1.0, A3=1.0j, spin_mixing=[0, 0, 0, 0])
    g = SCCG(manifold=m, omega=1.5)
    assert D3_phase_alignment(g) == pytest.approx(1 / np.sqrt(2), abs=1e-9)


def test_d4_mixed_fraction():
    """Equal-weight mediators, one spin-mixed and one pure: D4 = 0.5."""
    m = _two_mediator_manifold(A2=1.0, A3=1.0, spin_mixing=[0, 0, 0.5, 0.0])
    g = SCCG(manifold=m, omega=1.5)
    assert D4_spin_mixing(g) == pytest.approx(0.5, abs=1e-9)


def test_d4_all_pure_is_zero_all_mixed_is_one():
    m_pure = _two_mediator_manifold(A2=1.0, A3=1.0, spin_mixing=[0, 0, 0, 0])
    m_mixed = _two_mediator_manifold(A2=1.0, A3=1.0, spin_mixing=[0, 0, 0.3, 0.3])
    assert D4_spin_mixing(SCCG(manifold=m_pure, omega=1.5)) == pytest.approx(0.0, abs=1e-9)
    assert D4_spin_mixing(SCCG(manifold=m_mixed, omega=1.5)) == pytest.approx(1.0, abs=1e-9)


def test_d4_zero_pathway_boundary_is_zero():
    m = _two_mediator_manifold(A2=0.0, A3=0.0, spin_mixing=[0, 0, 0.3, 0.3])
    assert D4_spin_mixing(SCCG(manifold=m, omega=1.5)) == 0.0


def test_openmolcas_so_coefficients_use_rows_as_eigenstates():
    """Projector weights collapse spin-basis columns, not SO-state rows."""
    # Two SF roots: a doublet (two basis components) and a singlet (one).
    # Rows are SO eigenstates. A permutation makes a transpose error obvious.
    coefficients = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=complex)
    weights = _coefficients_to_sf_weights(coefficients, np.array([2, 1]))
    np.testing.assert_allclose(weights[0], [0.0, 1.0, 1.0])
    np.testing.assert_allclose(weights[1], [1.0, 0.0, 0.0])


def test_graph_excludes_self_edges():
    """The transition graph must not include diagonal dipole moments as edges."""
    m = _two_mediator_manifold(A2=1.0, A3=1.0, spin_mixing=[0, 0, 0, 0])
    # Make the diagonal deliberately nonzero so this test exercises the guard.
    m.dipoles[np.arange(4), np.arange(4), 0] = 7.0
    g = SCCG(manifold=m, omega=0.0)
    assert np.allclose(np.diag(g.weights), 0.0)


def test_default_graph_is_dipole_only_even_with_archived_soc():
    m = _two_mediator_manifold(1.0, 0.25, [0, 0, 0, 0])
    baseline = SCCG(m, 1.5).weights.copy()
    m.soc = np.full((4, 4), 1000.0)
    np.testing.assert_array_equal(SCCG(m, 1.5).weights, baseline)
    with pytest.raises(ValueError, match="dipole-only"):
        SCCG(m, 1.5, beta=1.0)


def test_constant_sweep_correlation_is_undefined():
    from sccg.mechanistic_sensitivity import rho
    assert rho([1.0, 1.0, 1.0], [0.1, 0.2, 0.3]) is None


def _d5_from_pathway_weights(p):
    """D5 evaluated directly on a pathway-weight vector p_i = W_Li * W_iR.

    Mirrors D5_pathway_evenness without needing a full manifold, so the
    limiting cases quoted in the SI (single dominant -> 0, uniform -> 1) are
    pinned by the test suite. D5 = H2(p)/log(n_eff), normalized Renyi-2 entropy.
    """
    p = np.asarray(p, float)
    p = p / p.sum()
    H2 = -np.log(np.sum(p ** 2) + 1e-12)
    n_eff = float(np.sum(p > 1e-6))
    return 0.0 if n_eff <= 1 else float(H2 / np.log(n_eff))


def test_d5_limiting_values():
    """Pin the evenness interpretation: 0 for a dominant pathway and 1 for a
    uniform distribution on the retained support (SI synthetic checks)."""
    assert _d5_from_pathway_weights([1, 1e-15, 1e-15, 1e-15]) == pytest.approx(0.0, abs=1e-6)
    assert _d5_from_pathway_weights([0.25, 0.25, 0.25, 0.25]) == pytest.approx(1.0, abs=1e-6)
    # one strong + three weak: strictly between the limits
    mid = _d5_from_pathway_weights([0.7, 0.1, 0.1, 0.1])
    assert 0.0 < mid < 1.0


def test_sparse_bridge_descriptors():
    m = sparse_bridge()
    g = SCCG(manifold=m, omega=1.5)
    assert 0.0 <= D1_endpoint_localization(g) <= 1.0
    assert D2_mediator_bridge(g) > 0.0
    # A dominant bridge should have low evenness.
    assert D5_pathway_evenness(g) < 0.5, "dominant bridge should give low D5"


def test_d2_zero_throughput_boundary_is_zero():
    """An endpoint profile with no optical weight follows the stated D2 convention."""
    m = _two_mediator_manifold(A2=0.0, A3=0.0, spin_mixing=[0, 0, 0, 0])
    g = SCCG(manifold=m, omega=1.5)
    assert D2_mediator_bridge(g) == 0.0


@pytest.mark.parametrize("amplitudes,expected", [((0., 0.), 0.), ((1., 0.), 0.),
                                                ((1., 1.), 1.)])
def test_d5_implemented_support_and_single_pathway_branches(amplitudes, expected):
    m = _two_mediator_manifold(*amplitudes, spin_mixing=[0, 0, 0, 0])
    assert D5_pathway_evenness(SCCG(manifold=m, omega=1.5)) == pytest.approx(expected, abs=1e-10)


def test_d5_raw_support_floor_is_distinct_from_d2_normalization():
    """The archived D5 floor uses raw G; D2 normalizes positive weak supports."""
    m = _two_mediator_manifold(1e-7, 1e-7, spin_mixing=[0, 0, 0, 0])
    graph = SCCG(manifold=m, omega=1.5)
    assert D2_mediator_bridge(graph) == pytest.approx(0.5)
    assert D5_pathway_evenness(graph) == 0.0


def test_d2_is_exactly_invariant_to_uniform_dipole_rescaling():
    """D2 is a normalized profile functional, not a throughput measure."""
    m = _two_mediator_manifold(A2=1.0, A3=0.25, spin_mixing=[0, 0, 0, 0])
    baseline = D2_mediator_bridge(SCCG(manifold=m, omega=1.5))
    m.dipoles *= 7.0
    m.dipoles_complex *= 7.0
    rescaled = D2_mediator_bridge(SCCG(manifold=m, omega=1.5))
    assert rescaled == pytest.approx(baseline, rel=1e-14, abs=1e-14)


def test_d2_alignment_concentration_identity():
    m = _two_mediator_manifold(A2=1.0, A3=0.25, spin_mixing=[0, 0, 0, 0])
    graph = SCCG(manifold=m, omega=1.5)
    parts = D2_decomposition(graph)
    assert parts["D2"] == pytest.approx(
        parts["cosine_alignment"] * parts["concentration"], abs=1e-14
    )
    assert parts["N_q"] >= 1.0
    assert parts["N_r"] >= 1.0


def test_dense_manifold_higher_evenness():
    m_sparse = sparse_bridge()
    m_dense = dense_redundant()
    g_sparse = SCCG(manifold=m_sparse, omega=1.5)
    g_dense = SCCG(manifold=m_dense, omega=2.5)
    D5_sparse = D5_pathway_evenness(g_sparse)
    D5_dense = D5_pathway_evenness(g_dense)
    assert D5_dense > D5_sparse, "this dense fixture must show higher pathway evenness"


def test_all_descriptors_returns_dict():
    m = sparse_bridge()
    g = SCCG(manifold=m, omega=1.5)
    out = all_descriptors(g)
    assert set(out.keys()) == {"D1", "D2", "D3", "D4", "D5"}
    # Implemented ones return floats; not-implemented return None
    assert isinstance(out["D1"], float)
    assert isinstance(out["D2"], float)
    assert isinstance(out["D5"], float)
    assert out["D3"] is None
    assert out["D4"] is None
