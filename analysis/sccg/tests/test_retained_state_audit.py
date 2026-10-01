"""Independent physical invariants for the matrix-only retained-state audit."""
import numpy as np
import pytest

from sccg.retained_state_audit import ranked_tv, measure, load_inputs, build, TIERS


def test_release_sweeps_and_complete_single_root_screen():
    from sccg.verify_release import verify
    verify()


def test_ranked_distance_compares_concentration_not_labels_or_dimension():
    assert ranked_tv(np.array([.8, .2]), np.array([0., .2, .8])) == pytest.approx(0)
    assert ranked_tv(np.array([1., 0.]), np.array([.5, .5])) == pytest.approx(.5)


@pytest.mark.parametrize("tier", list(TIERS))
def test_reduced_manifold_preserves_whole_endpoint_multiplets(tier):
    data = load_inputs(TIERS[tier][0])
    for n in (3, 4, 5):
        reduced = build(data, data["local_roots"] <= n)
        np.testing.assert_allclose(reduced["endpoint"].sum(axis=0), [5, 1], atol=2e-13)


def test_null_optical_states_and_scalar_contractions():
    e = np.array([0., .03, .15])
    mu = np.zeros((3, 3, 3))
    mu[0] = [[4., .3, 2.], [.3, 8., 1.], [2., 1., 7.]]
    endpoints = np.array([[1., 0.], [0., 1.], [0., 0.]])
    model = {"energy": e, "mu": mu, "endpoint": endpoints}
    base, explicit = measure(model, .1), measure(model, .1, loop=True)
    padded = {"energy": np.r_[e, .02, .5],
              "mu": np.pad(mu, ((0, 0), (0, 2), (0, 2))),
              "endpoint": np.pad(endpoints, ((0, 2), (0, 0)))}
    null = measure(padded, .1)
    for key in ("D2", "PR", "max_p", "D5"):
        assert base[key] == pytest.approx(explicit[key], abs=1e-13)
        assert base[key] == pytest.approx(null[key], abs=1e-13)


def test_support_identity_and_production_q5_mechanism():
    """G = D2*S_L*S_R exactly; omitting Q5 collapses S_L while D2 rises and G falls."""
    from sccg.retained_state_robustness import endpoint_columns, keep_mask, profiles
    data = load_inputs(TIERS["production"][0])
    cols = endpoint_columns(data, 1, 1)
    full = profiles(build(data), cols, 0.65, 0.10)
    omit = profiles(build(data, keep_mask(data, "omit_Q5")), cols, 0.65, 0.10)
    for row in (full, omit):
        assert row["G"] == pytest.approx(row["D2"] * row["S_L"] * row["S_R"], rel=1e-12)
    assert omit["S_L"] / full["S_L"] < 2e-3
    assert omit["S_R"] / full["S_R"] > 0.9
    assert omit["D2"] > 100 * full["D2"] and omit["G"] < full["G"]


def test_lower_boundary_scan_preserves_reversal_and_support_domain():
    from sccg.retained_state_robustness import keep_mask, lower_boundary_scan
    data = {tier: load_inputs(label) for tier, (label, _) in TIERS.items()}
    models = {tier: {case: build(d, keep_mask(d, case)) for case in ("full", "omit_Q5")}
              for tier, d in data.items()}
    scan = lower_boundary_scan(models, data)
    for lower, entry in scan.items():
        grid = entry["grid_eV"]
        assert grid[0] == pytest.approx(float(lower))
        assert grid[-1] == pytest.approx(.65)
        np.testing.assert_allclose(np.diff(grid), .005, atol=1e-14)
        assert entry["cases"]["full"]["correlations"]["two_MS_tiers"] > .98
        assert -.75 < entry["cases"]["omit_Q5"]["correlations"]["two_MS_tiers"] < -.69
        for case in entry["cases"].values():
            for curve in case["curves"].values():
                assert np.min(curve["G"]) > 3.3e-10
                np.testing.assert_allclose(curve["G"], curve["D2"] * curve["S_L"] * curve["S_R"], rtol=1e-12)
