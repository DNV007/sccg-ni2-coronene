"""Read-only reconstruction checks for the additional version 2 records."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
from pathlib import Path

import h5py
import numpy as np

from sccg import retained_state_audit as audit
from sccg.tools import export_support_accounting as accounting
from sccg.tools import permanent_dipole_difference_screen as dipoles


def compare(actual, expected, path=""):
    """Compare all fields, preserving relative accuracy for tiny supports."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            compare(actual[key], expected[key], path + "/" + key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, value in enumerate(expected):
            compare(actual[i], value, path + "/" + str(i))
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        numerical_residual = any(s in path for s in ("error", "residual"))
        np.testing.assert_allclose(actual, expected, rtol=2e-9,
                                   atol=1e-12 if numerical_residual else 0,
                                   err_msg=path)
    else:
        assert actual == expected, (path, actual, expected)


def verify():
    b1 = audit.B1
    screen = json.loads((b1 / "permanent_dipole_difference_screen.json").read_text())
    # Regenerate both complete records without modifying the distributed files.
    with tempfile.TemporaryDirectory(prefix="sccg-record-check-") as temporary:
        old_output = dipoles.OUTPUT
        try:
            dipoles.OUTPUT = Path(temporary) / "screen.json"
            dipoles.main()
            compare(json.loads(dipoles.OUTPUT.read_text()), screen)
        finally:
            dipoles.OUTPUT = old_output
        old_json, old_csv = accounting.OUT_JSON, accounting.OUT_CSV
        try:
            accounting.OUT_JSON = Path(temporary) / "accounting.json"
            accounting.OUT_CSV = Path(temporary) / "accounting.csv"
            accounting.main()
            compare(json.loads(accounting.OUT_JSON.read_text()),
                    json.loads((b1 / "support_accounting_record.json").read_text()))
            with accounting.OUT_CSV.open() as a, (b1 / "support_accounting_curves.csv").open() as b:
                actual, expected = list(csv.DictReader(a)), list(csv.DictReader(b))
            assert len(actual) == len(expected) == 168
            for row, target in zip(actual, expected):
                for key in target:
                    if key in ("tier", "retention"):
                        assert row[key] == target[key]
                    else:
                        np.testing.assert_allclose(float(row[key]), float(target[key]),
                                                   rtol=2e-9, atol=0)
        finally:
            accounting.OUT_JSON, accounting.OUT_CSV = old_json, old_csv

    # Compute G directly from unnormalized profiles, independently of D2*S_L*S_R.
    for tier, (label, _) in audit.TIERS.items():
        data = audit.load_inputs(label)
        mean = np.diagonal(data["sf_dipoles"], axis1=1, axis2=2).mean(axis=1)
        for condition, inputs in (("original", data),
                                  ("common_diagonal", dipoles.replace_diagonal(data, mean))):
            for case, mask in dipoles.masks(inputs).items():
                model = audit.build(inputs, mask)
                strength = np.sum(np.abs(model["mu"]) ** 2, axis=0)
                gaps = np.abs(model["energy"][:, None] - model["energy"][None, :])
                g = []
                for energy in audit.GRID:
                    weights = strength * np.exp(-0.5 * ((gaps - energy) / 0.1) ** 2)
                    np.fill_diagonal(weights, 0)
                    profiles = weights @ model["endpoint"]
                    g.append(float(np.sum(profiles[:, 0] * profiles[:, 1])))
                np.testing.assert_allclose(g, screen["curves"][condition][tier][case]["G"],
                                           rtol=2e-9, atol=0)

    ci_dir = b1 / "06_referee_studies" / "ci_root_screen_10"
    ci = json.loads((ci_dir / "ci_root_screen_result.json").read_text())
    compare(ci, json.loads((b1 / "ci_root_screen_result.json").read_text()))
    energies = {}
    for tag, label in (("Q", "quintet"), ("T", "triplet"), ("S", "singlet")):
        path = ci_dir / f"ci_{tag}.rasscf.h5"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ci["files"][tag]["sha256"]
        with h5py.File(path) as handle:
            values = handle["ROOT_ENERGIES"][:]
        assert len(values) == 10
        log = (ci_dir / f"ci_{tag}.out").read_text()
        assert "Happy landing" in log
        printed = re.findall(r"RASSCF root number\s+\d+\s+Total energy:\s+(-?\d+\.\d+)", log)
        # The archived HDF5 and printed energies differ by up to 7.53e-8 Eh.
        # HDF5 supplies the reported energies; check log correspondence to
        # 1e-7 Eh (2.72 micro-eV), well below the reported energy precision.
        np.testing.assert_allclose(values, np.asarray(printed, float), rtol=0, atol=1e-7)
        energies[label] = values
    zero = min(min(values) for values in energies.values())
    assert zero == ci["lowest_CI_energy_hartree"]
    extras = []
    for row in ci["roots"]:
        value = energies[row["multiplicity"]][row["root"] - 1]
        assert value == row["energy_hartree"]
        relative = (value - zero) * 27.211386245988
        np.testing.assert_allclose(relative, row["relative_energy_eV"], rtol=1e-12, atol=0)
        if row["root"] > 5 and relative < 0.65:
            extras.append(row)
    order = lambda row: (row["multiplicity"], row["root"])
    compare(sorted(extras, key=order), sorted(ci["additional_roots_below_0p65_eV"], key=order))
    assert len(extras) == 7
    assert (energies["triplet"][9] - zero) * 27.211386245988 < 0.65
    print("Version 2: 30 reconstructed dipole/retention cases, 168 support rows,")
    print("all support statistics/log curves, and 30 CI energies verified.")
