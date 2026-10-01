# Spin-Control Connectivity Graphs: Ni₂/coronene benchmark

Version 2.0.0 supplements the baseline computational record with the analyses used in the revised manuscript, **Unnormalized Endpoint Supports Expose State-Set Dependence in Spin-Control Connectivity Graphs**, by Jing Liu, Kanchan Sarkar, and Axel Groß.

This archive documents one fixed-geometry Ni₂/coronene benchmark. It contains graph-analysis software, numerical records, selected electronic-structure outputs, and figures. It does not establish retained-state convergence, molecular-candidate ranking performance, or predictive control fidelity.

## What changed in v2.0.0

- Full-grid endpoint-support accounting for `G = D2 S_L S_R`, including uniform four-root retention, selective quintet- and singlet-root omissions, mixed-tier pairings, and the logarithmic decomposition.
- The permanent-dipole-difference sensitivity record and its derived support-accounting curves.
- A fixed-orbital CAS(26,15) CI screen requesting ten roots per multiplicity. Among ranks 6–10, seven additional roots lie below 0.65 eV relative to the lowest CI root. The tenth triplet root also remains in that interval, so this is a lower bound on the CI-level state count. Higher-root MS-CASPT2 energies and an enlarged spin–orbit graph were not computed.
- The CI inputs, optimized geometry, starting orbital files, text outputs, HDF5 outputs, parser, and run notes.
- Updated figures associated with the revised manuscript and Supporting Information.

## Contents

- `analysis/sccg/`: graph and descriptor implementation, reconstruction and retained-state analyses, tests, and release verification tools.
- `calculations/B1/`: parsed spin–orbit manifolds, descriptor sweeps, reconstruction arrays, sensitivity outputs, support-accounting records, and the ten-root CI screen.
- `figures/`: vector figures used in the revised analysis; `opt_optimized.xyz` is the optimized 38-atom geometry in ångström.
- `pyproject.toml`, `uv.lock`, `.python-version`: analysis environment.
- `CITATION.cff`, `.zenodo.json`, `LICENSE.md`, and `LICENSES/`: citation metadata and licensing.

The Gaussian high-spin optimization output reports a nuclear-repulsion energy of 3152.7618058572 Eh for the optimized geometry. The DFT total energies quoted in the Supporting Information include this contribution. The geometry and this value are also recorded in the SI.

## CI screen: inputs and outputs

The self-contained run directory is `calculations/B1/06_referee_studies/ci_root_screen_10/`. Its three OpenMolcas inputs expect the working directory to contain `opt_optimized.xyz` and the matching `rasscf*.RasOrb` files. Each requests ten roots with `CIONLY`; no orbital optimization, CASPT2 step, or RASSI-SO calculation is included. The inherited production orbitals are supplied so the fixed-orbital CI screen can be repeated, subject to a compatible OpenMolcas installation and local execution environment.

The returned `ci_[QST].out` and `ci_[QST].rasscf.h5` files are the outputs used for the archived root-energy analysis. The result JSON records all thirty relative energies and checksums. Regenerate that summary from the project root with:

```sh
uv run python calculations/B1/06_referee_studies/ci_root_screen_10/parse_ci_root_screen.py
```

The run notes identify the geometry, active space, production orbital provenance, software context, and scheduler details. The screen finds at least seven additional roots below 0.65 eV at the fixed-orbital CI level; their MS-CASPT2 placements remain unknown.

## Reproduce the graph-analysis audit

Run from this repository root with Python 3.11 and [`uv`](https://docs.astral.sh/uv/):

```sh
uv sync --frozen --dev
uv run pytest
uv run python analysis/sccg/retained_state_audit.py
uv run python analysis/sccg/retained_state_robustness.py
uv run python analysis/sccg/verify_release.py
```

The retained-state programs write JSON and CSV records under `calculations/B1/` and may replace distributed copies. Use a disposable copy if you want to preserve the release records. These analyses start from the archived compact reconstruction arrays and do not rerun electronic structure. The read-only verification program reconstructs the 42 baseline descriptor rows, all thirteen non-endpoint single-root deletions, and both dipole treatments for five retained sets in three tiers. It checks the full support curves, correlations, logarithmic accounting, direct unnormalized shared weights, and all thirty CI energies against the archived HDF5 outputs. The graph and command-line defaults use dipole-only edges (`beta=0`), as in the article; spin–orbit coupling enters through the SO eigenstates and transition dipoles.

To regenerate the version 2 sensitivity and support-accounting records:

```sh
uv run python analysis/sccg/tools/permanent_dipole_difference_screen.py
uv run python analysis/sccg/tools/export_support_accounting.py
```

The release contains evaluated tables and figures as data. The commands above do not regenerate every distributed figure or manuscript table. Full HDF5 outputs are included for the ten-root CI screen. The original five-root production, bridge, and CASCI RASSI HDF5 files and Gaussian/OpenMolcas executables are not included. Repeating those electronic-structure calculations requires the software and inputs appropriate to the stated basis and active spaces.

## Scope and provenance

The production, compact MS-CASPT2 bridge, and compact CASCI model-space anchor each contain 45 spin–orbit states. The compact and production active spaces are not nested. Root-deletion calculations preserve the upstream five-root spin-free solutions and change only the state-interaction construction; they are sensitivity tests, not convergence tests. Cross-tier rank correlations refer to photon-energy grid points within this single system, not rankings of molecular candidates.

The electronic-structure calculations were performed with OpenMolcas 24.02 and Gaussian; the archived analysis environment is specified by the lockfile. `calculations/B1/manifest.yaml` records the model and tier definitions. Some log-file filesystem paths are redacted; numerical energies, matrices, and printed state properties are unchanged. Undefined correlations of constant curves are represented by JSON `null`. Paired grid-point resampling percentiles in `tier_correlations.json` are descriptive and are not validated confidence intervals for correlated deterministic sweeps.

Verify the distributed files before running scripts that overwrite records:

```sh
sha256sum -c MANIFEST.sha256
```

## Licensing and citation

Numerical data and figures are licensed under CC BY 4.0. Analysis software is licensed under MIT; see `LICENSES/`. These licenses apply to the authors' contributions and do not relicense third-party software, basis sets, or publications.

Please cite this archive using `CITATION.cff` and the version-specific DOI displayed on its Zenodo record. The corresponding article title is given above. Version 2.0.0 extends the baseline v1.0.0 record, [10.5281/zenodo.23047692](https://doi.org/10.5281/zenodo.23047692).
