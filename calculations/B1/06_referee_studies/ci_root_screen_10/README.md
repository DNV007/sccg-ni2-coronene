# Fixed-orbital CAS(26,15) CI root screen

Three OpenMolcas 24.02 calculations completed on JUSTUS on 30 September 2026.
Each quintet, triplet, and singlet calculation requested ten roots on inherited
production orbitals with `CIONLY` and `CIROOT = 10 10 1`. There is no orbital
optimization, CASPT2, or RASSI-SO step. All three text logs report successful
completion and ten roots. Recorded wall times are approximately 47–48 minutes.

## Inputs and provenance

This directory contains `ci_Q.input`, `ci_T.input`, `ci_S.input`,
`opt_optimized.xyz`, and the multiplicity-specific starting orbital files
`rasscf.RasOrb`, `rasscf_T.RasOrb`, and `rasscf_S.RasOrb`. The input decks
reference these filenames in the working directory. The geometry matches
`calculations/B1/01_cluster/opt_optimized.xyz`; the starting orbital files
were exported from the production RASSCF HDF5 files.

The runs requested one node, one task, one CPU, 16000 MB memory, an eight-hour
limit, and 200 GB local scratch. OpenMolcas used a 12 GB memory setting.
Slurm job identifiers were 22432222, 22432225, and 22432226. Application logs
and HDF5 outputs are included; scheduler logs and the site-specific submission
driver are not distributed.

## Results and regeneration

From the archive root, after installing the locked environment:

```sh
uv run python calculations/B1/06_referee_studies/ci_root_screen_10/parse_ci_root_screen.py
```

The parser reads `ROOT_ENERGIES` from the three HDF5 outputs and writes
`ci_root_screen_result.json`. That record includes checksums, all thirty
energies, and interpretation against `SCREEN_CRITERIA_2026-09-30.md`,
recorded before the calculations. An identical summary is also provided at
`calculations/B1/ci_root_screen_result.json`.
The HDF5 root energies are authoritative for this analysis. The text-log
energies differ from them by at most 7.53 × 10⁻⁸ Eh (2.05 micro-eV), which
does not affect the reported millielectronvolt precision or window classification.

Among ranks 6–10, Q6, T6–T10, and S6 lie below 0.65 eV relative to the lowest
CI root, T1. Since T10 remains inside that interval, the seven additional
roots establish a lower bound. Their MS-CASPT2 energies and effects on an
enlarged spin–orbit manifold remain uncomputed.
