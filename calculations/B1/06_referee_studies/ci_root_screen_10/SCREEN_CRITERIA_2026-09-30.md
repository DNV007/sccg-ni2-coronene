# Interpretation criteria recorded before two sensitivity screens

These criteria were recorded on 30 September 2026 before running the screens. The resulting calculation and analysis records are included in this release.

## Fixed-orbital CI root screen

Use the production multiplicity-specific orbital files without orbital reoptimization. Request at least ten spin-free CI roots for each of the quintet, triplet, and singlet blocks. Report each root's energy relative to the lowest CI root across those blocks. Retain the first five roots in each multiplicity as the existing selection rule, and examine roots 6--10.

- If any root 6--10 lies below 0.65 eV at the CI level, report its multiplicity, within-block rank, and energy as an unrepresented CI-level state in the analysis window. This is a reason for an enlarged electronic-structure calculation, not evidence of its MS-CASPT2 position.
- If none lies below 0.65 eV but a root 6--10 lies below a retained root of another multiplicity, report that interleaving. It tests the energy-contiguity of the five-per-multiplicity rule at the CI level.
- If neither occurs, report the negative CI-level result. The positions of higher MS-CASPT2 roots remain unknown.

CI-only energies are not mapped to MS-CASPT2 energies by a fitted compression factor. Root identities are not inferred from energy rank alone. Input, output, software version, and numerical root table must accompany any reported result.

## Spin-free permanent-dipole-difference screen

For each tier, replace every spin-free diagonal electric-dipole vector by one common vector (the componentwise mean of the original diagonals), retaining all off-diagonal spin-free dipoles, spin--orbit matrices, and endpoint definitions. Reconstruct the full five-root, uniform four-root, Q5-only, and S5-only spin--orbit graphs with the same photon-energy grid and Gaussian width as the baseline. A common diagonal vector contributes no off-diagonal spin--orbit transition dipole in a complete orthonormal retained eigenbasis; confirm numerically that choosing a different common vector leaves these off-diagonal results unchanged.

Record baseline and perturbed D2, S_L, S_R, G, participation ratio, leading-mediator identity and dominant spin-free parent across the grid. Compare production--bridge correlations for all four retention cases.

- Recast the low-energy mediator interpretation as permanent-dipole-difference-sensitive if the leading mediator's dominant parent changes or the 0.02-eV participation ratio changes by more than a factor of two in either tier.
- Reconsider the main support-loss interpretation if the signs or ordering of the three log contributions (Delta log G, -Delta log S_L, -Delta log S_R) at 0.65 eV change for N4, Q5-only, or S5-only deletion, or if the baseline/deletion D2 correlation signs change.
- Otherwise report the quantitative shifts and describe the conclusions as surviving this particular operator sensitivity, without claiming a generally validated transition operator.

All logarithms require positive D2, G, S_L, and S_R; zero-support cases, if any, are flagged rather than assigned finite log contributions.
