# Spin-Control Connectivity Graph: Ni₂/coronene benchmark

Data and analysis code supporting **Retained-State Selection Can Reverse
Spin-Control Descriptor Correlations**, by Jing Liu, Kanchan Sarkar, and Axel Groß.

This repository distributes the version 1.0.0 reproducibility archive for a
fixed-geometry Ni₂/coronene computational benchmark. It documents retained-state
sensitivity, not retained-state convergence, molecular-candidate ranking
performance, or predictive control fidelity.

## Download and verify

Download [the archive](sccg-ni2-coronene-1.0.0.tar.gz) and
[its SHA-256 checksum](sccg-ni2-coronene-1.0.0.tar.gz.sha256) into the same directory:

```sh
sha256sum -c sccg-ni2-coronene-1.0.0.tar.gz.sha256
tar -xzf sccg-ni2-coronene-1.0.0.tar.gz
cd sccg-ni2-coronene
sha256sum -c MANIFEST.sha256
```

The extracted README documents the contents, provenance, limitations, and
reproduction commands. The archive includes analysis software and tests,
parsed electronic-structure data, compact reconstruction arrays, sensitivity
records, calculation inputs, and figures. Starting orbital files and full
AO/density HDF5 outputs are not included.

## Citation and licensing

See [CITATION.cff](CITATION.cff) for citation metadata. A DOI will be added after
archiving on Zenodo; no DOI has yet been assigned in these files.

Numerical data and figures use CC BY 4.0; analysis software and associated
documentation use MIT. See [LICENSE.md](LICENSE.md) for scope and license texts.
