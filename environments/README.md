# Recorded computational environments

These files pin the direct scientific dependencies observed during the cloud rerun. They are not a lockfile for every transitive package or operating-system library. The execution platform was Linux; keep the Python environments separate because the raw-atlas, native mapping and statistical analyses used different dependency versions.

| Environment | Python | Use |
|---|---|---|
| Statistical analysis | 3.14.3 | Donor/infant statistics, methylation, transport and publication plots |
| Native mapping | 3.12.16 | scVI/scANVI reference fitting, saved-model inference and posterior-assignment summaries |
| Raw-atlas reconstruction | 3.10.20 | Scanpy, Harmony and raw 10x reconstruction |
| Count and bulk models | R 4.4.2 | edgeR 4.5.1 and limma 3.63.2; also jsonlite and statmod |

For example, in a dedicated environment with the indicated Python interpreter:

```bash
python -m pip install -r environments/requirements-statistics.txt
```

For the native environment, install the recorded CPU build of PyTorch first, then the scientific requirements:

```bash
python -m pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environments/requirements-native.txt
```

Use the relevant Bioconductor source archives for the recorded edgeR and limma development versions if an exact R reproduction is needed. Record `sessionInfo()` for any newly assembled R environment. Changing model-library versions can affect numerical fitting; comparisons should use the corresponding recorded environment.

The raw-rebuild requirements and their version snapshot document the reconstructed environment used in the continued revision. Random seeds and statistical settings reside in the scripts. CPU execution was used for native inference. External download stages additionally require `curl` on PATH. Install Liberation Sans through the operating system's font manager for all publication figures; no font binaries are distributed here.
