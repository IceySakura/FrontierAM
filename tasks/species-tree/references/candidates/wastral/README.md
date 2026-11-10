# Weighted ASTRAL (wASTRAL, hybrid weighting) — human reference candidate for species-tree v3

- **Method**: Weighted ASTRAL with hybrid weighting by branch support and branch length (Zhang & Mirarab, *Mol. Biol.
  Evol.* 39(12), 2022), from the ASTER package (https://github.com/chaoszhang/ASTER, commit
  `ddc3dc6b28b20cecfec18cede8c6247decc82b77`, 2026-06-25; `src/` copied unchanged, AGPL-3.0, `LICENSE`).
- **Why**: the v6.3 gpt-6.1-sol iterative submission weights quartets by branch support (local posterior probability)
  and pendant branch length — the weighting this published method introduced (results/BASELINE_REVIEW.md).
- **Settings**: the documented defaults for these inputs: hybrid mode (default), `-B` (local Bayesian supports in
  [1/3, 1], the gene trees' support type), one thread (`-t 1`), topology only (`-u 0`). No tuning on any data.
- **Build/run**: `./build.sh` compiles `src/astral-hybrid.cpp` (the package's own `wastral` target); `./run.sh GENES OUT`.
