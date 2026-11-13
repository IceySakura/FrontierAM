# Galaxy Gravity B.4 evaluation bundle

Version: `0.4.0-precalibration`. This is a pre-release operator bundle.

It contains the synchronized B.4 Harbor task, the hidden 80-case full-particle references, the frozen per-case CPU limits, and the theta=0.8 classic / theta=0.7 modified baselines. The semantic 0/5/10 anchors are recorded, but scale/family weights and the final aggregate score formula remain unset, so this bundle is for review and calibration rather than final scoring.

The ten timeout reruns are provenance only and are excluded from the formal 240-call records. Use the ARM64 image named in `task/task.toml`; the operator must provide the local Docker image.
