# Public calibration

Only the public `dev` and `dev_full` data were used. No private cases, panel
timings, or model submissions were inspected.

## Frozen policy

- Backend: SLEPc `MFNEXPOKIT` with `FNEXP`.
- Augmented Krylov dimension: 60 for one-shot, 80 for reuse.
- Tolerance: `multiplier * rtol`.
- Candidate multiplier grid: `0.01, 0.1, 1, 10, 100`.
- Selection rule: choose the largest grid value for which every request in
  both public manifests passes and the maximum scaled error is at most 0.25.
- Selected/default multiplier: **100**.

The first argument of `run.sh` overrides this multiplier solely to make the
public calibration reproducible. Timing did not select the multiplier.

The dimension policy is fixed by public scenario, not by case identity or
matrix contents. A 30-vector Expokit subspace was inaccurate on a public stiff
p10 case. A 60-vector subspace passed all full-size one-shot cases but missed
one reduced p10 reuse request, so reuse uses 80.

## Validation in the frozen image

One repeat per case, network disabled, one pinned logical CPU:

| Set | Sessions passed | Max scaled error | Median time | Max time |
| --- | ---: | ---: | ---: | ---: |
| `dev` | 56 / 56 | 0.0505943 | 0.07230 s | 4.32342 s |
| `dev_full` | 40 / 40 | 0.229377 | 0.33281 s | 27.2800 s |

`scaled error = max(abs(y-y_ref)) / (atol + rtol*max(abs(y_ref)))`; passing
requires it to be at most 1. These are public development measurements, not a
formal score. The benchmark score remains unmeasured until the reference panel
is rerun: raw time ratios are not available, mapping is `100/(1+R)`, and the
reference anchor is 50/100.
