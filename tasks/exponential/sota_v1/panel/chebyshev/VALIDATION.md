# Public validation

Date: 2026-11-05. Dataset: all 29 cases in the public
`task/dev/manifest.json` extracted from `public_pilot_v2/public_pilot.tar`.

The recorded run used the task image `frontieram-exponential:port-v1`, no
network, `nice -n 19`, host affinity CPUs 2–3, container cpuset CPUs 2–3, and
the supplied `task/tools/run_local.py`. Each program was run in a fresh process
three times per case. Both programs are single-threaded. These are local
development timings, not formal cgroup-isolated grading results.

## Results

| Program | Valid repetitions | Max scaled error | Sum of 24 scored-case median times | Peak RSS |
|---|---:|---:|---:|---:|
| Chebyshev (`run.sh`) | 87 / 87 | 0.1610558 | 0.121892 s | 36.08 MiB |
| Expokit reference (`reference_expokit/run.sh`) | 87 / 87 | 0.002439902 | 0.399832 s | 36.07 MiB |

All requests in all repetitions passed. The scaled error is
`||y-y_ref||_inf / (atol + rtol*||y_ref||_inf)`; the acceptance limit is 1.

On the scored public cases, the sum-of-medians time ratio was
`0.121892 / 0.399832 = 0.30486`. Split by session, the corresponding ratios
were 0.41022 for one-shot and 0.29685 for 16-request reuse. Chebyshev was faster
on all 24 scored sessions in this run.

Using the repository's prescribed five-level public aggregation gives raw
relative time `R = 0.4590232` against the locally rerun Expokit reference.
With the exponential-task mapping

`final score = 100 / (1 + R)`,

this is **68.539 / 100** (the same-machine Expokit reference is 50). This
development score is descriptive only; the public runner itself reports no
calibrated official score.

The generic x86-64 binary was separately run once over all 29 cases: 29/29
valid, with maximum scaled error 0.1610488.

Raw three-repeat reports are retained as `chebyshev_report.json` and
`expokit_report.json`.

## Calibration audit note

No safety or degree parameter was changed after seeing public results. An
initial protocol-wide dry run exposed a dimensional implementation error: the
coefficient recurrence produced `phi1(-tL)` rather than
`t*phi1(-tL)`. The missing algebraic factor `t` was restored; this is a
correctness repair, not case calibration. The preregistered endpoint and safety
factor in `CALIBRATION.md` were unchanged before the recorded three-repeat run.
