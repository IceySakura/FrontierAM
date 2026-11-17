# Task name

Continuum estimation from positive discretization levels

# Problem description

A deterministic numerical calculation provides approximations y at positive
discretization levels x. Estimate the limiting value as x tends to zero.
The scored problems have convergent, smooth error trends with unknown positive
convergence orders. Inputs include four or five distinct positive levels, on
possibly nonuniform grids. Simulator expressions and the limiting values are private.

# Algorithm objective

Implement one reusable CPU program that infers the limiting value from the
current x,y observations. Minimize error across the complete evaluation batch.
This is a one-dimensional fixed-observation problem: additional simulator queries
and uncertainty-interval outputs are outside this task.

# Algorithm constraints

Each invocation runs offline in Linux x86-64, with one logical CPU, 2 GiB RAM,
at most 128 candidate processes and a 60-second wall-clock limit including startup, imports, reading,
computation, writing and exit. Use one thread. The bundled numerical runtime is Python
3.11.15, NumPy 1.26.4 and SciPy 1.12.0; dependencies must work offline.
For development, invoke /workspace/runtime/bin/python3 explicitly.
It selects the bundled libraries without an installation or setup step.
The verifier puts that interpreter first on PATH, so run.sh may call python3.
Verification uses a fresh offline container. Each case starts a restricted
process with a fresh source copy and empty result/temporary directories.
RAM and swap are shared with the trusted verifier; the container has 2 GiB
additional swap. Source initialization is allowed within the invocation.
Temporary source and scratch space are each limited to 512 MiB and 20,000
files. Read only the current input, public
materials and submission files. No hidden files, answer tables, test-case
identification, cross-invocation state or network use is permitted.

# Delivery requirements

Submit source, README.md and run.sh in /workspace/submission. The evaluator runs
`/bin/sh /workspace/submission/run.sh` without arguments. No separate build step
is provided; any initialization belongs inside the 60-second invocation.

Read /workspace/inputs/input.json, an object containing exactly x and y arrays
of equal length. Values are finite JSON numbers; x values are strictly positive
and distinct. Do not assume the arrays are sorted.
Write /workspace/result/output.json containing exactly one field:
`{"estimate": 0.0}` with a finite JSON number. Booleans, numeric strings,
duplicate keys, NaN, Infinity and additional fields are invalid. Create the
result directory if needed and exit successfully.

For local development, run.sh may accept optional INPUT and OUTPUT paths;
its no-argument behavior must follow the paths above.
Submit only ordinary files: links and special files are rejected. The source
package is limited to 64 MiB and 10,000 files. output.json must be at most
64 KiB; collected result files and program logs are limited to 64 MiB. These
quotas include offline dependencies shipped with the submission.
One frozen program handles every case.

# Evaluation metric

For an estimate z and private limiting value t, the raw loss is
`abs(z-t)/max(1,abs(t))`. Average equally over all 60 scored cases.
An invalid, missing, late or unsuccessful output records per-case loss 1 and
makes the whole submission score zero. Evaluation infrastructure errors produce
no scientific score and are reported separately.

Let E0 and E5 be the frozen aggregate losses of a classical reference and an
independently verified advanced reference, with E0>E5>0. For valid submissions,
score is 0 when E>=E0, 5*(E0-E)/(E0-E5) when E5<=E<E0, and 10-5*E/E5
when 0<=E<E5. Exact limiting values have raw loss zero and score 10.
The whole-task verifier computes the project score once and writes
/logs/verifier/reward.json containing normalized reward=score/10 (higher
is better). E0 and E5 are the two references' aggregate losses measured on the evaluation
machine with the bundled numerical runtime.
A separate adverse-regime audit is diagnostic and does not change this score.

# Development materials and data entry

Read /workspace/resources.yaml for the complete public resource inventory.
The public dev/ directory contains eight fixed input/truth pairs. Run
`sh /workspace/public/self_check.sh` to check the examples, or
`python3 /workspace/public/check.py INPUT OUTPUT TRUTH` to measure a prediction.
Only positive-level observations are available for scored cases. No generator,
private coefficients or hidden reference values are included in this package.
