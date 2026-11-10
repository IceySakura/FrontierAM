# Pilot instructions — read first

This pilot gives **one development attempt of 2 hours (7200 wall-clock seconds)**,
matching the two-hour development allowance in INSTRUCTION.md, ENVIRONMENT.md and
RULES.json. The earlier 30-minute pilot limit no longer applies. All other
mathematical, accuracy, interface and final-session limits remain in force. The development window starts
when the coding agent is launched and ends at its final turn or the deadline.
The controller pauses the entire workspace before freezing the submission.
There is no continuation or algorithm repair after that point.

Read INSTRUCTION.md and API.md. All public materials are in /workspace/task.
Write a self-contained submission, including full source and executable run.sh,
in **/workspace/submission**. The evaluator executes ./run.sh with the submitted
directory as its working directory (mounted internally as /submission); use
relative paths for your own files and include any SDK code you need.
Files elsewhere in the development workspace are not submitted. Include a README
describing the algorithm, dependencies and public checks. No symlinks, hardlinks,
device files or other special files are accepted. Source and artifacts together
must not exceed 1 GiB.

If you supply build.sh, it is run once with bash in a fresh offline container
after development ends, with only the frozen submission available. It has
600 seconds wall time, 600 seconds aggregate container CPU and 4 GiB memory.
It cannot see public or private tests. Include every source needed to reproduce
the build. The original source snapshot is retained separately. A missing or
failing run.sh/build.sh receives no maintainer repair or additional attempt.

Only the workspace shell MCP tool is available. No network, host filesystem,
Docker socket, API credentials, external tools or subagents are provided.
Development and build share one CPU pinned to CPU 0, 4 GiB memory with no extra
swap, 32 PIDs/threads, 128 file descriptors, a read-only system image, 256 MiB
/tmp and a 1 GiB /workspace temporary filesystem. All code runs as uid 1000
with all capabilities dropped and no-new-privileges. Public tests are unlimited
within the development allowance. Hidden evaluation never feeds back into it.

## Frozen available Linux environment

Image: `sha256:f203391e0381bdca377548961998bbbcac9bfd2212cfd00ffb77c4b31ed230f2`; architecture: arm64.
The following software was inspected inside this exact image. These installed
general-purpose compilers and libraries, including their high-level APIs, may
be used freely. Do not download or install further dependencies.

- **platform**: Linux-7.0.0-34-generic-aarch64-with-glibc2.36
- **machine**: aarch64
- **python**: 3.12.16
- **numpy**: 2.5.4
- **scipy**: 1.18.1
- **gcc**: gcc (Debian 12.2.0-14+deb12u1) 12.2.0
- **g++**: g++ (Debian 12.2.0-14+deb12u1) 12.2.0
- **gfortran**: GNU Fortran (Debian 12.2.0-14+deb12u1) 12.2.0
- **make**: GNU Make 4.3
- **octave**: GNU Octave, version 7.3.0
- libeigen3-dev 3.4.0-4
- liblapack-dev 3.11.0-2
- liblapacke-dev 3.11.0-2
- libopenblas-dev 0.3.21+ds-4

Python standard-library modules and the installed Bash/coreutils utilities are
also available. No task-specific author implementation is supplied.

## Frozen pilot distribution

PUBLIC_DISTRIBUTION.json gives the complete public aggregate description of the
pre-frozen pilot. It contains distribution summaries only, with no private
seeds, exact final-case manifest, answers, baseline source or tuning records.
This is a difficulty pilot, not a claim about final benchmark performance.


## Repaired harness contract (v3, two-hour)

HARNESS.json specifies the common inference limits. A response may use up to
64,000 output tokens, including internal reasoning. At most 800 actual HTTP
attempts are allowed, including automatic retries and compaction requests. All
five models use their provider's default reasoning effort, the same 131,072-token
Codex context setting and the same 2-hour total wall budget. These are not
equal-token or equal-cost comparisons. All inference and retry waiting counts
toward that wall budget; there is no separate 180-second early request cutoff.
Codex compacts the conversation automatically near 110,000 tokens; compaction
uses the same wall and HTTP budgets. Continuation after the trial is unavailable.

Where the official model protocol supports a server timeout, it is explicitly
set to the smaller of the remaining development time and the documented maximum:
600 seconds for DeepSeek, 3600 seconds for Opus/MiniMax/Step. HY exposes no such
standard parameter, so it is omitted. Provider-side restrictions remain outside
the host harness's control; the shared 7200-second wall deadline always applies.

Only HTTP 429/500/502/503/504 responses with a documented retryable gateway
status may automatically retry the identical request, at most three times,
within the same original time and request budgets.
Retry-After takes priority over full jitter capped at 1, 2, then 4 seconds.
No provider failover or new development attempt is introduced by a retry.
An upstream failure or truncated response is recorded separately from algorithm
correctness; incomplete tool calls are never executed. At the normal wall-clock
deadline the workspace is frozen and the existing valid submission is evaluated.

This revised package is a new prospective protocol. Results from the original
8,192-token/180-second harness retain their original records and are not silently
combined into a five-model ranking under this revised contract.
