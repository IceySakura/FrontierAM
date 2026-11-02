# cb-insiders

`agent_name: cb-insiders` runs the original CodeBuddy CN Insiders IDE CLI in a
Gongfeng sandbox. The npm packages carry the unchanged upstream `.deb`; Nexus
extracts it and starts `buddycn-insiders --cli -- -p --output-format stream-json`.
The package owns the craft prompt, agent loop, plugins and tools. Nexus handles
installation, model routing, process lifetime and result collection.

| Runtime architecture | Private npm package |
| --- | --- |
| Linux x64 | `@tencent/nexus-cb-insiders-linux-x64` |
| Linux arm64 | `@tencent/nexus-cb-insiders-linux-arm64` |

Both use npm version `5.0.0-40236426.983c1e09`, corresponding to upstream build
`5.0.0.40236426-983c1e09`, commit
`983c1e090fc9863cc5b0aa66a361766bac083b2e`. The installer verifies the original
deb's SHA-256, its architecture, and the embedded product commit. It disables
npm lifecycle scripts. A runtime-local `package_path` to the verified deb is an
explicit development fallback when publication is not yet available.

Publication verified on **2026-10-03**: both packages are available from
`https://mirrors.tencent.com/npm/` at the exact version above and the `insiders`
tag. The existing WorkBuddy read account can download both. Downloaded tarballs
and embedded deb checksums match the original artifacts. A fresh Linux x64
Gongfeng sandbox installed the package through the Nexus npm path and passed
Read/Edit/Bash tool checks, including Langfuse feedback verification. The arm64
package was verified by download and checksum; arm64 CLI execution was not tested.

That installation check was a scripted protocol smoke with no model-service
calls. The earlier 300-case SWE run used the original deb upload path, so its
results remain a separate record. Use the exact version or `@insiders`; this
publication did not set a `latest` tag.

Package privacy updated on **2026-10-04**: the Tencent registry's package-level
`public` setting is now `false` for both architectures, matching
`@tencent/nexus-wb-cli`. The owner retains admin access, and the existing WorkBuddy
runtime account has an explicit read grant. Full downloads using that account
match the originally published tarball hashes. The version and tag did not change.

The npm `access: restricted` metadata alone did not set this registry-specific
flag. Separate anonymous requests now return `401`, but mixed concurrent
authorized/unauthorized probes sometimes return metadata or archive bytes to an
unauthorized request; the same behavior was observed on the WorkBuddy reference
package. The private configuration is applied, but strict access isolation is
not verified; package installation success does not establish strict registry access isolation.

## Runtime and model configuration

Use an Ubuntu/Debian runtime with glibc >= 2.29. Ubuntu 22.04 and 24.04 were used
for integration checks. CLI mode still needs Electron libraries and an X server;
the agent can install the missing libraries with apt when running as root.
With `auto_install_system_deps: false`, provision them in the image beforehand.
Each invocation creates and stops its own Xvfb, including when the runtime
exports a `DISPLAY` value without starting a desktop.

Set `runtime_image: current` and `worker_venv: {modules: []}` so the runtime
server and uploaded Nexus worker support the same trace protocol. The old
`feat-cc-support-proxy-convert` image rejects the newer worker's acknowledged
trace flush after the agent has already run.

Models are configured through the IDE's supported `models.json` and environment
product overlay. The local model id `nexus` maps to Nexus's OpenAI proxy;
`openai_clients.main` selects the actual provider model. Main and helper model
requests use the same route. Real provider keys stay on the host. Separate
CodeBuddy config and IDE user-data directories, memory credential storage, and
disabled remote product updates avoid desktop login prompts and cross-run
settings. The task image's HOME and language dependency caches remain available
to tools; moving HOME would hide preinstalled SWE dependencies.

`effort` sets the CLI's session preference. Enforce provider parameters with
`enforced_request_args` when a benchmark requires exact settings. The supplied
configuration uses `glm-5.3-flashx`, `reasoning_effort: low`, `max_tokens: 4096`,
`temperature: 1`, `top_p: 0.95`, and a non-streaming upstream request; the local
proxy serves the CLI's streaming protocol.

## Run a tool smoke

Export `GONGFENG_RUNTIME_TOKEN`, `OPENAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`, `CB_INSIDERS_NPM_USERNAME`, and
`CB_INSIDERS_NPM_TOKEN`. Optionally set `CB_INSIDERS_NPM_REGISTRY`.
The agent's Python input also accepts explicit `npm_username` / `npm_token`;
when omitted it falls back to the CB variables, then the WorkBuddy npm variables.

```bash
export NEXUS_AUTO_INSTALL=1
uv run nexus modules install swe langfuse_tracker
uv run nexus run single docs/examples/configs/cb-insiders/smoke.yaml
```

The identity task itself is only a passthrough. Inspect the `agent.cb-insiders`
span and its `success`, `status`, `exit_code`, `result`, and `assistant_text`.
Exit code zero without a final successful JSON result is an error. Timeouts,
process errors, and installation failures are reported separately. Raw JSONL
and the runtime transcript path remain available in the agent output.

## Run SWE smoke

Provide full SWE instances, including their image and evaluation configuration:

```bash
export SWE_DATASET_PATH=/absolute/path/cases.jsonl
export SWE_LIMIT=2
mkdir -p experiments/cb-insiders
uv run nexus run batch \
  -g docs/examples/configs/cb-insiders/swe_generator.py \
  -d experiments/cb-insiders/swe-smoke.db \
  --max-concurrent 2
```

`SWE_INSTANCE_IDS` optionally selects comma-separated instance ids. The generator
keeps answer-lookup detection and penalties enabled, and evaluates each generated
patch once. Check task `resolved`, `score`, `error`, and `termination_reason` in
addition to batch status. A successful CLI run does not by itself mean the patch
resolved the benchmark case.

LLM generations and tool results appear in the configured Langfuse project.
The CLI's lightweight JSONL may contain empty tool-result summaries; use the
recorded model requests for the complete feedback actually delivered to the
model. Do not infer a missing tool response from the lightweight summary alone.

The smoke defaults are 1800 seconds for the agent, 2700 for a runtime RPC, and
5400 for the runner. These are shorter than the 2026-10-03 full evaluation's
12000 / 12900 / 18000 seconds (its independent evaluation budget was 3600 seconds).
Set `SWE_AGENT_TIMEOUT`, `SWE_RUNTIME_TIMEOUT`, and
`SWE_RUNNER_TIMEOUT` to choose the budget explicitly. The generator requires
`0 < agent < runtime < runner` and records the effective budgets in Langfuse.
Increasing the budget does not fix a waiting package installer or test process.

`SWE_PRE_AGENT_SCRIPT` optionally supplies one shell command to the agent's
`pre_start_scripts`, which runs after dataset initialization and before CLI
startup. It uses Nexus's command executor to preserve the task image's native
tools and shell setup. The shared SWE `task_config.pre_agent_script` executes
inside the worker's raw environment. Its Nix `LD_LIBRARY_PATH` can make the
image's own Node fail with a `GLIBC_2.38 not found` error; do not use that hook
for this dependency installation.
For an Axios-only smoke on the SWE-Bench-Multilingual image, restore dependencies
after its `git clean -fdx` removes the preinstalled `node_modules`:

```bash
export SWE_INSTANCE_IDS=axios__axios-4731
export SWE_PRE_AGENT_SCRIPT='timeout 180s npm install --no-audit --no-fund'
```

This installs the checked-out repository's dependencies; it does not install a
newer Axios release. Use a command appropriate to the selected dataset when
running other repositories. Keep preparation bounded and let failures propagate.
Without local dependencies, `npx mocha` can fetch a different Mocha version and
wait for an interactive installation confirmation.

Each `agent_config.pre_start_scripts` entry accepts `code`, `interpreter`
(`bash` or `python`), `timeout` (default 300 seconds), optional `cwd`, and `env`.
Preparation errors include command output and are classified as infrastructure
failures at the `pre_start` stage. `extra_env` configures the CLI process; the
upstream harness filters its tool environment, so it does not promise to forward
arbitrary variables into Bash. Use an explicit setup script to prepare task
dependencies and verify the actual tool environment when a task needs it.

The Axios HTTP suite can also finish reporting tests while leaving the process
alive. For manual diagnosis, use a bounded local Mocha invocation with `--exit`,
which still reports test failures with a nonzero exit code. If piping output,
enable `pipefail` so `tail` or `grep` cannot mask a test failure or timeout:

```bash
bash -o pipefail -c \
  'timeout 60s node_modules/.bin/mocha test/unit/adapters/http.js --exit 2>&1 | tee /tmp/axios-http-tests.log'
```

These are diagnostic commands. The official SWE evaluator keeps its own test
selection and scoring; a passing development test does not establish resolution.

## HyWay configuration

`hyway_generator.py` is a separate example pinned to `us.openai.gpt-6.1-sol`
with `high` effort on both the CLI and provider request. It uses the AIHub
`openai/v1/responses` route and preserves the credential's `provider=aws_third`
and `account_id2` suffix. Use credentials authorized for that route; the GLM
Chat Completions smoke configuration above uses a different endpoint.

Set `SWE_DATASET_PATH` and either `GONGFENG_RUNTIME_TOKEN` or `GONGFENG_TOKEN`.
The first nonempty value takes precedence. Configure npm and Langfuse credentials
as above; `SWE_DISABLE_LANGFUSE=true` disables tracking for this example.
`SWE_INSTANCE_IDS`, `SWE_SKIP`, and `SWE_LIMIT` select the cases. Validate a small
batch and the actual provider request before using the generator in a HyWay plan.

## Build and publish the npm artifacts

```bash
uv run --no-sync python docs/examples/configs/cb-insiders/package_npm.py \
  --deb-dir experiments/cb-insiders/debs \
  --output-dir experiments/cb-insiders/npm \
  --download
```

The builder checks both upstream checksums and inspects each generated tarball.
It writes a manifest containing artifact hashes and source identities. To publish,
set `CB_INSIDERS_NPM_PUBLISH_USERNAME` and `CB_INSIDERS_NPM_PUBLISH_TOKEN`, then
run the same command with `--publish` and the intended private `--registry`.
Publishing requires write access; the WorkBuddy read account alone is insufficient.
Only the original deb, package metadata, provenance and a short README enter each
npm artifact. Credentials and experiment results are excluded.

The two architectures are packaged and checksum-verified separately. Execution
on x64 does not establish runtime compatibility on arm64; use an arm64 Gongfeng
image and the arm64 package for that validation.
