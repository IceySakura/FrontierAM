# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `nexus/`. Put runtime providers, executors, and protocol glue under `nexus/runtime/`; agent, task, and tool integrations under `nexus/extensions/`; and CLI entrypoints under `nexus/cli/`. Keep user-facing runnable example configs in `docs/examples/configs/`. Tests are grouped by scope: `tests/unit/` for isolated logic, `tests/integration/` for local multi-component flows, `tests/e2e/` for external-service flows such as `agr_mobile`, and `tests/stress/` for heavier scenarios. Keep generated E2E artifacts in gitignored output folders such as `tests/e2e/agr_mobile_output/`.

## Build, Test, and Development Commands
Use `uv` to manage the environment and commands.

- `uv sync --extra dev`: install dev.
- `uv run nexus modules`： find extension dependencies.
- `uv run nexus modules install litellm`： install dependencies for extensions like litellm.
- `uv run ruff check nexus tests`: run lint and import-order checks.
- `uv run ruff format nexus tests`: format Python code.
- `uv run pytest tests/unit -q`: run unit tests.
- `uv run pytest tests/e2e -m e2e -q`: run E2E tests that depend on external services.
- `python tests/e2e/test_e2e_agr_mobile.py template_parallel_e2e`: run the AGR mobile template scenario directly.

## Coding Style & Naming Conventions
Target Python `3.12` through `3.14`. Follow the Ruff rules in `pyproject.toml`: 120-character lines, single quotes, spaces for indentation, and sorted imports. Use `snake_case` for modules, functions, and variables, and `PascalCase` for classes. Name tasks and tools with short imperative identifiers. Add comments only where protocol behavior or control flow is not obvious.

## Testing Guidelines
Write tests with `pytest`. Name files `test_*.py` and keep test names behavior-focused, for example `test_agr_mobile_existing_sandbox_parallel_e2e`. Prefer unit coverage first, then add E2E only for real integration boundaries. For local E2E runs, copy the example config under `tests/e2e/config/` and keep real credentials or sandbox IDs out of git.

When iterating on runtime-side Python code, prefer configs with `runtime_configs.<name>.worker_venv: true`. The runner uploads the current wheel into a fresh worker venv, so changes under `nexus/` take effect without rebuilding the base `nexus-runtime` Docker image. Rebuild the image only when changing `deployment/runtime`, base runtime dependencies, or image-level system packages.

## Commit & Pull Request Guidelines
Recent commits use short imperative English subjects such as `Add agr mobile runtime and e2e coverage` or `feat: add ...`. Keep commits focused and logically isolated. A pre-commit hook runs `make check` (ruff format + ruff lint + ty typecheck) on every commit; fix any reported issues before committing, or bypass with `git commit --no-verify` only in emergencies. Pull requests should describe the user-visible change, list validation commands, note any config or secret handling, and include screenshots or output paths when GUI or mobile behavior changes.

Do not commit temporary implementation plans to the repository. If a design or spec must be kept in-tree, place it under `docs/developing/`.

## Security & Configuration Tips
Do not commit secrets, `.env` files, or customer sandbox identifiers. Store local credentials in ignored config files and keep only sanitized examples in the repo. When adding a provider or E2E flow, document required environment variables, external dependencies, and cleanup behavior.

## Unified Log Level Selection Criteria

Choose the level based on the complete event and purpose of a log message, not isolated words such as "success" or "result". Use the levels below for new logs. For existing logs classified as INFO, preserve their level, content, and frequency; do not remove them because multiple layers report the same outcome. Keep existing WARNING/ERROR logs at their current levels.

The entire `nexus/distributed/` package is excluded from this logging adjustment. Do not change its log levels, content, frequency, handlers, or output mechanisms, including indirect changes through shared logging configuration in other packages.

| Log semantics | Level |
|---|---|
| **Call payloads and raw content**: Model prompts, message history, reasoning text, and full answers; tool arguments and response bodies; raw HTTP requests/responses, SSE events, JSON content, and line-by-line echoes of normal stdout from external Agents. These logs support input/output debugging. Classify separate completion, scoring, or exception records using the corresponding rules below. | DEBUG |
| **Normal execution of individual requests and loop iterations**: The start of each iteration; dispatch, routing, execution, and normal success of model/tool/HTTP/RPC/database requests; per-request duration, token usage, cache hits/writes, thinking/reasoning cache injection, and deduplicated replay. Examples include `invoke: executing`, `LLM usage`, and `Langfuse Step`. This excludes overall Task/Agent completion and evaluation conclusions. | DEBUG |
| **Expected configuration, environment, and resource setup/cleanup**: Reading/merging configuration, listing registered modules and tools, reusing installed dependencies or caches, copying/uploading/downloading input files, mapping ports, and normal creation, readiness, and release of Runtimes, Proxies, and kernels. Final artifact locations, actual failures, degradation, and recovery are not routine setup details. | DEBUG |
| **Logs used only for progress display or formatting**: Separator lines, banners, phase headings, iteration numbers, per-file/shard progress, download percentages, and repeated completion notices for the same routine initialization step that carry no independent business outcome or exception information. Do not classify business completion or scoring logs from multiple layers this way merely because they are repetitive. | DEBUG |
| **Meaningful completion and scoring conclusions, including repeated records across layers**: Task, Agent, sub-Agent, turn, or Tracker `done/completed/finalized` records, final status, and execution statistics; verdict/pass for each rubric/check, P/R/F1 components, reward/score, validation results, and final summaries. Preserve INFO for component scores, totals, and existing records at every layer. | INFO |
| **Shared logging statements for normal and exceptional results**: The same statement or shared result callback emits fields such as `success`, `exit_code`, `exit_status`, `status`, `error`, or `is_error`, whose values vary with the execution outcome. Keep the entire record at INFO even when execution succeeds, the exit code is 0, or error is empty. If the record also contains body content, do not separately downgrade the success branch or that content. | INFO |
| **State events explaining termination, missing input, permissions, or changes in execution behavior**: Actual timeouts; exhausted token/iteration/cost budgets; missing required input/result files; user approval or rejection; interactive pauses; retry failures and recovery; actual fallback or compatibility degradation. Merely echoing `timeout=120` or reporting that fallback support is configured remains a setup detail. Keep existing WARNING/ERROR logs at their current levels. | INFO |
| **User-facing operation results and final artifact locations**: Necessary CLI operation feedback and command results; final batch success/failure/timeout statistics; and final names, IDs, URIs, or file paths of generated or published deliverables, evaluation reports, recordings, container images, and similar artifacts. Routine input-file transfers, temporary files, and intermediate screenshots from each iteration do not become final-artifact logs merely because they include paths. | INFO |
