# Codex SWE protocol checks

Run one fixed SWE sample through the legacy client or Gateway policy, then
validate the exported JSONL and SQLite spans. Gateway checks also validate the
canonical trajectory, tool identities, compaction boundaries, and subagent chains.

From the Nexus repository root, set `OPENAI_API_KEY`, `GONGFENG_RUNTIME_TOKEN`,
`SWE_INSTANCE_IDS` (one instance ID), and `SWE_TRACE_OUTPUT_DIR`, then run:

```bash
CODEX_E2E_ROUTE=gateway CODEX_VERSION=0.147.0 \
  bash tests/e2e/codex/run_swe.sh
```

Use `CODEX_E2E_ROUTE=legacy` for the direct client. Optional
`CODEX_REQUIRE_COMPACTION=true` and `CODEX_REQUIRE_MULTI_AGENT=true` require
evidence of those operations; multi-agent checks require Codex 0.146 or newer.
`SWE_MODEL` and `OPENAI_BASE_URL` select the model service. Artifacts are written
under `$SWE_TRACE_OUTPUT_DIR/$CODEX_VERSION/` and should remain outside Git.

The launcher uses `uv run` for Nexus and `PYTHON_BIN` (default `python`) for the
checker. Use an interpreter with Nexus dependencies installed. It enables the
current Nexus worker venv and runs one sandbox at a time. These checks call real
model and sandbox services; they do not run HyTuner training.
