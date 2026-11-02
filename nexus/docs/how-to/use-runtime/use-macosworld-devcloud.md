# macOSWorld on DevCloud

macOSWorld reuses the shared `devcloud` provider and its current Sandbox API lifecycle.
It selects its execution client and bridge through `runtime_kind: macosworld_devcloud`;
there is no separate provider registration. See [DevCloud macOS](use-devcloud-macos.md)
for lifecycle configuration and [adapter boundaries](../../developing/macosworld-adapters.md)
for agent and client registrations.

## Runnable configurations

Use the YAMLs in `docs/examples/configs/mac_os_world/`:

- `mac_os_world_claude_code_hunyuan_devcloud.yaml`: Claude Code installed in the Mac, HY upstream.
- `mac_os_world_codex_hunyuan_devcloud.yaml`: Codex installed in the Mac, HY upstream.
- `mac_os_world_gpt4o_paper_devcloud.yaml`: ordinary visual model with the upstream GPT-4o action parser.

The native harnesses run in the Mac. Their model traffic goes through the
sandbox-local bridge and host model client. Codex's Responses-to-Chat adaptation
is selected by `macosworld_codex_native_bridge`; it is not enabled for ordinary Codex.

```bash
uv run nexus run single \
  --use docs/examples/configs/mac_os_world/mac_os_world_codex_hunyuan_devcloud.yaml \
  --data macosworld/macosworld@v1.2.0 --split test \
  --id 0e224b29-14f7-3629-ee27-13a9789a997d -o outputs/macosworld/result.json
```

Provide credentials via the environment variables referenced by the chosen YAML.
Never commit populated credentials, instance identifiers or local result files.
HY platform evaluations using `ONE_STOP_SERVICE` retain platform-managed routing;
do not overwrite it with a local SDK endpoint or token.

## Explicit runtime selection

```yaml
runtime_provider_name: devcloud
runtime_provider_config:
  app_id: ${oc.env:DEVCLOUD_APP_ID}
  app_token: ${oc.env:DEVCLOUD_APP_TOKEN}
  sandbox_api_base_url: ${oc.env:DEVCLOUD_SANDBOX_API_BASE_URL}
  proxy_url: ${oc.env:DEVCLOUD_PROXY_URL,null}
  execute_mode: async
runtime_configs:
  main:
    runtime_kind: macosworld_devcloud
```

This is a configuration fragment, not a complete RunInput. Use the runnable YAML
for imports, runtime kind, model settings, time budgets and task configuration.

The explicit proxy covers sandbox management, command/screenshot/file traffic,
bridge upload, health checks and WebSocket transport. It does not change process
proxy environment variables. Without an explicit proxy, the shared transports retain their existing connection behavior.

Proxy propagation and asynchronous execution are shared DevCloud options; they do not require
the macOSWorld adapter. Asynchronous execution submits a uniquely identified background command through
`/execute` and polls its result through `/file`. Read-only polling tolerates
transient incomplete results within a fixed deadline. An uncertain submission is
not replayed, because doing so could duplicate a desktop action. Host cancellation
does not establish that the remote command has stopped.

## Initialization, evaluator and cleanup

`task_config.prepare_screen_capture: true` handles only the screen-recording
permission dialog attributed to `GUICloneVMService`, before the official task
setup. It is not a general dialog dismissal feature and must not dismiss task or
safety dialogs. A failed preparation remains an environment failure.

Use the official evaluator after agent execution. Effective zero scores and
infrastructure errors remain distinct. Screenshot resizing does not change the
actual desktop layout or language. The model version and harness choice must be
reported alongside results; a successful integration is not proof of paper-score
reproduction. See [paper alignment](../use-agent/macosworld-paper-alignment.md).

Each run records its sandbox ownership and requests release after evaluation and
artifact capture. Delete acceptance is not release confirmation: the provider
polls the returned task identifier. A timed-out delete submission with no task ID
is not blindly replayed. Record unresolved release state; do not delete a reused
instance based solely on an old IP or instance ID.
