# macOSWorld adapter isolation

macOSWorld selects its adaptations explicitly. An OS name, model name, import,
or use of the shared CUA/CC/Codex agent does not enable these adaptations.
The shared input/output schemas and default execution policies preserve the main baseline
`9628c6dac80ec427a8563fecb74421fa77ab2b42`. Explicit proxy and asynchronous
execution options are shared DevCloud capabilities, independent of benchmark selection.

Runtime setup and cleanup are documented in the [macOSWorld DevCloud guide](../how-to/use-runtime/use-macosworld-devcloud.md).

## Entry points and configuration migration

| Layer | Shared entry | macOSWorld entry |
| --- | --- | --- |
| Runtime provider | `devcloud` | Same `devcloud` provider; `runtime_kind: macosworld_devcloud` selects the execution client |
| GPT CUA agent | `cua_gpt_agent` | `macosworld_cua_gpt` |
| Claude CUA agent | `cua_anthropic_agent` | `macosworld_cua_anthropic` |
| Native Codex agent | `codex_native_bridge` | `macosworld_codex_native_bridge` |
| Native Claude Code agent | `claude_code_native_bridge` | Same agent, with `macosworld_devcloud` runtime |
| Chat client | `openai` | `macosworld_openai` |
| Responses client | `openai_responses` | `macosworld_openai_responses` |
| HY client | `infer_proxy` | `macosworld_infer_proxy` |
| Claude Messages client | `anthropic_messages` | `macosworld_anthropic_messages` |

The macOSWorld generator and runnable YAMLs select these registrations. When
migrating a platform evaluation version, update `runtime_provider_name`, the
runtime configuration's `runtime_kind`, `agent_name`, client names and imports
alongside the pinned code commit. Update `nexus_main_client_name` in platform
parameters as well; a task-level override must not replace the selected client
with the old name. Do not change only the commit while keeping old entry names.
Existing platform versions pinned to older commits remain unchanged.

For HY `ONE_STOP_SERVICE`, keep the platform-managed model routing and client
when the source version has no explicit client override. Do not replace that
route with a local InferProxy URL or credentials. The client-name migration above
applies to explicitly configured SDK clients; the runtime and harness adapters
also work with the platform's existing bridge client.

Dataset rows, model routing, model source, sampling settings, budgets, and the
original evaluator are not changed by this refactor. The `devcloud` provider now
uses the current Sandbox API for all callers. Old macOSWorld configurations must
change `runtime_provider_name: macosworld_devcloud` to `devcloud`; keep the runtime kind.
Old lifecycle identity fields are ignored with a warning; the server must support the new API.

## Boundaries

The macOSWorld CUA adapters and screenshot-based VLM agent live under
`nexus/extensions/agents/cua/macosworld_cua/` and
`nexus/extensions/agents/cua/macosworld_vlm/`. Their registrations and host
dependencies are declared by the existing `cua` module (`nexus modules install cua`).
Agent registration names remain unchanged; configurations using explicit Python
imports must use these nested paths when upgrading to this layout. Benchmark-specific
unit tests are grouped under `tests/unit/mac_os_world/`.
The native Codex adapter belongs to `nexus/extensions/agents/codex/macosworld/`
and is registered by the `codex` module. Runnable examples and platform parameter
templates are collected in `docs/examples/configs/mac_os_world/`; task source
files do not contain deployment presets. Shared agent tests stay in their original
directories and do not import macOSWorld implementations.

- CUA adapters in `nexus/extensions/agents/cua/macosworld_cua/` select prompt,
  action translation, termination, image history and diagnostics. They reuse the
  shared agent loops through default-preserving callbacks. Budgets, screenshot
  history, tool restrictions and diagnostic state live entirely in the adapters.
  Policy objects are created per run; no global monkey patch
  or environment-variable switch selects behavior.
- `nexus/runtime/providers/devcloud/provider.py` owns the single Sandbox API lifecycle
  and bounded release confirmation. `nexus/runtime/providers/macosworld_devcloud/`
  contains only execution/bridge adaptations, selected explicitly by runtime kind.
  Screenshot validation/retries and strict execute-error policies belong to this runtime.
  HTTP/file proxy propagation and detached execution live in the shared DevCloud client;
  the adapter inherits them. The worker and polling implementation exist only once.
  Actions with uncertain submission results are not replayed. Unknown deletion
  state is reported as unknown, never as confirmed release.
- The runtime provides a bridge adapter factory. macOSWorld's bridge adapter only adds
  the private protocol overlay to the shared DevCloud bridge. Explicit proxy upload,
  readiness and host transport are shared capabilities. BridgeHostClient owns a separate
  session only when proxy_url is set; without it, existing path prefixes, polling/upgrade
  transport, keepalive and connection behavior are retained. Other adapters do not opt in
  automatically, and no global session or environment variable is changed.
- Native Codex selects `responses-chat` through the macOSWorld agent input.
  The host adapter creates a private overlay archive containing a standalone
  `macosworld_bridge` guest package and selects its launcher. The shared archive
  is never modified. Guest code lives in `macosworld_devcloud/bridge_guest/`;
  host overlay adapter and GUI helpers also live under this adapter package, outside
  the shared bridge's source fingerprint. The shared app only exposes an optional
  converter callback and does not import a bench-specific module. Its CLI still
  accepts only the original `anthropic` and `responses` protocols.
- Client adapters sanitize copied configuration, preserving enforced request
  arguments. Extra Anthropic tool types and HY image normalization do not alter
  the ordinary clients' default validation or requests.
- Internal operation timeout diagnostics are raised at the macOSWorld Task
  boundary. The global runner's timeout/error contract is unchanged.
- Native bridge host dependencies are declared by the macOSWorld task extension;
  ordinary CC/Codex module dependency declarations are unchanged.

## Verification

Regression coverage includes the shared DevCloud Sandbox API lifecycle and cleanup,
ordinary versus macOSWorld execution-client selection, concurrent shared/adapted Claude runs, GPT prompt/action
selection, unchanged shared schemas, proxy HTTP/WebSocket traffic, image
conversion, Codex protocol selection, client requests, task timeout ownership,
and configuration/module registration. The isolated bundle test constructs each
protocol's application without the source checkout. Importing only the app module
would miss lazy imports of host-only adapter dependencies. Unit tests use mocked
model/sandbox APIs or local HTTP/WebSocket servers; they do not establish
real-platform success.

A platform rollout still needs a new evaluation version with the explicit entry
names and a small end-to-end run before reusing full-scale settings.

## Remaining shared extension points and impact

| Shared file | Purpose | Default-path contract |
| --- | --- | --- |
| `agents/codex/agent.py` | Reuse native execution with explicit protocol/error callbacks | Original registry uses Responses and no error formatter |
| `agents/cua/cua_gpt_agent/agent.py` | Per-run prompt/action/request/observation/output callbacks | Original prompt, request history, action execution, output schema and termination; no Mac policy settings |
| `agents/cua/cua_anthropic_agent/agent.py` | Per-run geometry/tools/action/result callbacks | Original geometry, tools, parser/executor, image truncation and output; no Mac budgets or tool restrictions |
| `agents/installed_common/native_bridge.py` | Runtime adapter and host-client factories | Without a factory, original runtime mapping and host client |
| `openai_clients/anthropic_messages.py` | Overridable payload builder | Delegates to the original builder; Vertex and its allowlist are unchanged |
| `runtime/providers/devcloud/client.py` | Optional proxy and asynchronous OSWorld execution | No proxy and sync by default; original command/screenshot error policy |
| `runtime/bridge/adapters/devcloud_macos.py` | Proxy upload, readiness and host-client selection | Unset proxy preserves previous upload and health routing |
| `runtime/bridge/host_client.py` | Explicit proxy WebSocket transport | Unset proxy preserves polling/upgrade and URL prefix; sessions remain per instance |
| `runtime/bridge/adapters/base.py` | Health-request options hook | Returns an empty dict; original polling and identity checks |
| `runtime/bridge/app.py` | Explicit Responses converter injection | Omitted callback keeps the original converter and accepted protocols |

These interfaces are justified by reuse of existing execution, authentication and
lifecycle code. They do not automatically select an adapter by OS/model name.
Objects and policy state are per invocation, not mutable globals. Tests cover
shared/adapted calls together, original schemas, model payloads and private bundle
startup without the checkout. Future edits to private guest files do not alter
ordinary bridge cache keys; changing the shared app hook itself necessarily does.

This is compatibility evidence, not a guarantee covering every external bench.
The real-device regression covers one selected task/harness; unit tests cover CUA
policy boundaries separately. Do not infer full-suite or platform validation from it.
