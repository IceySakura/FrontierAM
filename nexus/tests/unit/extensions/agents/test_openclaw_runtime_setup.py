import asyncio
import json
import os
import signal
import sys

from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.extensions.agents.openclaw import runtime_setup
from nexus.extensions.agents.openclaw.runtime_setup import (
    CURL_FALLBACK_PATCH_MARKER,
    DEFAULT_GATEWAY_AUTH_TOKEN,
    DEFAULT_LLM_IDLE_TIMEOUT_MS,
    DEFAULT_PROVIDER_ID,
    NEXUS_BRAVE_API_KEY,
    WEB_SEARCH_ENDPOINT_PATCH_MARKER,
    WebSearchCompat,
    _build_openclaw_env,
    _patch_full_config,
    _patch_gateway_config,
    _patch_model_config,
    _patch_web_fetch_curl_fallback,
    _patch_web_search_bridge_endpoint,
    _prepare_web_search_compat,
    _read_transcript,
    _start_web_search_bridge_for_openclaw,
    build_openclaw_config,
    classify_openclaw_response,
    parse_openclaw_response,
)


def _stub_openclaw_gateway(monkeypatch) -> None:
    """Gateway start/stop need a real node process; unit tests no-op them."""

    async def _fake_start(**kwargs):
        return None

    async def _fake_stop(proc) -> None:
        return None

    monkeypatch.setattr(runtime_setup, '_start_openclaw_gateway', _fake_start)
    monkeypatch.setattr(runtime_setup, '_stop_openclaw_gateway', _fake_stop)


def _write_openclaw_config(config_path: Path, *, model: str = 'bedrock/claude-opus') -> None:
    config_path.write_text(
        json.dumps(
            {
                'models': {
                    'providers': {
                        DEFAULT_PROVIDER_ID: {
                            'models': [
                                {
                                    'id': model,
                                    'name': model,
                                    'contextWindow': 16_000,
                                    'maxTokens': 4_096,
                                    'requestTimeoutMs': 120_000,
                                }
                            ]
                        }
                    }
                }
            }
        ),
        encoding='utf-8',
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('failure_point', 'expected_stage'),
    [
        ('install', 'install'),
        ('node', 'runtime_dependency'),
        ('entrypoint', 'entrypoint'),
        ('proxy', 'proxy'),
    ],
)
async def test_setup_failures_are_classified_as_infrastructure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_point: str,
    expected_stage: str,
) -> None:
    openclaw_bin = tmp_path / 'prefix' / 'node_modules' / '.bin' / 'openclaw'
    js_entry = tmp_path / 'prefix' / 'node_modules' / 'openclaw' / 'entry.js'
    js_entry.parent.mkdir(parents=True)
    js_entry.touch()

    async def fake_install_openclaw(**kwargs):
        if failure_point == 'install':
            raise RuntimeError('install failed')
        return openclaw_bin

    class FailingProxy:
        def __init__(self, **kwargs):
            pass

        async def start(self):
            raise RuntimeError('proxy failed')

    monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: SimpleNamespace(sio=object()))
    monkeypatch.setattr(runtime_setup, 'install_openclaw', fake_install_openclaw)
    monkeypatch.setattr(runtime_setup, '_patch_web_fetch_ssrf', lambda path: None)
    monkeypatch.setattr(runtime_setup, '_patch_web_fetch_ua', lambda path: None)
    monkeypatch.setattr(runtime_setup, '_patch_web_fetch_curl_fallback', lambda path: None)
    monkeypatch.setattr(
        runtime_setup,
        'find_nexus_runtime_binary',
        lambda name: None if failure_point == 'node' else tmp_path / name,
    )
    monkeypatch.setattr(
        runtime_setup,
        'resolve_npm_bin_entry',
        lambda *args: None if failure_point == 'entrypoint' else js_entry,
    )
    monkeypatch.setattr(runtime_setup, '_get_openclaw_proxy_class', lambda: FailingProxy)

    result = await runtime_setup.setup_and_run_openclaw(
        task='solve task',
        model='test-model',
        working_dir=str(tmp_path / 'workspace'),
    )

    assert result['success'] is False
    assert result['exit_code'] == -1
    assert result['failure_kind'] == 'infrastructure'
    assert result['failure_stage'] == expected_stage


@pytest.mark.parametrize(
    ('stdout', 'exit_code', 'expected_status', 'expected_failure'),
    [
        (json.dumps({'payloads': [{'text': 'visible answer'}]}), 0, 'completed', None),
        (
            json.dumps(
                {
                    'payloads': [
                        {'text': 'private reasoning', 'isReasoning': True},
                        {'text': 'visible answer'},
                        {'text': 'provider error', 'isError': True},
                    ]
                }
            ),
            0,
            'completed',
            None,
        ),
        ('', 0, 'error', 'OpenClaw returned no valid ReplyPayload'),
        ('not json', 0, 'error', 'OpenClaw returned no valid ReplyPayload'),
        (json.dumps({'meta': {}}), 0, 'error', 'OpenClaw ReplyPayload has no payloads list'),
        (json.dumps({'payloads': []}), 0, 'error', 'OpenClaw returned no visible non-error reply'),
        (
            json.dumps({'payloads': [{'text': 'private reasoning', 'isReasoning': True}]}),
            0,
            'error',
            'OpenClaw returned no visible non-error reply',
        ),
        (
            json.dumps({'payloads': [{'text': 'provider error', 'isError': True}]}),
            0,
            'error',
            'OpenClaw returned no visible non-error reply',
        ),
        (
            json.dumps({'payloads': [{'text': 'answer'}], 'meta': {'aborted': True}}),
            0,
            'error',
            'OpenClaw reported an aborted embedded run',
        ),
        (
            json.dumps({'payloads': [{'text': 'answer'}], 'meta': {'error': 'provider unavailable'}}),
            0,
            'error',
            'OpenClaw reported an error: provider unavailable',
        ),
        (
            json.dumps({'payloads': [{'text': '422 status code (no body)'}], 'meta': {'stopReason': 'error'}}),
            0,
            'error',
            "OpenClaw stopped with reason 'error'",
        ),
        (
            json.dumps({'payloads': [{'text': 'partial'}], 'meta': {'stopReason': 'context_overflow'}}),
            0,
            'truncation',
            "OpenClaw stopped with reason 'context_overflow'",
        ),
        (
            json.dumps({'payloads': [{'text': 'partial'}], 'meta': {'stopReason': 'timeout'}}),
            0,
            'timeout',
            "OpenClaw stopped with reason 'timeout'",
        ),
        (json.dumps({'payloads': [{'text': 'answer'}]}), 1, 'error', None),
    ],
)
def test_classify_openclaw_response_requires_a_visible_successful_reply(
    stdout,
    exit_code,
    expected_status,
    expected_failure,
) -> None:
    assert classify_openclaw_response(stdout, exit_code) == (expected_status, expected_failure)


def test_parse_openclaw_response_finds_envelope_after_diagnostic_objects() -> None:
    stdout = '\n'.join(
        [
            'diagnostic {"level":"warn","message":"loading"}',
            json.dumps({'payloads': [{'text': 'visible answer'}], 'meta': {'stopReason': 'stop'}}),
        ]
    )

    assert parse_openclaw_response(stdout) == {
        'payloads': [{'text': 'visible answer'}],
        'meta': {'stopReason': 'stop'},
    }


@pytest.mark.parametrize(
    'stdout',
    [
        json.dumps({'payloads': [], 'meta': None}),
        json.dumps({'payloads': [], 'meta': []}),
        json.dumps({'payloads': [], 'meta': {'agentMeta': None}}),
    ],
)
def test_read_transcript_tolerates_malformed_meta(stdout: str, tmp_path: Path) -> None:
    assert _read_transcript(stdout, tmp_path) == []


def test_build_openclaw_config_sets_default_llm_idle_timeout() -> None:
    config = build_openclaw_config(model='bedrock/claude-opus', working_dir='/workspace', proxy_port=18789)

    provider = config['models']['providers'][DEFAULT_PROVIDER_ID]
    model_entry = provider['models'][0]
    assert provider['timeoutSeconds'] == DEFAULT_LLM_IDLE_TIMEOUT_MS // 1000
    assert 'requestTimeoutMs' not in model_entry


def test_patch_gateway_config_enables_local_loopback(tmp_path: Path) -> None:
    config_path = tmp_path / 'openclaw.json'
    _write_openclaw_config(config_path)

    _patch_gateway_config(config_path, port=19876)

    config = json.loads(config_path.read_text(encoding='utf-8'))
    assert config['gateway'] == {
        'mode': 'local',
        'port': 19876,
        'bind': 'loopback',
        'auth': {'mode': 'token', 'token': DEFAULT_GATEWAY_AUTH_TOKEN},
    }


def test_patch_model_config_sets_default_provider_timeout(tmp_path: Path) -> None:
    config_path = tmp_path / 'openclaw.json'
    _write_openclaw_config(config_path)

    _patch_model_config(
        config_path=config_path,
        provider_id=DEFAULT_PROVIDER_ID,
        model='bedrock/claude-opus',
        context_window=200_000,
        max_tokens=65_536,
    )

    config = json.loads(config_path.read_text(encoding='utf-8'))
    provider = config['models']['providers'][DEFAULT_PROVIDER_ID]
    model_entry = provider['models'][0]
    assert provider['timeoutSeconds'] == DEFAULT_LLM_IDLE_TIMEOUT_MS // 1000
    assert model_entry['contextWindow'] == 200_000
    assert model_entry['maxTokens'] == 65_536
    assert 'requestTimeoutMs' not in model_entry


def test_patch_model_config_accepts_custom_provider_timeout(tmp_path: Path) -> None:
    config_path = tmp_path / 'openclaw.json'
    _write_openclaw_config(config_path)

    _patch_model_config(
        config_path=config_path,
        provider_id=DEFAULT_PROVIDER_ID,
        model='bedrock/claude-opus',
        context_window=200_000,
        max_tokens=65_536,
        llm_idle_timeout_ms='900000',
    )

    config = json.loads(config_path.read_text(encoding='utf-8'))
    provider = config['models']['providers'][DEFAULT_PROVIDER_ID]
    assert provider['timeoutSeconds'] == 900


def test_patch_model_config_rounds_timeout_up_to_provider_seconds(tmp_path: Path) -> None:
    config_path = tmp_path / 'openclaw.json'
    _write_openclaw_config(config_path)

    _patch_model_config(
        config_path=config_path,
        provider_id=DEFAULT_PROVIDER_ID,
        model='bedrock/claude-opus',
        context_window=200_000,
        max_tokens=65_536,
        llm_idle_timeout_ms=1_500,
    )

    config = json.loads(config_path.read_text(encoding='utf-8'))
    provider = config['models']['providers'][DEFAULT_PROVIDER_ID]
    assert provider['timeoutSeconds'] == 2


def test_patch_model_config_ignores_non_object_model_entries(tmp_path: Path) -> None:
    config_path = tmp_path / 'openclaw.json'
    config_path.write_text(
        json.dumps(
            {
                'models': {
                    'providers': {
                        DEFAULT_PROVIDER_ID: {
                            'models': [
                                'bedrock/claude-opus',
                                {
                                    'id': 'bedrock/claude-opus',
                                    'name': 'bedrock/claude-opus',
                                },
                            ]
                        }
                    }
                }
            }
        ),
        encoding='utf-8',
    )

    _patch_model_config(
        config_path=config_path,
        provider_id=DEFAULT_PROVIDER_ID,
        model='bedrock/claude-opus',
        context_window=200_000,
        max_tokens=65_536,
        llm_idle_timeout_ms=900_000,
    )

    config = json.loads(config_path.read_text(encoding='utf-8'))
    provider = config['models']['providers'][DEFAULT_PROVIDER_ID]
    assert provider['timeoutSeconds'] == 900
    assert provider['models'][0] == 'bedrock/claude-opus'
    assert provider['models'][1]['contextWindow'] == 200_000


def _write_openclaw_install(tmp_path: Path, *, version: str | None = None) -> Path:
    """Lay out a minimal ``node_modules/openclaw`` tree and return its CLI entry point."""
    package_dir = tmp_path / 'node_modules' / 'openclaw'
    (package_dir / 'dist').mkdir(parents=True)
    (package_dir / 'bin').mkdir()
    if version is not None:
        (package_dir / 'package.json').write_text(json.dumps({'version': version}), encoding='utf-8')
    binary = package_dir / 'bin' / 'openclaw.mjs'
    binary.write_text('#!/usr/bin/env node\n', encoding='utf-8')
    return binary


@pytest.mark.parametrize('preexisting', [False, True], ids=['fresh', 'stale-key'])
def test_patch_model_config_omits_provider_timeout_on_releases_that_reject_it(
    tmp_path: Path, preexisting: bool
) -> None:
    """Releases up to 2026.4.25 fail config validation on ``timeoutSeconds``, so the gateway never starts."""
    config_path = tmp_path / 'openclaw.json'
    _write_openclaw_config(config_path)
    if preexisting:
        stale = json.loads(config_path.read_text(encoding='utf-8'))
        stale['models']['providers'][DEFAULT_PROVIDER_ID]['timeoutSeconds'] = 600
        config_path.write_text(json.dumps(stale), encoding='utf-8')

    _patch_model_config(
        config_path=config_path,
        provider_id=DEFAULT_PROVIDER_ID,
        model='bedrock/claude-opus',
        context_window=200_000,
        max_tokens=65_536,
        openclaw_version='2026.3.13',
    )

    config = json.loads(config_path.read_text(encoding='utf-8'))
    provider = config['models']['providers'][DEFAULT_PROVIDER_ID]
    assert 'timeoutSeconds' not in provider
    assert provider['models'][0]['contextWindow'] == 200_000
    assert provider['models'][0]['maxTokens'] == 65_536


def test_patch_model_config_keeps_provider_timeout_on_supported_releases(tmp_path: Path) -> None:
    config_path = tmp_path / 'openclaw.json'
    _write_openclaw_config(config_path)

    _patch_model_config(
        config_path=config_path,
        provider_id=DEFAULT_PROVIDER_ID,
        model='bedrock/claude-opus',
        context_window=200_000,
        max_tokens=65_536,
        openclaw_version='2026.5.28',
    )

    config = json.loads(config_path.read_text(encoding='utf-8'))
    assert config['models']['providers'][DEFAULT_PROVIDER_ID]['timeoutSeconds'] == DEFAULT_LLM_IDLE_TIMEOUT_MS // 1000


def test_read_openclaw_package_version_reads_the_installed_package(tmp_path: Path) -> None:
    binary = _write_openclaw_install(tmp_path, version='2026.3.13')

    assert runtime_setup._read_openclaw_package_version(binary) == '2026.3.13'


def test_read_openclaw_package_version_returns_none_when_the_package_is_missing(tmp_path: Path) -> None:
    assert runtime_setup._read_openclaw_package_version(tmp_path / 'openclaw') is None


def test_openclaw_version_key_orders_the_measured_releases() -> None:
    """Every version below was measured in the sandbox matrix, in this order.

    The gates are thresholds over this ordering, so getting it right is what makes them
    correct -- including the beta ordinals, since the wait-abort boundary falls mid-series.
    """
    ordered = [
        '2026.3.11',
        '2026.3.13',
        '2026.4.25',  # last release to reject provider timeoutSeconds
        '2026.4.26-beta.1',  # first to accept it; adjacent to 2026.4.25 in the registry
        '2026.5.12',
        '2026.5.28',
        '2026.6.1-beta.3',
        '2026.6.1',
        '2026.6.2-beta.1',
        '2026.6.5-beta.1',
        '2026.6.5-beta.2',
        '2026.6.5-beta.3',
        '2026.6.5-beta.6',
        '2026.6.5',
        '2026.6.6',
        '2026.6.34',
        '2026.7.1-beta.6',
        '2026.9.2',
    ]
    keys = [runtime_setup._openclaw_version_key(v) for v in ordered]

    assert all(key is not None for key in keys)
    assert keys == sorted(keys)


@pytest.mark.parametrize(
    ('version', 'timeout_seconds', 'wait_abort'),
    [
        ('2026.3.13', False, False),
        # 2026.4.25 is the last release to reject provider timeoutSeconds, 2026.4.26-beta.1 the
        # first to accept it, and the registry has nothing between them.
        ('2026.4.25', False, False),
        ('2026.4.26-beta.1', True, False),
        ('2026.5.12', True, False),
        # 2026.6.5-beta.2 is the last release that drains its subagents on its own; from beta.3
        # on the CLI hangs on agent.wait instead, so the boundary sits between these two.
        ('2026.6.5-beta.2', True, False),
        ('2026.6.5-beta.3', True, True),
        ('2026.6.5', True, True),
        # 2026.7.1-beta.6 reshaped the registry's completion path; it is in the matrix because it
        # is the newest release nexus' default node can still run, so the anchor is checked there.
        ('2026.7.1-beta.6', True, True),
        # A missing or unparsable version takes both fixes: withholding one costs more.
        (None, True, True),
        ('nonsense', True, True),
    ],
)
def test_version_gates_match_the_measured_behaviour(
    version: str | None, timeout_seconds: bool, wait_abort: bool
) -> None:
    assert runtime_setup._supports_provider_timeout_seconds(version) is timeout_seconds
    assert runtime_setup._needs_subagent_wait_abort(version) is wait_abort


def test_patch_subagent_wait_abort_leaves_old_bundles_untouched(tmp_path: Path) -> None:
    binary = _write_openclaw_install(tmp_path)
    bundle = binary.parent.parent / 'dist' / 'run-wait-abc123.js'
    original = 'async function waitForAgentRun(params) {\n\t\t\tmethod: "agent.wait",\n}\n'
    bundle.write_text(original, encoding='utf-8')

    runtime_setup._patch_subagent_wait_abort(binary, '2026.5.28')

    assert bundle.read_text(encoding='utf-8') == original


_RUN_WAIT_BUNDLE = """async function waitForAgentRun(params) {
\ttry {
\t\treturn await callGateway({
\t\t\tmethod: "agent.wait",
\t\t\trunId: params.runId,
\t\t});
\t} catch (err) {
\t\tconst error = formatErrorMessage(err);
\t\treturn { status: "error", error };
\t}
}
"""

# The three shapes OpenClaw's completion path has taken, reduced to the lines the patch
# depends on.  2026.7 wrapped the body in a `try` (re-indenting the anchor) and 2026.8 moved
# it into a free function, which is why the marker cannot include the enclosing declaration.
_REGISTRY_BUNDLES = {
    '2026.6.x-inline-arrow': """\tconst completeSubagentRun = async (completeParams) => {
\t\tparams.clearPendingLifecycleError(completeParams.runId);
\t\tconst entry = params.runs.get(completeParams.runId);
\t\tif (!entry) return;
\t};
""",
    '2026.7.x-inner-try': """\tconst completeSubagentRun = async (completeParams) => {
\t\tconst releaseCompletionLock = await acquireTerminalCompletionLock(completeParams.runId);
\t\tlet entry;
\t\ttry {
\t\t\tparams.clearPendingLifecycleError(completeParams.runId);
\t\t\tentry = params.runs.get(completeParams.runId);
\t\t\tif (!entry) return;
\t\t} finally {
\t\t\treleaseCompletionLock();
\t\t}
\t};
""",
    '2026.8.x-free-function': """async function completeSubagentRunAttempt(context, completeParams) {
\tconst params = context.options;
\tlet entry;
\ttry {
\t\tentry = params.runs.get(completeParams.runId);
\t\tif (!entry) return;
\t\tif (completeParams.expectedEntry && entry !== completeParams.expectedEntry) return;
\t\tparams.clearPendingLifecycleError(completeParams.runId);
\t\tconst currentEntry = entry;
\t} finally {}
}
""",
}


@pytest.mark.parametrize('shape', sorted(_REGISTRY_BUNDLES), ids=sorted(_REGISTRY_BUNDLES))
def test_patch_subagent_wait_abort_lands_on_every_registry_shape(tmp_path: Path, shape: str) -> None:
    binary = _write_openclaw_install(tmp_path)
    dist = binary.parent.parent / 'dist'
    run_wait = dist / 'run-wait-abc123.js'
    registry = dist / 'subagent-registry-xyz789.js'
    run_wait.write_text(_RUN_WAIT_BUNDLE, encoding='utf-8')
    registry.write_text(_REGISTRY_BUNDLES[shape], encoding='utf-8')

    runtime_setup._patch_subagent_wait_abort(binary, '2026.9.9')

    wait_src = run_wait.read_text(encoding='utf-8')
    registry_src = registry.read_text(encoding='utf-8')
    assert 'signal: params.signal ?? globalThis.__nexusWaitSignal(params.runId),' in wait_src
    assert 'if (err && err.name === "AbortError") return { status: "pending" };' in wait_src
    # The abort has to sit directly behind the anchor, inside the completion path -- not merely
    # somewhere in the file -- or a still-running child could keep the CLI pinned open.
    assert (
        'params.clearPendingLifecycleError(completeParams.runId);\n\tglobalThis.__nexusAbortWait(completeParams.runId);'
    ) in registry_src
    assert registry_src.count('__nexusAbortWait(completeParams.runId)') == 1


def test_patch_subagent_wait_abort_is_idempotent(tmp_path: Path) -> None:
    binary = _write_openclaw_install(tmp_path)
    dist = binary.parent.parent / 'dist'
    run_wait = dist / 'run-wait-abc123.js'
    registry = dist / 'subagent-registry-xyz789.js'
    run_wait.write_text(_RUN_WAIT_BUNDLE, encoding='utf-8')
    registry.write_text(_REGISTRY_BUNDLES['2026.6.x-inline-arrow'], encoding='utf-8')

    runtime_setup._patch_subagent_wait_abort(binary, '2026.9.9')
    once = (run_wait.read_text(encoding='utf-8'), registry.read_text(encoding='utf-8'))
    runtime_setup._patch_subagent_wait_abort(binary, '2026.9.9')

    assert (run_wait.read_text(encoding='utf-8'), registry.read_text(encoding='utf-8')) == once


@pytest.mark.parametrize('value', [0, -1, '0', 'not-a-number', True])
def test_patch_model_config_rejects_invalid_llm_idle_timeout(tmp_path: Path, value: object) -> None:
    config_path = tmp_path / 'openclaw.json'
    _write_openclaw_config(config_path)

    with pytest.raises(ValueError, match='llm_idle_timeout_ms must be a positive number'):
        _patch_model_config(
            config_path=config_path,
            provider_id=DEFAULT_PROVIDER_ID,
            model='bedrock/claude-opus',
            context_window=200_000,
            max_tokens=65_536,
            llm_idle_timeout_ms=value,  # type: ignore[arg-type]
        )


def test_web_fetch_curl_fallback_handles_fetch_exceptions(tmp_path: Path) -> None:
    openclaw_bin = tmp_path / 'node_modules' / 'openclaw' / 'bin' / 'openclaw.mjs'
    dist_dir = tmp_path / 'node_modules' / 'openclaw' / 'dist'
    openclaw_bin.parent.mkdir(parents=True)
    dist_dir.mkdir(parents=True)
    openclaw_bin.write_text('', encoding='utf-8')

    web_tools = dist_dir / 'web-tools.js'
    web_tools.write_text(
        """async function runWebFetch(params) {
\tconst cacheKey = "cache-key";
\tconst start = Date.now();
\tlet res;
\ttry {
\t\tres = await fetchWithWebToolsNetworkGuard(params.url, {});
\t} catch (error) {
\t\tconst payload = await maybeFetchWithFirecrawl(params, error);
\t\tif (payload) return payload;
\t\tthrow error;
\t}
\ttry {
\t\tif (!res.ok) {
\t\t\tconst payload = await maybeFetchWithFirecrawl(params, res);
\t\t\tif (payload) return payload;
\t\t\tconst rawDetail = (await readResponseText(res, {})).text;
\t\t\tconst wrappedDetail = wrapWebFetchContent(rawDetail, params.maxChars);
\t\t\tthrow new Error(`Web fetch failed (${res.status}): ${wrappedDetail.text}`);
\t\t}
\t} catch (error) {
\t\tthrow error;
\t}
}
""",
        encoding='utf-8',
    )

    _patch_web_fetch_curl_fallback(openclaw_bin)

    patched = web_tools.read_text(encoding='utf-8')
    assert CURL_FALLBACK_PATCH_MARKER in patched
    assert 'async function nexusCurlWebFetchFallback' in patched
    assert 'error?.message || "fetch error"' in patched
    assert 'if (curlPayload) return curlPayload;' in patched
    assert 'res.status === 403 || res.status === 429' in patched

    _patch_web_fetch_curl_fallback(openclaw_bin)

    assert web_tools.read_text(encoding='utf-8') == patched


def _write_openclaw_search_config(path: Path) -> None:
    path.write_text(
        """{
  "tools": {
    "web": {
      "search": {
        "enabled": true,
        "provider": "brave",
        "apiKey": "stale-real-key"
      }
    }
  },
  "models": {
    "providers": {
      "nexusproxy": {
        "models": [
          {
            "id": "gpt-test",
            "name": "gpt-test"
          }
        ]
      }
    }
  }
}""",
        encoding='utf-8',
    )


def _write_openclaw_bin(tmp_path: Path) -> tuple[Path, Path]:
    openclaw_bin = tmp_path / 'node_modules' / 'openclaw' / 'bin' / 'openclaw.mjs'
    dist_dir = tmp_path / 'node_modules' / 'openclaw' / 'dist'
    openclaw_bin.parent.mkdir(parents=True)
    dist_dir.mkdir(parents=True)
    openclaw_bin.write_text('', encoding='utf-8')
    return openclaw_bin, dist_dir


def test_full_config_uses_brave_provider_base_url_for_new_openclaw(tmp_path: Path) -> None:
    config_path = tmp_path / 'openclaw.json'
    _write_openclaw_search_config(config_path)

    _patch_full_config(
        config_path=config_path,
        provider_id='nexusproxy',
        model='gpt-test',
        web_search_enabled=True,
        web_search_base_url='http://127.0.0.1:38491',
        web_search_uses_brave_base_url=True,
    )

    patched = __import__('json').loads(config_path.read_text(encoding='utf-8'))
    search_cfg = patched['tools']['web']['search']
    assert search_cfg == {
        'enabled': True,
        'provider': 'brave',
    }
    brave_cfg = patched['plugins']['entries']['brave']['config']['webSearch']
    assert brave_cfg == {
        'apiKey': NEXUS_BRAVE_API_KEY,
        'baseUrl': 'http://127.0.0.1:38491',
        'mode': 'web',
    }


def test_full_config_does_not_write_brave_base_url_for_old_openclaw(tmp_path: Path) -> None:
    config_path = tmp_path / 'openclaw.json'
    _write_openclaw_search_config(config_path)

    _patch_full_config(
        config_path=config_path,
        provider_id='nexusproxy',
        model='gpt-test',
        web_search_enabled=True,
        web_search_base_url='http://127.0.0.1:38491',
        web_search_uses_brave_base_url=False,
    )

    patched = __import__('json').loads(config_path.read_text(encoding='utf-8'))
    assert patched['tools']['web']['search'] == {
        'enabled': True,
        'provider': 'brave',
        'apiKey': NEXUS_BRAVE_API_KEY,
    }
    assert patched.get('plugins', {}).get('entries', {}).get('brave') is None


async def test_setup_and_run_openclaw_installs_brave_plugin_for_new_openclaw(monkeypatch, tmp_path: Path) -> None:
    from nexus.runtime.executors.command import CommandRunStatus

    openclaw_bin = tmp_path / 'openclaw-prefix' / 'node_modules' / '.bin' / 'openclaw'
    js_entry = tmp_path / 'openclaw-prefix' / 'node_modules' / 'openclaw' / 'bin' / 'openclaw.mjs'
    openclaw_bin.parent.mkdir(parents=True)
    js_entry.parent.mkdir(parents=True)
    openclaw_bin.write_text('', encoding='utf-8')
    js_entry.write_text('', encoding='utf-8')

    class FakeProxy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    class FakeBridge:
        async def stop(self) -> None:
            return None

    async def fake_start_web_search_bridge_for_openclaw(**kwargs):
        return FakeBridge(), WebSearchCompat(
            enabled=True,
            base_url='http://127.0.0.1:38491',
            uses_brave_base_url=True,
        )

    commands: list[str] = []

    async def fake_run_command(request):
        commands.append(request.command)
        if ' onboard ' in f' {request.command} ':
            config_path = Path(request.extra_env['OPENCLAW_CONFIG_PATH'])
            config_path.parent.mkdir(parents=True, exist_ok=True)
            _write_openclaw_config(config_path, model='gpt-test')
        if ' infer web providers ' in f' {request.command} ':
            return SimpleNamespace(
                status=CommandRunStatus.Finished,
                return_code=0,
                stdout=json.dumps(
                    {
                        'search': [
                            {
                                'id': 'brave',
                                'available': True,
                                'configured': True,
                                'selected': True,
                            }
                        ]
                    }
                ),
                stderr='',
            )
        stdout = json.dumps({'payloads': [{'text': 'visible answer'}]}) if ' agent ' in f' {request.command} ' else '{}'
        return SimpleNamespace(
            status=CommandRunStatus.Finished,
            return_code=0,
            stdout=stdout,
            stderr='',
        )

    async def fake_install_openclaw(**kwargs):
        return openclaw_bin

    monkeypatch.setattr(
        'nexus.runtime.context.require_runtime_control',
        lambda: SimpleNamespace(sio=None),
    )
    monkeypatch.setattr(
        'nexus.runtime.server.services.openai_proxy.OpenAIProxy',
        FakeProxy,
    )
    monkeypatch.setattr('nexus.runtime.executors.command.run_command', fake_run_command)
    monkeypatch.setattr(runtime_setup, 'install_openclaw', fake_install_openclaw)
    monkeypatch.setattr(
        runtime_setup, 'find_nexus_runtime_binary', lambda name: '/usr/bin/node' if name == 'node' else None
    )
    monkeypatch.setattr(runtime_setup, 'resolve_npm_bin_entry', lambda *args, **kwargs: js_entry)
    monkeypatch.setattr(runtime_setup, '_patch_web_fetch_ssrf', lambda openclaw_path: None)
    monkeypatch.setattr(runtime_setup, '_patch_web_fetch_ua', lambda openclaw_path: None)
    monkeypatch.setattr(runtime_setup, '_patch_web_fetch_curl_fallback', lambda openclaw_path: None)
    monkeypatch.setattr(runtime_setup, '_find_runtime_curl_binary', lambda: None)
    monkeypatch.setattr(
        runtime_setup,
        '_start_web_search_bridge_for_openclaw',
        fake_start_web_search_bridge_for_openclaw,
    )
    _stub_openclaw_gateway(monkeypatch)

    result = await runtime_setup.setup_and_run_openclaw(
        task='search for test',
        model='gpt-test',
        working_dir=str(tmp_path / 'workspace'),
        timeout=30,
        version='2026.5.12',
    )

    assert result['success'] is True
    assert len(commands) == 5
    assert any(' onboard ' in f' {command} ' for command in commands)
    assert any(' plugins install @openclaw/brave-plugin ' in f' {command} ' for command in commands)
    assert any(' plugins enable brave ' in f' {command} ' for command in commands)
    assert any(' infer web providers ' in f' {command} ' for command in commands)
    assert any(' agent ' in f' {command} ' for command in commands)


async def test_setup_and_run_openclaw_uses_tencent_registry_for_brave_plugin_install(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from nexus.runtime.executors.command import CommandRunStatus

    openclaw_bin = tmp_path / 'openclaw-prefix' / 'node_modules' / '.bin' / 'openclaw'
    js_entry = tmp_path / 'openclaw-prefix' / 'node_modules' / 'openclaw' / 'bin' / 'openclaw.mjs'
    openclaw_bin.parent.mkdir(parents=True)
    js_entry.parent.mkdir(parents=True)
    openclaw_bin.write_text('', encoding='utf-8')
    js_entry.write_text('', encoding='utf-8')

    class FakeProxy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    class FakeBridge:
        async def stop(self) -> None:
            return None

    async def fake_start_web_search_bridge_for_openclaw(**kwargs):
        return FakeBridge(), WebSearchCompat(
            enabled=True,
            base_url='http://127.0.0.1:38491',
            uses_brave_base_url=True,
        )

    plugin_envs: list[dict[str, str]] = []

    async def fake_run_command(request):
        if ' onboard ' in f' {request.command} ':
            config_path = Path(request.extra_env['OPENCLAW_CONFIG_PATH'])
            config_path.parent.mkdir(parents=True, exist_ok=True)
            _write_openclaw_config(config_path, model='gpt-test')
        if ' plugins install @openclaw/brave-plugin ' in f' {request.command} ':
            plugin_envs.append(request.extra_env)
        if ' infer web providers ' in f' {request.command} ':
            return SimpleNamespace(
                status=CommandRunStatus.Finished,
                return_code=0,
                stdout=json.dumps(
                    {
                        'search': [
                            {
                                'id': 'brave',
                                'available': True,
                                'configured': True,
                                'selected': True,
                            }
                        ]
                    }
                ),
                stderr='',
            )
        stdout = json.dumps({'payloads': [{'text': 'visible answer'}]}) if ' agent ' in f' {request.command} ' else '{}'
        return SimpleNamespace(
            status=CommandRunStatus.Finished,
            return_code=0,
            stdout=stdout,
            stderr='',
        )

    async def fake_install_openclaw(**kwargs):
        return openclaw_bin

    monkeypatch.setattr(
        'nexus.runtime.context.require_runtime_control',
        lambda: SimpleNamespace(sio=None),
    )
    monkeypatch.setattr(
        'nexus.runtime.server.services.openai_proxy.OpenAIProxy',
        FakeProxy,
    )
    monkeypatch.setattr('nexus.runtime.executors.command.run_command', fake_run_command)
    monkeypatch.setattr(runtime_setup, 'install_openclaw', fake_install_openclaw)
    monkeypatch.setattr(
        runtime_setup, 'find_nexus_runtime_binary', lambda name: '/usr/bin/node' if name == 'node' else None
    )
    monkeypatch.setattr(runtime_setup, 'resolve_npm_bin_entry', lambda *args, **kwargs: js_entry)
    monkeypatch.setattr(runtime_setup, '_patch_web_fetch_ssrf', lambda openclaw_path: None)
    monkeypatch.setattr(runtime_setup, '_patch_web_fetch_ua', lambda openclaw_path: None)
    monkeypatch.setattr(runtime_setup, '_patch_web_fetch_curl_fallback', lambda openclaw_path: None)
    monkeypatch.setattr(runtime_setup, '_find_runtime_curl_binary', lambda: None)
    monkeypatch.setattr(
        runtime_setup,
        '_start_web_search_bridge_for_openclaw',
        fake_start_web_search_bridge_for_openclaw,
    )
    _stub_openclaw_gateway(monkeypatch)

    result = await runtime_setup.setup_and_run_openclaw(
        task='search for test',
        model='gpt-test',
        working_dir=str(tmp_path / 'workspace'),
        timeout=30,
        version='2026.5.12',
    )

    assert result['success'] is True
    assert plugin_envs
    assert plugin_envs[0]['NPM_CONFIG_REGISTRY'] == 'http://mirrors.tencent.com/npm/'
    assert plugin_envs[0]['npm_config_registry'] == 'http://mirrors.tencent.com/npm/'


async def test_setup_and_run_openclaw_disables_search_when_brave_provider_is_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from nexus.runtime.executors.command import CommandRunStatus

    openclaw_bin = tmp_path / 'openclaw-prefix' / 'node_modules' / '.bin' / 'openclaw'
    js_entry = tmp_path / 'openclaw-prefix' / 'node_modules' / 'openclaw' / 'bin' / 'openclaw.mjs'
    openclaw_bin.parent.mkdir(parents=True)
    js_entry.parent.mkdir(parents=True)
    openclaw_bin.write_text('', encoding='utf-8')
    js_entry.write_text('', encoding='utf-8')

    class FakeProxy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    class FakeBridge:
        stopped = False

        async def stop(self) -> None:
            self.stopped = True

    fake_bridge = FakeBridge()

    async def fake_start_web_search_bridge_for_openclaw(**kwargs):
        return fake_bridge, WebSearchCompat(
            enabled=True,
            base_url='http://127.0.0.1:38491',
            uses_brave_base_url=True,
        )

    commands: list[str] = []
    config_paths: list[Path] = []
    agent_config: dict[str, object] = {}
    agent_env: dict[str, str] = {}

    async def fake_run_command(request):
        commands.append(request.command)
        if ' onboard ' in f' {request.command} ':
            config_path = Path(request.extra_env['OPENCLAW_CONFIG_PATH'])
            config_paths.append(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            _write_openclaw_config(config_path, model='gpt-test')
        if ' infer web providers ' in f' {request.command} ':
            return SimpleNamespace(
                status=CommandRunStatus.Finished,
                return_code=0,
                stdout=json.dumps(
                    {
                        'search': [
                            {
                                'id': 'duckduckgo',
                                'available': True,
                                'configured': True,
                                'selected': False,
                            }
                        ]
                    }
                ),
                stderr='',
            )
        if ' agent ' in f' {request.command} ':
            config_path = Path(request.extra_env['OPENCLAW_CONFIG_PATH'])
            agent_config.update(json.loads(config_path.read_text(encoding='utf-8')))
            agent_env.update(request.extra_env)
        stdout = json.dumps({'payloads': [{'text': 'visible answer'}]}) if ' agent ' in f' {request.command} ' else '{}'
        return SimpleNamespace(
            status=CommandRunStatus.Finished,
            return_code=0,
            stdout=stdout,
            stderr='',
        )

    async def fake_install_openclaw(**kwargs):
        return openclaw_bin

    monkeypatch.setattr(
        'nexus.runtime.context.require_runtime_control',
        lambda: SimpleNamespace(sio=None),
    )
    monkeypatch.setattr(
        'nexus.runtime.server.services.openai_proxy.OpenAIProxy',
        FakeProxy,
    )
    monkeypatch.setattr('nexus.runtime.executors.command.run_command', fake_run_command)
    monkeypatch.setattr(runtime_setup, 'install_openclaw', fake_install_openclaw)
    monkeypatch.setattr(
        runtime_setup, 'find_nexus_runtime_binary', lambda name: '/usr/bin/node' if name == 'node' else None
    )
    monkeypatch.setattr(runtime_setup, 'resolve_npm_bin_entry', lambda *args, **kwargs: js_entry)
    monkeypatch.setattr(runtime_setup, '_patch_web_fetch_ssrf', lambda openclaw_path: None)
    monkeypatch.setattr(runtime_setup, '_patch_web_fetch_ua', lambda openclaw_path: None)
    monkeypatch.setattr(runtime_setup, '_patch_web_fetch_curl_fallback', lambda openclaw_path: None)
    monkeypatch.setattr(runtime_setup, '_find_runtime_curl_binary', lambda: None)
    monkeypatch.setattr(
        runtime_setup,
        '_start_web_search_bridge_for_openclaw',
        fake_start_web_search_bridge_for_openclaw,
    )
    _stub_openclaw_gateway(monkeypatch)

    result = await runtime_setup.setup_and_run_openclaw(
        task='search for test',
        model='gpt-test',
        working_dir=str(tmp_path / 'workspace'),
        timeout=30,
        version='2026.5.12',
    )

    assert result['success'] is True
    assert any(' plugins install @openclaw/brave-plugin ' in f' {command} ' for command in commands)
    assert any(' plugins enable brave ' in f' {command} ' for command in commands)
    assert any(' infer web providers ' in f' {command} ' for command in commands)
    assert fake_bridge.stopped is True

    assert config_paths
    assert agent_config['tools']['web']['search'] == {'enabled': False}
    assert agent_config.get('plugins', {}).get('entries', {}).get('brave') is None
    assert agent_env['BRAVE_API_KEY'] == ''
    assert agent_env['NEXUS_SERPER_BRIDGE_BASE_URL'] == ''


def test_full_config_disables_search_when_compat_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / 'openclaw.json'
    _write_openclaw_search_config(config_path)

    _patch_full_config(
        config_path=config_path,
        provider_id='nexusproxy',
        model='gpt-test',
        web_search_enabled=False,
        web_search_base_url=None,
        web_search_uses_brave_base_url=False,
    )

    patched = __import__('json').loads(config_path.read_text(encoding='utf-8'))
    assert patched['tools']['web']['search'] == {'enabled': False}
    assert patched.get('plugins', {}).get('entries', {}).get('brave') is None


def test_old_endpoint_patch_is_idempotent_and_env_based(tmp_path: Path) -> None:
    openclaw_bin, dist_dir = _write_openclaw_bin(tmp_path)
    web_tools = dist_dir / 'web-tools.js'
    web_tools.write_text(
        """const BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search";
async function runWebSearch() {
  const url = new URL(BRAVE_SEARCH_ENDPOINT);
}
""",
        encoding='utf-8',
    )

    assert _patch_web_search_bridge_endpoint(openclaw_bin) is True
    patched = web_tools.read_text(encoding='utf-8')
    assert WEB_SEARCH_ENDPOINT_PATCH_MARKER in patched
    assert 'process.env.NEXUS_SERPER_BRIDGE_BASE_URL' in patched
    assert 'https://api.search.brave.com' not in patched

    assert _patch_web_search_bridge_endpoint(openclaw_bin) is True
    assert web_tools.read_text(encoding='utf-8') == patched


def test_old_endpoint_patch_accepts_existing_env_based_patch_without_marker(tmp_path: Path) -> None:
    openclaw_bin, dist_dir = _write_openclaw_bin(tmp_path)
    web_tools = dist_dir / 'web-tools.js'
    web_tools.write_text(
        """const BRAVE_SEARCH_ENDPOINT = (
  (process.env.NEXUS_SERPER_BRIDGE_BASE_URL || "http://127.0.0.1:0").replace(/\\/+$/u, "") +
  "/res/v1/web/search"
);
async function runWebSearch() {
  const url = new URL(BRAVE_SEARCH_ENDPOINT);
}
""",
        encoding='utf-8',
    )

    assert _patch_web_search_bridge_endpoint(openclaw_bin) is True


def test_old_endpoint_patch_replaces_stale_hardcoded_local_bridge(tmp_path: Path) -> None:
    openclaw_bin, dist_dir = _write_openclaw_bin(tmp_path)
    web_tools = dist_dir / 'web-tools.js'
    web_tools.write_text(
        """const BRAVE_SEARCH_ENDPOINT = "http://127.0.0.1:43123/res/v1/web/search";
async function runWebSearch() {
  const url = new URL(BRAVE_SEARCH_ENDPOINT);
}
""",
        encoding='utf-8',
    )

    assert _patch_web_search_bridge_endpoint(openclaw_bin) is True

    patched = web_tools.read_text(encoding='utf-8')
    assert 'http://127.0.0.1:43123' not in patched
    assert 'process.env.NEXUS_SERPER_BRIDGE_BASE_URL' in patched


def test_prepare_web_search_compat_disables_when_patch_fails(tmp_path: Path) -> None:
    openclaw_bin, dist_dir = _write_openclaw_bin(tmp_path)
    (dist_dir / 'web-tools.js').write_text('async function runWebSearch() {}', encoding='utf-8')

    compat = _prepare_web_search_compat(
        openclaw_bin=openclaw_bin,
        bridge_base_url='http://127.0.0.1:38491',
        serper_configured=True,
    )

    assert compat == WebSearchCompat(enabled=False, base_url=None, uses_brave_base_url=False)


def test_prepare_web_search_compat_does_not_misread_legacy_schema_docs_as_new_runtime(tmp_path: Path) -> None:
    openclaw_bin, dist_dir = _write_openclaw_bin(tmp_path)
    web_tools = dist_dir / 'web-tools.js'
    web_tools.write_text(
        """const BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search";
const docs = "tools.web.search.kimi.baseUrl tools.web.search.perplexity.baseUrl";
async function runWebSearch() {
  const url = new URL(BRAVE_SEARCH_ENDPOINT);
}
""",
        encoding='utf-8',
    )

    compat = _prepare_web_search_compat(
        openclaw_bin=openclaw_bin,
        bridge_base_url='http://127.0.0.1:38491',
        serper_configured=True,
    )

    assert compat == WebSearchCompat(enabled=True, base_url='http://127.0.0.1:38491', uses_brave_base_url=False)
    assert WEB_SEARCH_ENDPOINT_PATCH_MARKER in web_tools.read_text(encoding='utf-8')


def test_prepare_web_search_compat_uses_new_openclaw_base_url_without_js_patch(tmp_path: Path) -> None:
    openclaw_bin, dist_dir = _write_openclaw_bin(tmp_path)
    web_tools = dist_dir / 'web-tools.js'
    web_tools.write_text(
        'const BRAVE_SEARCH_ENDPOINT_PATH = "/res/v1/web/search"; function resolveBraveBaseUrl() {}',
        encoding='utf-8',
    )

    compat = _prepare_web_search_compat(
        openclaw_bin=openclaw_bin,
        bridge_base_url='http://127.0.0.1:38491',
        serper_configured=True,
    )

    assert compat == WebSearchCompat(enabled=True, base_url='http://127.0.0.1:38491', uses_brave_base_url=True)
    assert WEB_SEARCH_ENDPOINT_PATCH_MARKER not in web_tools.read_text(encoding='utf-8')


def test_build_openclaw_env_masks_serper_key_and_bypasses_bridge_proxy() -> None:
    env = _build_openclaw_env(
        home_dir=Path('/tmp/home'),
        state_dir=Path('/tmp/state'),
        config_path=Path('/tmp/state/openclaw.json'),
        proxy_port=18083,
        web_search=WebSearchCompat(enabled=True, base_url='http://127.0.0.1:38491', uses_brave_base_url=True),
        curl_binary=None,
        base_env={
            'SERPER_API_KEY': 'real-serper-key',
            'SERPER_API_TYPE': 'distillation',
            'SEARCH_SERPER_URL': 'http://search-gateway.local/search',
            'SEARCH_SERPER_APP_ID': 'app-id',
            'SEARCH_SERPER_APP_KEY': 'app-key',
            'SERPER_KEY_ID': 'key-id',
            'no_proxy': 'example.com',
        },
    )

    assert env['BRAVE_API_KEY'] == NEXUS_BRAVE_API_KEY
    assert env['SERPER_API_KEY'] == ''
    assert env['SERPER_API_TYPE'] == ''
    assert env['SEARCH_SERPER_URL'] == ''
    assert env['SEARCH_SERPER_APP_ID'] == ''
    assert env['SEARCH_SERPER_APP_KEY'] == ''
    assert env['SERPER_KEY_ID'] == ''
    assert env['NEXUS_SERPER_BRIDGE_BASE_URL'] == 'http://127.0.0.1:38491'
    assert '127.0.0.1:18083' in env['no_proxy']
    assert '127.0.0.1:38491' in env['no_proxy']
    assert env['NO_PROXY'] == env['no_proxy']


async def test_start_web_search_bridge_stops_candidate_when_compat_setup_fails(monkeypatch, tmp_path: Path) -> None:
    class FakeBridge:
        base_url = 'http://127.0.0.1:38491'

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False
            self.stopped = False

        async def start(self) -> None:
            self.started = True

        async def stop(self) -> None:
            self.stopped = True

    fake_bridge = FakeBridge()

    def bridge_factory(**kwargs):
        fake_bridge.kwargs = kwargs
        return fake_bridge

    def raise_compat(**kwargs):
        raise RuntimeError('compat failed')

    monkeypatch.setattr(
        'nexus.extensions.agents.openclaw.runtime_setup._prepare_web_search_compat',
        raise_compat,
    )

    bridge, compat = await _start_web_search_bridge_for_openclaw(
        openclaw_bin=tmp_path / 'openclaw.mjs',
        serper_config=object(),
        timeout=30,
        bridge_factory=bridge_factory,
    )

    assert bridge is None
    assert compat == WebSearchCompat(enabled=False, base_url=None, uses_brave_base_url=False)
    assert fake_bridge.started is True
    assert fake_bridge.stopped is True


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - only if the pid gets recycled by another user
        return True
    return True


async def _spawn_fake_gateway_with_child(tmp_path: Path) -> tuple[asyncio.subprocess.Process, int]:
    """Start a process group shaped like the gateway: a leader plus a surviving descendant.

    The leader ignores SIGTERM so the test also covers the SIGKILL escalation, and it writes the
    descendant's pid out so the test can assert on it after the group is gone.
    """
    pid_file = tmp_path / 'child.pid'
    script = (
        'import os, signal, subprocess, sys, time\n'
        'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f'open({str(pid_file)!r}, "w").write(str(child.pid))\n'
        'time.sleep(120)\n'
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        '-c',
        script,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(100):
        if pid_file.is_file() and pid_file.read_text().strip():
            break
        await asyncio.sleep(0.05)
    child_pid = int(pid_file.read_text().strip())
    assert _process_is_alive(child_pid)
    return proc, child_pid


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != 'posix', reason='process groups are POSIX-only')
async def test_stop_openclaw_gateway_kills_descendant_processes(tmp_path: Path, monkeypatch) -> None:
    """The gateway hosts the exec tool, so its backgrounded commands must not outlive it."""
    monkeypatch.setattr(runtime_setup, '_GATEWAY_STOP_GRACE_SECONDS', 1)
    proc, child_pid = await _spawn_fake_gateway_with_child(tmp_path)

    await runtime_setup._stop_openclaw_gateway(proc)

    assert proc.returncode is not None
    for _ in range(100):
        if not _process_is_alive(child_pid):
            break
        await asyncio.sleep(0.05)
    assert not _process_is_alive(child_pid), f'descendant {child_pid} survived gateway shutdown'


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != 'posix', reason='process groups are POSIX-only')
async def test_stop_openclaw_gateway_spares_the_group_when_isolation_did_not_take(tmp_path: Path, monkeypatch) -> None:
    """A gateway sharing Nexus's process group must never be signalled as a group."""
    monkeypatch.setattr(runtime_setup, '_GATEWAY_STOP_GRACE_SECONDS', 1)
    proc, child_pid = await _spawn_fake_gateway_with_child(tmp_path)
    # Pretend setsid never happened, which is the only case where killpg could reach Nexus.
    # Read our own pgid first: runtime_setup.os is the os module, so the patch below is global.
    nexus_pgid = os.getpgid(0)
    monkeypatch.setattr(runtime_setup.os, 'getpgid', lambda _pid: nexus_pgid)
    killed_groups: list[int] = []
    monkeypatch.setattr(runtime_setup.os, 'killpg', lambda pgid, sig: killed_groups.append(pgid))

    try:
        await runtime_setup._stop_openclaw_gateway(proc)

        assert killed_groups == []
        assert proc.returncode is not None
    finally:
        with suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)
