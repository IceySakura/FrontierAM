"""Build the isolated DSH profile used by the Nexus ACP host."""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path

import yaml


PROFILE_NAME = 'nexus-acp'
PROTECTED_ENV_KEYS = frozenset({
    'DSH_HOME',
    'DSH_PERMISSION_MODE',
    'DSH_TELEMETRY_DISABLED',
    'DSH_TELEMETRY_MODE',
    'DSH_TELEMETRY_OTLP_URL',
    'NEXUS_PROXY_API_KEY',
    'PATH',
    'PWD',
})


@dataclass(frozen=True)
class DshProfile:
    home: Path
    profile_dir: Path
    session_root: Path


def validate_extra_env(extra_env: dict[str, str] | None) -> None:
    """Reject overrides that would bypass the managed ACP/profile boundary."""
    conflicts = PROTECTED_ENV_KEYS.intersection(extra_env or {})
    if conflicts:
        names = ', '.join(sorted(conflicts))
        raise ValueError(f'DSH extra_env cannot override Nexus-managed variables: {names}')


def _disabled(row_id: str) -> dict[str, object]:
    return {'id': row_id, 'disabled': True}


def build_profile_patch(
    *,
    proxy_base_url: str,
    session_root: Path,
    context_window: int,
    max_output_tokens: int,
) -> list[dict[str, object]]:
    """Return the ACP-only DSH composition patch."""
    patch: list[dict[str, object]] = [
        _disabled('hmr'),
        {'id': 'agent-default-model', 'config': {'provider': 'nexus', 'model': 'nexus-model'}},
        _disabled('session-title-llm'),
        _disabled('plan-mode'),
        {
            'id': 'llm-pi-ai',
            'config': {
                'providers': {
                    'nexus': {
                        'displayName': 'Nexus Proxy',
                        'apiKeyEnv': 'NEXUS_PROXY_API_KEY',
                        'api': 'openai-completions',
                        'baseURL': proxy_base_url.rstrip('/'),
                        'models': [
                            {
                                'id': 'nexus-model',
                                'name': 'Nexus Model',
                                'contextWindow': context_window,
                                'maxTokens': max_output_tokens,
                            }
                        ],
                    }
                }
            },
        },
        {
            'id': 'session-persistence-jsonl',
            'config': {
                'root': str(session_root),
                'compression': 'none',
                'packChunks': False,
            },
        },
        _disabled('llm-deepseek'),
    ]
    patch.extend(
        _disabled(row_id)
        for row_id in (
            # Match DSH's standard coding preset: expose read/write/edit as the
            # single filesystem editing interface and leave str_replace_editor
            # to the upstream minimal preset or explicit custom compositions.
            'tool-str-replace-editor',
            'tool-jobs',
            'tool-subagent-control',
            'tool-subagent-list-agents',
            'tool-subagent',
            'tool-subagent-fork',
            'tool-subagent-report',
            'workflow-worker-thread',
            'tool-workflow',
            'tool-ralph',
            'web',
            'web-search-deepseek',
            'tool-web',
        )
    )
    patch.append({
        'insert': [
            {
                'id': 'nexus-acp',
                'name': '@deepseek-ai/dsh-acp',
                'config': {'provider': 'nexus', 'model': 'nexus-model'},
            }
        ]
    })
    return patch


def prepare_dsh_profile(
    *,
    install_dir: str,
    runtime_installation: Path,
    session_id: str,
    acp_version: str,
    proxy_base_url: str,
    context_window: int,
    max_output_tokens: int,
) -> DshProfile:
    """Create a per-Nexus-session DSH home and profile."""
    homes_root = (Path(install_dir) / 'session-homes').resolve()
    home = (homes_root / session_id).resolve()
    if home.parent != homes_root:
        raise ValueError(f'Invalid DSH session id for profile path: {session_id!r}')
    profile_dir = home / 'profiles' / PROFILE_NAME
    session_root = home / 'sessions'
    profile_dir.mkdir(parents=True, exist_ok=True)
    session_root.mkdir(parents=True, exist_ok=True)
    runtime_node_modules = (runtime_installation / 'node_modules').resolve()
    if not runtime_node_modules.is_dir():
        raise RuntimeError(f'DeepSeek Harness runtime node_modules is missing: {runtime_node_modules}')
    profile_node_modules = profile_dir / 'node_modules'
    if profile_node_modules.is_symlink():
        profile_node_modules.unlink()
    elif profile_node_modules.exists():
        raise RuntimeError(f'DeepSeek Harness profile node_modules must be a symlink: {profile_node_modules}')
    profile_node_modules.symlink_to(runtime_node_modules, target_is_directory=True)

    package = {
        'name': 'nexus-dsh-acp-profile',
        'private': True,
        'dependencies': {'@deepseek-ai/dsh-acp': acp_version},
        'dsh': {'profile': {'bundles': ['@deepseek-ai/dsh-base']}},
    }
    (profile_dir / 'package.json').write_text(json.dumps(package, indent=2) + '\n')
    patch = build_profile_patch(
        proxy_base_url=proxy_base_url,
        session_root=session_root,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
    )
    (profile_dir / 'cordis.patch.yml').write_text(
        yaml.safe_dump(patch, sort_keys=False, allow_unicode=True),
    )
    return DshProfile(home=home, profile_dir=profile_dir, session_root=session_root)


__all__ = [
    'DshProfile',
    'PROFILE_NAME',
    'PROTECTED_ENV_KEYS',
    'build_profile_patch',
    'prepare_dsh_profile',
    'validate_extra_env',
]
