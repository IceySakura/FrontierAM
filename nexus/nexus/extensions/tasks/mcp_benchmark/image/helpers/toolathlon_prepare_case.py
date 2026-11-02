#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil

from pathlib import Path
from pprint import pformat


GLOBAL_CONFIGS_DEFAULTS = {
    'aihubmix_key': '',
    'openrouter_key': '',
    'qwen_official_key': '',
    'kimi_official_key': '',
    'deepseek_official_key': '',
    'anthropic_official_key': '',
    'openai_official_key': '',
    'google_official_key': '',
    'xai_official_key': '',
    'proxy': '',
    'podman_or_docker': 'kind',
    'notion_preprocess_with_playwright': False,
}


def _token_values() -> dict[str, str]:
    payload = json.loads(os.getenv('NEXUS_TOOLATHLON_TOKEN_VALUES_JSON') or '{}')
    return {str(key): str(value) for key, value in payload.items()}


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def _write_json_if_missing(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        return
    _write(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _ensure_symlink(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if path.resolve() == target.resolve():
            return
        path.unlink()
    elif path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.symlink_to(target)


def _render_token_module(base_values: dict[str, str], *, import_task_file: bool, case_root: Path) -> str:
    base_literal = pformat(base_values, width=100, sort_dicts=True)
    maybe_merge = ''
    if import_task_file:
        maybe_merge = (
            f'_task_token_path = Path({str(case_root / "token_key_session.py")!r})\n'
            'if _task_token_path.exists():\n'
            '    try:\n'
            '        _namespace = runpy.run_path(str(_task_token_path))\n'
            "        _raw = _namespace.get('all_token_key_session', {})\n"
            '        try:\n'
            '            all_token_key_session.update(dict(_raw))\n'
            '        except Exception:\n'
            '            pass\n'
            '    except Exception:\n'
            '        pass\n'
        )
    return (
        'from __future__ import annotations\n\n'
        'import runpy\n\n'
        'from pathlib import Path\n\n'
        'class Dict(dict):\n'
        '    def __getattr__(self, item):\n'
        '        try:\n'
        '            return self[item]\n'
        '        except KeyError as exc:\n'
        '            raise AttributeError(item) from exc\n\n\n'
        f'_BASE_TOKEN_VALUES = {base_literal}\n'
        'all_token_key_session = Dict(_BASE_TOKEN_VALUES)\n'
        f'{maybe_merge}'
        '\n'
        'for _key, _value in dict(all_token_key_session).items():\n'
        '    globals().setdefault(_key, _value)\n'
    )


def _render_global_configs_module() -> str:
    payload = pformat(GLOBAL_CONFIGS_DEFAULTS, width=100, sort_dicts=True)
    return (
        'class Dict(dict):\n'
        '    def __getattr__(self, item):\n'
        '        try:\n'
        '            return self[item]\n'
        '        except KeyError as exc:\n'
        '            raise AttributeError(item) from exc\n\n\n'
        f'global_configs = Dict({payload})\n'
    )


def _render_init_module() -> str:
    return (
        'from .global_configs import global_configs\n'
        'from .token_key_session import all_token_key_session\n\n'
        "__all__ = ['global_configs', 'all_token_key_session']\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-root', required=True)
    args = parser.parse_args()

    case_root = Path(args.case_root)
    dataset_root = Path(os.environ['NEXUS_TOOLATHLON_DATASET_ROOT'])
    workspace_root = case_root.parents[2]
    toolathlon_compat_root = case_root.parents[1]
    configs_dir = case_root / 'configs'
    base_values = _token_values()

    _write(configs_dir / '__init__.py', _render_init_module())
    _write(configs_dir / 'global_configs.py', _render_global_configs_module())
    _write(
        configs_dir / 'token_key_session.py',
        _render_token_module(base_values, import_task_file=True, case_root=case_root),
    )

    root_token_path = case_root / 'token_key_session.py'
    if not root_token_path.exists():
        _write(root_token_path, _render_token_module(base_values, import_task_file=False, case_root=case_root))

    (configs_dir / 'mcp_servers').mkdir(parents=True, exist_ok=True)
    _ensure_symlink(workspace_root / 'configs', configs_dir)
    _ensure_symlink(workspace_root / 'utils', dataset_root / 'utils')
    _ensure_symlink(toolathlon_compat_root / 'configs', configs_dir)
    _ensure_symlink(toolathlon_compat_root / 'utils', dataset_root / 'utils')


if __name__ == '__main__':
    main()
