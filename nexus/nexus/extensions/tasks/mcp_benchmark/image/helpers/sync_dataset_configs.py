#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import runpy
import shutil
import sys

from pathlib import Path
from types import ModuleType

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[6]
TASK_ROOT = PROJECT_ROOT / 'nexus' / 'extensions' / 'tasks' / 'mcp_benchmark'
WORKSPACE_MCP_ROOT = PROJECT_ROOT / 'workspace' / 'mcp'


def _ensure_pkg(name: str, path: Path | None = None) -> None:
    if name in sys.modules:
        return
    module = ModuleType(name)
    if path is not None:
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to load module {name} from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_localize_module():
    # Load only the benchmark package subtree without importing nexus/__init__.py
    _ensure_pkg('nexus', PROJECT_ROOT / 'nexus')
    _ensure_pkg('nexus.extensions', PROJECT_ROOT / 'nexus' / 'extensions')
    _ensure_pkg('nexus.extensions.tasks', PROJECT_ROOT / 'nexus' / 'extensions' / 'tasks')
    _ensure_pkg('nexus.extensions.tasks.mcp_benchmark', TASK_ROOT)
    _load_module('nexus.extensions.tasks.mcp_benchmark.constants', TASK_ROOT / 'constants.py')
    return _load_module('nexus.extensions.tasks.mcp_benchmark.localize', TASK_ROOT / 'localize.py')


_LOCALIZE = _load_localize_module()
TOOLATHLON_SERVER_FILE_NAMES = _LOCALIZE.TOOLATHLON_SERVER_FILE_NAMES
localize_atlas_server = _LOCALIZE.localize_atlas_server
localize_toolathlon_rendered = _LOCALIZE.localize_toolathlon_rendered
localize_universe_server = _LOCALIZE.localize_universe_server


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


class _CompatDict(dict):
    def __getattr__(self, item: str):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


def _ensure_addict_compat() -> None:
    if 'addict' in sys.modules:
        return
    module = ModuleType('addict')
    module.Dict = _CompatDict
    sys.modules['addict'] = module


def _load_workspace_token_values() -> dict[str, object] | None:
    workspace_keys = WORKSPACE_MCP_ROOT / 'keys.py'
    if not workspace_keys.exists():
        return None

    old_cwd = os.getcwd()
    try:
        os.chdir(WORKSPACE_MCP_ROOT)
        _ensure_addict_compat()
        namespace = runpy.run_path(str(workspace_keys))
    finally:
        os.chdir(old_cwd)

    raw = namespace.get('all_token_key_session', {})
    try:
        return dict(raw)
    except Exception:
        return None


def _write_toolathlon_token_key_session(target: Path) -> None:
    values = _load_workspace_token_values()
    if values is None:
        shutil.copy2(target.parent / 'token_key_session_example.py', target)
        return

    payload = repr(values)
    target.write_text(
        'class _CompatDict(dict):\n'
        '    def __getattr__(self, item):\n'
        '        try:\n'
        '            return self[item]\n'
        '        except KeyError as exc:\n'
        '            raise AttributeError(item) from exc\n'
        '\n'
        f'all_token_key_session = _CompatDict({payload})\n'
    )


def _ensure_toolathlon_runtime_configs(configs_dir: Path) -> None:
    configs_dir.mkdir(parents=True, exist_ok=True)

    global_configs = configs_dir / 'global_configs.py'
    global_configs.write_text(
        'from addict import Dict\n'
        'global_configs = Dict(\n'
        '    podman_or_docker="docker",\n'
        '    notion_preprocess_with_playwright=False,\n'
        ')\n'
    )

    token_key_session = configs_dir / 'token_key_session.py'
    _write_toolathlon_token_key_session(token_key_session)


def _sync_toolathlon(src_root: Path, dst_root: Path) -> None:
    src = src_root / 'toolathlon'
    dst = dst_root / 'toolathlon'
    _copy_tree(src, dst)
    configs_dir = dst / 'configs' / 'mcp_servers'

    for file_path in configs_dir.glob('*.yaml'):
        data = yaml.safe_load(file_path.read_text())
        server_name = data.get('name') or file_path.stem
        localized = localize_toolathlon_rendered(server_name, data)
        file_path.write_text(yaml.safe_dump(localized, sort_keys=False, allow_unicode=True))

    for logical_name, file_name in TOOLATHLON_SERVER_FILE_NAMES.items():
        if logical_name == file_name:
            continue
        src_file = configs_dir / f'{file_name}.yaml'
        dst_file = configs_dir / f'{logical_name}.yaml'
        if src_file.exists() and not dst_file.exists():
            shutil.copy2(src_file, dst_file)

    _ensure_toolathlon_runtime_configs(dst / 'configs')


def _sync_atlas(src_root: Path, dst_root: Path) -> None:
    src = src_root / 'atlas'
    dst = dst_root / 'atlas'
    _copy_tree(src, dst)
    template_path = dst / 'services' / 'agent-environment' / 'src' / 'agent_environment' / 'mcp_server_template.json'
    payload = json.loads(template_path.read_text())
    payload['mcpServers'] = {
        name: localize_atlas_server(name, config) for name, config in payload['mcpServers'].items()
    }
    template_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _sync_universe(src_root: Path, dst_root: Path) -> None:
    src = src_root / 'universe'
    dst = dst_root / 'universe'
    _copy_tree(src, dst)
    server_list_path = dst / 'mcpuniverse' / 'mcp' / 'configs' / 'server_list.json'
    payload = json.loads(server_list_path.read_text())
    payload = {name: localize_universe_server(name, config) for name, config in payload.items()}
    server_list_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    _sync_toolathlon(source_root, output_root)
    _sync_atlas(source_root, output_root)
    _sync_universe(source_root, output_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
