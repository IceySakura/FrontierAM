#!/usr/bin/env python3
"""Patch MCP server YAML configs to inject proxy env vars for all servers.

eval's MCPServerManager reads YAML files and passes their ``params.env`` dict
directly to ``MCPServerStdio`` → ``StdioServerParameters(env=...)``.  The
``mcp`` SDK uses this dict to *replace* (not merge) ``os.environ`` when
spawning the subprocess — so any MCP server that has an ``env:`` section will
NOT inherit the container's HTTP_PROXY / HTTPS_PROXY / etc. unless they are
explicitly listed in that env section.

This script patches every ``*.yaml`` file under ``--config-dir`` to ensure the
following keys are present in ``params.env``:

  Common (all servers):
  - HTTP_PROXY / HTTPS_PROXY / NO_PROXY  (corporate proxy)
  - SSL_CERT_FILE / REQUESTS_CA_BUNDLE   (Python requests/urllib3 CA cert)

  Node.js servers only (``npx`` / ``node`` commands):
  - NODE_EXTRA_CA_CERTS                  (Squid MITM CA cert)
  - NODE_PATH                            (undici require path)
  - NODE_OPTIONS                         (undici proxy bootstrap)

The patch is idempotent: running it twice produces the same result.

Usage::

    python3 toolathlon_mcp_yaml_proxy_patcher.py \\
        --config-dir /workspace/.../configs/mcp_servers \\
        --npm-node-modules /opt/mcp-benchmarks/npm/node_modules
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

import yaml


# Node.js command names that require extra Node-specific env vars.
_NODE_COMMANDS = {'npx', 'node', 'npx.cmd', 'node.exe'}


def _is_node_server(config: dict) -> bool:
    """Return True if this YAML config spawns a Node.js process."""
    params = config.get('params') or config.get('client') or {}
    cmd = params.get('command', '')
    return Path(cmd).name in _NODE_COMMANDS


def _patch_config(path: Path, common_env: dict[str, str], node_env: dict[str, str]) -> bool:
    """Patch a single YAML file.  Returns True if the file was modified."""
    text = path.read_text(encoding='utf-8')
    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        return False

    # Determine which sub-key holds the params (upstream uses 'params', nexus may use 'client')
    params_key = 'params' if 'params' in config else 'client'
    params = config.setdefault(params_key, {})
    env = params.get('env')
    if env is None:
        env = {}
        params['env'] = env

    # Build the full set of env vars to inject for this server.
    proxy_env = dict(common_env)
    if _is_node_server(config):
        proxy_env.update(node_env)

    changed = False
    for key, value in proxy_env.items():
        if env.get(key) != value:
            env[key] = value
            changed = True

    if changed:
        new_text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        path.write_text(new_text, encoding='utf-8')

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description='Inject proxy env into all MCP server YAML configs')
    parser.add_argument('--config-dir', required=True, help='directory containing *.yaml MCP server configs')
    parser.add_argument(
        '--npm-node-modules',
        default='/opt/mcp-benchmarks/npm/node_modules',
        help='path to node_modules for NODE_PATH (default: /opt/mcp-benchmarks/npm/node_modules)',
    )
    args = parser.parse_args()

    ca_cert = '/etc/ssl/certs/ca-certificates.crt'
    undici_bootstrap = '/tmp/_undici_proxy_eval.cjs'

    # Env vars injected into every server (Node.js, Python, etc.)
    common_env: dict[str, str] = {
        'HTTP_PROXY': 'http://hyrl-proxy.woa.com:3128',
        'HTTPS_PROXY': 'http://hyrl-proxy.woa.com:3128',
        'NO_PROXY': '.woa.com,localhost,127.0.0.1',
        'SSL_CERT_FILE': ca_cert,
        'REQUESTS_CA_BUNDLE': ca_cert,
    }

    # Extra env vars only for Node.js servers.
    node_env: dict[str, str] = {
        'NODE_EXTRA_CA_CERTS': ca_cert,
        'NODE_PATH': args.npm_node_modules,
        'NODE_OPTIONS': f'--require {undici_bootstrap}',
    }

    config_dir = Path(args.config_dir)
    if not config_dir.is_dir():
        print(f'Config dir not found: {config_dir}', file=sys.stderr)
        sys.exit(1)

    yaml_files = sorted(config_dir.glob('*.yaml'))
    patched = 0
    skipped = 0
    for yaml_path in yaml_files:
        try:
            if _patch_config(yaml_path, common_env, node_env):
                patched += 1
                print(f'Patched:  {yaml_path.name}')
            else:
                skipped += 1
        except Exception as exc:  # noqa: BLE001
            print(f'ERROR patching {yaml_path.name}: {exc}', file=sys.stderr)

    print(f'Done: {patched} patched, {skipped} skipped (already up to date)')


if __name__ == '__main__':
    main()
