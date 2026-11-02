"""Route the sandbox runtime Node's global ``fetch`` through the platform's explicit proxy.

CodeBuddy's ``WebSearch`` / ``WebFetch`` use that ``fetch``; see ``docs/impl/codebuddy_web_search.md``.

When ``NEXUS_PROXY_HOSTS`` is set (comma-separated hostnames), only those hosts go through the
proxy; everything else goes direct. That is required on suanli sandboxes created with a
``network_policy``: the platform proxy refuses the public internet, while ``copilot.tencent.com``
only answers through the proxy. When the variable is empty, the preload uses
``EnvHttpProxyAgent`` (proxy everything except ``NO_PROXY``).
"""

from __future__ import annotations

import logging

from typing import Any


logger = logging.getLogger(__name__)

PROXY_PRELOAD_DIR = '/tmp/nexus-node-proxy'
PROXY_PRELOAD_PATH = f'{PROXY_PRELOAD_DIR}/preload.js'
PROXY_PRELOAD_NODE_OPTION = f'--require {PROXY_PRELOAD_PATH}'
PROXY_PRELOAD_TIMEOUT_S = 300.0

# NOTE(wallyshi): the runtime's Node 22.20 global fetch ignores HTTP(S)_PROXY (NODE_USE_ENV_PROXY needs
# 22.21+), so a preload swaps undici's global dispatcher. The preload only acts in the runtime's own
# node, so node binaries the agent runs are left alone.
PROXY_PRELOAD_SETUP = rf"""
set -eu
export PATH=/nix/runtime/nix-env/bin:$PATH
D={PROXY_PRELOAD_DIR}
if [ ! -f "$D/node_modules/undici/package.json" ]; then
  mkdir -p "$D"
  npm install --prefix "$D" --no-save --no-audit --no-fund --loglevel=error \
    --registry http://mirrors.tencent.com/npm/ undici@6
fi
cat > "{PROXY_PRELOAD_PATH}" <<'JS'
const fs = require('fs');
let runtimeNode = '';
try {{ runtimeNode = fs.realpathSync('/nix/runtime/nix-env/bin/node'); }} catch (error) {{}}
const proxy = process.env.HTTPS_PROXY || process.env.https_proxy || process.env.HTTP_PROXY || process.env.http_proxy;
if (process.execPath === runtimeNode && proxy) {{
  if (process.env.NEXUS_WORKBUDDY_PROXY_CHILDREN === '1') {{
    const childProcess = require('node:child_process');
    const {{ createRequire, syncBuiltinESMExports }} = require('node:module');
    const inheritedKeys = [
      'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
      'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy',
      'NODE_EXTRA_CA_CERTS', 'SSL_CERT_FILE', 'SSL_CERT_DIR',
      'NEXUS_PROXY_HOSTS', 'NEXUS_WORKBUDDY_PROXY_CHILDREN',
    ];
    // NOTE(wallyshi): WorkBuddy uses child_process for its sidecar, then node-pty for the Linux agent.
    const wrapSpawn = (spawn) => function(command, args, options) {{
      const paths = Array.isArray(args) ? args : [];
      const isWorker = paths.some((arg) => typeof arg === 'string' && (
        arg.endsWith('/sidecar-entry.js') || (
          arg.endsWith('/cli/bin/codebuddy') && paths.includes('--serve')
        )
      ));
      if (command === runtimeNode && isWorker) {{
        const env = {{ ...(options?.env || process.env) }};
        for (const key of inheritedKeys) {{
          if (process.env[key] !== undefined) env[key] = process.env[key];
        }}
        const preload = '--require {PROXY_PRELOAD_PATH}';
        if (!(env.NODE_OPTIONS || '').includes(preload)) {{
          env.NODE_OPTIONS = `${{preload}} ${{env.NODE_OPTIONS || ''}}`.trim();
        }}
        options = {{ ...options, env }};
      }}
      return spawn.call(this, command, args, options);
    }};
    childProcess.spawn = wrapSpawn(childProcess.spawn);
    syncBuiltinESMExports();
    if ((process.argv[1] || '').endsWith('/sidecar-entry.js') && process.platform !== 'win32') {{
      const pty = createRequire(process.argv[1])('@lydell/node-pty');
      pty.spawn = wrapSpawn(pty.spawn);
    }}
  }}
  const undici = require('{PROXY_PRELOAD_DIR}/node_modules/undici');
  const {{ setGlobalDispatcher, EnvHttpProxyAgent, ProxyAgent, Agent }} = undici;
  const hosts = (process.env.NEXUS_PROXY_HOSTS || '')
    .split(',')
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);
  if (hosts.length === 0) {{
    setGlobalDispatcher(new EnvHttpProxyAgent());
  }} else {{
    const direct = new Agent();
    const proxied = new ProxyAgent(proxy);
    const matches = (hostname) => hosts.some(
      (host) => hostname === host || hostname.endsWith('.' + host),
    );
    setGlobalDispatcher({{
      dispatch(opts, handler) {{
        let hostname = '';
        try {{ hostname = new URL(opts.origin).hostname.toLowerCase(); }} catch (error) {{}}
        return (matches(hostname) ? proxied : direct).dispatch(opts, handler);
      }},
      close() {{ return Promise.all([direct.close(), proxied.close()]); }},
      destroy(error) {{ return Promise.all([direct.destroy(error), proxied.destroy(error)]); }},
    }});
  }}
}}
JS
"""


def with_proxy_preload(extra_env: dict[str, str] | None) -> dict[str, str]:
    """Return ``extra_env`` with the preload put in front of any ``NODE_OPTIONS`` it already sets."""
    env = dict(extra_env or {})
    configured = env.get('NODE_OPTIONS', '').strip()
    env['NODE_OPTIONS'] = f'{PROXY_PRELOAD_NODE_OPTION} {configured}'.strip()
    return env


async def install_proxy_preload(runtime: Any) -> None:
    """Install the preload in the sandbox; rerunning only rewrites the preload file.

    Raises:
        RuntimeError: The install command failed.
    """
    result = await runtime.run_command(
        PROXY_PRELOAD_SETUP, timeout=PROXY_PRELOAD_TIMEOUT_S, extra_env={'NODE_OPTIONS': ''}
    )
    if result.return_code != 0:
        detail = (result.stderr or result.stdout or str(result.status)).strip()[-1000:]
        raise RuntimeError(f'Node proxy preload install failed: {detail}')
    logger.info('Node proxy preload ready at %s', PROXY_PRELOAD_PATH)


__all__ = ['PROXY_PRELOAD_NODE_OPTION', 'PROXY_PRELOAD_PATH', 'install_proxy_preload', 'with_proxy_preload']
