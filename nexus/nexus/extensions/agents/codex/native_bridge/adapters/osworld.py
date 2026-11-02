"""AGS OSWorld installation adapter for the native Codex execution kernel."""

from __future__ import annotations

import json

from nexus.runtime.client.base import ExecutorMixin
from nexus.runtime.executors.models.command import CommandRunStatus
from nexus.runtime.providers.ags.osworld import AGSOSWorldExecutorRuntime, require_ags_osworld_runtime

from .posix import BasePosixCodexNativeBridgeAdapter


_NODE_DIR = '/home/user/node20'
_NODE_URL = 'https://nodejs.org/dist/v20.18.0/node-v20.18.0-linux-x64.tar.xz'
_ENV_MARKER = 'NEXUS_CODEX_NODE_ENV='
# Prefer a usable managed or system Node; bootstrap only when neither works.
NODE_BOOTSTRAP = f"""
set -eu
node_ok() {{
  "$1" -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 16 ? 0 : 1)' >/dev/null 2>&1
}}
if [ -x {_NODE_DIR}/bin/npm ] && node_ok {_NODE_DIR}/bin/node; then
  export PATH={_NODE_DIR}/bin:$PATH
elif command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1 && node_ok "$(command -v node)"; then
  export PATH="$(dirname "$(command -v node)"):$(dirname "$(command -v npm)"):$PATH"
else
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) ;;
    *) echo 'Managed OSWorld Node bootstrap requires Linux x86_64' >&2; exit 2 ;;
  esac
  node_archive=$(mktemp /tmp/nexus-codex-node.XXXXXX)
  trap 'rm -f "$node_archive"' EXIT
  if command -v curl >/dev/null 2>&1; then
    curl -fLsS --connect-timeout 30 --max-time 180 {_NODE_URL} -o "$node_archive"
  else
    wget -nv --timeout=90 {_NODE_URL} -O "$node_archive"
  fi
  mkdir -p {_NODE_DIR}
  tar -xf "$node_archive" -C {_NODE_DIR} --strip-components=1
  export PATH={_NODE_DIR}/bin:$PATH
fi
node_ok "$(command -v node)"
npm --version >/dev/null
node -e 'console.log("{_ENV_MARKER}" + JSON.stringify({{PATH: process.env.PATH, node: process.execPath, version: process.versions.node}}))'
"""


async def prepare_node_env(runtime: ExecutorMixin) -> dict[str, str]:
    result = await runtime.run_command(command=NODE_BOOTSTRAP, timeout=900.0)
    if result.status != CommandRunStatus.Finished or result.return_code != 0:
        raise RuntimeError(f'Codex Node preparation failed: status={result.status} rc={result.return_code}')
    for line in reversed((result.stdout or '').splitlines()):
        if not line.startswith(_ENV_MARKER):
            continue
        try:
            value = json.loads(line[len(_ENV_MARKER) :])
            path, executable, version = value['PATH'], value['node'], value['version']
            if not all(isinstance(item, str) and item for item in (path, executable, version)):
                raise ValueError('missing Node environment')
            if not executable.startswith('/') or int(version.split('.')[0]) < 16 or '\n' in path:
                raise ValueError('unsupported Node environment')
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError('Codex Node preparation returned an invalid environment') from exc
        return {'PATH': path}
    raise RuntimeError('Codex Node preparation did not report its environment')


class OSWorldCodexNativeBridgeAdapter(BasePosixCodexNativeBridgeAdapter):
    def __init__(self, runtime: AGSOSWorldExecutorRuntime, *, benchmark_policy: bool = False):
        super().__init__(require_ags_osworld_runtime(runtime))
        self.benchmark_policy = benchmark_policy

    async def prepare_install(self, spec):
        if self.benchmark_policy and (
            (spec.binary_url and spec.binary_url.split('?')[0].endswith('.tar.gz'))
            or (not spec.binary_url and await self._has_npm())
        ):
            self.execution_env = await prepare_node_env(self.runtime)

    async def _has_npm(self):
        result = await self.runtime.run_command(command='command -v npm', timeout=30)
        return result.return_code == 0 and bool((result.stdout or '').strip())

    async def _npm_available(self):
        if self.benchmark_policy:
            return await self._has_npm()
        return await super()._npm_available()

    def install_registries(self, primary):
        if self.benchmark_policy:
            return list(dict.fromkeys([primary.rstrip('/'), 'https://registry.npmmirror.com']))
        return [primary]

    async def execute_install(self, command, timeout=1800):
        if self.benchmark_policy:
            return await self._run_install_command(command, timeout=timeout)
        return await super().execute_install(command, timeout=timeout)
