#!/usr/bin/env bash
set -euo pipefail

# Taiji writes its init completion marker under /data after this command.
mkdir -p /data

# Temporary Taiji workaround: the platform-generated network setup runs under
# `set -e` and writes kernel sysctls. Unprivileged containers expose those
# sysctls as read-only, so setup exits before the user command can start even
# though these tuning failures do not prevent Redis from running.
#
# Append `|| true` to every generated `sysctl` command. This does not change
# kernel settings or disable error handling for the rest of the script. Remove
# this patch once Taiji treats unsupported sysctls as non-fatal itself.
# `TAIJI_NETWORK_ENVS` exists only to override the platform path for testing.
NETWORK_ENVS="${TAIJI_NETWORK_ENVS:-/root/dl/runtime/script/network_envs.sh}"
if [[ ! -f "$NETWORK_ENVS" ]]; then
  echo "Taiji network environment script not found: $NETWORK_ENVS" >&2
  exit 1
fi

# Use embedded Python for an idempotent, line-preserving edit of Taiji's
# generated shell script; already-patched commands are left unchanged.
python3 - "$NETWORK_ENVS" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
lines = path.read_text().splitlines(keepends=True)
patched = 0
found = 0
for index, line in enumerate(lines):
    command = line.strip()
    if command.startswith('sysctl '):
        found += 1
        if not command.endswith('|| true'):
            newline = '\n' if line.endswith('\n') else ''
            lines[index] = f'{line.rstrip()} || true{newline}'
            patched += 1
if not found:
    raise SystemExit(f'{path}: no sysctl commands were found')
if patched:
    path.write_text(''.join(lines))
print(f'{path}: {found} sysctl commands found, {patched} made tolerant')
PY

# This is the Python client used for Cluster topology discovery and slot
# routing; server binaries are checked/installed on each node by ensure_redis.
python3 -m pip install 'redis==8.1.0'
