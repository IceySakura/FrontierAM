#!/usr/bin/env bash
# Register a per-user service using the current desktop session environment.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <manifest-url-or-path>" >&2
    exit 2
fi
source_url="$1"
package_dir="$(cd "$(dirname "$0")" && pwd)"
install_dir="$HOME/.local/lib/nexus-bootstrap"
state_dir="$HOME/.local/state/nexus-bootstrap"
command -v bwrap >/dev/null
command -v python3 >/dev/null
command -v systemctl >/dev/null
mkdir -p "$install_dir" "$state_dir" "$HOME/.config/systemd/user"
install -m 755 "$package_dir/nexus-bootstrap" "$install_dir/nexus-bootstrap"
# EnvironmentFile is parsed by systemd, not sourced as shell code.
python3 - "$install_dir" "$state_dir" "$source_url" <<'PY'
import json, os, sys
from pathlib import Path
install, state, source = sys.argv[1:]
values = {'NEXUS_BOOTSTRAP_SOURCE': source, 'NEXUS_BOOTSTRAP_STATE': state}
for key in ('DISPLAY', 'XAUTHORITY', 'XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS', 'NEXUS_RUNTIME_PORT'):
    if key in os.environ:
        values[key] = os.environ[key]
config = Path(install) / 'bootstrap.env'
config.write_text(''.join(k + '=' + json.dumps(v, ensure_ascii=False) + '\n' for k, v in values.items()))
config.chmod(0o600)
PY
cat > "$install_dir/run.sh" <<'RUN'
#!/bin/sh
exec "$HOME/.local/lib/nexus-bootstrap/nexus-bootstrap" \
    --source "$NEXUS_BOOTSTRAP_SOURCE" --state-root "$NEXUS_BOOTSTRAP_STATE" --allow-unsigned
RUN
chmod 755 "$install_dir/run.sh"
cat > "$HOME/.config/systemd/user/nexus-bootstrap.service" <<'UNIT'
[Unit]
Description=Nexus runtime bootstrap

[Service]
EnvironmentFile=%h/.local/lib/nexus-bootstrap/bootstrap.env
ExecStart=%h/.local/lib/nexus-bootstrap/run.sh
Restart=on-failure
RestartSec=5
KillMode=control-group

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now nexus-bootstrap.service
