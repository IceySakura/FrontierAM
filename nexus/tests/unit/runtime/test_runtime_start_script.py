"""Unit coverage for runtime start script guardrails."""

import json

from pathlib import Path


def test_start_script_aborts_when_gui_stack_fails():
    script = Path('deployment/runtime/start.sh').read_text()

    assert 'set -euo pipefail' in script
    assert 'if [[ "${NEXUS_GUI_AUTO_START:-0}" == "1" ]]; then' in script
    assert 'if ! start_gui_stack; then' in script
    assert 'echo "GUI stack startup failed; aborting runtime server startup." >&2' in script
    assert 'exit 1' in script
    assert 'GUI stack auto-start disabled; call gui_start executor to enable desktop/VNC.' in script


def test_start_script_bootstraps_dind_only_when_enabled():
    script = Path('deployment/runtime/start.sh').read_text()
    bootstrap = Path('deployment/runtime/docker-bootstrap.sh').read_text()
    flake = Path('deployment/runtime/flake.nix').read_text()

    assert 'NEXUS_ENABLE_DIND_VALUE="${NEXUS_ENABLE_DIND:-}"' in script
    assert 'if [[ "${NEXUS_ENABLE_DIND_VALUE,,}" == "true" ]]; then' in script
    assert 'command -v dockerd' in script
    assert '/nix/runtime/nix-env/bin/docker-bootstrap.sh' in script
    assert 'docker bootstrap failed, aborting runtime server startup.' in script

    assert 'iptables --version' in bootstrap
    assert 'grep -q legacy' in bootstrap
    assert 'iptables -t nat -A POSTROUTING' in bootstrap
    assert '"iptables": false' in bootstrap
    assert '"ip6tables": false' in bootstrap
    assert '"storage-driver": "overlay2"' in bootstrap
    assert '"registry-mirrors"' in bootstrap
    assert 'https://mirror.ccs.tencentyun.com' in bootstrap
    assert 'https://dockerhub.woa.com' in bootstrap
    assert 'https://csighub.woa.com' in bootstrap
    assert '"data-root"' not in bootstrap

    assert 'docker-buildx' in flake
    assert 'docker-compose' in flake
    assert 'iproute2' in flake
    assert 'iptables-legacy' in flake
    assert 'docker_bootstrap' in flake


def test_dockerd_starts_with_empty_proxy_env():
    """Snapshot outer proxy for inner containers, then start dockerd with those vars empty."""
    bootstrap = Path('deployment/runtime/docker-bootstrap.sh').read_text()
    snapshot_idx = bootstrap.index('HTTP_PROXY HTTPS_PROXY FTP_PROXY NO_PROXY')
    launch_idx = bootstrap.index('HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= NO_PROXY= no_proxy= nohup dockerd')
    assert snapshot_idx < launch_idx


def test_daemon_json_applies_mtu_to_user_defined_bridge_networks():
    """Compose networks must inherit the DinD MTU instead of Docker's 1500 default."""
    bootstrap = Path('deployment/runtime/docker-bootstrap.sh').read_text()
    marker = 'cat >/etc/docker/daemon.json <<DAEMON_EOF\n'
    start = bootstrap.index(marker) + len(marker)
    template = bootstrap[start : bootstrap.index('\nDAEMON_EOF\n', start)]

    daemon = json.loads(template.replace('$mtu', '65549'))

    assert daemon['mtu'] == 65549
    assert daemon['default-network-opts'] == {'bridge': {'com.docker.network.driver.mtu': '65549'}}


def test_start_script_does_not_start_openbox_by_default_and_can_start_custom_session():
    script = Path('deployment/runtime/start.sh').read_text()
    gui_stack = Path('deployment/runtime/gui_stack.sh').read_text()
    flake = Path('deployment/runtime/flake.nix').read_text()

    assert 'source "$GUI_STACK_SCRIPT"' in script
    assert 'export NEXUS_GUI_WINDOW_MANAGER="${NEXUS_GUI_WINDOW_MANAGER:-none}"' in gui_stack
    assert 'openbox' not in script
    assert 'openbox' not in gui_stack
    assert 'openbox' not in flake
    assert 'export NEXUS_GUI_DESKTOP_SESSION="${NEXUS_GUI_DESKTOP_SESSION:-}"' in gui_stack
    assert 'export NEXUS_GUI_GNOME_SESSION_NAME="${NEXUS_GUI_GNOME_SESSION_NAME:-gnome}"' in gui_stack
    assert 'export NEXUS_GUI_SCREEN_WIDTH="${NEXUS_GUI_SCREEN_WIDTH:-1440}"' in gui_stack
    assert 'export NEXUS_GUI_SCREEN_HEIGHT="${NEXUS_GUI_SCREEN_HEIGHT:-900}"' in gui_stack
    assert 'export NEXUS_GUI_APPLY_UBUNTU_THEME="${NEXUS_GUI_APPLY_UBUNTU_THEME:-1}"' in gui_stack
    assert 'export NEXUS_GUI_SESSION_COMMAND="${NEXUS_GUI_SESSION_COMMAND:-}"' in gui_stack
    assert 'export NEXUS_GUI_SESSION_USER="${NEXUS_GUI_SESSION_USER:-}"' in gui_stack
    assert 'export NEXUS_GUI_XVFB_COMMAND="${NEXUS_GUI_XVFB_COMMAND:-}"' in gui_stack
    assert 'ensure_session_user' in gui_stack
    assert 'useradd --uid 1000 --create-home --shell /bin/bash "$NEXUS_GUI_SESSION_USER"' in gui_stack
    assert 'start_desktop_session_as_user' in gui_stack
    assert 'runuser -u "$NEXUS_GUI_SESSION_USER"' in gui_stack
    assert 'HOME="/home/$NEXUS_GUI_SESSION_USER"' in gui_stack
    assert 'runuser -u "$session_user" -- bash -c' in gui_stack
    assert 'run_session_gsettings gsettings set org.gnome.shell enabled-extensions' in gui_stack
    assert 'export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"' in gui_stack
    assert 'export GSK_RENDERER="${GSK_RENDERER:-cairo}"' in gui_stack
    assert 'export MESA_LOADER_DRIVER_OVERRIDE="${MESA_LOADER_DRIVER_OVERRIDE:-llvmpipe}"' in gui_stack
    assert 'start_system_dbus' in gui_stack
    assert 'if ! start_system_dbus; then' in gui_stack
    assert 'start_gnome_system_services' in gui_stack
    assert 'systemd-logind is ready.' in gui_stack
    assert 'systemd-logind did not become ready; continuing with GNOME non-systemd fallback.' in gui_stack
    assert 'polkitd --no-debug' in gui_stack
    assert 'accounts-daemon' in gui_stack
    assert 'local display_ready=0' in gui_stack
    assert 'Xvfb failed to become ready. Recent logs:' in gui_stack
    assert 'xvfb_command=/usr/bin/Xvfb' in gui_stack
    assert 'xvfb_args+=(+extension GLX +iglx)' in gui_stack
    assert 'NEXUS_GUI_DESKTOP_SESSION=gnome-shell requested' in gui_stack
    assert 'gnome-shell --x11 --replace' in gui_stack
    assert 'NEXUS_GUI_DESKTOP_SESSION=gnome-session requested' in gui_stack
    assert 'gnome-session --disable-acceleration-check --session=${NEXUS_GUI_GNOME_SESSION_NAME}' in gui_stack
    assert 'export XDG_CURRENT_DESKTOP="${XDG_CURRENT_DESKTOP:-GNOME}"' in gui_stack
    assert 'export DESKTOP_SESSION="${DESKTOP_SESSION:-$NEXUS_GUI_GNOME_SESSION_NAME}"' in gui_stack
    assert 'wait_for_desktop_session' in gui_stack
    assert 'apply_ubuntu_theme_settings' in gui_stack
    assert 'gsettings set org.gnome.desktop.interface gtk-theme Yaru-dark' in gui_stack
    assert 'gsettings set org.gnome.desktop.background picture-uri' in gui_stack
    assert (
        'gsettings set org.gnome.desktop.background picture-uri-dark "file:///usr/share/backgrounds/Numbat_wallpaper_light_3480x2160.png"'
        in gui_stack
    )
    assert "['ubuntu-dock@ubuntu.com', 'ubuntu-appindicators@ubuntu.com', 'ding@rastersoft.com']" in gui_stack
    assert 'gnome-extensions enable ubuntu-dock@ubuntu.com' in gui_stack
    assert 'gnome-extensions enable ding@rastersoft.com' in gui_stack
    assert 'gsettings set org.gnome.shell.extensions.dash-to-dock dock-position LEFT' in gui_stack
    assert (
        "['google-chrome.desktop', 'gimp.desktop', 'libreoffice-calc.desktop', "
        "'libreoffice-impress.desktop', 'libreoffice-writer.desktop', "
        "'thunderbird.desktop', 'vlc.desktop', 'code.desktop']"
    ) in gui_stack
    assert 'chromium.desktop' not in gui_stack
    assert 'hide_gnome_overview' in gui_stack
    assert 'sleep 1' in gui_stack
    assert 'xdotool key Escape' in gui_stack
    assert 'GNOME desktop session is ready.' in gui_stack
    assert 'GNOME desktop session failed to become ready. Recent logs:' in gui_stack
    assert '""|"none")' in gui_stack
    assert 'Skipping window manager because NEXUS_GUI_WINDOW_MANAGER=' in gui_stack
    assert 'start_window_manager' in gui_stack
    assert 'start_desktop_session' in gui_stack
    assert '/tmp/nexus-gui-session.log' in gui_stack


def test_gui_stack_script_can_be_started_by_executor():
    gui_stack = Path('deployment/runtime/gui_stack.sh').read_text()
    backend = Path('nexus/runtime/server/gui.py').read_text()
    executor = Path('nexus/runtime/executors/gui.py').read_text()

    assert 'if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then' in gui_stack
    assert 'start)' in gui_stack
    assert 'start_gui_stack' in gui_stack
    assert '_start_stack_sync' in backend
    assert "[GUI_BASH, str(script), 'start']" in backend
    assert "@executor_registry.register('gui_start')" in executor
