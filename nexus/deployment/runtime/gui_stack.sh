#!/usr/bin/env bash

set -euo pipefail

export PATH="/nix/runtime/nix-env/bin:$PATH"
export FONTCONFIG_FILE=/nix/runtime/nix-env/etc/fonts/fonts.conf
export LD_LIBRARY_PATH=/nix/runtime/nix-env/lib
export NEXUS_GUI_DISPLAY="${NEXUS_GUI_DISPLAY:-:99}"
export NEXUS_GUI_SCREEN_WIDTH="${NEXUS_GUI_SCREEN_WIDTH:-1440}"
export NEXUS_GUI_SCREEN_HEIGHT="${NEXUS_GUI_SCREEN_HEIGHT:-900}"
export NEXUS_GUI_SCREEN_DEPTH="${NEXUS_GUI_SCREEN_DEPTH:-24}"
export NEXUS_GUI_XVFB_COMMAND="${NEXUS_GUI_XVFB_COMMAND:-}"
export NEXUS_GUI_VNC_HOST="${NEXUS_GUI_VNC_HOST:-127.0.0.1}"
export NEXUS_GUI_VNC_PORT="${NEXUS_GUI_VNC_PORT:-5900}"
export NEXUS_GUI_NOVNC_HOST="${NEXUS_GUI_NOVNC_HOST:-0.0.0.0}"
export NEXUS_GUI_NOVNC_PORT="${NEXUS_GUI_NOVNC_PORT:-6080}"
export NEXUS_GUI_WINDOW_MANAGER="${NEXUS_GUI_WINDOW_MANAGER:-none}"
export NEXUS_GUI_DESKTOP_SESSION="${NEXUS_GUI_DESKTOP_SESSION:-}"
export NEXUS_GUI_GNOME_SESSION_NAME="${NEXUS_GUI_GNOME_SESSION_NAME:-gnome}"
export NEXUS_GUI_APPLY_UBUNTU_THEME="${NEXUS_GUI_APPLY_UBUNTU_THEME:-1}"
export NEXUS_GUI_SESSION_COMMAND="${NEXUS_GUI_SESSION_COMMAND:-}"
export NEXUS_GUI_SESSION_USER="${NEXUS_GUI_SESSION_USER:-}"
export NEXUS_GUI_REQUIRE_SYSTEM_DBUS="${NEXUS_GUI_REQUIRE_SYSTEM_DBUS:-0}"
export DISPLAY="$NEXUS_GUI_DISPLAY"
export XAUTHORITY="${XAUTHORITY:-/tmp/nexus.Xauthority}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/nexus-xdg-runtime}"
export XDG_SESSION_TYPE="${XDG_SESSION_TYPE:-x11}"
export GDK_BACKEND="${GDK_BACKEND:-x11}"
export GSK_RENDERER="${GSK_RENDERER:-cairo}"
export CLUTTER_BACKEND="${CLUTTER_BACKEND:-x11}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export MESA_LOADER_DRIVER_OVERRIDE="${MESA_LOADER_DRIVER_OVERRIDE:-llvmpipe}"

touch "$XAUTHORITY"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

gui_novnc_ready() {
    curl -fsS "http://127.0.0.1:${NEXUS_GUI_NOVNC_PORT}/vnc.html" >/dev/null 2>&1
}

start_system_dbus() {
    if [[ -S /run/dbus/system_bus_socket ]]; then
        return 0
    fi
    if ! command -v dbus-daemon >/dev/null 2>&1; then
        echo "dbus-daemon not found; skipping system D-Bus startup"
        if [[ "$NEXUS_GUI_DESKTOP_SESSION" == "gnome-shell" || "$NEXUS_GUI_DESKTOP_SESSION" == "gnome-session" || "$NEXUS_GUI_REQUIRE_SYSTEM_DBUS" == "1" ]]; then
            return 1
        fi
        return 0
    fi

    mkdir -p /run/dbus
    echo "Starting system D-Bus..."
    if dbus-daemon --system --fork --nopidfile; then
        return 0
    fi

    echo "System D-Bus failed to start." >&2
    if [[ "$NEXUS_GUI_DESKTOP_SESSION" == "gnome-shell" || "$NEXUS_GUI_DESKTOP_SESSION" == "gnome-session" || "$NEXUS_GUI_REQUIRE_SYSTEM_DBUS" == "1" ]]; then
        return 1
    fi
    return 0
}

start_optional_service() {
    local name="$1"
    local command="$2"
    local log_file="$3"

    echo "Starting optional GNOME service ${name}: ${command}"
    /nix/runtime/nix-env/bin/bash -lc "$command" >"$log_file" 2>&1 &
}

start_gnome_system_services() {
    if [[ "$NEXUS_GUI_DESKTOP_SESSION" != "gnome-session" ]]; then
        return 0
    fi

    mkdir -p /run/systemd /run/systemd/seats /run/systemd/sessions /run/systemd/users
    if command -v systemd-machine-id-setup >/dev/null 2>&1 && [[ ! -s /etc/machine-id ]]; then
        systemd-machine-id-setup >/tmp/nexus-machine-id.log 2>&1 || true
    fi

    if command -v systemd-logind >/dev/null 2>&1 && ! pgrep -x systemd-logind >/dev/null 2>&1; then
        start_optional_service "systemd-logind" "systemd-logind" "/tmp/nexus-logind.log"
    elif [[ -x /lib/systemd/systemd-logind ]] && ! pgrep -x systemd-logind >/dev/null 2>&1; then
        start_optional_service "systemd-logind" "/lib/systemd/systemd-logind" "/tmp/nexus-logind.log"
    fi

    if command -v polkitd >/dev/null 2>&1 && ! pgrep -x polkitd >/dev/null 2>&1; then
        start_optional_service "polkitd" "polkitd --no-debug" "/tmp/nexus-polkitd.log"
    elif [[ -x /usr/lib/polkit-1/polkitd ]] && ! pgrep -x polkitd >/dev/null 2>&1; then
        start_optional_service "polkitd" "/usr/lib/polkit-1/polkitd --no-debug" "/tmp/nexus-polkitd.log"
    fi

    if command -v accounts-daemon >/dev/null 2>&1 && ! pgrep -x accounts-daemon >/dev/null 2>&1; then
        start_optional_service "accounts-daemon" "accounts-daemon" "/tmp/nexus-accounts-daemon.log"
    elif [[ -x /usr/libexec/accounts-daemon ]] && ! pgrep -x accounts-daemon >/dev/null 2>&1; then
        start_optional_service "accounts-daemon" "/usr/libexec/accounts-daemon" "/tmp/nexus-accounts-daemon.log"
    fi

    for _ in $(seq 1 20); do
        if dbus-send \
            --system \
            --dest=org.freedesktop.login1 \
            --type=method_call \
            --print-reply \
            /org/freedesktop/login1 \
            org.freedesktop.DBus.Peer.Ping \
            >/dev/null 2>&1; then
            echo "systemd-logind is ready."
            return 0
        fi
        sleep 0.25
    done

    echo "systemd-logind did not become ready; continuing with GNOME non-systemd fallback." >&2
    tail -n 100 /tmp/nexus-logind.log >&2 || true
    return 0
}

start_window_manager() {
    case "$NEXUS_GUI_WINDOW_MANAGER" in
        ""|"none")
            echo "Skipping window manager because NEXUS_GUI_WINDOW_MANAGER=${NEXUS_GUI_WINDOW_MANAGER:-none}"
            ;;
        *)
            echo "Starting custom window manager: ${NEXUS_GUI_WINDOW_MANAGER}"
            /nix/runtime/nix-env/bin/bash -lc "$NEXUS_GUI_WINDOW_MANAGER" >/tmp/nexus-window-manager.log 2>&1 &
            ;;
    esac
}

ensure_session_user() {
    if [[ -z "$NEXUS_GUI_SESSION_USER" ]]; then
        return 0
    fi

    if id "$NEXUS_GUI_SESSION_USER" >/dev/null 2>&1; then
        return 0
    fi

    if ! getent group "$NEXUS_GUI_SESSION_USER" >/dev/null 2>&1; then
        groupadd --gid 1000 "$NEXUS_GUI_SESSION_USER" 2>/dev/null || true
    fi
    useradd --uid 1000 --create-home --shell /bin/bash "$NEXUS_GUI_SESSION_USER" 2>/dev/null \
        || useradd --create-home --shell /bin/bash "$NEXUS_GUI_SESSION_USER"
}

start_desktop_session_as_user() {
    if [[ -z "$NEXUS_GUI_SESSION_USER" ]]; then
        /nix/runtime/nix-env/bin/bash -lc "$NEXUS_GUI_SESSION_COMMAND" >/tmp/nexus-gui-session.log 2>&1 &
        return 0
    fi

    ensure_session_user

    if ! id "$NEXUS_GUI_SESSION_USER" >/dev/null 2>&1; then
        echo "NEXUS_GUI_SESSION_USER=${NEXUS_GUI_SESSION_USER} does not exist." >&2
        return 1
    fi

    local session_uid=""
    session_uid="$(id -u "$NEXUS_GUI_SESSION_USER")"
    local session_runtime_dir="/run/user/${session_uid}"
    mkdir -p "$session_runtime_dir" "/home/$NEXUS_GUI_SESSION_USER"
    chown "$NEXUS_GUI_SESSION_USER:$NEXUS_GUI_SESSION_USER" "$session_runtime_dir" "/home/$NEXUS_GUI_SESSION_USER"
    chmod 700 "$session_runtime_dir"

    echo "Starting desktop session as ${NEXUS_GUI_SESSION_USER} with HOME=/home/${NEXUS_GUI_SESSION_USER}"
    runuser -u "$NEXUS_GUI_SESSION_USER" -- env \
        DISPLAY="$DISPLAY" \
        XAUTHORITY="$XAUTHORITY" \
        XDG_RUNTIME_DIR="$session_runtime_dir" \
        XDG_SESSION_TYPE="$XDG_SESSION_TYPE" \
        GDK_BACKEND="$GDK_BACKEND" \
        GSK_RENDERER="$GSK_RENDERER" \
        CLUTTER_BACKEND="$CLUTTER_BACKEND" \
        LIBGL_ALWAYS_SOFTWARE="$LIBGL_ALWAYS_SOFTWARE" \
        MESA_LOADER_DRIVER_OVERRIDE="$MESA_LOADER_DRIVER_OVERRIDE" \
        XDG_CURRENT_DESKTOP="${XDG_CURRENT_DESKTOP:-GNOME}" \
        DESKTOP_SESSION="${DESKTOP_SESSION:-$NEXUS_GUI_GNOME_SESSION_NAME}" \
        GNOME_SHELL_SESSION_MODE="${GNOME_SHELL_SESSION_MODE:-$NEXUS_GUI_GNOME_SESSION_NAME}" \
        HOME="/home/$NEXUS_GUI_SESSION_USER" \
        USER="$NEXUS_GUI_SESSION_USER" \
        LOGNAME="$NEXUS_GUI_SESSION_USER" \
        PATH="$PATH" \
        FONTCONFIG_FILE="$FONTCONFIG_FILE" \
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        /nix/runtime/nix-env/bin/bash -lc "$NEXUS_GUI_SESSION_COMMAND" >/tmp/nexus-gui-session.log 2>&1 &
}

start_desktop_session() {
    if [[ "$NEXUS_GUI_DESKTOP_SESSION" == "gnome-shell" && -z "$NEXUS_GUI_SESSION_COMMAND" ]]; then
        if ! command -v gnome-shell >/dev/null 2>&1; then
            echo "NEXUS_GUI_DESKTOP_SESSION=gnome-shell requested, but gnome-shell was not found." >&2
            return 1
        fi

        NEXUS_GUI_SESSION_COMMAND="dbus-run-session -- ${GUI_BASH:-/nix/runtime/nix-env/bin/bash} -lc 'gnome-shell --x11 --replace'"
    fi
    if [[ "$NEXUS_GUI_DESKTOP_SESSION" == "gnome-session" && -z "$NEXUS_GUI_SESSION_COMMAND" ]]; then
        if ! command -v gnome-session >/dev/null 2>&1; then
            echo "NEXUS_GUI_DESKTOP_SESSION=gnome-session requested, but gnome-session was not found." >&2
            return 1
        fi

        export XDG_CURRENT_DESKTOP="${XDG_CURRENT_DESKTOP:-GNOME}"
        export DESKTOP_SESSION="${DESKTOP_SESSION:-$NEXUS_GUI_GNOME_SESSION_NAME}"
        export GNOME_SHELL_SESSION_MODE="${GNOME_SHELL_SESSION_MODE:-$NEXUS_GUI_GNOME_SESSION_NAME}"
        NEXUS_GUI_SESSION_COMMAND="dbus-run-session -- ${GUI_BASH:-/nix/runtime/nix-env/bin/bash} -lc 'gnome-session --disable-acceleration-check --session=${NEXUS_GUI_GNOME_SESSION_NAME}'"
    fi

    if [[ -z "$NEXUS_GUI_SESSION_COMMAND" ]]; then
        return 0
    fi
    if pgrep -f '[g]nome-session-binary|[g]nome-shell' >/dev/null 2>&1; then
        echo "GUI desktop session is already running."
        return 0
    fi

    echo "Starting GUI desktop session: ${NEXUS_GUI_SESSION_COMMAND}"
    start_desktop_session_as_user
}

apply_ubuntu_theme_settings() {
    if [[ "$NEXUS_GUI_DESKTOP_SESSION" != "gnome-session" || "$NEXUS_GUI_GNOME_SESSION_NAME" != "ubuntu" ]]; then
        return 0
    fi
    if [[ "$NEXUS_GUI_APPLY_UBUNTU_THEME" != "1" ]]; then
        return 0
    fi

    local session_pid=""
    session_pid="$(pgrep -f '[g]nome-session-binary' | head -n 1 || true)"
    if [[ -z "$session_pid" ]]; then
        echo "Unable to apply Ubuntu theme settings because gnome-session-binary was not found." >&2
        return 0
    fi

    local session_user=""
    session_user="$(ps -o user= -p "$session_pid" | tr -d ' ' || true)"
    if [[ -n "$NEXUS_GUI_SESSION_USER" ]]; then
        session_user="$NEXUS_GUI_SESSION_USER"
    fi

    local session_bus=""
    if [[ -n "$session_user" && "$session_user" != "root" ]]; then
        session_bus="$(
            runuser -u "$session_user" -- bash -c \
                'tr "\0" "\n" <"/proc/$1/environ" | sed -n "s/^DBUS_SESSION_BUS_ADDRESS=//p" | head -n 1' \
                bash "$session_pid" || true
        )"
    else
        session_bus="$(tr '\0' '\n' <"/proc/${session_pid}/environ" | sed -n 's/^DBUS_SESSION_BUS_ADDRESS=//p' | head -n 1 || true)"
    fi
    if [[ -z "$session_bus" ]]; then
        echo "Unable to apply Ubuntu theme settings because DBUS_SESSION_BUS_ADDRESS was not found." >&2
        return 0
    fi

    local session_runtime_dir="$XDG_RUNTIME_DIR"
    if [[ -n "$session_user" && "$session_user" != "root" ]]; then
        session_runtime_dir="/run/user/$(id -u "$session_user")"
    fi

    run_session_gsettings() {
        if [[ -n "$session_user" && "$session_user" != "root" ]]; then
            runuser -u "$session_user" -- env \
                DISPLAY="$DISPLAY" \
                XDG_RUNTIME_DIR="$session_runtime_dir" \
                DBUS_SESSION_BUS_ADDRESS="$session_bus" \
                "$@" >/dev/null 2>&1 || true
        else
            DBUS_SESSION_BUS_ADDRESS="$session_bus" "$@" >/dev/null 2>&1 || true
        fi
    }

    echo "Applying Ubuntu desktop theme settings..."
    run_session_gsettings gsettings set org.gnome.desktop.interface gtk-theme Yaru-dark
    run_session_gsettings gsettings set org.gnome.desktop.interface icon-theme Yaru
    run_session_gsettings gsettings set org.gnome.desktop.interface cursor-theme Yaru
    run_session_gsettings gsettings set org.gnome.desktop.interface font-name "Ubuntu 11"
    run_session_gsettings gsettings set org.gnome.desktop.interface document-font-name "Ubuntu 11"
    run_session_gsettings gsettings set org.gnome.desktop.interface monospace-font-name "Ubuntu Mono 13"
    run_session_gsettings gsettings set org.gnome.desktop.interface color-scheme prefer-dark
    run_session_gsettings gsettings set org.gnome.desktop.background picture-uri "file:///usr/share/backgrounds/Numbat_wallpaper_light_3480x2160.png"
    run_session_gsettings gsettings set org.gnome.desktop.background picture-uri-dark "file:///usr/share/backgrounds/Numbat_wallpaper_light_3480x2160.png"
    run_session_gsettings gsettings set org.gnome.desktop.screensaver picture-uri "file:///usr/share/backgrounds/Numbat_wallpaper_light_3480x2160.png"
    run_session_gsettings gsettings set org.gnome.shell enabled-extensions "['ubuntu-dock@ubuntu.com', 'ubuntu-appindicators@ubuntu.com', 'ding@rastersoft.com']"
    run_session_gsettings gsettings set org.gnome.shell favorite-apps "['google-chrome.desktop', 'gimp.desktop', 'libreoffice-calc.desktop', 'libreoffice-impress.desktop', 'libreoffice-writer.desktop', 'thunderbird.desktop', 'vlc.desktop', 'code.desktop']"
    run_session_gsettings gsettings set org.gnome.shell.extensions.dash-to-dock dock-position LEFT
    run_session_gsettings gsettings set org.gnome.shell.extensions.dash-to-dock extend-height true
    run_session_gsettings gsettings set org.gnome.shell.extensions.dash-to-dock dock-fixed true
    run_session_gsettings gsettings set org.gnome.shell.extensions.dash-to-dock show-trash true
    run_session_gsettings gsettings set org.gnome.shell.extensions.dash-to-dock show-mounts true
    run_session_gsettings gsettings set org.gnome.shell.extensions.dash-to-dock dash-max-icon-size 48
    run_session_gsettings gsettings set org.gnome.shell.extensions.ding show-home true
    run_session_gsettings gsettings set org.gnome.shell.extensions.ding show-trash true
    run_session_gsettings gnome-extensions enable ubuntu-dock@ubuntu.com
    run_session_gsettings gnome-extensions enable ubuntu-appindicators@ubuntu.com
    run_session_gsettings gnome-extensions enable ding@rastersoft.com
}

hide_gnome_overview() {
    if [[ "$NEXUS_GUI_DESKTOP_SESSION" != "gnome-shell" && "$NEXUS_GUI_DESKTOP_SESSION" != "gnome-session" ]]; then
        return 0
    fi
    if ! command -v xdotool >/dev/null 2>&1; then
        return 0
    fi

    # GNOME can start in the Activities overview inside Xvfb; wait until Shell can receive keys.
    sleep 1
    DISPLAY="$NEXUS_GUI_DISPLAY" xdotool key Escape >/dev/null 2>&1 || true
    sleep 0.5
    DISPLAY="$NEXUS_GUI_DISPLAY" xdotool key Escape >/dev/null 2>&1 || true
}

wait_for_desktop_session() {
    if [[ "$NEXUS_GUI_DESKTOP_SESSION" != "gnome-shell" && "$NEXUS_GUI_DESKTOP_SESSION" != "gnome-session" ]]; then
        return 0
    fi

    for _ in $(seq 1 60); do
        if command -v ps >/dev/null 2>&1 && ps -ef | grep -q '[g]nome-shell'; then
            echo "GNOME desktop session is ready."
            apply_ubuntu_theme_settings
            hide_gnome_overview
            return 0
        fi
        sleep 0.5
    done

    echo "GNOME desktop session failed to become ready. Recent logs:" >&2
    tail -n 100 /tmp/nexus-gui-session.log >&2 || true
    return 1
}

wait_for_display() {
    local display_ready=0
    for _ in $(seq 1 20); do
        if xrandr >/dev/null 2>&1; then
            display_ready=1
            break
        fi
        sleep 0.25
    done

    if [[ "$display_ready" != "1" ]]; then
        echo "Xvfb failed to become ready. Recent logs:" >&2
        tail -n 100 /tmp/nexus-xvfb.log >&2 || true
        return 1
    fi
}

start_xvfb() {
    if xrandr >/dev/null 2>&1; then
        echo "Xvfb is already ready on ${DISPLAY}."
        return 0
    fi

    local xvfb_command="$NEXUS_GUI_XVFB_COMMAND"
    if [[ -z "$xvfb_command" && -x /usr/bin/Xvfb ]]; then
        xvfb_command=/usr/bin/Xvfb
    fi
    if [[ -z "$xvfb_command" ]]; then
        xvfb_command="$(command -v Xvfb)"
    fi

    local xvfb_args=(
        "${DISPLAY}"
        -screen 0 "${NEXUS_GUI_SCREEN_WIDTH}x${NEXUS_GUI_SCREEN_HEIGHT}x${NEXUS_GUI_SCREEN_DEPTH}"
        -ac
        +extension RANDR
        +extension COMPOSITE
        +extension RENDER
        +extension DAMAGE
        +extension DPMS
    )
    if "$xvfb_command" -help 2>&1 | grep -Eq '(^|[[:space:]])GLX($|[[:space:]])'; then
        xvfb_args+=(+extension GLX +iglx)
    fi

    echo "Starting Xvfb on ${DISPLAY}..."
    "$xvfb_command" "${xvfb_args[@]}" >/tmp/nexus-xvfb.log 2>&1 &

    wait_for_display
}

start_vnc_stack() {
    if gui_novnc_ready; then
        echo "noVNC is already ready."
        return 0
    fi

    if ! pgrep -f "[x]11vnc .*${DISPLAY}" >/dev/null 2>&1; then
        echo "Starting x11vnc on ${NEXUS_GUI_VNC_HOST}:${NEXUS_GUI_VNC_PORT}..."
        x11vnc \
            -display "${DISPLAY}" \
            -rfbport "${NEXUS_GUI_VNC_PORT}" \
            -listen "${NEXUS_GUI_VNC_HOST}" \
            -forever \
            -shared \
            -nopw \
            >/tmp/nexus-x11vnc.log 2>&1 &
    fi

    if ! pgrep -f "[n]ovnc .*${NEXUS_GUI_NOVNC_PORT}" >/dev/null 2>&1; then
        echo "Starting noVNC on ${NEXUS_GUI_NOVNC_HOST}:${NEXUS_GUI_NOVNC_PORT}..."
        novnc \
            --listen "${NEXUS_GUI_NOVNC_PORT}" \
            --vnc "${NEXUS_GUI_VNC_HOST}:${NEXUS_GUI_VNC_PORT}" \
            >/tmp/nexus-novnc.log 2>&1 &
    fi

    for _ in $(seq 1 60); do
        if gui_novnc_ready; then
            echo "noVNC is ready."
            return 0
        fi
        sleep 0.5
    done

    echo "GUI stack failed to become ready. Recent logs:" >&2
    tail -n 100 /tmp/nexus-x11vnc.log >&2 || true
    tail -n 100 /tmp/nexus-novnc.log >&2 || true
    return 1
}

start_gui_stack() {
    if [[ "${NEXUS_DISABLE_GUI_STACK:-0}" == "1" ]]; then
        echo "GUI stack disabled by NEXUS_DISABLE_GUI_STACK=1"
        return 0
    fi

    if ! start_system_dbus; then
        echo "System D-Bus startup failed." >&2
        return 1
    fi
    if ! start_gnome_system_services; then
        echo "GNOME system service startup failed." >&2
        return 1
    fi
    if ! start_xvfb; then
        return 1
    fi
    if ! start_window_manager; then
        echo "Window manager startup failed." >&2
        return 1
    fi
    if ! start_desktop_session; then
        echo "Desktop session startup failed." >&2
        return 1
    fi
    if ! wait_for_desktop_session; then
        return 1
    fi
    xsetroot -solid "#1d2330" >/dev/null 2>&1 || true

    start_vnc_stack
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    command="${1:-start}"
    case "$command" in
        start)
            start_gui_stack
            ;;
        status)
            gui_novnc_ready
            ;;
        *)
            echo "Usage: $0 [start|status]" >&2
            exit 2
            ;;
    esac
fi
