"""Unit coverage for shared Linux GUI session shell helpers."""

import shlex
import shutil
import subprocess

import pytest


def test_session_user_command_can_create_user_and_run_in_home():
    from nexus.runtime.utils.gui import session_user_command

    command = session_user_command(
        'printf hello',
        user='user',
        home_dir='/home/user',
        bash='/bin/bash',
        display=':99',
        missing_user_policy='create',
    )

    assert 'useradd --uid 1000 --create-home --shell /bin/bash user' in command
    assert 'runuser -u user' in command
    assert 'DISPLAY="${DISPLAY:-:99}"' in command
    assert 'XDG_RUNTIME_DIR="/run/user/${uid}"' in command
    assert 'DBUS_SESSION_BUS_ADDRESS=$session_bus' in command
    assert 'pgrep -u "$(id -u)" -f "[g]nome-session-binary|[g]nome-shell"' in command
    assert 'XDG_CURRENT_DESKTOP="${XDG_CURRENT_DESKTOP:-GNOME}"' in command
    assert 'DESKTOP_SESSION="${DESKTOP_SESSION:-${NEXUS_GUI_GNOME_SESSION_NAME:-ubuntu}}"' in command
    assert 'GNOME_SHELL_SESSION_MODE="${GNOME_SHELL_SESSION_MODE:-${NEXUS_GUI_GNOME_SESSION_NAME:-ubuntu}}"' in command
    assert 'GDK_BACKEND="${GDK_BACKEND:-x11}"' in command
    assert 'GSK_RENDERER="${GSK_RENDERER:-cairo}"' in command
    assert 'CLUTTER_BACKEND="${CLUTTER_BACKEND:-x11}"' in command
    assert 'HOME=/home/user USER=user LOGNAME=user' in command
    assert 'cd /home/user && printf hello' in command


def test_session_user_command_can_fallback_when_user_is_missing():
    from nexus.runtime.utils.gui import session_user_command

    command = session_user_command(
        'printf hello',
        user='user',
        home_dir='/home/user',
        missing_user_policy='fallback',
    )

    assert 'if id user >/dev/null 2>&1; then' in command
    assert 'else mkdir -p /home/user;' in command
    assert 'runuser -u user' in command
    assert 'DISPLAY="${DISPLAY:-:99}"' in command
    assert "bash -lc 'cd /home/user && printf hello'" in command
    assert 'XDG_CURRENT_DESKTOP="${XDG_CURRENT_DESKTOP:-GNOME}"' in command


def test_session_user_command_returns_plain_command_without_session_user():
    from nexus.runtime.utils.gui import session_user_command

    command = session_user_command('printf hello', user=None, cwd='/tmp')

    assert command == 'cd /tmp && printf hello'


@pytest.mark.skipif(shutil.which('bash') is None, reason='Requires Bash')
@pytest.mark.parametrize(('current_uid', 'switch_count'), [('1000', 0), ('0', 2)])
def test_session_user_command_only_switches_when_uid_differs(tmp_path, current_uid, switch_count):
    from nexus.runtime.utils.gui import session_user_command

    calls = tmp_path / 'runuser-calls'
    calls.touch()
    # Simulate identities without requiring root or modifying system directories.
    prelude = f"""
id() {{ if [ "$#" -gt 1 ]; then echo 1000; else echo {current_uid}; fi; }}
mkdir() {{ :; }}
chown() {{ :; }}
chmod() {{ :; }}
runuser() {{ printf 'switch\\n' >> {shlex.quote(str(calls))}; shift 3; "$@"; }}
"""
    command = session_user_command(
        'printf "%s\\n" SESSION_COMMAND_OK "$PWD"',
        user='desktop-test',
        home_dir=str(tmp_path),
        bash=shutil.which('bash') or 'bash',
    )
    result = subprocess.run(['bash', '-c', prelude + command], capture_output=True, text=True, check=True)

    assert result.stdout.splitlines() == ['SESSION_COMMAND_OK', str(tmp_path)]
    assert calls.read_text().splitlines() == ['switch'] * switch_count
