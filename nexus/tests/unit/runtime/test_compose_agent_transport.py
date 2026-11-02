import socket
import subprocess
import sys
import threading
import time

from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
RUNTIME_BIN = REPO_ROOT / 'deployment' / 'docker' / 'runtime-bin'
SOCKET_BRIDGE = RUNTIME_BIN / 'nexus-socket-bridge'
COMPOSE_EXEC = RUNTIME_BIN / 'nexus-compose-exec'


def test_socket_bridge_is_packaged_and_compose_helper_is_removed() -> None:
    pyproject = (REPO_ROOT / 'pyproject.toml').read_text()

    assert SOCKET_BRIDGE.is_file()
    assert SOCKET_BRIDGE.stat().st_mode & 0o111
    assert not COMPOSE_EXEC.exists()
    assert '"deployment/docker/runtime-bin" = "nexus/_bundle/deployment/docker/runtime-bin"' in pyproject


def test_socket_bridge_uses_dynamic_port_and_relays(tmp_path: Path) -> None:
    upstream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    upstream.bind(('127.0.0.1', 0))
    upstream.listen(1)
    upstream_port = upstream.getsockname()[1]

    def echo_once() -> None:
        connection, _ = upstream.accept()
        with connection:
            while data := connection.recv(65536):
                connection.sendall(data)
        upstream.close()

    thread = threading.Thread(target=echo_once, daemon=True)
    thread.start()
    ready_file = tmp_path / 'bridge.ready'
    source = SOCKET_BRIDGE.read_text().replace(
        '#!/nix/runtime/nix-env/bin/python3',
        f'#!{sys.executable}',
        1,
    )
    executable = tmp_path / 'nexus-socket-bridge'
    executable.write_text(source)
    executable.chmod(0o755)
    process = subprocess.Popen([
        str(executable),
        '--tcp-listen',
        '127.0.0.1:0',
        '--tcp-target',
        f'127.0.0.1:{upstream_port}',
        '--ready-file',
        str(ready_file),
    ])
    try:
        for _ in range(100):
            if ready_file.is_file():
                break
            assert process.poll() is None
            time.sleep(0.02)
        pid, raw_port = ready_file.read_text().split()
        bridge_port = int(raw_port)

        assert int(pid) == process.pid
        with socket.create_connection(('127.0.0.1', bridge_port), timeout=2) as client:
            client.sendall(b'compose-relay')
            assert client.recv(65536) == b'compose-relay'
    finally:
        process.terminate()
        process.wait(timeout=5)
        thread.join(timeout=5)
