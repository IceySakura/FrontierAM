from types import SimpleNamespace

import pytest

from nexus.runtime.providers.docker import DockerRuntimeProvider, _docker_environment_args, build_runtime_port_summary


def test_docker_provider_prints_port_mappings_by_default():
    provider = DockerRuntimeProvider()

    assert provider.print_port_mappings is True


def test_build_runtime_port_summary_includes_debug_gui_and_vnc_urls():
    runtime = SimpleNamespace(
        base_url='http://172.17.0.1:32909',
        port_mappings={
            5900: 'http://172.17.0.1:32907',
            6080: 'http://172.17.0.1:32908',
        },
    )

    lines = build_runtime_port_summary(runtime, display_host='127.0.0.1')

    assert lines == [
        'Runtime server: http://127.0.0.1:32909',
        'Debug GUI: http://127.0.0.1:32909/debug/gui',
        'noVNC: http://127.0.0.1:32908/vnc.html',
        'VNC: 127.0.0.1:32907',
    ]


def test_build_runtime_port_summary_handles_missing_optional_ports():
    runtime = SimpleNamespace(base_url='http://172.17.0.1:32909', port_mappings={})

    lines = build_runtime_port_summary(runtime, display_host='127.0.0.1')

    assert lines == [
        'Runtime server: http://127.0.0.1:32909',
        'Debug GUI: http://127.0.0.1:32909/debug/gui',
    ]


def test_build_runtime_port_summary_uses_explicit_remote_host():
    runtime = SimpleNamespace(
        base_url='http://172.17.0.1:32909',
        port_mappings={5900: 'http://172.17.0.1:32907'},
    )

    lines = build_runtime_port_summary(runtime, display_host='192.168.1.50')

    assert lines == [
        'Runtime server: http://192.168.1.50:32909',
        'Debug GUI: http://192.168.1.50:32909/debug/gui',
        'VNC: 192.168.1.50:32907',
    ]


def test_docker_environment_args_quote_proxy_values():
    args = _docker_environment_args(
        {
            'http_proxy': 'http://proxy.internal:3128',
            'GRADLE_OPTS': '-Dhttps.proxyHost=proxy.internal -Dhttps.proxyPort=3128',
        }
    )

    assert args == [
        '--env http_proxy=http://proxy.internal:3128',
        "--env 'GRADLE_OPTS=-Dhttps.proxyHost=proxy.internal -Dhttps.proxyPort=3128'",
    ]


def test_docker_environment_args_reject_invalid_name():
    with pytest.raises(ValueError, match='Invalid Docker environment variable name'):
        _docker_environment_args({'bad=name': 'value'})
