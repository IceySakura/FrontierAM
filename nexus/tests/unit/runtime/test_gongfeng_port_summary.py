from types import SimpleNamespace

from nexus.runtime.providers.gongfeng import GongfengRuntimeProvider, build_gongfeng_runtime_port_summary


def test_gongfeng_provider_prints_port_mappings_by_default():
    provider = GongfengRuntimeProvider(api_base_url='http://gongfeng.example.com', token='token')

    assert provider.print_port_mappings is True


def test_build_gongfeng_runtime_port_summary_includes_debug_gui_and_vnc_urls():
    runtime = SimpleNamespace(
        base_url='http://gongfeng-runtime.example.com:32080',
        port_mappings={
            5900: 'http://gongfeng-runtime.example.com:32090',
            6080: 'http://gongfeng-runtime.example.com:32091',
        },
    )

    lines = build_gongfeng_runtime_port_summary(runtime)

    assert lines == [
        'Runtime server: http://gongfeng-runtime.example.com:32080',
        'Debug GUI: http://gongfeng-runtime.example.com:32080/debug/gui',
        'noVNC: http://gongfeng-runtime.example.com:32091/vnc.html',
        'VNC: gongfeng-runtime.example.com:32090',
    ]


def test_build_gongfeng_runtime_port_summary_handles_missing_optional_ports():
    runtime = SimpleNamespace(base_url='http://gongfeng-runtime.example.com:32080/', port_mappings={})

    lines = build_gongfeng_runtime_port_summary(runtime)

    assert lines == [
        'Runtime server: http://gongfeng-runtime.example.com:32080/',
        'Debug GUI: http://gongfeng-runtime.example.com:32080/debug/gui',
    ]
