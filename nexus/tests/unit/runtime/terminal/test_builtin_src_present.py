"""Guard that the bash-capture C sources are present with key entrypoints.

The capture mechanism is now bash-capture's link-time --wrap hooks (not the old
loadable builtin and not LD_PRELOAD), so the sources live as cmd_wrap.c /
readline_wrap.c / bc_common.c / bc_common.h under deployment/runtime/bash.
"""

from pathlib import Path

from importlib import resources


def _bash_src_dir() -> Path:
    nexus_pkg = Path(str(resources.files('nexus')))
    return nexus_pkg.parent / 'deployment' / 'runtime' / 'bash'


def test_cmd_wrap_present_with_wrap_symbol():
    c = (_bash_src_dir() / 'cmd_wrap.c').read_text()
    assert '__wrap_execute_command' in c
    assert 'bc_command_is_top_level_async' in c


def test_cmd_wrap_uses_bash_command_header_instead_of_local_abi():
    c = (_bash_src_dir() / 'cmd_wrap.c').read_text()
    assert '#include "command.h"' in c
    assert 'typedef struct bc_command' not in c
    assert '#define BC_CM_' not in c
    assert '#define BC_CMD_AMPERSAND' not in c


def test_readline_wrap_present_with_wrap_symbol():
    c = (_bash_src_dir() / 'readline_wrap.c').read_text()
    assert '__wrap_readline' in c
    assert '__real_readline' in c
    assert 'bc_reset' in c
    assert 'LD_PRELOAD' not in c
    assert 'dlsym' not in c


def test_bc_common_emits_nexus_multifile():
    d = _bash_src_dir()
    h = (d / 'bc_common.h').read_text()
    c = (d / 'bc_common.c').read_text()
    assert 'bc_emit_cmd' in h
    assert 'status.json' in c
    assert 'cmd_content_' in c
    assert 'NEXUS_CAPTURE_DIR' in c


def test_old_sources_removed():
    d = _bash_src_dir()
    assert not (d / 'nexus_capture.c').exists()
    assert not (d / 'readline_preload.c').exists()
    assert not (d / 'build.sh').exists()


def test_capture_shared_state_does_not_use_mmap_file():
    d = _bash_src_dir()
    assert (d / 'bc_common.c').exists()
    combined = ''.join((d / name).read_text() for name in ('bc_common.c', 'bc_common.h', 'cmd_wrap.c', 'readline_wrap.c'))
    assert 'mmap' not in combined
    assert 'MAP_SHARED' not in combined
    assert '.bash_capture_' not in combined
    assert '__atomic' not in combined
