import shutil

from pathlib import Path

import pytest

from nexus.runtime.server.venv import manager


@pytest.mark.skipif(not manager.is_windows(), reason='Windows extended-length paths')
def test_dependency_copy_supports_long_source_and_destination_paths(tmp_path: Path) -> None:
    source = tmp_path / 'source' / ('nested' * 30) / ('package' * 20)
    destination = tmp_path / 'destination' / ('nested' * 30) / ('package' * 20)
    source = manager._copy_path(source)
    destination = manager._copy_path(destination)
    source.mkdir(parents=True)
    (source / 'data.json').write_text('{"copied": true}')

    shutil.copytree(source, destination)

    assert (destination / 'data.json').read_text() == '{"copied": true}'
