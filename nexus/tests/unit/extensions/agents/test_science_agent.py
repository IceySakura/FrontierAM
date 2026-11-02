from pathlib import Path

import pytest

from nexus.utils.tool_schema_overrides import load_tool_schema_overrides


def test_load_tool_schema_overrides_from_inline_list():
    overrides = [{'type': 'function', 'function': {'name': 'search'}}]

    assert load_tool_schema_overrides(overrides) == overrides


def test_load_tool_schema_overrides_from_tool_config_file(tmp_path: Path):
    config_path = tmp_path / 'tool_config.yaml'
    config_path.write_text(
        """
tools:
  - class_name: custom.search
    tool_schema:
      type: function
      function:
        name: search
  - class_name: custom.python
    tool_schema:
      type: function
      function:
        name: python
""".strip(),
        encoding='utf-8',
    )

    assert load_tool_schema_overrides(str(config_path)) == [
        {'type': 'function', 'function': {'name': 'search'}},
        {'type': 'function', 'function': {'name': 'python'}},
    ]


def test_load_tool_schema_overrides_rejects_missing_file(tmp_path: Path):
    missing_path = tmp_path / 'missing.yaml'

    with pytest.raises(FileNotFoundError):
        load_tool_schema_overrides(str(missing_path))
