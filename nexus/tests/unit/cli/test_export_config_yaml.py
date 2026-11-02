"""Exporting a stored run config must survive strings OmegaConf cannot parse."""

from __future__ import annotations

import json

import pytest
import yaml

from omegaconf import OmegaConf
from omegaconf.errors import GrammarParseError

from nexus.cli.commands.export import _dump_run_config_yaml


# bash's strip-longest-suffix inside ${...}: SWE-bench Pro's NodeBB cases ship this in
# spec_dict.run_script.sh, and it is not an OmegaConf interpolation.
BASH_SCRIPT = 'for idx in "${!test_files[@]}"; do\n  test_files[$idx]="${test_files[$idx]%% | *}"\ndone\n'


@pytest.mark.parametrize('as_json_text', [False, True])
def test_a_shell_script_that_looks_like_an_interpolation_survives(as_json_text: bool) -> None:
    config = {'task_config': {'instance': {'spec_dict': {'run_script': {'sh': BASH_SCRIPT}}}}}
    stored = json.dumps(config) if as_json_text else config
    dumped = _dump_run_config_yaml(stored)

    # Round-trip equality, not a byte or substring comparison against the input: a double-quoted
    # YAML scalar escapes newlines and folds at a width PyYAML chooses, which here lands in the
    # middle of `%% | *`. The property that matters is that the value comes back unchanged; that
    # the `%%` is never *interpreted* is pinned by the OmegaConf test below.
    loaded = yaml.safe_load(dumped)
    assert loaded == config
    assert loaded['task_config']['instance']['spec_dict']['run_script']['sh'] == BASH_SCRIPT


def test_omegaconf_still_rejects_the_script_so_the_fallback_is_needed() -> None:
    with pytest.raises(GrammarParseError):
        OmegaConf.create({'sh': BASH_SCRIPT})


def test_an_ordinary_config_is_dumped_exactly_as_omegaconf_did() -> None:
    config = {'n': 3, 'y': 'yes', 'interp': '${oc.env:HOME}', 'ml': 'a\nb\n', 'lst': [1, {'k': None}]}
    assert _dump_run_config_yaml(config) == OmegaConf.to_yaml(OmegaConf.create(config))


def test_the_fallback_keeps_omegaconfs_format() -> None:
    """Keys like ``n``/``y`` stay quoted, as OmegaConf's dumper does and ``yaml.safe_dump`` does not."""
    dumped = _dump_run_config_yaml({'n': 3, 'sh': BASH_SCRIPT})
    assert dumped.startswith("'n': 3\n")


def test_the_fallback_survives_omegaconf_dropping_its_private_dumper(monkeypatch: pytest.MonkeyPatch) -> None:
    """A moved private symbol must cost quoting, not the import of the whole export module.

    ``get_omega_conf_dumper`` lives in ``omegaconf._utils`` and ``omegaconf`` has no upper
    bound, so the symbol is only guaranteed for the versions we have seen.
    """
    import omegaconf._utils

    # Drop just the symbol: stubbing the whole module breaks omegaconf's own internals.
    monkeypatch.delattr(omegaconf._utils, 'get_omega_conf_dumper', raising=False)

    dumped = _dump_run_config_yaml({'n': 3, 'sh': BASH_SCRIPT})

    assert yaml.safe_load(dumped)['sh'] == BASH_SCRIPT
    assert dumped.startswith('n: 3\n')  # safe_dump leaves the key unquoted; OmegaConf's dumper quotes it


def test_unicode_is_kept_readable() -> None:
    assert '音乐剧剧评' in _dump_run_config_yaml({'prompt': '音乐剧剧评'})
    assert '音乐剧剧评' in _dump_run_config_yaml({'prompt': '音乐剧剧评', 'sh': BASH_SCRIPT})
