from nexus.dataset.uri import parse_dataset_uri

import pytest


def test_parse_dataset_uri_defaults_namespace_and_version():
    parsed = parse_dataset_uri('swebench_verified')

    assert parsed.namespace == 'nexus-curated'
    assert parsed.dataset == 'swebench_verified'
    assert parsed.version == 'latest'
    assert parsed.split is None


def test_parse_dataset_uri_full_name_and_split():
    parsed = parse_dataset_uri('taiji.nexus.research1/swebench_verified@v1.0.0?split=test')

    assert parsed.namespace == 'research1'
    assert parsed.dataset == 'swebench_verified'
    assert parsed.version == 'v1.0.0'
    assert parsed.split == 'test'


def test_parse_dataset_uri_rejects_empty_split():
    with pytest.raises(ValueError, match='split.*empty'):
        parse_dataset_uri('research/tasks@v1?split=')
