from nexus.dataset.cache import DEFAULT_DATASET_CACHE_DIR
from nexus.utils.paths import DATASET_CACHE_DIR, NEXUS_HOME


def test_default_dataset_cache_dir_lives_under_nexus_home():
    assert DEFAULT_DATASET_CACHE_DIR == DATASET_CACHE_DIR
    assert DEFAULT_DATASET_CACHE_DIR == NEXUS_HOME / 'datasets'
