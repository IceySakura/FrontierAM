"""
Configuration for search_stateful tools.
Provides default paths for filter files and cache directories.
"""

import os

from pathlib import Path


# Get the directory where this config file is located
_CURRENT_DIR = Path(__file__).parent

# Default data directory (relative to this module)
DEFAULT_DATA_DIR = _CURRENT_DIR / 'data'

# Default filter file paths
DEFAULT_TESTSET_FILTER_FILE = str(DEFAULT_DATA_DIR / 'testset_name_to_filter.txt')
DEFAULT_URL_BLOCK_LIST_FILE = str(DEFAULT_DATA_DIR / 'url_block_list.txt')

# Cache paths are resolved lazily by the backend so importing extensions does not
# eagerly touch environment-specific shared directories.
DEFAULT_SEARCH_CACHE_PATH: str | None = os.getenv('DEFAULT_SEARCH_CACHE_PATH')
DEFAULT_VISIT_CACHE_PATH: str | None = os.getenv('DEFAULT_VISIT_CACHE_PATH')

# Legacy shared credentials（同时用于 serper 和 jina，保留向后兼容）.
DEFAULT_USER: str = os.getenv('SEARCH_STATEFUL_USER', '')
DEFAULT_APIKEY: str = os.getenv('SEARCH_STATEFUL_APIKEY', '')

# Serper 专用环境变量（与 nexus.extensions.tools.web_search 对齐）.
# 未设置时回退到旧的 SEARCH_STATEFUL_USER/SEARCH_STATEFUL_APIKEY.
DEFAULT_SERPER_APP_ID: str = os.getenv('SEARCH_SERPER_APP_ID', DEFAULT_USER)
DEFAULT_SERPER_APP_KEY: str = os.getenv('SEARCH_SERPER_APP_KEY', DEFAULT_APIKEY)

# Jina 专用环境变量（与 nexus.extensions.tools.web_search 对齐）.
# 未设置时回退到旧的 SEARCH_STATEFUL_USER/SEARCH_STATEFUL_APIKEY.
DEFAULT_JINA_APP_ID: str = os.getenv('DEFAULT_JINA_APP_ID', DEFAULT_USER)
DEFAULT_JINA_APP_KEY: str = os.getenv('DEFAULT_JINA_APP_KEY', DEFAULT_APIKEY)

# Jina fetch `provider` query value in Authorization (Bearer ...?provider=...).
DEFAULT_JINA_PROVIDER: str = os.getenv('SEARCH_STATEFUL_JINA_PROVIDER', 'aws_jina')


def _env_bool(name: str, default: bool) -> bool:
    """Parse boolean environment variable with common truthy/falsey values."""
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if normalized in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    return default


DEFAULT_USE_CACHE: bool = _env_bool('SEARCH_STATEFUL_USE_CACHE', True)


def get_default_config() -> dict:
    """
    Get default configuration for search_stateful tools.

    Returns:
        dict: Configuration with default file paths and cache paths
    """
    return {
        'testset_name_to_filter_file': DEFAULT_TESTSET_FILTER_FILE,
        'url_block_list_file': DEFAULT_URL_BLOCK_LIST_FILE,
        # Legacy 共享凭据（回退用）.
        'user': DEFAULT_USER,
        'apikey': DEFAULT_APIKEY,
        # Serper 专用凭据.
        'serper_app_id': DEFAULT_SERPER_APP_ID,
        'serper_app_key': DEFAULT_SERPER_APP_KEY,
        # Jina 专用凭据.
        'jina_app_id': DEFAULT_JINA_APP_ID,
        'jina_app_key': DEFAULT_JINA_APP_KEY,
        'jina_provider': DEFAULT_JINA_PROVIDER,
        'search_cache_path': DEFAULT_SEARCH_CACHE_PATH,
        'visit_cache_path': DEFAULT_VISIT_CACHE_PATH,
        'use_cache': DEFAULT_USE_CACHE,
    }
