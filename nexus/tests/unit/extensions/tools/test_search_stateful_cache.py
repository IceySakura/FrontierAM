"""
Unit tests for search_stateful cache functionality.

Tests cache hit/miss behavior for the fetch operation.
"""

import asyncio
import json
import os
import shutil
import tempfile
import threading
import time

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tenacity import wait_none

from nexus.extensions.tools.search_stateful import file_cache
from nexus.extensions.tools.search_stateful.backend import BackendError, GoogleApiBackend


class TestSearchStatefulCache:
    """Test cache functionality for search_stateful backend."""

    @pytest.fixture(autouse=True)
    def no_retry_wait(self, monkeypatch):
        # fetch backs off up to ~30s of jitter per retry, so the error-path tests would otherwise sleep for a minute.
        monkeypatch.setattr(GoogleApiBackend.fetch.retry, 'wait', wait_none())

    @pytest.fixture(scope='class')
    def config(self):
        """Provide fake credentials for mocked backend calls."""
        return {
            'user': 'test-user',
            'apikey': 'test-apikey',
        }

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp(prefix='test_search_cache_')
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def backend_with_cache(self, temp_cache_dir, config):
        """Create a backend instance with cache enabled."""
        backend = GoogleApiBackend(
            user=config['user'],
            apikey=config['apikey'],
            use_cache=True,
            visit_cache_path=temp_cache_dir,
        )
        yield backend
        # The background writer must finish before temp_cache_dir is removed.
        if backend.visit_cache is not None:
            backend.visit_cache.flush()

    @pytest.fixture
    def backend_without_cache(self, temp_cache_dir, config):
        """Create a backend instance with cache disabled."""
        return GoogleApiBackend(
            user=config['user'],
            apikey=config['apikey'],
            use_cache=False,
            visit_cache_path=temp_cache_dir,
        )

    @pytest.fixture
    def mock_http_response(self):
        """Mock HTTP response for Jina API."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'Mock response text'
        # json() should return a dict directly, not a coroutine
        mock_response.json = Mock(
            return_value={
                'data': {
                    'text': 'This is test content from Wikipedia about Second presidency of Donald Trump.',
                    'title': 'Second presidency of Donald Trump',
                }
            }
        )
        return mock_response

    def test_initialize_caches_falls_back_to_local_dir(self, monkeypatch, tmp_path):
        """Default cache paths should fall back to a user-writable directory."""
        default_search_path = '/unwritable/search'
        default_visit_path = '/unwritable/visit'
        original_file_cache = file_cache.FileCache

        monkeypatch.setattr(file_cache, '_DEFAULT_SEARCH_CACHE_PATH', default_search_path)
        monkeypatch.setattr(file_cache, '_DEFAULT_VISIT_CACHE_PATH', default_visit_path)
        monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'xdg-cache'))

        def fake_file_cache(base_dir):
            if base_dir in {default_search_path, default_visit_path}:
                raise PermissionError('permission denied')
            return original_file_cache(base_dir)

        monkeypatch.setattr(file_cache, 'FileCache', fake_file_cache)

        search_cache, visit_cache = file_cache.initialize_caches()

        assert Path(search_cache.base_dir) == tmp_path / 'xdg-cache' / 'nexus' / 'search_stateful' / 'search'
        assert Path(visit_cache.base_dir) == tmp_path / 'xdg-cache' / 'nexus' / 'search_stateful' / 'visit'

    def test_backend_uses_reinitialized_cache_instance(self, config, tmp_path):
        """Backend should use the cache instance created for its own configured path."""
        first_dir = str(tmp_path / 'first')
        second_dir = str(tmp_path / 'second')

        file_cache.initialize_caches(search_cache_path=first_dir, visit_cache_path=first_dir)
        backend = GoogleApiBackend(
            user=config['user'],
            apikey=config['apikey'],
            use_cache=True,
            visit_cache_path=second_dir,
        )

        assert backend.visit_cache is not None
        assert Path(backend.visit_cache.base_dir) == Path(second_dir)

    def test_initialize_caches_reuses_cache_per_directory(self, monkeypatch, tmp_path):
        """Repeated backend construction must not re-create caches or touch the directory again."""
        created_dirs = []
        original_file_cache = file_cache.FileCache

        def counting_file_cache(base_dir):
            created_dirs.append(base_dir)
            return original_file_cache(base_dir)

        monkeypatch.setattr(file_cache, 'FileCache', counting_file_cache)
        search_dir, visit_dir = str(tmp_path / 'search'), str(tmp_path / 'visit')

        first = file_cache.initialize_caches(search_cache_path=search_dir, visit_cache_path=visit_dir)
        second = file_cache.initialize_caches(search_cache_path=search_dir, visit_cache_path=visit_dir)

        assert first[0] is second[0]
        assert first[1] is second[1]
        assert created_dirs == [search_dir, visit_dir]

    @pytest.mark.asyncio
    async def test_load_async_times_out_without_blocking_event_loop(self, monkeypatch, tmp_path):
        """A slow disk read is reported as a miss after the timeout while the loop keeps running."""
        monkeypatch.setattr(file_cache, 'LOAD_TIMEOUT_S', 0.1)
        cache = file_cache.FileCache(str(tmp_path))
        cache.save('query', {'results': [1]})
        release_read = threading.Event()
        original_read = cache._read

        def slow_read(key_md5):
            release_read.wait(timeout=5)
            return original_read(key_md5)

        monkeypatch.setattr(cache, '_read', slow_read)
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        heartbeat_task = asyncio.create_task(heartbeat())
        start = time.monotonic()
        record = await cache.load_async('query')
        elapsed = time.monotonic() - start
        heartbeat_task.cancel()
        release_read.set()

        assert record is None
        assert elapsed < 1.0
        assert ticks >= 5
        assert cache._stats['load_timeout'] == 1

    @pytest.mark.asyncio
    async def test_load_async_skips_disk_when_load_workers_are_busy(self, monkeypatch, tmp_path):
        """Once every load worker is stuck, further loads return a miss immediately instead of queueing."""
        monkeypatch.setattr(file_cache, 'LOAD_WORKERS', 1)
        monkeypatch.setattr(file_cache, 'LOAD_TIMEOUT_S', 5.0)
        cache = file_cache.FileCache(str(tmp_path))
        read_started, release_read = threading.Event(), threading.Event()

        def blocked_read(key_md5):
            read_started.set()
            release_read.wait(timeout=5)

        monkeypatch.setattr(cache, '_read', blocked_read)
        first_load = asyncio.create_task(cache.load_async('first'))
        await asyncio.to_thread(read_started.wait, 5)

        start = time.monotonic()
        second = await cache.load_async('second')
        assert second is None
        assert time.monotonic() - start < 0.5
        assert cache._stats['load_busy'] == 1

        release_read.set()
        assert await first_load is None

    @pytest.mark.asyncio
    async def test_save_in_background_is_readable_before_and_after_disk_write(self, monkeypatch, tmp_path):
        """A queued save is visible to loads immediately, then lands on disk with one meta line per record."""
        cache = file_cache.FileCache(str(tmp_path))
        release_write = threading.Event()
        original_write = cache._write_data

        def blocked_write(key_md5, record):
            release_write.wait(timeout=5)
            return original_write(key_md5, record)

        monkeypatch.setattr(cache, '_write_data', blocked_write)

        start = time.monotonic()
        cache.save_in_background('first', {'results': [1]})
        cache.save_in_background('second', {'results': [2]})
        assert time.monotonic() - start < 0.5

        pending = await cache.load_async('first')
        assert pending is not None
        assert pending['results'] == [1]
        assert not os.path.exists(cache.get_path(cache.md5('first')))

        release_write.set()
        await asyncio.to_thread(cache.flush)

        assert cache._pending == {}
        assert cache.load('second')['results'] == [2]
        with open(cache.meta_path, encoding='utf-8') as meta_file:
            meta_queries = [json.loads(line)['query'] for line in meta_file]
        assert sorted(meta_queries) == ['first', 'second']
        assert not [name for name in os.listdir(cache.data_dir) if name.endswith('.tmp')]

    @pytest.mark.asyncio
    async def test_fetch_with_cache_first_call_saves_to_cache(
        self, backend_with_cache, mock_http_response, temp_cache_dir
    ):
        """Test that first fetch saves to cache when use_cache=True."""
        test_url = 'https://en.wikipedia.org/wiki/Second_presidency_of_Donald_Trump'

        with patch(
            'nexus.extensions.tools.search_stateful.backend.http_client.post', new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_http_response

            # First fetch - should call API and save to cache
            result = await backend_with_cache.fetch(test_url)

            # Verify API was called
            assert mock_post.called
            mock_post.assert_called_once()

            # Verify result
            assert result.url == test_url
            assert result.text is not None
            assert len(result.text) > 0
            assert 'Second presidency of Donald Trump' in result.text or 'test content' in result.text.lower()

            # Verify cache was saved
            assert backend_with_cache.visit_cache is not None
            cached_data = backend_with_cache.visit_cache.load(test_url)
            assert cached_data is not None
            assert 'data' in cached_data

    @pytest.mark.asyncio
    async def test_fetch_with_cache_second_call_hits_cache(
        self, backend_with_cache, mock_http_response, temp_cache_dir
    ):
        """Test that second fetch hits cache when use_cache=True."""
        test_url = 'https://en.wikipedia.org/wiki/Second_presidency_of_Donald_Trump'

        with patch(
            'nexus.extensions.tools.search_stateful.backend.http_client.post', new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_http_response

            # First fetch - saves to cache
            result1 = await backend_with_cache.fetch(test_url)
            first_call_count = mock_post.call_count

            # Second fetch - should hit cache, no API call
            result2 = await backend_with_cache.fetch(test_url)
            second_call_count = mock_post.call_count

            # Verify API was only called once (first time)
            assert first_call_count == 1
            assert second_call_count == 1, 'API should not be called on second fetch when cache is hit'

            # Verify results are the same
            assert result1.url == result2.url
            assert result1.text == result2.text

    @pytest.mark.asyncio
    async def test_fetch_without_cache_always_calls_api(
        self, backend_without_cache, mock_http_response, temp_cache_dir
    ):
        """Test that fetch always calls API when use_cache=False."""
        test_url = 'https://en.wikipedia.org/wiki/Second_presidency_of_Donald_Trump'

        with patch(
            'nexus.extensions.tools.search_stateful.backend.http_client.post', new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_http_response

            # First fetch
            result1 = await backend_without_cache.fetch(test_url)
            assert mock_post.call_count == 1

            # Second fetch - should still call API
            result2 = await backend_without_cache.fetch(test_url)
            assert mock_post.call_count == 2, 'API should be called every time when cache is disabled'

            # Verify results
            assert result1.url == result2.url
            assert result1.text is not None
            assert result2.text is not None

    @pytest.mark.asyncio
    async def test_fetch_without_cache_does_not_save_to_cache(
        self, backend_without_cache, mock_http_response, temp_cache_dir
    ):
        """Test that fetch does not save to cache when use_cache=False."""
        test_url = 'https://en.wikipedia.org/wiki/Second_presidency_of_Donald_Trump'

        with patch(
            'nexus.extensions.tools.search_stateful.backend.http_client.post', new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_http_response

            # Fetch with cache disabled
            result = await backend_without_cache.fetch(test_url)

            # Verify result is correct
            assert result.url == test_url
            assert result.text is not None

            assert backend_without_cache.visit_cache is None
            assert list(Path(temp_cache_dir).iterdir()) == []

    @pytest.mark.asyncio
    async def test_fetch_handles_api_error(self, backend_with_cache):
        """Test that fetch handles API errors correctly."""
        test_url = 'https://en.wikipedia.org/wiki/Second_presidency_of_Donald_Trump'

        with patch(
            'nexus.extensions.tools.search_stateful.backend.http_client.post', new_callable=AsyncMock
        ) as mock_post:
            # Mock API error
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = 'Internal Server Error'
            mock_post.return_value = mock_response

            # Should raise BackendError
            with pytest.raises(BackendError) as exc_info:
                await backend_with_cache.fetch(test_url)

            assert 'Failed to fetch URL' in str(exc_info.value) or '500' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_handles_empty_response(self, backend_with_cache):
        """Test that fetch handles empty API response correctly."""
        test_url = 'https://en.wikipedia.org/wiki/Second_presidency_of_Donald_Trump'

        with patch(
            'nexus.extensions.tools.search_stateful.backend.http_client.post', new_callable=AsyncMock
        ) as mock_post:
            # Mock empty response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = 'Mock response text'
            mock_response.json = Mock(
                return_value={
                    'data': {
                        'text': None,  # Empty text
                        'title': 'Test Title',
                    }
                }
            )
            mock_post.return_value = mock_response

            # Should raise BackendError
            with pytest.raises(BackendError) as exc_info:
                await backend_with_cache.fetch(test_url)

            assert 'text为空' in str(exc_info.value) or 'No contents returned' in str(exc_info.value)
