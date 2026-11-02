"""Unit tests for nexus.utils.cache module."""

import asyncio
import pickle

from pathlib import Path

import pytest

from nexus.utils.cache import (
    CacheEntry,
    CacheResultType,
    CacheStage,
    cache_context,
    cacheable,
    clear_cache,
    get_cache_stage,
    get_cache_stats,
    get_current_cache,
    is_cache_enabled,
    is_caching,
    is_replaying,
    load_cache,
    save_cache,
)


# ============== CacheEntry Tests ==============


class TestCacheEntry:
    """Tests for CacheEntry class."""

    def test_creation_with_defaults(self):
        """Test CacheEntry creation with default values."""
        entry = CacheEntry(func_name='test_func', args_hash='abc123')
        assert entry.func_name == 'test_func'
        assert entry.args_hash == 'abc123'
        assert entry.result_type == CacheResultType.INCOMPLETE.value
        assert entry.result_value is None
        assert entry.children == []
        assert entry.timestamp is not None

    def test_creation_with_all_fields(self):
        """Test CacheEntry creation with all fields specified."""
        child = CacheEntry(func_name='child_func', args_hash='child_hash')
        entry = CacheEntry(
            func_name='parent_func',
            args_hash='parent_hash',
            result_type=CacheResultType.SUCCESS.value,
            result_value=pickle.dumps('result').hex(),
            children=[child],
            timestamp='2024-01-01T00:00:00',
        )
        assert entry.func_name == 'parent_func'
        assert entry.result_type == CacheResultType.SUCCESS.value
        assert len(entry.children) == 1
        assert entry.timestamp == '2024-01-01T00:00:00'

    def test_to_dict(self):
        """Test conversion to dictionary."""
        child = CacheEntry(func_name='child', args_hash='child_hash')
        entry = CacheEntry(
            func_name='parent',
            args_hash='parent_hash',
            result_type=CacheResultType.SUCCESS.value,
            result_value='hexvalue',
            children=[child],
            timestamp='2024-01-01T00:00:00',
        )
        d = entry.to_dict()
        assert d['func_name'] == 'parent'
        assert d['args_hash'] == 'parent_hash'
        assert d['result_type'] == 'success'
        assert d['result_value'] == 'hexvalue'
        assert len(d['children']) == 1
        assert d['children'][0]['func_name'] == 'child'
        assert d['timestamp'] == '2024-01-01T00:00:00'

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'func_name': 'test',
            'args_hash': 'hash123',
            'result_type': 'success',
            'result_value': 'hexdata',
            'children': [{'func_name': 'child', 'args_hash': 'child_hash'}],
            'timestamp': '2024-01-01T00:00:00',
        }
        entry = CacheEntry.from_dict(data)
        assert entry.func_name == 'test'
        assert entry.args_hash == 'hash123'
        assert entry.result_type == 'success'
        assert len(entry.children) == 1
        assert entry.children[0].func_name == 'child'

    def test_from_dict_with_defaults(self):
        """Test from_dict handles missing optional fields."""
        data = {'func_name': 'test', 'args_hash': 'hash'}
        entry = CacheEntry.from_dict(data)
        assert entry.result_type == CacheResultType.SUCCESS.value
        assert entry.children == []

    def test_is_complete_success(self):
        """Test is_complete returns True for SUCCESS."""
        entry = CacheEntry(func_name='f', args_hash='h', result_type=CacheResultType.SUCCESS.value)
        assert entry.is_complete() is True

    def test_is_complete_exception(self):
        """Test is_complete returns True for EXCEPTION."""
        entry = CacheEntry(func_name='f', args_hash='h', result_type=CacheResultType.EXCEPTION.value)
        assert entry.is_complete() is True

    def test_is_complete_incomplete(self):
        """Test is_complete returns False for INCOMPLETE."""
        entry = CacheEntry(func_name='f', args_hash='h', result_type=CacheResultType.INCOMPLETE.value)
        assert entry.is_complete() is False


# ============== Cache Context Tests ==============


class TestCacheContext:
    """Tests for cache_context context manager."""

    def test_empty_context(self):
        """Test cache_context with no existing data."""
        with cache_context() as cache:
            assert cache is not None
            assert cache['entries'] == []
            assert cache['metadata'] == {}
            assert is_cache_enabled()
            assert is_caching()
            assert not is_replaying()

    def test_provided_empty_dict(self):
        """Test cache_context with provided empty dict."""
        cache_dict = {}
        with cache_context(cache_dict) as cache:
            assert cache is cache_dict
            assert 'entries' in cache
            assert 'metadata' in cache
            assert is_caching()

    def test_cache_with_existing_entries_starts_replay(self):
        """Test that existing entries trigger REPLAY stage."""
        entry_dict = {
            'func_name': 'test',
            'args_hash': 'hash',
            'result_type': 'success',
            'result_value': pickle.dumps('result').hex(),
            'children': [],
        }
        cache_dict = {'entries': [entry_dict], 'metadata': {}}
        with cache_context(cache_dict):
            assert is_replaying()
            assert get_cache_stage() == CacheStage.REPLAY

    def test_context_cleanup(self):
        """Test that context variables are cleaned up after exit."""
        with cache_context():
            assert is_cache_enabled()

        assert not is_cache_enabled()
        assert get_cache_stage() is None
        assert get_current_cache() is None

    def test_entries_converted_to_dicts_on_exit(self):
        """Test that CacheEntry objects are converted to dicts on exit."""
        cache_dict = {}
        with cache_context(cache_dict):
            entries = get_current_cache()['entries']
            entries.append(CacheEntry(func_name='test', args_hash='hash'))

        # After context exit, entries should be dicts
        assert len(cache_dict['entries']) == 1
        assert isinstance(cache_dict['entries'][0], dict)
        assert cache_dict['entries'][0]['func_name'] == 'test'

    def test_ignore_args_hash_option(self):
        """Test ignore_args_hash option is set correctly."""
        # This is implicitly tested through the decorator tests
        with cache_context(ignore_args_hash=True):
            assert is_cache_enabled()

    def test_overwrite_tags_option(self):
        """Test overwrite_tags option is set correctly."""
        with cache_context(overwrite_tags=['llm']):
            assert is_cache_enabled()


# ============== File I/O Tests ==============


class TestFileIO:
    """Tests for load_cache and save_cache functions."""

    def test_load_nonexistent_file(self, tmp_path: Path):
        """Test loading from non-existent file returns empty cache."""
        cache = load_cache(tmp_path / 'nonexistent.pkl')
        assert cache == {'entries': [], 'metadata': {}}

    def test_save_and_load(self, tmp_path: Path):
        """Test saving and loading cache."""
        cache_path = tmp_path / 'cache.pkl'
        original = {
            'entries': [
                {
                    'func_name': 'test',
                    'args_hash': 'hash',
                    'result_type': 'success',
                    'result_value': 'hex',
                    'children': [],
                }
            ],
            'metadata': {'version': '1.0'},
        }
        save_cache(original, cache_path)
        loaded = load_cache(cache_path)

        assert loaded['metadata'] == original['metadata']
        assert len(loaded['entries']) == 1
        assert loaded['entries'][0]['func_name'] == 'test'

    def test_save_creates_parent_directories(self, tmp_path: Path):
        """Test that save_cache creates parent directories."""
        cache_path = tmp_path / 'nested' / 'dir' / 'cache.pkl'
        save_cache({'entries': [], 'metadata': {}}, cache_path)
        assert cache_path.exists()

    def test_save_with_cache_entry_objects(self, tmp_path: Path):
        """Test saving CacheEntry objects (converted to dicts)."""
        cache_path = tmp_path / 'cache.pkl'
        entry = CacheEntry(func_name='test', args_hash='hash')
        cache_dict = {'entries': [entry], 'metadata': {}}
        save_cache(cache_dict, cache_path)

        loaded = load_cache(cache_path)
        assert loaded['entries'][0]['func_name'] == 'test'

    def test_load_corrupted_file(self, tmp_path: Path):
        """Test loading corrupted file returns empty cache."""
        cache_path = tmp_path / 'corrupted.pkl'
        cache_path.write_bytes(b'not valid pickle data')
        cache = load_cache(cache_path)
        assert cache == {'entries': [], 'metadata': {}}

    def test_save_and_load_with_nested_entries(self, tmp_path: Path):
        """Test saving and loading cache with nested entries."""
        cache_path = tmp_path / 'nested.pkl'
        child_entry = CacheEntry(
            func_name='child',
            args_hash='child_hash',
            result_type='success',
            result_value=pickle.dumps('child_result').hex(),
        )
        parent_entry = CacheEntry(
            func_name='parent',
            args_hash='parent_hash',
            result_type='success',
            result_value=pickle.dumps('parent_result').hex(),
            children=[child_entry],
        )
        cache_dict = {'entries': [parent_entry], 'metadata': {}}
        save_cache(cache_dict, cache_path)

        loaded = load_cache(cache_path)
        assert len(loaded['entries']) == 1
        assert loaded['entries'][0]['func_name'] == 'parent'
        assert len(loaded['entries'][0]['children']) == 1
        assert loaded['entries'][0]['children'][0]['func_name'] == 'child'


# ============== Cacheable Decorator Tests ==============


class TestCacheableSync:
    """Tests for @cacheable decorator with sync functions."""

    def test_basic_recording(self):
        """Test basic function recording."""
        call_count = 0

        @cacheable()
        def add(a: int, b: int) -> int:
            nonlocal call_count
            call_count += 1
            return a + b

        cache_dict = {}
        with cache_context(cache_dict):
            result = add(1, 2)
            assert result == 3
            assert call_count == 1

        assert len(cache_dict['entries']) == 1
        assert cache_dict['entries'][0]['func_name'] == 'TestCacheableSync.test_basic_recording.<locals>.add'

    def test_replay(self):
        """Test function replay from cache."""
        call_count = 0

        @cacheable()
        def multiply(a: int, b: int) -> int:
            nonlocal call_count
            call_count += 1
            return a * b

        # First pass: record
        cache_dict = {}
        with cache_context(cache_dict):
            result1 = multiply(3, 4)
            assert result1 == 12
            assert call_count == 1

        # Second pass: replay
        with cache_context(cache_dict):
            result2 = multiply(3, 4)
            assert result2 == 12
            assert call_count == 1  # Not called again

    def test_multiple_calls(self):
        """Test multiple function calls are recorded in order."""
        call_count = 0

        @cacheable()
        def increment(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + 1

        cache_dict = {}
        with cache_context(cache_dict):
            r1 = increment(1)
            r2 = increment(2)
            r3 = increment(3)
            assert [r1, r2, r3] == [2, 3, 4]
            assert call_count == 3

        # Replay
        call_count = 0
        with cache_context(cache_dict):
            r1 = increment(1)
            r2 = increment(2)
            r3 = increment(3)
            assert [r1, r2, r3] == [2, 3, 4]
            assert call_count == 0

    def test_ignore_args(self):
        """Test ignore_args parameter."""
        call_count = 0

        @cacheable(ignore_args=['timestamp'])
        def process(data: str, timestamp: str) -> str:
            nonlocal call_count
            call_count += 1
            return f'{data}@{timestamp}'

        cache_dict = {}
        with cache_context(cache_dict):
            process('hello', '2024-01-01')
            assert call_count == 1

        # Replay with different timestamp should still hit cache
        with cache_context(cache_dict):
            process('hello', '2024-12-31')
            assert call_count == 1  # Not called again

    def test_cache_exceptions_enabled(self):
        """Test caching exceptions when enabled."""

        @cacheable(cache_exceptions=True)
        def failing():
            raise ValueError('test error')

        cache_dict = {}
        with cache_context(cache_dict):
            with pytest.raises(ValueError, match='test error'):
                failing()

        # Replay should raise same exception
        with cache_context(cache_dict):
            with pytest.raises(ValueError, match='test error'):
                failing()

        # Check entry has exception type
        assert cache_dict['entries'][0]['result_type'] == CacheResultType.EXCEPTION.value

    def test_cache_exceptions_disabled(self):
        """Test exceptions not cached when disabled."""
        call_count = 0

        @cacheable(cache_exceptions=False)
        def maybe_fail(fail: bool):
            nonlocal call_count
            call_count += 1
            if fail:
                raise RuntimeError('failure')
            return 'success'

        cache_dict = {}
        with cache_context(cache_dict):
            with pytest.raises(RuntimeError):
                maybe_fail(fail=True)
            assert call_count == 1

        # Entry should be incomplete
        assert cache_dict['entries'][0]['result_type'] == CacheResultType.INCOMPLETE.value

    def test_without_cache_context(self):
        """Test function works without cache context."""

        @cacheable()
        def simple(x: int) -> int:
            return x * 2

        result = simple(5)
        assert result == 10

    def test_without_cache_context_skips_argument_serialization(self):
        """Disabled caching must not require arguments to be pickleable."""

        @cacheable()
        def identity(value):
            return value

        def local_function():
            return None

        assert identity(local_function) is local_function


class TestCacheableAsync:
    """Tests for @cacheable decorator with async functions."""

    async def test_basic_async_recording(self):
        """Test basic async function recording."""
        call_count = 0

        @cacheable()
        async def async_add(a: int, b: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.001)
            return a + b

        cache_dict = {}
        with cache_context(cache_dict):
            result = await async_add(1, 2)
            assert result == 3
            assert call_count == 1

        assert len(cache_dict['entries']) == 1

    async def test_async_replay(self):
        """Test async function replay."""
        call_count = 0

        @cacheable()
        async def async_multiply(a: int, b: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return a * b

        cache_dict = {}
        with cache_context(cache_dict):
            await async_multiply(3, 4)
            assert call_count == 1

        with cache_context(cache_dict):
            result = await async_multiply(3, 4)
            assert result == 12
            assert call_count == 1  # Not called

    async def test_async_exception_caching(self):
        """Test async exception caching."""

        @cacheable(cache_exceptions=True)
        async def async_fail():
            await asyncio.sleep(0.001)
            raise ValueError('async error')

        cache_dict = {}
        with cache_context(cache_dict):
            with pytest.raises(ValueError, match='async error'):
                await async_fail()

        with cache_context(cache_dict):
            with pytest.raises(ValueError, match='async error'):
                await async_fail()

    async def test_without_cache_context_skips_argument_serialization(self):
        """Disabled async caching must not require arguments to be pickleable."""

        @cacheable()
        async def identity(value):
            return value

        def local_function():
            return None

        assert await identity(local_function) is local_function


# ============== Nested/Hierarchical Caching Tests ==============


class TestHierarchicalCaching:
    """Tests for nested/hierarchical caching."""

    def test_nested_calls_recording(self):
        """Test that nested calls are recorded as children."""
        inner_count = 0
        outer_count = 0

        @cacheable()
        def inner_func(x: int) -> int:
            nonlocal inner_count
            inner_count += 1
            return x * 2

        @cacheable()
        def outer_func(x: int) -> int:
            nonlocal outer_count
            outer_count += 1
            return inner_func(x) + 1

        cache_dict = {}
        with cache_context(cache_dict):
            result = outer_func(5)
            assert result == 11
            assert inner_count == 1
            assert outer_count == 1

        # Check structure: outer has inner as child
        assert len(cache_dict['entries']) == 1
        outer_entry = cache_dict['entries'][0]
        assert 'outer_func' in outer_entry['func_name']
        assert len(outer_entry['children']) == 1
        assert 'inner_func' in outer_entry['children'][0]['func_name']

    def test_nested_calls_replay(self):
        """Test that nested calls replay correctly."""
        inner_count = 0
        outer_count = 0

        @cacheable()
        def inner_func(x: int) -> int:
            nonlocal inner_count
            inner_count += 1
            return x * 2

        @cacheable()
        def outer_func(x: int) -> int:
            nonlocal outer_count
            outer_count += 1
            return inner_func(x) + 1

        # Record
        cache_dict = {}
        with cache_context(cache_dict):
            outer_func(5)

        # Replay
        inner_count = 0
        outer_count = 0
        with cache_context(cache_dict):
            result = outer_func(5)
            assert result == 11
            assert inner_count == 0
            assert outer_count == 0

    def test_outer_fails_inner_succeeds_recovery(self):
        """Test recovery when outer fails but inner succeeds.

        This is the key scenario: inner_func succeeds and is cached,
        then outer_func fails. On retry, inner_func should return from cache.
        """
        inner_count = 0
        outer_count = 0
        should_fail = True

        @cacheable()
        def inner_func(x: int) -> int:
            nonlocal inner_count
            inner_count += 1
            return x * 2

        @cacheable()
        def outer_func(x: int) -> int:
            nonlocal outer_count, should_fail
            outer_count += 1
            result = inner_func(x)  # This succeeds
            if should_fail:
                raise RuntimeError('outer failed after inner succeeded')
            return result + 1

        cache_dict = {}

        # First attempt: inner succeeds, outer fails
        with cache_context(cache_dict):
            with pytest.raises(RuntimeError):
                outer_func(5)
            assert inner_count == 1
            assert outer_count == 1

        # Check that outer is incomplete but has inner as child
        assert len(cache_dict['entries']) == 1
        outer_entry = cache_dict['entries'][0]
        assert outer_entry['result_type'] == CacheResultType.INCOMPLETE.value
        assert len(outer_entry['children']) == 1
        inner_entry = outer_entry['children'][0]
        assert inner_entry['result_type'] == CacheResultType.SUCCESS.value

        # Retry: inner should come from cache, outer should execute
        should_fail = False
        with cache_context(cache_dict):
            result = outer_func(5)
            assert result == 11
            assert inner_count == 1  # Not called again
            assert outer_count == 2  # Called again

    def test_multiple_nested_levels(self):
        """Test three levels of nesting."""
        level1_count = 0
        level2_count = 0
        level3_count = 0

        @cacheable()
        def level3(x: int) -> int:
            nonlocal level3_count
            level3_count += 1
            return x

        @cacheable()
        def level2(x: int) -> int:
            nonlocal level2_count
            level2_count += 1
            return level3(x) * 2

        @cacheable()
        def level1(x: int) -> int:
            nonlocal level1_count
            level1_count += 1
            return level2(x) + 1

        cache_dict = {}
        with cache_context(cache_dict):
            result = level1(5)
            assert result == 11

        # Check structure
        assert len(cache_dict['entries']) == 1
        l1 = cache_dict['entries'][0]
        assert len(l1['children']) == 1
        l2 = l1['children'][0]
        assert len(l2['children']) == 1
        l3 = l2['children'][0]
        assert l3['children'] == []

        # Replay
        level1_count = level2_count = level3_count = 0
        with cache_context(cache_dict):
            result = level1(5)
            assert result == 11
            assert level1_count == 0
            assert level2_count == 0
            assert level3_count == 0

    def test_multiple_children(self):
        """Test outer function calling multiple inner functions."""
        child1_count = 0
        child2_count = 0
        parent_count = 0

        @cacheable()
        def child1(x: int) -> int:
            nonlocal child1_count
            child1_count += 1
            return x + 1

        @cacheable()
        def child2(x: int) -> int:
            nonlocal child2_count
            child2_count += 1
            return x * 2

        @cacheable()
        def parent(x: int) -> int:
            nonlocal parent_count
            parent_count += 1
            a = child1(x)
            b = child2(x)
            return a + b

        cache_dict = {}
        with cache_context(cache_dict):
            result = parent(5)
            assert result == 16  # (5+1) + (5*2) = 6 + 10 = 16

        # Check structure
        parent_entry = cache_dict['entries'][0]
        assert len(parent_entry['children']) == 2

        # Replay
        child1_count = child2_count = parent_count = 0
        with cache_context(cache_dict):
            result = parent(5)
            assert result == 16
            assert child1_count == 0
            assert child2_count == 0
            assert parent_count == 0

    async def test_async_nested_recovery(self):
        """Test async nested recovery scenario."""
        inner_count = 0
        should_fail = True

        @cacheable()
        async def inner_async(x: int) -> int:
            nonlocal inner_count
            inner_count += 1
            await asyncio.sleep(0.001)
            return x * 2

        @cacheable()
        async def outer_async(x: int) -> int:
            nonlocal should_fail
            result = await inner_async(x)
            if should_fail:
                raise RuntimeError('outer async failed')
            return result + 1

        cache_dict = {}

        # First attempt
        with cache_context(cache_dict):
            with pytest.raises(RuntimeError):
                await outer_async(5)
            assert inner_count == 1

        # Retry
        should_fail = False
        with cache_context(cache_dict):
            result = await outer_async(5)
            assert result == 11
            assert inner_count == 1  # Not called again

    def test_multiple_outer_calls_recording_and_replay(self):
        """Test multiple outer_func calls are recorded and replayed correctly."""
        inner_count = 0
        outer_count = 0

        @cacheable()
        def inner_func(x: int) -> int:
            nonlocal inner_count
            inner_count += 1
            return x * 2

        @cacheable()
        def outer_func(x: int) -> int:
            nonlocal outer_count
            outer_count += 1
            return inner_func(x) + 1

        # Record: call outer_func 3 times with different args
        cache_dict = {}
        with cache_context(cache_dict):
            r1 = outer_func(1)
            r2 = outer_func(2)
            r3 = outer_func(3)
            assert [r1, r2, r3] == [3, 5, 7]  # (1*2+1), (2*2+1), (3*2+1)
            assert inner_count == 3
            assert outer_count == 3

        # Verify structure: 3 top-level entries, each with 1 child
        assert len(cache_dict['entries']) == 3
        for i, entry in enumerate(cache_dict['entries']):
            assert 'outer_func' in entry['func_name']
            assert len(entry['children']) == 1
            assert 'inner_func' in entry['children'][0]['func_name']

        # Replay: all should come from cache
        inner_count = 0
        outer_count = 0
        with cache_context(cache_dict):
            r1 = outer_func(1)
            r2 = outer_func(2)
            r3 = outer_func(3)
            assert [r1, r2, r3] == [3, 5, 7]
            assert inner_count == 0
            assert outer_count == 0

    def test_multiple_outer_calls_with_same_args(self):
        """Test multiple outer_func calls with same args are recorded separately."""
        inner_count = 0
        outer_count = 0

        @cacheable()
        def inner_func(x: int) -> int:
            nonlocal inner_count
            inner_count += 1
            return x * 2 + inner_count  # Different result each time

        @cacheable()
        def outer_func(x: int) -> int:
            nonlocal outer_count
            outer_count += 1
            return inner_func(x) + outer_count

        # Record: call outer_func 3 times with SAME args
        cache_dict = {}
        with cache_context(cache_dict):
            r1 = outer_func(5)
            r2 = outer_func(5)
            r3 = outer_func(5)
            # Each call produces different result due to counters
            assert r1 != r2 != r3
            assert inner_count == 3
            assert outer_count == 3

        # Verify structure: 3 separate entries even with same args
        assert len(cache_dict['entries']) == 3

        # Replay: each call should return its own cached result
        inner_count = 0
        outer_count = 0
        with cache_context(cache_dict):
            r1_replay = outer_func(5)
            r2_replay = outer_func(5)
            r3_replay = outer_func(5)
            assert r1_replay == r1
            assert r2_replay == r2
            assert r3_replay == r3
            assert inner_count == 0
            assert outer_count == 0

    def test_multiple_outer_calls_partial_failure_recovery(self):
        """Test recovery when some outer calls fail.

        Scenario:
        - Call 1: outer succeeds (inner succeeds)
        - Call 2: outer fails after inner succeeds
        - Call 3: outer succeeds (inner succeeds)
        On retry, call 2's inner should be cached.
        """
        inner_count = 0
        outer_count = 0
        fail_on_call = 2

        @cacheable()
        def inner_func(x: int) -> int:
            nonlocal inner_count
            inner_count += 1
            return x * 2

        @cacheable()
        def outer_func(x: int) -> int:
            nonlocal outer_count, fail_on_call
            outer_count += 1
            result = inner_func(x)
            if outer_count == fail_on_call:
                raise RuntimeError(f'outer failed on call {outer_count}')
            return result + 1

        cache_dict = {}

        # First pass: call 1 succeeds, call 2 fails, call 3 succeeds
        with cache_context(cache_dict):
            r1 = outer_func(1)
            assert r1 == 3
            with pytest.raises(RuntimeError, match='call 2'):
                outer_func(2)
            r3 = outer_func(3)
            assert r3 == 7
            assert inner_count == 3
            assert outer_count == 3

        # Verify structure
        assert len(cache_dict['entries']) == 3
        assert cache_dict['entries'][0]['result_type'] == 'success'
        assert cache_dict['entries'][1]['result_type'] == 'incomplete'  # Failed
        assert cache_dict['entries'][2]['result_type'] == 'success'
        # But inner of failed call should be cached
        assert cache_dict['entries'][1]['children'][0]['result_type'] == 'success'

        # Retry: call 1 and 3 from cache, call 2 retries but inner from cache
        inner_count = 0
        outer_count = 0
        fail_on_call = -1  # Don't fail this time
        with cache_context(cache_dict):
            r1 = outer_func(1)
            r2 = outer_func(2)  # Now succeeds
            r3 = outer_func(3)
            assert [r1, r2, r3] == [3, 5, 7]
            assert inner_count == 0  # All inner from cache
            assert outer_count == 1  # Only call 2 re-executed

    def test_deeply_nested_multiple_calls(self):
        """Test multiple calls with deep nesting (3 levels), called twice."""
        l1_count = 0
        l2_count = 0
        l3_count = 0

        @cacheable()
        def level3(x: int) -> int:
            nonlocal l3_count
            l3_count += 1
            return x

        @cacheable()
        def level2(x: int) -> int:
            nonlocal l2_count
            l2_count += 1
            return level3(x) * 2

        @cacheable()
        def level1(x: int) -> int:
            nonlocal l1_count
            l1_count += 1
            return level2(x) + 1

        cache_dict = {}

        # Record: call level1 twice
        with cache_context(cache_dict):
            r1 = level1(5)
            r2 = level1(10)
            assert r1 == 11  # 5*2+1
            assert r2 == 21  # 10*2+1
            assert l1_count == 2
            assert l2_count == 2
            assert l3_count == 2

        # Verify structure: 2 top-level entries, each with nested children
        assert len(cache_dict['entries']) == 2
        for entry in cache_dict['entries']:
            assert len(entry['children']) == 1  # level2
            assert len(entry['children'][0]['children']) == 1  # level3

        # Replay
        l1_count = l2_count = l3_count = 0
        with cache_context(cache_dict):
            r1 = level1(5)
            r2 = level1(10)
            assert r1 == 11
            assert r2 == 21
            assert l1_count == 0
            assert l2_count == 0
            assert l3_count == 0

    def test_interleaved_nested_calls(self):
        """Test interleaved pattern: outer1 -> inner, outer2 -> inner, ..."""
        inner_count = 0

        @cacheable()
        def inner_func(x: int) -> int:
            nonlocal inner_count
            inner_count += 1
            return x * 10

        @cacheable()
        def outer_a(x: int) -> int:
            return inner_func(x) + 1

        @cacheable()
        def outer_b(x: int) -> int:
            return inner_func(x) + 2

        cache_dict = {}

        # Record: interleaved calls
        with cache_context(cache_dict):
            ra1 = outer_a(1)
            rb1 = outer_b(2)
            ra2 = outer_a(3)
            rb2 = outer_b(4)
            assert [ra1, rb1, ra2, rb2] == [11, 22, 31, 42]
            assert inner_count == 4

        # Verify structure: 4 top-level entries
        assert len(cache_dict['entries']) == 4
        assert 'outer_a' in cache_dict['entries'][0]['func_name']
        assert 'outer_b' in cache_dict['entries'][1]['func_name']
        assert 'outer_a' in cache_dict['entries'][2]['func_name']
        assert 'outer_b' in cache_dict['entries'][3]['func_name']

        # Replay
        inner_count = 0
        with cache_context(cache_dict):
            ra1 = outer_a(1)
            rb1 = outer_b(2)
            ra2 = outer_a(3)
            rb2 = outer_b(4)
            assert [ra1, rb1, ra2, rb2] == [11, 22, 31, 42]
            assert inner_count == 0


# ============== Utility Function Tests ==============


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_is_cache_enabled_outside_context(self):
        """Test is_cache_enabled returns False outside context."""
        assert is_cache_enabled() is False

    def test_is_cache_enabled_inside_context(self):
        """Test is_cache_enabled returns True inside context."""
        with cache_context():
            assert is_cache_enabled() is True

    def test_get_cache_stage_outside_context(self):
        """Test get_cache_stage returns None outside context."""
        assert get_cache_stage() is None

    def test_get_cache_stage_replay(self):
        """Test get_cache_stage returns REPLAY with existing entries."""
        cache_dict = {
            'entries': [
                {'func_name': 'f', 'args_hash': 'h', 'result_type': 'success', 'result_value': 'v', 'children': []}
            ]
        }
        with cache_context(cache_dict):
            assert get_cache_stage() == CacheStage.REPLAY

    def test_get_cache_stage_cache(self):
        """Test get_cache_stage returns CACHE with empty entries."""
        with cache_context():
            assert get_cache_stage() == CacheStage.CACHE

    def test_is_replaying(self):
        """Test is_replaying function."""
        assert is_replaying() is False

        cache_dict = {
            'entries': [
                {'func_name': 'f', 'args_hash': 'h', 'result_type': 'success', 'result_value': 'v', 'children': []}
            ]
        }
        with cache_context(cache_dict):
            assert is_replaying() is True

    def test_is_caching(self):
        """Test is_caching function."""
        assert is_caching() is False

        with cache_context():
            assert is_caching() is True

    def test_get_current_cache(self):
        """Test get_current_cache function."""
        assert get_current_cache() is None

        cache_dict = {'entries': [], 'metadata': {'test': 'value'}}
        with cache_context(cache_dict):
            current = get_current_cache()
            assert current is not None
            assert current['metadata']['test'] == 'value'

    def test_clear_cache(self):
        """Test clear_cache function."""
        cache_dict = {
            'entries': [
                {'func_name': 'f1', 'args_hash': 'h1'},
                {'func_name': 'f2', 'args_hash': 'h2'},
            ],
            'metadata': {'key': 'value'},
        }
        clear_cache(cache_dict)
        assert cache_dict['entries'] == []
        assert cache_dict['metadata'] == {'key': 'value'}  # Metadata preserved


class TestGetCacheStats:
    """Tests for get_cache_stats function."""

    def test_empty_cache(self):
        """Test stats for empty cache."""
        stats = get_cache_stats({})
        assert stats == {'total': 0, 'top_level': 0, 'by_func': {}, 'by_tag': {}, 'replay_hits': 0, 'new_cached': 0}

    def test_cache_without_entries(self):
        """Test stats for cache without entries key."""
        stats = get_cache_stats({'metadata': {}})
        assert stats['total'] == 0

    def test_flat_entries(self):
        """Test stats for flat (non-nested) entries."""
        cache_dict = {
            'entries': [
                {'func_name': 'func_a', 'args_hash': 'h1', 'children': []},
                {'func_name': 'func_b', 'args_hash': 'h2', 'children': []},
                {'func_name': 'func_a', 'args_hash': 'h3', 'children': []},
            ]
        }
        stats = get_cache_stats(cache_dict)
        assert stats['total'] == 3
        assert stats['top_level'] == 3
        assert stats['by_func']['func_a'] == 2
        assert stats['by_func']['func_b'] == 1

    def test_nested_entries(self):
        """Test stats for nested entries."""
        cache_dict = {
            'entries': [
                {
                    'func_name': 'outer',
                    'args_hash': 'h1',
                    'children': [
                        {'func_name': 'inner1', 'args_hash': 'h2', 'children': []},
                        {
                            'func_name': 'inner2',
                            'args_hash': 'h3',
                            'children': [{'func_name': 'innermost', 'args_hash': 'h4', 'children': []}],
                        },
                    ],
                }
            ]
        }
        stats = get_cache_stats(cache_dict)
        assert stats['total'] == 4
        assert stats['top_level'] == 1
        assert stats['by_func']['outer'] == 1
        assert stats['by_func']['inner1'] == 1
        assert stats['by_func']['inner2'] == 1
        assert stats['by_func']['innermost'] == 1

    def test_with_cache_entry_objects(self):
        """Test stats work with CacheEntry objects."""
        inner = CacheEntry(func_name='inner', args_hash='h2')
        outer = CacheEntry(func_name='outer', args_hash='h1', children=[inner])
        cache_dict = {'entries': [outer]}
        stats = get_cache_stats(cache_dict)
        assert stats['total'] == 2
        assert stats['top_level'] == 1


# ============== Edge Cases ==============


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_same_function_different_args(self):
        """Test same function called with different args."""
        call_count = 0

        @cacheable()
        def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        cache_dict = {}
        with cache_context(cache_dict):
            r1 = func(1)
            r2 = func(2)
            r3 = func(3)
            assert [r1, r2, r3] == [2, 4, 6]
            assert call_count == 3

        assert len(cache_dict['entries']) == 3

        # Replay
        call_count = 0
        with cache_context(cache_dict):
            r1 = func(1)
            r2 = func(2)
            r3 = func(3)
            assert [r1, r2, r3] == [2, 4, 6]
            assert call_count == 0

    def test_ignore_args_hash_in_context(self):
        """Test ignore_args_hash context option."""
        call_count = 0

        @cacheable()
        def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Record with x=1
        cache_dict = {}
        with cache_context(cache_dict):
            func(1)

        # Replay with x=999 but ignore_args_hash=True
        call_count = 0
        with cache_context(cache_dict, ignore_args_hash=True):
            result = func(999)
            assert result == 2  # Returns cached result for x=1
            assert call_count == 0

    def test_class_method(self):
        """Test caching works with class methods."""
        call_count = 0

        class Calculator:
            @cacheable(ignore_args=['self'])
            def add(self, a: int, b: int) -> int:
                nonlocal call_count
                call_count += 1
                return a + b

        calc = Calculator()
        cache_dict = {}
        with cache_context(cache_dict):
            result = calc.add(1, 2)
            assert result == 3
            assert call_count == 1

        with cache_context(cache_dict):
            result = calc.add(1, 2)
            assert result == 3
            assert call_count == 1

    def test_function_with_kwargs(self):
        """Test caching with keyword arguments."""
        call_count = 0

        @cacheable()
        def func(a: int, b: int = 10, **kwargs) -> dict:
            nonlocal call_count
            call_count += 1
            return {'a': a, 'b': b, 'kwargs': kwargs}

        cache_dict = {}
        with cache_context(cache_dict):
            result = func(1, b=20, extra='value')
            assert result['a'] == 1
            assert result['b'] == 20
            assert call_count == 1

        with cache_context(cache_dict):
            result = func(1, b=20, extra='value')
            assert result['a'] == 1
            assert call_count == 1

    def test_non_picklable_result_fails(self):
        """Test that non-picklable results cause errors."""

        @cacheable()
        def return_lambda():
            return lambda x: x + 1

        # This should fail because lambdas can't be pickled
        cache_dict = {}
        with cache_context(cache_dict):
            with pytest.raises(AttributeError, match=r"Can't (get|pickle) local object"):
                return_lambda()

    def test_tags_parameter(self):
        """Test that tags parameter is accepted."""

        @cacheable(tags=['llm', 'completion'])
        def func():
            return 'result'

        cache_dict = {}
        with cache_context(cache_dict):
            result = func()
            assert result == 'result'

    def test_nested_context_not_supported(self):
        """Test nested cache_context behavior."""
        # Nested contexts override the previous one
        outer_cache = {}
        inner_cache = {}

        with cache_context(outer_cache):
            assert is_cache_enabled()
            with cache_context(inner_cache):
                assert is_cache_enabled()
                assert get_current_cache() is inner_cache
            # After inner context exits, outer is restored
            assert get_current_cache() is outer_cache

    def test_empty_function(self):
        """Test caching function with no return value."""
        call_count = 0

        @cacheable()
        def no_return():
            nonlocal call_count
            call_count += 1

        cache_dict = {}
        with cache_context(cache_dict):
            no_return()
            assert call_count == 1

        with cache_context(cache_dict):
            result = no_return()
            assert result is None
            assert call_count == 1


# ============== Integration Scenarios ==============


class TestIntegrationScenarios:
    """Tests for realistic integration scenarios."""

    async def test_llm_completion_simulation(self):
        """Simulate caching LLM completion calls."""
        api_calls = 0

        @cacheable(tags=['llm', 'completion'], ignore_args=['self'])
        async def create_completion(messages: list[dict], model: str = 'gpt-4') -> dict:
            nonlocal api_calls
            api_calls += 1
            await asyncio.sleep(0.001)  # Simulate API latency
            return {'content': f'Response to: {messages[-1]["content"]}', 'model': model}

        # Record
        cache_dict = {}
        with cache_context(cache_dict):
            r1 = await create_completion([{'role': 'user', 'content': 'Hello'}])
            r2 = await create_completion([{'role': 'user', 'content': 'How are you?'}])
            assert api_calls == 2

        # Replay (no API calls)
        api_calls = 0
        with cache_context(cache_dict):
            r1_replay = await create_completion([{'role': 'user', 'content': 'Hello'}])
            r2_replay = await create_completion([{'role': 'user', 'content': 'How are you?'}])
            assert api_calls == 0
            assert r1_replay == r1
            assert r2_replay == r2

    def test_tool_execution_simulation(self):
        """Simulate caching tool execution."""
        tool_calls = 0

        @cacheable(tags=['tool'])
        def execute_tool(tool_name: str, args: dict) -> dict:
            nonlocal tool_calls
            tool_calls += 1
            return {'tool': tool_name, 'result': f'executed with {args}'}

        @cacheable()
        def agent_step(action: str) -> str:
            result = execute_tool('search', {'query': action})
            return f'Agent completed: {result["result"]}'

        cache_dict = {}
        with cache_context(cache_dict):
            step_result = agent_step('find documents')
            assert tool_calls == 1

        # Replay
        tool_calls = 0
        with cache_context(cache_dict):
            replay_result = agent_step('find documents')
            assert tool_calls == 0
            assert replay_result == step_result

    def test_full_session_simulation(self):
        """Simulate a full agent session with recording and replay."""
        llm_calls = []
        tool_calls = []

        @cacheable(tags=['llm'])
        def llm_call(prompt: str) -> str:
            llm_calls.append(prompt)
            if 'search' in prompt.lower():
                return 'ACTION: search("test query")'
            return 'DONE: completed task'

        @cacheable(tags=['tool'])
        def tool_call(name: str, args: str) -> str:
            tool_calls.append((name, args))
            return f'Results for {args}'

        @cacheable()
        def run_agent():
            llm_call('User wants to search for information')
            tool_result = tool_call('search', 'test query')
            return llm_call(f'Tool returned: {tool_result}')

        # Record
        cache_dict = {}
        with cache_context(cache_dict):
            result = run_agent()
            assert len(llm_calls) == 2
            assert len(tool_calls) == 1

        # Replay
        llm_calls.clear()
        tool_calls.clear()
        with cache_context(cache_dict):
            replay_result = run_agent()
            assert len(llm_calls) == 0  # No actual calls
            assert len(tool_calls) == 0
            assert replay_result == result

    def test_ignore_args_hash_sequential_same_func_no_cross_pollution(self):
        """Regression: with ignore_args_hash=True, sequential calls to the same
        cacheable function must not return a stale result from a newly-appended
        entry that was recorded moments earlier in the same replay session."""
        call_log: list[str] = []

        @cacheable(tags=['llm'])
        def llm_completion(messages: list[str]) -> dict:
            call_log.append('llm')
            return {'tool_calls': [{'name': 'do_work', 'args': 'a'}, {'name': 'do_work', 'args': 'b'}]}

        @cacheable(tags=['tool'])
        def execute_tool(name: str, args: str) -> str:
            call_log.append(f'tool:{args}')
            return f'result_{args}'

        # --- Record: only cache the LLM call, NOT the tool calls ---
        record_cache: dict = {}
        with cache_context(record_cache):
            resp = llm_completion(['user prompt'])

        assert len(record_cache['entries']) == 1
        assert call_log == ['llm']

        # --- Replay with ignore_args_hash ---
        call_log.clear()
        with cache_context(record_cache, ignore_args_hash=True):
            resp = llm_completion(['user prompt'])
            assert call_log == []  # LLM from cache

            r1 = execute_tool('do_work', 'a')
            r2 = execute_tool('do_work', 'b')

        assert r1 == 'result_a'
        assert r2 == 'result_b'  # must NOT be 'result_a'
        assert call_log == ['tool:a', 'tool:b']
