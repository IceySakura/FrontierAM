"""Cache decorator for recording and replaying function calls.

This module provides a hierarchical caching mechanism that allows recording function calls
and their results in a tree structure, then replaying them later. This is useful for:
- Deterministic replay of agent trajectories
- Debugging and testing with recorded interactions
- Reducing API costs by caching LLM responses
- Fine-grained recovery when outer functions fail but inner functions succeed

The cache uses a tree structure where each function call can have child calls,
preserving the call hierarchy and enabling proper nested caching.

Example:
    # Recording mode
    cache_data = {}
    with cache_context(cache_data):
        result = await llm_completion(messages)  # Executed and recorded
    save_cache(cache_data, 'trajectory.pkl')

    # Replay mode
    cache_data = load_cache('trajectory.pkl')
    with cache_context(cache_data):
        result = await llm_completion(messages)  # From cache
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import pickle

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, cast


logger = logging.getLogger(__name__)


# ============== Enums ==============


class CacheResultType(Enum):
    """Type of cached result."""

    SUCCESS = 'success'
    EXCEPTION = 'exception'
    INCOMPLETE = 'incomplete'  # Function started but didn't complete


class CacheStage(Enum):
    """Cache operation stage."""

    REPLAY = 'replay'  # Only read from cache, no writing
    CACHE = 'cache'  # Only write to cache, no reading


# ============== Cache Entry ==============


class CacheEntry:
    """A single cache entry representing a function call.

    Attributes:
        func_name: Fully qualified function name
        args_hash: MD5 hash of the function arguments
        result_type: Type of result (success, exception, incomplete)
        result_value: Hex-encoded pickled result (None if incomplete)
        children: List of child function calls made during this call
        timestamp: ISO format timestamp of when this entry was created
    """

    __slots__ = ('func_name', 'args_hash', 'result_type', 'result_value', 'children', 'timestamp')

    def __init__(
        self,
        func_name: str,
        args_hash: str,
        result_type: str = CacheResultType.INCOMPLETE.value,
        result_value: str | None = None,
        children: list[CacheEntry] | None = None,
        timestamp: str | None = None,
    ):
        self.func_name = func_name
        self.args_hash = args_hash
        self.result_type = result_type
        self.result_value = result_value
        self.children = children if children is not None else []
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'func_name': self.func_name,
            'args_hash': self.args_hash,
            'result_type': self.result_type,
            'result_value': self.result_value,
            'children': [c.to_dict() for c in self.children],
            'timestamp': self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        """Create from dictionary."""
        return cls(
            func_name=data['func_name'],
            args_hash=data['args_hash'],
            result_type=data.get('result_type', CacheResultType.SUCCESS.value),
            result_value=data.get('result_value'),
            children=[cls.from_dict(c) for c in data.get('children', [])],
            timestamp=data.get('timestamp'),
        )

    def is_complete(self) -> bool:
        """Check if this entry has a complete result."""
        return self.result_type in (CacheResultType.SUCCESS.value, CacheResultType.EXCEPTION.value)


# ============== Stack Frame ==============


@dataclass
class StackFrame:
    """A frame in the cache replay stack.

    Tracks position within a list of cache entries (either top-level or children).

    Attributes:
        entries: List of cache entries at this level
        index: Current position in the entries list
        current_entry: The entry being built during CACHE stage (for adding children)
    """

    entries: list[CacheEntry]
    index: int = 0
    current_entry: CacheEntry | None = None


# ============== Context Variables ==============


_cache_dict: ContextVar[dict[str, Any] | None] = ContextVar('cache_dict', default=None)
_cache_stage: ContextVar[CacheStage] = ContextVar('cache_stage', default=CacheStage.CACHE)
_cache_stack: ContextVar[list[StackFrame] | None] = ContextVar('cache_stack', default=None)
_cache_ignore_args_hash: ContextVar[bool] = ContextVar('cache_ignore_args_hash', default=False)
_cache_overwrite_tags: ContextVar[list[str] | None] = ContextVar('cache_overwrite_tags', default=None)
_cache_runtime_stats: ContextVar[dict[str, int] | None] = ContextVar('cache_runtime_stats', default=None)


# ============== Context Manager ==============


@contextmanager
def cache_context(
    cache_dict: dict[str, Any] | None = None,
    ignore_args_hash: bool = False,
    overwrite_tags: list[str] | None = None,
):
    """Cache context manager with hierarchical caching.

    Args:
        cache_dict: Cache dictionary. If provided with existing entries, starts in REPLAY stage.
                   If None or empty, starts in CACHE stage.
        ignore_args_hash: If True, cache matching will be done by function name only,
                         ignoring argument differences.
        overwrite_tags: Tags for entries that should be overwritten if found.

    Yields:
        dict: The cache dictionary that can be saved and reused.

    Example:
        # Pure recording mode
        cache_data = {}
        with cache_context(cache_data):
            result = my_cached_function(x=1, y=2)
        save_cache(cache_data, 'cache.pkl')

        # Replay mode
        cache_data = load_cache('cache.pkl')
        with cache_context(cache_data):
            result = my_cached_function(x=1, y=2)  # From cache
    """
    if cache_dict is None:
        cache_dict = {'entries': [], 'metadata': {}}

    # Ensure required fields exist
    if 'entries' not in cache_dict:
        cache_dict['entries'] = []
    if 'metadata' not in cache_dict:
        cache_dict['metadata'] = {}

    # Convert dict entries to CacheEntry objects
    entries = cache_dict['entries']
    if entries and isinstance(entries[0], dict):
        cache_dict['entries'] = [CacheEntry.from_dict(e) for e in entries]

    # Determine initial stage
    initial_stage = CacheStage.REPLAY if cache_dict['entries'] else CacheStage.CACHE

    # Initialize stack with top-level entries
    initial_stack = [StackFrame(cache_dict['entries'], 0)]

    # Initialize runtime stats
    runtime_stats = {'replay_hits': 0, 'new_cached': 0}

    # Set context variables
    token_dict = _cache_dict.set(cache_dict)
    token_stage = _cache_stage.set(initial_stage)
    token_stack = _cache_stack.set(initial_stack)
    token_ignore = _cache_ignore_args_hash.set(ignore_args_hash)
    token_overwrite = _cache_overwrite_tags.set(overwrite_tags)
    token_stats = _cache_runtime_stats.set(runtime_stats)

    try:
        yield cache_dict
    finally:
        # Convert CacheEntry objects back to dicts for serialization
        if cache_dict and 'entries' in cache_dict:
            cache_dict['entries'] = [e.to_dict() if isinstance(e, CacheEntry) else e for e in cache_dict['entries']]

        # Store runtime stats in cache_dict for retrieval
        if cache_dict:
            cache_dict['_runtime_stats'] = runtime_stats.copy()

        # Reset context variables
        _cache_dict.reset(token_dict)
        _cache_stage.reset(token_stage)
        _cache_stack.reset(token_stack)
        _cache_ignore_args_hash.reset(token_ignore)
        _cache_overwrite_tags.reset(token_overwrite)
        _cache_runtime_stats.reset(token_stats)


# ============== File I/O ==============


def load_cache(path: str | Path) -> dict[str, Any]:
    """Load cache from a file.

    Args:
        path: Path to the cache file (pickle format)

    Returns:
        Cache dictionary with 'entries' and 'metadata' keys.
        Returns empty cache if file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        logger.debug(f'Cache file not found: {path}, returning empty cache')
        return {'entries': [], 'metadata': {}}

    try:
        data = pickle.loads(path.read_bytes())
        entry_count = len(data.get('entries', []))
        logger.debug(f'Loaded cache with {entry_count} top-level entries from {path}')
        return data
    except Exception as e:
        logger.warning(f'Failed to load cache from {path}: {e}, returning empty cache')
        return {'entries': [], 'metadata': {}}


def save_cache(cache_dict: dict[str, Any], path: str | Path) -> None:
    """Save cache to a file.

    Args:
        cache_dict: Cache dictionary to save
        path: Path to save the cache file (pickle format)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure entries are dicts for serialization
    save_dict = cache_dict.copy()
    if 'entries' in save_dict:
        save_dict['entries'] = [e.to_dict() if isinstance(e, CacheEntry) else e for e in save_dict['entries']]

    path.write_bytes(pickle.dumps(save_dict, protocol=pickle.HIGHEST_PROTOCOL))
    logger.debug(f'Saved cache with {len(save_dict.get("entries", []))} top-level entries to {path}')


# ============== Helper Functions ==============


def _compute_args_hash(args: dict[str, Any], ignore_args: list[str] | None = None) -> str:
    """Compute hash of function arguments."""
    if ignore_args:
        args = {k: v for k, v in args.items() if k not in ignore_args}
    return hashlib.md5(pickle.dumps(args, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def _extract_call_args(func: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Extract function call arguments into a dictionary."""
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    call_args = dict(bound.arguments)

    params = list(sig.parameters.keys())
    if params and params[0] in ('self', 'cls'):
        call_args.pop(params[0], None)

    return call_args


# ============== Decorator ==============

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


def cacheable(
    ignore_args: list[str] | None = None,
    cache_exceptions: bool = False,
    tags: list[str] | None = None,  # noqa: ARG001 - reserved for future tag-based filtering
) -> Callable[[F], F]:
    """Cache decorator for recording and replaying function calls.

    This decorator supports both sync and async functions with hierarchical caching.
    Child function calls are recorded within their parent's cache entry, preserving
    the call structure for accurate replay.

    When an outer function fails but inner functions succeed, the inner results
    are preserved in the cache. On retry, inner functions return from cache while
    the outer function re-executes.

    Args:
        ignore_args: List of argument names to ignore when matching cache entries.
        cache_exceptions: Whether to cache exceptions.
        tags: List of tags to associate with cached entries (for selective invalidation).

    Returns:
        Decorator function.

    Example:
        @cacheable(tags=['llm'])
        async def create_completion(messages, model, **kwargs):
            return await client.chat.completions.create(...)

        @cacheable(tags=['generate'])
        async def generate(prompt_ids, **kwargs):
            # Even if create_completion fails after this succeeds,
            # this result will be cached and reused on retry
            return await server.generate(prompt_ids, **kwargs)
    """

    def decorator(func: F) -> F:
        func_name = getattr(func, '__qualname__', getattr(func, '__name__', '<unknown>'))
        is_async = inspect.iscoroutinefunction(func)

        def _check_and_enter_cache(call_args: dict[str, Any]) -> tuple[bool, Any, CacheEntry | None]:
            """Check cache and enter child scope if matched.

            Returns:
                (has_complete_cache, cached_result, matched_entry)
                - has_complete_cache: True if we have a complete cached result
                - cached_result: The cached result (if complete)
                - matched_entry: The matched entry (for entering children scope)
            """
            cache_dict = _cache_dict.get()
            stack = _cache_stack.get()

            if cache_dict is None or stack is None:
                return False, None, None

            if _cache_stage.get() != CacheStage.REPLAY:
                return False, None, None

            args_hash = _compute_args_hash(call_args, ignore_args)
            ignore_hash = _cache_ignore_args_hash.get()

            # Get current frame
            frame = stack[-1]

            if frame.index >= len(frame.entries):
                _cache_stage.set(CacheStage.CACHE)
                return False, None, None

            entry = frame.entries[frame.index]

            # Check if entry matches
            if entry.func_name == func_name and (ignore_hash or entry.args_hash == args_hash):
                # Match found - advance index
                frame.index += 1

                if entry.is_complete():
                    # Complete cache - return result
                    result_value = pickle.loads(bytes.fromhex(entry.result_value))  # type: ignore[arg-type]
                    logger.debug(f'Cache hit (complete): {func_name}')

                    # Increment replay_hits
                    stats = _cache_runtime_stats.get()
                    if stats is not None:
                        stats['replay_hits'] += 1

                    if entry.result_type == CacheResultType.EXCEPTION.value:
                        return True, ('exception', result_value), entry
                    return True, ('success', result_value), entry
                # Incomplete cache - need to re-execute but can use children
                logger.debug(f'Cache hit (incomplete): {func_name}, using children for nested calls')
                return False, None, entry

            return False, None, None

        def _create_new_entry(call_args: dict[str, Any]) -> CacheEntry:
            """Create a new cache entry for recording."""
            args_hash = _compute_args_hash(call_args, ignore_args)
            entry = CacheEntry(
                func_name=func_name,
                args_hash=args_hash,
                result_type=CacheResultType.INCOMPLETE.value,
                result_value=None,
                children=[],
            )
            if tags:
                # Store tags in a way that can be retrieved later
                pass  # Tags are implicit in the entry for now
            return entry

        def _exit_scope_and_record(
            entry: CacheEntry, result: Any, result_type: CacheResultType, was_replay_match: bool
        ) -> None:
            """Exit current scope and record result."""
            stack = _cache_stack.get()
            cache_dict = _cache_dict.get()

            if stack is None or cache_dict is None:
                return

            # Update entry with result
            entry.result_type = result_type.value
            entry.result_value = pickle.dumps(result).hex()
            entry.timestamp = datetime.now().isoformat()

            # Pop children frame if we pushed one
            if len(stack) > 1 and stack[-1].current_entry is entry:
                stack.pop()

            # If this was a new entry (not from replay), add it to parent's children or top-level
            if not was_replay_match:
                # Increment new_cached counter
                stats = _cache_runtime_stats.get()
                if stats is not None:
                    stats['new_cached'] += 1

                if len(stack) > 1:
                    # Add to parent's children
                    parent_frame = stack[-1]
                    if parent_frame.current_entry is not None:
                        parent_frame.current_entry.children.append(entry)
                else:
                    # Add to top-level entries
                    top_entries = cache_dict['entries']
                    if isinstance(top_entries, list):
                        top_entries.append(entry)

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not is_cache_enabled():
                    return await func(*args, **kwargs)

                call_args = _extract_call_args(func, args, kwargs)
                stack = _cache_stack.get()

                # Check cache
                has_cache, cache_result, matched_entry = _check_and_enter_cache(call_args)

                if has_cache and cache_result is not None:
                    result_type, result_value = cache_result
                    if result_type == 'exception':
                        raise result_value
                    return result_value

                # Need to execute - create or reuse entry
                if matched_entry is not None:
                    # Reusing incomplete entry - enter its children scope
                    entry = matched_entry
                    was_replay_match = True
                    if stack:
                        stack.append(StackFrame(entry.children, 0, entry))
                else:
                    # New entry
                    entry = _create_new_entry(call_args)
                    was_replay_match = False
                    if stack:
                        stack.append(StackFrame(entry.children, 0, entry))

                try:
                    result = await func(*args, **kwargs)
                    _exit_scope_and_record(entry, result, CacheResultType.SUCCESS, was_replay_match)
                    return result
                except Exception as e:
                    if cache_exceptions:
                        _exit_scope_and_record(entry, e, CacheResultType.EXCEPTION, was_replay_match)
                    else:
                        # Still record as incomplete so children are preserved
                        if stack and len(stack) > 1 and stack[-1].current_entry is entry:
                            stack.pop()
                        if not was_replay_match:
                            # Add incomplete entry to preserve children
                            cache_dict = _cache_dict.get()
                            if cache_dict and stack:
                                if len(stack) > 1:
                                    parent_frame = stack[-1]
                                    if parent_frame.current_entry is not None:
                                        parent_frame.current_entry.children.append(entry)
                                else:
                                    cache_dict['entries'].append(entry)
                    raise

            return cast(F, async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not is_cache_enabled():
                return func(*args, **kwargs)

            call_args = _extract_call_args(func, args, kwargs)
            stack = _cache_stack.get()

            # Check cache
            has_cache, cache_result, matched_entry = _check_and_enter_cache(call_args)

            if has_cache and cache_result is not None:
                result_type, result_value = cache_result
                if result_type == 'exception':
                    raise result_value
                return result_value

            # Need to execute - create or reuse entry
            if matched_entry is not None:
                entry = matched_entry
                was_replay_match = True
                if stack:
                    stack.append(StackFrame(entry.children, 0, entry))
            else:
                entry = _create_new_entry(call_args)
                was_replay_match = False
                if stack:
                    stack.append(StackFrame(entry.children, 0, entry))

            try:
                result = func(*args, **kwargs)
                _exit_scope_and_record(entry, result, CacheResultType.SUCCESS, was_replay_match)
                return result
            except Exception as e:
                if cache_exceptions:
                    _exit_scope_and_record(entry, e, CacheResultType.EXCEPTION, was_replay_match)
                else:
                    if stack and len(stack) > 1 and stack[-1].current_entry is entry:
                        stack.pop()
                    if not was_replay_match:
                        cache_dict = _cache_dict.get()
                        if cache_dict and stack:
                            if len(stack) > 1:
                                parent_frame = stack[-1]
                                if parent_frame.current_entry is not None:
                                    parent_frame.current_entry.children.append(entry)
                            else:
                                cache_dict['entries'].append(entry)
                raise

        return cast(F, sync_wrapper)

    return decorator


# ============== Utility Functions ==============


def is_cache_enabled() -> bool:
    """Check if currently in a cache context."""
    return _cache_dict.get() is not None


def get_cache_stage() -> CacheStage | None:
    """Get current cache stage."""
    if not is_cache_enabled():
        return None
    return _cache_stage.get()


def is_replaying() -> bool:
    """Check if currently in REPLAY stage."""
    return get_cache_stage() == CacheStage.REPLAY


def is_caching() -> bool:
    """Check if currently in CACHE stage."""
    return get_cache_stage() == CacheStage.CACHE


def get_current_cache() -> dict[str, Any] | None:
    """Get current cache dictionary."""
    return _cache_dict.get()


# ============== Cache Manipulation ==============


def clear_cache(cache_dict: dict[str, Any]) -> None:
    """Clear all cache entries."""
    cache_dict['entries'] = []


def _count_entries(entries: list) -> tuple[int, dict[str, int], dict[str, int]]:
    """Recursively count entries, by function, and by tag."""
    total = 0
    by_func: dict[str, int] = {}
    by_tag: dict[str, int] = {}

    for entry in entries:
        if isinstance(entry, CacheEntry):
            e = entry
        elif isinstance(entry, dict):
            e = CacheEntry.from_dict(entry)
        else:
            continue

        total += 1
        by_func[e.func_name] = by_func.get(e.func_name, 0) + 1

        # Recursively count children
        if e.children:
            child_total, child_by_func, child_by_tag = _count_entries(e.children)
            total += child_total
            for k, v in child_by_func.items():
                by_func[k] = by_func.get(k, 0) + v
            for k, v in child_by_tag.items():
                by_tag[k] = by_tag.get(k, 0) + v

    return total, by_func, by_tag


def get_runtime_stats() -> dict[str, int]:
    """Get current runtime statistics.

    Returns:
        Dictionary with 'replay_hits' and 'new_cached' counts.
        Returns zeros if not in cache context.
    """
    stats = _cache_runtime_stats.get()
    if stats is None:
        return {'replay_hits': 0, 'new_cached': 0}
    return stats.copy()


def get_cache_stats(cache_dict: dict[str, Any]) -> dict[str, Any]:
    """Get statistics about cache entries.

    Args:
        cache_dict: Cache dictionary to analyze

    Returns:
        Dictionary with cache statistics including nested entries and runtime stats
    """
    if not cache_dict or 'entries' not in cache_dict:
        return {'total': 0, 'top_level': 0, 'by_func': {}, 'by_tag': {}, 'replay_hits': 0, 'new_cached': 0}

    entries = cache_dict['entries']
    total, by_func, by_tag = _count_entries(entries)

    # Get runtime stats if available (stored after cache_context exits)
    runtime_stats = cache_dict.get('_runtime_stats', {'replay_hits': 0, 'new_cached': 0})

    return {
        'total': total,
        'top_level': len(entries),
        'by_func': by_func,
        'by_tag': by_tag,
        'replay_hits': runtime_stats.get('replay_hits', 0),
        'new_cached': runtime_stats.get('new_cached', 0),
    }


def invalidate_by_tags(
    cache_dict: dict[str, Any],
    tags: list[str],  # noqa: ARG001 - not implemented yet
) -> dict[str, Any]:
    """Remove cache entries matching specified tags.

    Note: Tag-based invalidation is not fully implemented in hierarchical cache.
    This function is kept for API compatibility.
    """
    logger.warning('invalidate_by_tags is not fully implemented for hierarchical cache')
    return cache_dict


def filter_by_tags(
    cache_dict: dict[str, Any],
    tags: list[str],  # noqa: ARG001 - not implemented yet
) -> dict[str, Any]:
    """Keep only cache entries matching specified tags.

    Note: Tag-based filtering is not fully implemented in hierarchical cache.
    This function is kept for API compatibility.
    """
    logger.warning('filter_by_tags is not fully implemented for hierarchical cache')
    return cache_dict


# ============== Exports ==============

__all__ = [
    # Enums
    'CacheResultType',
    'CacheStage',
    'CacheEntry',
    # Context manager
    'cache_context',
    # File I/O
    'load_cache',
    'save_cache',
    # Decorator
    'cacheable',
    # Status functions
    'is_cache_enabled',
    'get_cache_stage',
    'is_replaying',
    'is_caching',
    'get_current_cache',
    'get_runtime_stats',
    # Cache manipulation
    'clear_cache',
    'invalidate_by_tags',
    'filter_by_tags',
    'get_cache_stats',
]
