from __future__ import annotations

import asyncio
import atexit
import base64
import contextlib
import contextvars
import datetime
import inspect
import json
import logging
import os
import pickle
import random
import re
import sys
import time
import traceback

from collections.abc import Awaitable, Callable
from concurrent.futures import Executor
from functools import partial, wraps
from typing import Any, TypeVar

import aiohttp

from tenacity import retry, retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential_jitter


logger = logging.getLogger(__name__)

# Type variable for the return type of the decorated function
T = TypeVar('T')


async def maybe_await(value: Any) -> Any:
    """Await an awaitable value, or return a synchronous value unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value


async def to_thread_complete(func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run in a thread, deferring cancellation until it finishes so callers can release resources safely."""
    return await await_complete(asyncio.to_thread(func, *args, **kwargs))


async def run_in_executor_complete(executor: Executor, func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run in a dedicated executor with the same context/cancellation contract as to_thread_complete."""
    context = contextvars.copy_context()
    future = asyncio.get_running_loop().run_in_executor(executor, partial(context.run, func, *args, **kwargs))
    return await await_complete(future)


async def await_complete(awaitable: Awaitable[T]) -> T:
    """Finish an owned operation before propagating cancellation and releasing its resources."""
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    if cancelled:
        with contextlib.suppress(BaseException):
            task.result()
        raise asyncio.CancelledError
    return task.result()


def loop_local_cache(func: Callable[..., T]) -> Callable[..., T]:  # noqa
    """An event-loop-local version of functools.cache decorator.

    This decorator caches the result of a function call separately for each asyncio event loop,
    ensuring that each event loop has its own cached instance.
    """
    # Dictionary to store loop-local caches
    local_caches: dict[int, dict[tuple, Any]] = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the current event loop
        try:
            loop = asyncio.get_running_loop()
            # Use the loop's memory address as the unique identifier
            loop_id = id(loop)
        except RuntimeError:
            # If no event loop is running, use a default ID
            loop_id = 0

        # Create a cache for this event loop if it doesn't exist
        if loop_id not in local_caches:
            local_caches[loop_id] = {}

        # Create a key from the function arguments
        key = (args, tuple(sorted(kwargs.items())))

        # Check if the result is already cached for this event loop
        if key not in local_caches[loop_id]:
            # Call the function and cache the result for this event loop
            local_caches[loop_id][key] = func(*args, **kwargs)

        return local_caches[loop_id][key]

    return wrapper


def register_and_disable_close(sync):
    def decorator(func):
        async def async_fake_close():
            pass

        @wraps(func)
        def wrapper(*args, **kwargs):
            object_ = func(*args, **kwargs)

            if hasattr(object_, 'close'):
                orignal_close = object_.close

                if sync:
                    object_.close = lambda: None
                    atexit.register(orignal_close)
                else:
                    object_.close = async_fake_close

                    async def close():
                        await orignal_close()

                    atexit.register(lambda: asyncio.run(close()))

            return object_

        return wrapper

    return decorator


@loop_local_cache
def aiohttp_session():
    return aiohttp.ClientSession()


def format_exception(exc: Exception) -> str:
    return f'{type(exc).__name__}: {exc}'


async def get_process_returncode(proc: asyncio.subprocess.Process, *, timeout: float = 2.0) -> int | None:
    return_code = proc.returncode
    if return_code is not None:
        return return_code
    try:
        return await asyncio.wait_for(proc.wait(), timeout=timeout)
    except Exception:
        return proc.returncode


async def terminate_process(proc: asyncio.subprocess.Process, *, timeout: float = 2.0) -> None:
    if proc.returncode is not None:
        return
    with contextlib.suppress(Exception):
        proc.terminate()
    if proc.returncode is not None:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        if proc.returncode is not None:
            return
        with contextlib.suppress(Exception):
            proc.kill()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=timeout)


def max_concurrency(limit: int):
    """Decorator to limit the maximum number of concurrent executions of an async function"""
    semaphore = asyncio.Semaphore(limit)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


async def with_random_jitter(min_ms=0, max_ms=1000):
    """Add random delay to avoid heavy concurrent requests"""
    jitter_ms = random.randint(min_ms, max_ms)
    await asyncio.sleep(jitter_ms / 1000)


def with_sync_random_jitter(min_ms=0, max_ms=1000):
    """Add random delay to avoid heavy concurrent requests"""
    jitter_ms = random.randint(min_ms, max_ms)
    time.sleep(jitter_ms / 1000)


def param_concurrency(param_keys=None):
    """
    Decorator to limit concurrent executions of an async function based on parameter values.
    Functions with identical parameter values (specified by param_keys) will execute serially.

    Args:
        param_keys: List of parameter names to consider for concurrency control.
                   If None, all parameters will be considered.
    """
    # Dictionary to store semaphores for each unique parameter combination
    semaphores = {}

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create a key based on the specified parameters
            if param_keys is None:
                # Use all parameters if none specified
                key_parts = list(args)
                for k, v in sorted(kwargs.items()):
                    key_parts.append((k, v))
            else:
                # Use only the specified parameters
                key_parts = []
                # Handle positional arguments that match param_keys
                func_params = func.__code__.co_varnames[: func.__code__.co_argcount]
                for i, arg_name in enumerate(func_params):
                    if i < len(args) and arg_name in param_keys:
                        key_parts.append((arg_name, args[i]))

                # Handle keyword arguments that match param_keys
                for k in param_keys:
                    if k in kwargs:
                        key_parts.append((k, kwargs[k]))

            # Create a hashable key from the parameters
            param_key = str(key_parts)

            # Get or create a semaphore for this parameter combination
            if param_key not in semaphores:
                semaphores[param_key] = asyncio.Semaphore(1)

            # Execute the function with the semaphore
            async with semaphores[param_key]:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def now():
    return datetime.datetime.now(datetime.timezone.utc)


async def run_shell_cmd(cmd, cwd=None):
    logger.debug(f'Running shell command: {cmd} in {cwd or os.getcwd()}')
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=cwd
    )
    _stdout, _stderr = await proc.communicate()
    if proc.returncode != 0:
        error_msg = _stderr.decode('utf-8').strip()
        logger.error(f'stdout before error: {_stdout.decode("utf-8").strip()}')
        raise RuntimeError(f'Command failed with error: {error_msg}')
    return _stdout.decode('utf-8').strip()


def log_execution_time(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logging.getLogger(__name__).debug(f'Task completed in {duration:.2f} seconds')
            return result
        except Exception:
            duration = time.time() - start
            logging.getLogger(__name__).debug(f'Task failed after {duration:.2f} seconds')
            raise

    return wrapper


def load_jsonl(file_path):
    with open(file_path) as f:
        return [json.loads(line) for line in f.readlines()]


# Default cumulative retry limit; NEXUS_RETRY_TOTAL_TIMEOUT can override it.
DEFAULT_RETRY_TOTAL_TIMEOUT = 0.0  # Opt in to an additional elapsed-time limit through configuration.


def _resolve_retry_total_timeout(explicit: float | None = None) -> float:
    """Resolve the retry limit: explicit value, environment, then default; non-positive values disable it."""
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get('NEXUS_RETRY_TOTAL_TIMEOUT')
    if raw is None or raw.strip() == '':
        return DEFAULT_RETRY_TOTAL_TIMEOUT
    try:
        return float(raw)
    except ValueError:
        logger.warning('NEXUS_RETRY_TOTAL_TIMEOUT=%r 无法解析为数字，回退默认 %ss', raw, DEFAULT_RETRY_TOTAL_TIMEOUT)
        return DEFAULT_RETRY_TOTAL_TIMEOUT


def on_retry_error(s):
    e = s.outcome.exception()
    logger.error(f'give up retrying. error: {e}')
    raise e


# Bound individual asynchronous calls separately from the accumulated retry time.
DEFAULT_CALL_DEADLINE = 0.0  # SDK and task deadlines remain independent of this optional limit.


def _resolve_call_deadline(explicit: float | None = None) -> float:
    """Resolve the per-call limit: explicit value, environment, then default; non-positive values disable it."""
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get('NEXUS_CALL_DEADLINE')
    if raw is None or raw.strip() == '':
        return DEFAULT_CALL_DEADLINE
    try:
        return float(raw)
    except ValueError:
        logger.warning('NEXUS_CALL_DEADLINE=%r 无法解析为数字，回退默认 %ss', raw, DEFAULT_CALL_DEADLINE)
        return DEFAULT_CALL_DEADLINE


class CallDeadlineExceededError(TimeoutError):
    """One local call deadline expired; an outer retry may try again."""


def with_call_deadline(deadline: float | None = None):
    """Bound one asynchronous attempt inside the retry layer.

    Return synchronous callables unchanged. Disable the limit for non-positive
    values; callers can retry an expired asynchronous attempt.
    """
    d = _resolve_call_deadline(deadline)

    def decorator(func):
        if not is_async_callable(func) or not d or d <= 0:
            return func

        @wraps(func)
        async def wrapper(*args, **kwargs):
            call = asyncio.ensure_future(func(*args, **kwargs))
            try:
                # wait() keeps local expiry separate from cancellation of the
                # caller, even if the provider converts CancelledError itself.
                done, _ = await asyncio.wait({call}, timeout=d)
                if done:
                    return call.result()
                call.cancel()
                await asyncio.gather(call, return_exceptions=True)
            except BaseException:
                call.cancel()
                await asyncio.gather(call, return_exceptions=True)
                raise
            raise CallDeadlineExceededError(
                f'call exceeded client-side deadline {d}s; func={getattr(func, "__name__", func)!r}'
            )

        return wrapper

    return decorator


def before_retry_sleep(s):
    e = s.outcome.exception()
    tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    msg = f'function call error for {s.attempt_number} time(s), will retry...\nError: {e}\nTraceback:\n{tb_str}'
    if s.attempt_number > 2:
        logger.warning(msg)
    else:
        logger.debug(msg)


def _is_retryable_exception(exc: BaseException) -> bool:
    """Return whether an exception may benefit from retrying.

    Control-flow exceptions, including cancellation, must propagate without
    retrying stopped or timed-out work.
    Context-window errors and HTTP 400 responses are deterministic for an
    unchanged request (e.g. a text-only model rejecting an image part). Retrying
    them only adds backoff latency and can keep a batch item alive for tens of
    minutes without any chance of success.
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if not isinstance(current, Exception):
            return False
        if type(current).__name__ in {'ContextWindowExceededError', 'ContextLengthExceededError'}:
            return False
        if getattr(current, 'status_code', None) == 400:
            return False
        if isinstance(current, CallDeadlineExceededError):
            return True
        current = current.__cause__ or current.__context__
    return True


def is_async_callable(obj: Any) -> bool:
    """
    异步判定：支持标准 async def、functools.wraps 包装后的函数、
    以及像 OpenAI AsyncOpenAI 这种动态生成的异步方法。
    """
    # 直接判定是否为异步函数对象
    if inspect.iscoroutinefunction(obj):
        return True

    # 检查被装饰器（如 functools.wraps）包装后的原始函数
    actual_obj = inspect.unwrap(obj)
    return bool(inspect.iscoroutinefunction(actual_obj))


def configurable_retry(max_attempts):
    """Retry until the attempt count or cumulative retry limit is reached.

    The cumulative limit is checked between attempts. A separate per-call deadline
    is needed to bound an asynchronous request that does not return.
    """
    max_attempts = int(max_attempts)
    total_timeout = _resolve_retry_total_timeout()

    def decorator(func):
        def _stop():
            # Stop when either the attempt count or the configured time limit is reached.
            s = stop_after_attempt(max_attempts)
            if total_timeout and total_timeout > 0:
                s = s | stop_after_delay(total_timeout)
            return s

        @wraps(func)
        @retry(
            wait=wait_exponential_jitter(
                max=10 * 60,
                jitter=30,
            ),
            stop=_stop(),
            before_sleep=before_retry_sleep,
            retry_error_callback=on_retry_error,
            retry=retry_if_exception(_is_retryable_exception),
        )
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        @wraps(func)
        @retry(
            wait=wait_exponential_jitter(
                max=10 * 60,
                jitter=30,
            ),
            stop=_stop(),
            before_sleep=before_retry_sleep,
            retry_error_callback=on_retry_error,
            retry=retry_if_exception(_is_retryable_exception),
        )
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return async_wrapper if is_async_callable(func) else sync_wrapper

    return decorator


async def safe_call_with_retry(func, max_attempts, *args, **kwargs):
    """
    Safely call a function with retries and return a wrapped result on final failure.
    """
    try:
        # Call the function with retry wrapper
        result = await configurable_retry(max_attempts)(func)(*args, **kwargs)
        # Wrap success
        return {'success': True, 'data': result, 'error': None}
    except Exception as e:
        # Wrap final failure
        return {'success': False, 'data': None, 'error': f'Max retry attempts reached. Error: {str(e)}'}


def pickle_dumps(obj) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode()


def pickle_loads(s):
    return pickle.loads(base64.b64decode(s))


def get_value_with_style_compat(dictionary: dict, key: str, default=None):
    if key in dictionary:
        return dictionary[key]

    if any(c.isupper() for c in key):
        snake_key = ''.join(['_' + c.lower() if c.isupper() else c for c in key]).lstrip('_')
        return dictionary.get(snake_key, default)
    words = key.split('_')
    camel_key = words[0] + ''.join(word.capitalize() for word in words[1:])
    return dictionary.get(camel_key, default)


def ttl_cache(seconds: float):
    def decorator(func):
        cache = {}

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                key = str(args) + str(kwargs)
                now = time.time()

                if key in cache:
                    result, timestamp = cache[key]
                    if now - timestamp < seconds:
                        return result

                result = await func(*args, **kwargs)
                cache[key] = (result, now)
                return result

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            now = time.time()

            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < seconds:
                    return result

            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result

        return sync_wrapper

    return decorator


def get_current_python_version() -> str:
    """Get the current Python interpreter version as 'major.minor'."""
    return f'{sys.version_info.major}.{sys.version_info.minor}'


def get_running_nexus_source() -> str:
    """Return the filesystem path of the currently-running ``nexus`` package.

    Resolves to either the source checkout project root (the directory
    containing ``pyproject.toml``) or the installed package dir in
    ``site-packages``. This guarantees the artifact always reflects the
    *running* code, independent of the process CWD.
    """
    from pathlib import Path

    import nexus as _nexus_pkg

    nexus_dir = Path(_nexus_pkg.__file__).resolve().parent
    # Source checkout: nexus/ sits next to pyproject.toml
    if (nexus_dir.parent / 'pyproject.toml').exists():
        return str(nexus_dir.parent)
    # Installed: point at the package dir, packaging._find_dist_info handles it
    return str(nexus_dir)


def random_urlsafe_id(num_bytes: int) -> str:
    return base64.urlsafe_b64encode(os.urandom(num_bytes)).decode('ascii').rstrip('=')


def serialize_object(obj: Any) -> Any:
    """Convert an object to a JSON-serializable form.

    Args:
        obj: The object to serialize

    Returns:
        A JSON-serializable representation of the object
    """
    # Basic types that are already JSON-serializable
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [serialize_object(item) for item in obj]

    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: serialize_object(v) for k, v in obj.items()}

    # Handle objects with model_dump method (e.g., Pydantic models)
    if hasattr(obj, 'model_dump') and callable(obj.model_dump):
        try:
            return serialize_object(obj.model_dump())
        except Exception:
            pass

    # Handle objects with dict method
    if hasattr(obj, 'dict') and callable(obj.dict):
        try:
            return serialize_object(obj.dict())
        except Exception:
            pass

    # Handle objects with to_dict method
    if hasattr(obj, 'to_dict') and callable(obj.to_dict):
        try:
            return serialize_object(obj.to_dict())
        except Exception:
            pass

    # Handle objects with __dict__ attribute
    if hasattr(obj, '__dict__'):
        try:
            # Filter out private attributes and methods
            filtered_dict = {k: v for k, v in obj.__dict__.items() if not k.startswith('_') and not callable(v)}
            if filtered_dict:  # Only use __dict__ if it has public attributes
                return serialize_object(filtered_dict)
        except Exception:
            pass

    # Handle common types that need special serialization
    if hasattr(obj, '__class__'):
        class_name = obj.__class__.__name__

        # Handle datetime objects
        if class_name in ('datetime', 'date', 'time'):
            return obj.isoformat()

        # Handle UUID objects
        if class_name == 'UUID':
            return str(obj)

        # Handle Path objects
        if class_name in ('Path', 'PosixPath', 'WindowsPath'):
            return str(obj)

        # Handle numpy arrays
        if class_name == 'ndarray':
            try:
                return obj.tolist()
            except Exception:
                pass

        # Handle pandas DataFrame/Series
        if class_name in ('DataFrame', 'Series'):
            try:
                return obj.to_dict()
            except Exception:
                pass

    # Last resort: convert to string representation
    try:
        return str(obj)
    except Exception:
        return f'<unserializable {type(obj).__name__}>'


def pformat(
    obj: Any,
    *,
    max_str_len: int = 200,
    compact_threshold: int = 80,
    indent: int = 2,
) -> str:
    """Pretty format an object (especially Pydantic models) for display.

    - Small objects: single line
    - Large objects: multi-line with indentation (like JSON indent=2)
    - Long strings: truncated with ellipsis

    Args:
        obj: The object to format
        max_str_len: Maximum length for string values before truncation
        compact_threshold: If compact representation is shorter than this, use single line
        indent: Indentation spaces for multi-line output

    Returns:
        Formatted string representation
    """

    def truncate_strings(data: Any) -> Any:
        """Recursively truncate long strings in nested data structures."""
        if isinstance(data, str):
            if len(data) > max_str_len:
                half = max_str_len // 2
                return f'{data[:half]}...({len(data)} chars)...{data[-half:]}'
            return data
        if isinstance(data, dict):
            return {k: truncate_strings(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [truncate_strings(item) for item in data]
        return data

    # Convert to serializable form
    data = serialize_object(obj)
    data = truncate_strings(data)

    # Try compact representation first
    try:
        compact = json.dumps(data, ensure_ascii=False, default=str, separators=(',', ':'))
        if len(compact) <= compact_threshold:
            return compact
        # Use indented format for larger objects
        return json.dumps(data, ensure_ascii=False, default=str, indent=indent)
    except Exception:
        return str(obj)


def pformat_for_display(obj: Any, **kwargs: Any) -> str:
    """Like :func:`pformat` but respects the ``__pformat_exclude__`` convention.

    A pydantic model may declare a class-level ``__pformat_exclude__`` set of
    field names that should be dropped when rendered for **display** (CLI logs,
    demo output). The excluded fields remain present on the model itself and in
    ``model_dump_json`` output — this only affects the string produced here.

    Non-pydantic objects, or models without ``__pformat_exclude__``, are
    formatted exactly like :func:`pformat`.
    """
    exclude = getattr(type(obj), '__pformat_exclude__', None)
    if exclude and hasattr(obj, 'model_dump'):
        try:
            return pformat(obj.model_dump(exclude=set(exclude)), **kwargs)
        except Exception:
            pass
    return pformat(obj, **kwargs)


def escape_non_oc_interpolations(obj: Any) -> Any:
    """Escape non-OmegaConf interpolations to prevent parsing errors.

    OmegaConf's ANTLR-based parser will fail on ${...} patterns that don't conform
    to its grammar (e.g., ${func()} in JavaScript code). This function escapes all
    ${...} patterns except ${oc.*} which are valid OmegaConf resolvers.

    Args:
        obj: Any object (recursively processes dicts, lists, and strings)

    Returns:
        Object with non-oc interpolations escaped as \\${...}
    """
    if isinstance(obj, str):
        # Escape any ${...} that is not a valid OmegaConf resolver (${oc.*}).
        # Also handle preceding backslashes: OmegaConf treats \${..} as escaped
        # (literal ${..}), but \\${..} as escaped-backslash + live interpolation.
        # We capture any preceding backslashes and double them, then add one more
        # to ensure OmegaConf always sees an odd count (= escaped interpolation).
        return re.sub(
            r'(\\*)\$\{(?!oc\.)',
            lambda m: '\\' * (len(m.group(1)) * 2 + 1) + '${',
            obj,
        )
    if isinstance(obj, dict):
        # Also escape string keys because some payload dicts may contain template-like keys.
        escaped: dict[Any, Any] = {}
        for k, v in obj.items():
            kk = escape_non_oc_interpolations(k) if isinstance(k, str) else k
            escaped[kk] = escape_non_oc_interpolations(v)
        return escaped
    if isinstance(obj, list):
        return [escape_non_oc_interpolations(item) for item in obj]
    return obj


_TRUE_BOOL_VALUES = {'1', 'true', 'yes', 'on'}
_FALSE_BOOL_VALUES = {'0', 'false', 'no', 'off', ''}


def coerce_bool(value: Any, *, name: str) -> bool:
    """Coerce common config bool forms (bool/None/str) while rejecting ambiguous values.

    Accepts native ``bool``; ``None`` -> ``False``; and case-insensitive strings
    ``1/true/yes/on`` -> ``True``, ``0/false/no/off/""`` -> ``False``. Anything
    else raises ``ValueError`` referencing *name*.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_BOOL_VALUES:
            return True
        if normalized in _FALSE_BOOL_VALUES:
            return False
    raise ValueError(f'{name} must be a boolean, got {value!r}')
