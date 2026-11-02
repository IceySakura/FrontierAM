"""OSWorld-owned sandbox release: one request, bounded response wait."""

import asyncio
import json
import logging
import threading
import time

from pathlib import Path

from nexus.runtime.providers.ags.provider import AGSRuntimeProvider, _exception_code
from nexus.utils.timeout import async_timeout


logger = logging.getLogger(__name__)
STOP_WAIT_SECONDS = 5.0


def _stop_once(provider, instance_id, sandbox):
    if sandbox is not None:
        # A missing sandbox is already absent and needs no further request.
        sandbox.kill(request_timeout=5)
        return {}
    from tencentcloud.ags.v20250920 import ags_client, models
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile

    # Dedicated short-timeout client: do not mutate a shared long-running SDK
    # client or invoke the provider's stop retry loop.
    profile = ClientProfile(httpProfile=HttpProfile(endpoint='ags.tencentcloudapi.com', reqTimeout=5))
    client = ags_client.AgsClient(
        credential.Credential(provider.secret_id, provider.secret_key), provider.region, profile
    )
    request = models.StopSandboxInstanceRequest()
    request.InstanceId = instance_id
    result = client.StopSandboxInstance(request)
    return {'request_id': getattr(result, 'RequestId', None)}


def install_release(provider, name, records, directory):
    """Install before acquisition so partial acquisition cleanup is bounded too."""
    if not isinstance(provider, AGSRuntimeProvider):
        return
    if name in provider._release_handlers:
        return

    async def release(owner, runtime_name):
        record = {
            'runtime_name': runtime_name,
            'instance_id': owner._instance_ids.get(runtime_name),
            'status': 'not_allocated',
        }
        records.append(record)

        def save():
            if directory is not None:
                try:
                    path = Path(directory) / 'release-status.json'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(records, indent=2))
                except OSError:
                    logger.warning('Cannot persist OSWorld release status')
            logger.info('OSWorld release: %s', record)

        runtime = owner._active_runtimes.pop(runtime_name, None)
        handle = getattr(runtime, 'handle', None)
        metadata = getattr(handle, 'metadata', {}) or {}
        instance = metadata.get('instance', {})
        if instance.get('ExpiresAt'):
            record['expires_at'] = instance['ExpiresAt']
        if instance.get('TimeoutSeconds') is not None:
            record['timeout_seconds'] = instance['TimeoutSeconds']
        cancelled = False
        if runtime is not None:
            try:
                async with async_timeout(5):
                    await owner._close_runtime(runtime)
            except asyncio.CancelledError:
                cancelled = True
                record['local_cleanup_error_type'] = 'CancelledError'
            except Exception as exc:
                record['local_cleanup_error_type'] = type(exc).__name__
        external = runtime_name in owner._external_instance_names
        owner._external_instance_names.discard(runtime_name)
        sandbox = owner._template_sandboxes.pop(runtime_name, None)
        instance_id = owner._instance_ids.pop(runtime_name, None)
        record['tool_id'] = owner._tool_ids.pop(runtime_name, None)
        # Tool deletion is not a prerequisite for delivering the task result.
        # Record dynamically created tools for external cleanup, never delete a
        # template while its instance may still be stopping.
        tool = owner._tools_to_delete.pop(runtime_name, None)
        if tool:
            record['tool_cleanup_pending'] = tool
        if external:
            record['status'] = 'detached'
            save()
            if cancelled:
                raise asyncio.CancelledError
            return
        if not instance_id:
            save()
            if cancelled:
                raise asyncio.CancelledError
            return
        record.update(instance_id=instance_id, status='stop_pending')
        done = threading.Event()
        result = {}

        def send():
            try:
                result.update(status='stop_accepted', **_stop_once(owner, instance_id, sandbox))
            except Exception as exc:
                result.update(status='stop_failed', error_type=type(exc).__name__, error_code=_exception_code(exc))
                request_id = getattr(exc, 'requestId', None)
                if isinstance(request_id, str):
                    result['request_id'] = request_id
            finally:
                done.set()
                # A late response is observable but never rewrites an already
                # returned snapshot or touches a closed asyncio event loop.
                logger.info('OSWorld stop response: instance_id=%s result=%s', instance_id, result)

        started = time.monotonic()
        # Daemon thread is deliberately outside asyncio's default executor:
        # shutdown_default_executor must not wait on a stalled SDK call.
        try:
            threading.Thread(target=send, name='osworld-stop', daemon=True).start()
            while not done.is_set() and time.monotonic() - started < STOP_WAIT_SECONDS:
                await asyncio.sleep(min(0.05, max(0, STOP_WAIT_SECONDS - (time.monotonic() - started))))
        except asyncio.CancelledError:
            cancelled = True
        except Exception as exc:
            record.update(status='stop_failed', error_type=type(exc).__name__)
        if done.is_set():
            record.update(result)
        record['elapsed_s'] = round(time.monotonic() - started, 3)
        save()
        if cancelled:
            raise asyncio.CancelledError

    provider._release_handlers[name] = release
