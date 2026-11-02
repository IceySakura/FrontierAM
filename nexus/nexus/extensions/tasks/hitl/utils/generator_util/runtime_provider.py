"""The sandbox provider a run acquires its instances from.

Provider-specific: ``gongfeng`` is the one HITL runs on, and a different provider name
yields an empty config for the runner to fill from its own defaults.
"""

import os

from typing import Any

from nexus.extensions.tasks.hitl.utils.generator_util.env import _env_int


def _build_runtime_provider_config() -> dict[str, Any]:
    """The provider config for ``SWE_RUNTIME_PROVIDER_NAME``, or {} for an unknown one."""
    provider = os.environ.get('SWE_RUNTIME_PROVIDER_NAME', '').strip() or 'gongfeng'
    config: dict[str, Any] = {}

    if provider == 'gongfeng':
        config = {
            'api_base_url': os.environ.get('SWE_RUNTIME_BASE_URL', ''),
            'token': os.environ.get('SWE_RUNTIME_TOKEN', ''),
            'runtime_image': os.environ.get('SWE_RUNTIME_IMAGE', '').strip() or 'current',
            'request_timeout': float(os.environ.get('SWE_RUNTIME_TIMEOUT', '').strip() or '7200'),
            'resource_spec': {
                'cpu': _env_int('SWE_RUNTIME_CPU', default=2),
                'memory': _env_int('SWE_RUNTIME_MEMORY', default=10),
            },
            'rpc_protocol': os.environ.get('SWE_RPC_PROTOCOL', '').strip() or 'http',
            # Sandbox lifecycle label passed to gongfeng /runtime/start; the backend
            # defines what each one reclaims. One of prod, eval or test.
            'purpose': os.environ.get('SWE_RUNTIME_PURPOSE', '').strip() or 'test',
            # Client-side budget, in seconds, for waiting out a queue when quota is
            # full or the backend is shedding load. On expiry the instance fails and
            # the batch carries on. 0 means no deadline, under which an instance can
            # hang for hours and hold up the whole batch.
            'queue_wait_max': _env_int('GONGFENG_QUEUE_WAIT_MAX', default=1200),
            # Random sleep of 0 to this many seconds before each acquire(), spreading
            # concurrent starts. 0 turns it off.
            'startup_jitter_max': float(_env_int('GONGFENG_STARTUP_JITTER_MAX', default=12)),
        }
        # How long the sandbox itself lives, as a Go duration ('12h', '1h30m'); the
        # backend's own default applies when unset. Left out rather than passed empty,
        # since a non-empty ttl force-routes the backend to the suanli provider.
        ttl = os.environ.get('SWE_RUNTIME_TTL', '').strip()
        if ttl:
            config['ttl'] = ttl

    return config


__all__ = ['_build_runtime_provider_config']
