"""Runtime-image preflight helpers for local GUI integration entrypoints."""

from __future__ import annotations

from nexus.runtime.providers.docker import build_runtime_image, get_package_version


RUNTIME_IMAGE_PREBUILT_ENV = 'NEXUS_GUI_RUNTIME_IMAGE_PREBUILT'


def require_desktop_image() -> str:
    """Require the prebuilt desktop image; Nexus no longer bundles its builder."""
    import subprocess

    from tests.integration.gui_agent.config_common import DEFAULT_BASE_IMAGE

    probe = subprocess.run(
        ['docker', 'image', 'inspect', DEFAULT_BASE_IMAGE], capture_output=True, text=True, check=False,
    )
    if probe.returncode:
        raise RuntimeError(f'Prebuilt desktop image {DEFAULT_BASE_IMAGE!r} is required for local GUI integration')
    return DEFAULT_BASE_IMAGE


async def ensure_runtime_image() -> None:
    """Force-build the default local runtime image once for an entrypoint run.

    Shell entrypoints perform the same preflight before dispatching to YAML or
    Python mode and set ``NEXUS_GUI_RUNTIME_IMAGE_PREBUILT=1`` so nested Python
    execution does not rebuild the image a second time. Docker provider acquire
    still uses ``force_rebuild=False`` and only consumes the prebuilt image.
    """
    import os

    if os.getenv(RUNTIME_IMAGE_PREBUILT_ENV) == '1':
        return

    await build_runtime_image(f'nexus-runtime:{get_package_version()}', force_rebuild=True)
    os.environ[RUNTIME_IMAGE_PREBUILT_ENV] = '1'


__all__ = ['RUNTIME_IMAGE_PREBUILT_ENV', 'ensure_runtime_image']
