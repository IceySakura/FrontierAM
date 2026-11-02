"""Runtime-related CLI commands."""

from __future__ import annotations

import argparse
import asyncio
import logging
import shlex

from pathlib import Path

from nexus.utils.log import setup_logging


logger = logging.getLogger(__name__)


def serve_runtime(args: argparse.Namespace) -> int:
    """Run the Nexus runtime server on the current machine."""
    try:
        from nexus.runtime.server.app import asgi_app, configure

        configure(host=args.host, port=args.port)
        import uvicorn

        uvicorn.run(
            asgi_app,
            host=args.host,
            port=args.port,
            ws_max_size=100 * 1024 * 1024,
        )
        return 0
    except ModuleNotFoundError as exc:
        logger.error(
            'Runtime server dependency is missing (%s). Run: uv sync --extra runtime && '
            'uv run nexus modules install --all --runtime',
            exc.name,
        )
        return 1
    except Exception:
        logger.exception('Runtime server failed to start')
        return 1


def register_runtime_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the cross-platform ``nexus runtime serve`` command."""
    runtime_parser = subparsers.add_parser('runtime', help='Manage the local Nexus runtime server')
    runtime_subparsers = runtime_parser.add_subparsers(dest='runtime_command')
    serve_parser = runtime_subparsers.add_parser('serve', help='Serve the runtime API on this machine')
    serve_parser.add_argument('--host', default='127.0.0.1', help='Bind host (default: 127.0.0.1)')
    serve_parser.add_argument('--port', type=int, default=8000, help='Bind port (default: 8000)')
    serve_parser.set_defaults(_parser=serve_parser)


async def _get_nexus_runtime_images() -> list[dict[str, str]]:
    """Get all local docker images with name 'nexus-runtime'.

    Returns:
        List of dicts with 'repository', 'tag', and 'id' keys.
    """
    from nexus.utils.common import run_shell_cmd

    try:
        result = await run_shell_cmd(
            "docker images --format '{{.Repository}}|{{.Tag}}|{{.ID}}' --filter 'reference=nexus-runtime'"
        )
        images = []
        for line in result.strip().split('\n'):
            if not line:
                continue
            parts = line.split('|')
            if len(parts) == 3:
                images.append({'repository': parts[0], 'tag': parts[1], 'id': parts[2]})
        return images
    except RuntimeError:
        return []


async def _get_containers_by_image(image_id: str) -> list[str]:
    """Get all container IDs created from a specific image.

    Args:
        image_id: The image ID to filter by.

    Returns:
        List of container IDs.
    """
    from nexus.utils.common import run_shell_cmd

    try:
        result = await run_shell_cmd(f"docker ps -a --filter 'ancestor={shlex.quote(image_id)}' --format '{{{{.ID}}}}'")
        return [cid for cid in result.strip().split('\n') if cid]
    except RuntimeError:
        return []


async def _get_dangling_images() -> list[str]:
    """Get all dangling images (images without repo and tag).

    Returns:
        List of image IDs.
    """
    from nexus.utils.common import run_shell_cmd

    try:
        result = await run_shell_cmd("docker images --filter 'dangling=true' --format '{{.ID}}'")
        return [iid for iid in result.strip().split('\n') if iid]
    except RuntimeError:
        return []


async def _remove_containers(container_ids: list[str]) -> int:
    """Remove containers by their IDs.

    Args:
        container_ids: List of container IDs to remove.

    Returns:
        Number of containers removed.
    """
    from nexus.utils.common import run_shell_cmd

    removed = 0
    for cid in container_ids:
        try:
            await run_shell_cmd(f'docker rm -f {shlex.quote(cid)}')
            logger.info('Removed container: %s', cid)
            removed += 1
        except RuntimeError as e:
            logger.warning('Failed to remove container %s: %s', cid, e)
    return removed


async def _remove_images(image_ids: list[str]) -> int:
    """Remove images by their IDs.

    Args:
        image_ids: List of image IDs to remove.

    Returns:
        Number of images removed.
    """
    from nexus.utils.common import run_shell_cmd

    removed = 0
    for iid in image_ids:
        try:
            await run_shell_cmd(f'docker rmi -f {shlex.quote(iid)}')
            logger.info('Removed image: %s', iid)
            removed += 1
        except RuntimeError as e:
            logger.warning('Failed to remove image %s: %s', iid, e)
    return removed


async def _clean_runtime_images(dangling: bool = False) -> tuple[int, int]:
    """Clean old nexus-runtime images and their containers.

    Args:
        dangling: If True, also remove dangling images (no repo/tag).

    Returns:
        Tuple of (containers_removed, images_removed).
    """
    from nexus.runtime.providers.docker import get_package_version

    current_version = get_package_version()
    logger.info('Current nexus version: %s', current_version)

    # Get all nexus-runtime images
    images = await _get_nexus_runtime_images()
    logger.info('Found %d nexus-runtime images', len(images))

    # Find images to remove (not matching current version)
    images_to_remove = []
    for img in images:
        if img['tag'] != current_version:
            images_to_remove.append(img)
            logger.info('Will remove: nexus-runtime:%s (ID: %s)', img['tag'], img['id'])
        else:
            logger.info('Keeping current version: nexus-runtime:%s (ID: %s)', img['tag'], img['id'])

    # Remove containers first
    total_containers_removed = 0
    for img in images_to_remove:
        containers = await _get_containers_by_image(img['id'])
        if containers:
            logger.info('Removing %d containers for image %s...', len(containers), img['id'])
            total_containers_removed += await _remove_containers(containers)

    # Remove old images
    total_images_removed = await _remove_images([img['id'] for img in images_to_remove])

    # Remove dangling images if requested
    if dangling:
        dangling_ids = await _get_dangling_images()
        if dangling_ids:
            logger.info('Found %d dangling images to remove', len(dangling_ids))
            # Remove containers for dangling images
            for iid in dangling_ids:
                containers = await _get_containers_by_image(iid)
                if containers:
                    total_containers_removed += await _remove_containers(containers)
            total_images_removed += await _remove_images(dangling_ids)

    return total_containers_removed, total_images_removed


def build_runtime(args: argparse.Namespace) -> int:
    """Build the nexus runtime Docker image."""
    from nexus.runtime.providers.docker import build_runtime_image, get_package_version

    setup_logging(log_level=logging.DEBUG)

    image_name = args.image or f'nexus-runtime:{get_package_version()}'
    build_context = Path(args.context) if args.context else None

    try:
        asyncio.run(
            build_runtime_image(
                image_name=image_name,
                force_rebuild=args.force,
                build_context=build_context,
                push=args.push,
            )
        )
        return 0
    except Exception as e:
        logger.error('Error building runtime image: %s', e)
        return 1


def clean_runtime(args: argparse.Namespace) -> int:
    """Clean old nexus-runtime images and their containers."""
    setup_logging(log_level=logging.DEBUG)

    try:
        containers_removed, images_removed = asyncio.run(_clean_runtime_images(dangling=args.dangling))
        logger.info('Cleaned up %s containers and %s images', containers_removed, images_removed)
        return 0
    except Exception as e:
        logger.error('Error cleaning runtime images: %s', e)
        return 1


def register_build_runtime_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the build-runtime subcommand."""
    build_runtime_parser = subparsers.add_parser(
        'build-runtime',
        help='Build the nexus runtime Docker image',
    )
    build_runtime_parser.add_argument(
        '--image',
        '-i',
        help='Name/tag for the built image (default: nexus-runtime:<version>)',
    )
    build_runtime_parser.add_argument(
        '--force',
        '-f',
        action='store_true',
        help='Force rebuild even if image already exists',
    )
    build_runtime_parser.add_argument(
        '--context',
        '-c',
        help='Custom build context directory (default: auto-detect)',
    )
    build_runtime_parser.add_argument(
        '--push',
        action='store_true',
        help='Push image to mirrors.tencent.com after build',
    )


def register_clean_runtime_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the clean-runtime subcommand."""
    clean_runtime_parser = subparsers.add_parser(
        'clean-runtime',
        help='Clean old nexus-runtime images and their containers',
    )
    clean_runtime_parser.add_argument(
        '--dangling',
        '-d',
        action='store_true',
        help='Also remove dangling images (images without repo/tag)',
    )
