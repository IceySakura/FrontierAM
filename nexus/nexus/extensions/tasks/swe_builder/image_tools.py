"""Image management tools for SWE Builder.

These tools handle Docker image commit and cleanup operations.
Note: For image upload, use upload_handler.py directly (production-grade implementation).
"""

from __future__ import annotations

import logging

import docker

from pydantic import BaseModel, Field

from nexus.core import require_context, tool_registry
from nexus.core.tool import ToolContext, ToolResponse, ToolType


logger = logging.getLogger(__name__)


# ============== Tool Input/Output Models ==============


class CommitRuntimeImageInput(BaseModel):
    """Input for commit_runtime_image tool."""

    runtime_name: str = Field(default='main', description='Runtime name')
    image_name: str = Field(..., description='New image name (e.g., swe-env:instance-123)')
    author: str = Field(default='nexus-agent', description='Image author')
    message: str = Field(default='', description='Commit message')


class CommitRuntimeImageOutput(BaseModel):
    """Output for commit_runtime_image tool."""

    success: bool
    image_name: str
    image_id: str
    size_mb: float
    container_name: str
    message: str


class CleanupImagesInput(BaseModel):
    """Input for cleanup_images tool."""

    image_names: list[str] = Field(..., description='List of image names to remove')
    force: bool = Field(default=False, description='Force remove even if in use')


class CleanupImagesOutput(BaseModel):
    """Output for cleanup_images tool."""

    success: bool
    results: list[dict]
    removed_count: int
    total_count: int


# ============== Tool Implementations ==============


def _commit_image_sync(container_name: str, params: CommitRuntimeImageInput) -> dict:
    """Synchronous Docker commit operation (runs in thread pool)."""
    client = None
    try:
        # Use docker-py SDK
        client = docker.from_env(timeout=600)

        # Get container object
        container = client.containers.get(container_name)

        # Commit container to image
        repository = params.image_name.split(':')[0] if ':' in params.image_name else params.image_name
        tag = params.image_name.split(':')[1] if ':' in params.image_name else 'latest'

        image = container.commit(
            repository=repository,
            tag=tag,
            author=params.author,
            message=params.message,
        )

        # Get image info
        image_size_mb = image.attrs['Size'] / (1024 * 1024)

        logger.debug('=' * 80)
        logger.info('[COMMIT_IMAGE] ✅ Image committed successfully')
        logger.info(f'[COMMIT_IMAGE]   Image ID: {image.short_id}')
        logger.info(f'[COMMIT_IMAGE]   Image size: {image_size_mb:.1f} MB')
        logger.debug('=' * 80)

        return CommitRuntimeImageOutput(
            success=True,
            image_name=params.image_name,
            image_id=image.short_id,
            size_mb=round(image_size_mb, 1),
            container_name=container_name,
            message=f'Container {container_name} committed to image {params.image_name}',
        ).model_dump()

    except docker.errors.NotFound as exc:
        raise RuntimeError(f"Container '{container_name}' not found") from exc
    except Exception as e:
        logger.error(f'[COMMIT_IMAGE] ❌ Failed to commit image: {e}')
        raise
    finally:
        if client:
            client.close()


@tool_registry.register(
    'swe_builder/commit_runtime_image',
    meta={
        'name': 'commit_runtime_image',
        'description': (
            'Commit runtime container to Docker image. '
            'Use cases: Save configured development environment, create reusable base image, backup current work state. '
            'Note: Only saves filesystem state, not running processes. '
            'Image size depends on total files in container. '
            'Recommend cleaning up temporary files before commit.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def commit_runtime_image(params: CommitRuntimeImageInput, context: ToolContext) -> ToolResponse:
    """Commit runtime container to Docker image."""
    import asyncio

    ctx = require_context()
    provider = ctx.runtime_provider

    # Check provider type
    if not hasattr(provider, '_user_containers'):
        raise RuntimeError('Current runtime provider does not support Docker operations')

    # Get container name
    from typing import cast

    user_containers = cast(dict[str, str], provider._user_containers)
    if params.runtime_name not in user_containers:
        raise KeyError(f"Runtime '{params.runtime_name}' not found")

    container_name = user_containers[params.runtime_name]

    logger.debug('=' * 80)
    logger.debug('[COMMIT_IMAGE] Starting image commit')
    logger.debug(f'[COMMIT_IMAGE]   Container: {container_name}')
    logger.debug(f'[COMMIT_IMAGE]   Image name: {params.image_name}')
    logger.debug('=' * 80)

    # Run blocking Docker operations in thread pool
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _commit_image_sync, container_name, params)

    # Wrap result in ToolResponse
    return ToolResponse(output=[{'type': 'text', 'text': result['message']}], meta=result)


def _cleanup_images_sync(params: CleanupImagesInput) -> dict:
    """Synchronous Docker cleanup operation (runs in thread pool)."""
    client = None
    try:
        client = docker.from_env(timeout=300)
        results = []

        for image_name in params.image_names:
            try:
                client.images.remove(image_name, force=params.force)
                logger.debug(f'[CLEANUP_IMAGES] ✅ Removed: {image_name}')
                results.append({'image': image_name, 'removed': True})
            except docker.errors.ImageNotFound:
                logger.warning(f'[CLEANUP_IMAGES] ⚠️  Not found: {image_name}')
                results.append({'image': image_name, 'removed': False, 'reason': 'not_found'})
            except Exception as e:
                logger.error(f'[CLEANUP_IMAGES] ❌ Failed to remove {image_name}: {e}')
                results.append({'image': image_name, 'removed': False, 'reason': str(e)})

        removed_count = sum(1 for r in results if r['removed'])
        logger.debug('=' * 80)
        logger.info(f'[CLEANUP_IMAGES] Completed: {removed_count}/{len(params.image_names)} removed')
        logger.debug('=' * 80)

        return CleanupImagesOutput(
            success=True, results=results, removed_count=removed_count, total_count=len(params.image_names)
        ).model_dump()

    finally:
        if client:
            client.close()


@tool_registry.register(
    'swe_builder/cleanup_images',
    meta={
        'name': 'cleanup_images',
        'description': (
            'Remove local Docker images to free disk space. '
            'Use cases: Cleanup after upload, periodic cleanup of unused images, free disk space. '
            'Note: If image is in use by container, needs force=True to remove. '
            'Cannot be recovered after deletion. '
            'Recommend verifying upload success before cleanup.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def cleanup_images(params: CleanupImagesInput, context: ToolContext) -> ToolResponse:
    """Remove local Docker images to free disk space."""
    import asyncio

    logger.debug('=' * 80)
    logger.debug('[CLEANUP_IMAGES] Starting cleanup')
    logger.debug(f'[CLEANUP_IMAGES]   Images: {params.image_names}')
    logger.debug(f'[CLEANUP_IMAGES]   Force: {params.force}')
    logger.debug('=' * 80)

    # Run blocking Docker operations in thread pool
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _cleanup_images_sync, params)

    # Wrap result in ToolResponse
    message = f'Cleanup completed: {result["removed_count"]}/{result["total_count"]} images removed'
    return ToolResponse(output=[{'type': 'text', 'text': message}], meta=result)
