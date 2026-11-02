"""
Docker image upload handler for SWE Builder.

This module handles uploading Docker images to gongfeng registry.

Credentials must be provided via environment variables:
  REGISTRY_USERNAME, REGISTRY_PASSWORD
"""

import asyncio
import logging
import os
import time

from datetime import datetime
from typing import Any

import docker
import docker.errors


logger = logging.getLogger(__name__)

# Registry configuration (完全复用 swe-builder-jiacx 的配置)
REGISTRY_PREFIX = 'mirrors.tencent.com/hunyuan_yanguan/t3build_repolaunch_midtrain'
REGISTRY_HOST = 'mirrors.tencent.com'
REGISTRY_USERNAME_DEFAULT = ''
REGISTRY_PASSWORD_DEFAULT = ''

# Metadata configuration (完全复用 swe-builder-jiacx 的配置)
DATASET_TYPE = 'T3Build_SWEBuilder_RL'
DATASET_NAME = 't3_web_scrape'  # 硬编码，不能改
SPLIT_DEFAULT = 't3'
WORKDIR = '/testbed'


def _docker_login(client: docker.DockerClient) -> None:
    """Login to Docker registry.

    Args:
        client: Docker client instance

    Raises:
        Exception: If login fails
    """
    registry_username = os.getenv('REGISTRY_USERNAME', REGISTRY_USERNAME_DEFAULT)
    registry_password = os.getenv('REGISTRY_PASSWORD', REGISTRY_PASSWORD_DEFAULT)

    logger.debug(f'[UPLOAD] Logging in to registry: {REGISTRY_HOST}')
    logger.debug(f'[UPLOAD] Using username: {registry_username}')

    client.login(username=registry_username, password=registry_password, registry=REGISTRY_HOST)

    logger.debug('[UPLOAD] Login successful')


def _get_local_image(client: docker.DockerClient, image_name: str) -> docker.models.images.Image:
    """Get local Docker image, handling case sensitivity.

    Args:
        client: Docker client instance
        image_name: Local image name

    Returns:
        Docker image object

    Raises:
        docker.errors.ImageNotFound: If image not found
    """
    try:
        return client.images.get(image_name)
    except docker.errors.APIError as e:
        # Handle case sensitivity issues
        if 'must be lowercase' in str(e) or 'invalid reference format' in str(e):
            image_name_lower = image_name.lower()
            logger.debug(f'[UPLOAD] Trying lowercase version: {image_name_lower}')
            return client.images.get(image_name_lower)
        raise


def _push_image_with_retry(client: docker.DockerClient, target_tag: str, max_retries: int = 3) -> None:
    """Push Docker image with exponential backoff retry.

    Args:
        client: Docker client instance
        target_tag: Target image tag to push
        max_retries: Maximum number of retry attempts (default: 3, reduced from 5)

    Raises:
        Exception: If push fails after all retries
    """
    import sys

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f'[UPLOAD] Pushing image (attempt {attempt}/{max_retries})...')
            logger.debug(f'[UPLOAD] Target: {target_tag}')

            # Track progress to show we're not stuck
            last_log_time = time.time()
            layer_count = 0

            # Stream push output and check for errors
            for line in client.images.push(target_tag, stream=True, decode=True):
                if 'error' in line:
                    raise Exception(line['error'])

                # Log progress periodically (every 30 seconds or every 5 layers)
                if 'status' in line and 'id' in line:
                    layer_count += 1
                    current_time = time.time()

                    # Log every 5 layers OR every 30 seconds
                    if layer_count % 5 == 0 or (current_time - last_log_time) > 30:
                        logger.debug(f'[UPLOAD] Progress: {line["id"]}: {line["status"]}')
                        last_log_time = current_time
                        # Flush to ensure log appears immediately
                        sys.stdout.flush()

            logger.info(f'[UPLOAD] Push successful: {target_tag} ({layer_count} layers)')
            break  # Success, exit retry loop

        except Exception as push_err:
            if attempt < max_retries:
                # Reduced backoff: 10s, 20s, 40s (instead of 30s, 60s, 120s, 240s, 480s)
                wait_time = 10 * (2 ** (attempt - 1))
                logger.warning(f'[UPLOAD] Push failed (attempt {attempt}/{max_retries}): {push_err}')
                logger.debug(f'[UPLOAD] Retrying in {wait_time}s...')

                # Use asyncio.sleep if possible, otherwise fallback to time.sleep
                # Note: We're in a sync function, so we can't use async sleep here
                # This will still block, but with reduced wait time it's less problematic
                time.sleep(wait_time)
            else:
                logger.error(f'[UPLOAD] Push failed after {max_retries} attempts')
                raise


def _cleanup_local_image(client: docker.DockerClient, image_names: list[str]) -> None:
    """Clean up local Docker images.

    Args:
        client: Docker client instance
        image_names: List of image names to remove
    """
    for image_name in image_names:
        try:
            client.images.remove(image_name, force=True)
            logger.debug(f'[UPLOAD] Removed local image: {image_name}')
        except Exception as e:
            logger.warning(f'[UPLOAD] Failed to remove local image {image_name}: {e}')


def generate_nexus_metadata(
    instance_with_results: dict[str, Any],
    remote_image: str,
    upload_time: str,
) -> dict[str, Any]:
    """
    Generate nexus-compatible metadata from task output.
    完全复用 swe-builder-jiacx/app/upload_utils.py 的元数据格式。

    Args:
        instance_with_results: Instance data with test results
        remote_image: Full remote image name
        upload_time: Upload timestamp

    Returns:
        Nexus-compatible metadata dictionary
    """
    # 复用 swe-builder-jiacx 的元数据格式（不做任何修改）
    eval_script = instance_with_results.get('eval_script_content', '')

    # Build spec_dict from test_spec if available
    test_spec = instance_with_results.get('test_spec', '')
    spec_dict = instance_with_results.get('spec_dict') or {}
    if test_spec and not spec_dict:
        spec_dict = {'test_cmd': [test_spec]}

    metadata = {
        'image': remote_image,
        'status': 'success',
        'dataset_type': DATASET_TYPE,  # T3Build_SWEBuilder_RL
        'language': instance_with_results.get('language', 'python'),
        'dataset_name': DATASET_NAME,  # t3_web_scrape
        'split': instance_with_results.get('split', SPLIT_DEFAULT),
        'image_upload_time': upload_time,
        'workdir': WORKDIR,
        'PASS_TO_PASS': instance_with_results.get('test_results', {}).get('pass2pass', []),
        'FAIL_TO_PASS': instance_with_results.get('test_results', {}).get('fail2pass', []),
        'eval_script': eval_script,  # Used by T3Build_SWEBuilder_RL / T3Build_SWEBuilder_Midtrain
        'spec_dict': spec_dict,  # test_cmd for running the full test suite
    }

    # Include original instance fields
    for key in ['instance_id', 'repo', 'base_commit', 'problem_statement', 'patch', 'test_patch']:
        if key in instance_with_results:
            metadata[key] = instance_with_results[key]

    return metadata


async def upload_swe_builder_image(
    instance_id: str,
    image_name: str,
    task_output: Any,  # SWEBuilderTaskOutput
    upload_strategy: str = 'all_success',
    cleanup_local: bool = True,
) -> dict[str, Any]:
    """
    Upload SWE Builder generated image to gongfeng registry.
    完全复用 swe-builder-jiacx/app/upload_utils.py 的上传逻辑。

    Args:
        instance_id: Instance ID
        image_name: Local image name (e.g., setup.instance_id:latest)
        task_output: SWE Builder task output containing test results
        upload_strategy: Upload strategy
            - "all_success": Upload all successfully built images (default)
            - "fail2pass_only": Only upload images that passed fail2pass tests
        cleanup_local: Whether to clean up local image after upload (default: True)

    Returns:
        Upload result dictionary containing:
        - uploaded: bool (whether upload succeeded)
        - remote_image: str (full remote image name)
        - metadata: dict (nexus-compatible metadata)
        - reason: str (reason for skip/failure, if applicable)
    """
    import sys

    logger.debug('=' * 80)
    logger.debug(f'[UPLOAD] Starting upload for instance: {instance_id}')
    logger.debug(f'[UPLOAD] Local image: {image_name}')
    logger.debug(f'[UPLOAD] Registry prefix: {REGISTRY_PREFIX}')
    logger.debug(f'[UPLOAD] Upload strategy: {upload_strategy}')
    logger.debug(f'[UPLOAD] Cleanup local: {cleanup_local}')
    logger.debug('=' * 80)
    sys.stdout.flush()  # Force flush to ensure logs appear immediately

    # Check upload strategy
    if upload_strategy == 'fail2pass_only':
        if not task_output.resolved:
            logger.info('[UPLOAD] Skipping upload - instance did not pass fail2pass tests')
            logger.info(f'[UPLOAD] Resolved: {task_output.resolved}, Score: {task_output.score}')
            return {'uploaded': False, 'reason': 'fail2pass_not_resolved', 'remote_image': '', 'metadata': {}}
        logger.info('[UPLOAD] Instance passed fail2pass tests, proceeding with upload')

    # Run blocking Docker operations in thread pool
    loop = asyncio.get_running_loop()

    try:
        # Execute upload in thread pool to avoid blocking
        return await loop.run_in_executor(None, _upload_image_sync, instance_id, image_name, task_output, cleanup_local)

    except Exception as e:
        logger.error(f'[UPLOAD] ❌ Upload failed: {e}', exc_info=True)
        return {'uploaded': False, 'reason': str(e), 'remote_image': '', 'metadata': {}}


def _upload_image_sync(instance_id: str, image_name: str, task_output: Any, cleanup_local: bool) -> dict[str, Any]:
    """
    Synchronous image upload implementation.
    This runs in a thread pool executor.
    """
    client = None
    target_tag = None

    try:
        # Create Docker client
        client = docker.from_env(timeout=600)

        # Login to registry
        _docker_login(client)

        # Get local image (handle case sensitivity)
        logger.debug(f'[UPLOAD] Getting local image: {image_name}')
        image = _get_local_image(client, image_name)

        # Tag image with registry prefix (完全复用格式)
        # Always derive target_tag from the caller-supplied image_name (lowercased),
        # not from image.tags — the image object may carry extra tags (e.g., the
        # base image it was built from) that would produce a wrong remote path.
        target_tag = f'{REGISTRY_PREFIX}.{image_name.lower()}'
        logger.debug(f'[UPLOAD] Tagging image as: {target_tag}')
        image.tag(target_tag)

        # Push image with retry
        _push_image_with_retry(client, target_tag)

        # Generate metadata (完全复用格式)
        upload_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        metadata = generate_nexus_metadata(task_output.instance_with_results, target_tag, upload_time)

        logger.debug(f'[UPLOAD] Generated metadata with {len(metadata)} fields')
        logger.debug('=' * 80)
        logger.info('[UPLOAD] ✅ Upload completed successfully')
        logger.debug(f'[UPLOAD] Remote image: {target_tag}')
        logger.debug('=' * 80)

        return {'uploaded': True, 'remote_image': target_tag, 'metadata': metadata}

    except docker.errors.ImageNotFound:
        logger.error(f'[UPLOAD] ❌ Local image not found: {image_name}')
        return {'uploaded': False, 'reason': 'image_not_found', 'remote_image': '', 'metadata': {}}

    except Exception as e:
        logger.error(f'[UPLOAD] ❌ Upload failed: {e}', exc_info=True)
        return {'uploaded': False, 'reason': str(e), 'remote_image': '', 'metadata': {}}

    finally:
        # Always clean up local images regardless of success/failure.
        # Without this, a failed push leaves the mirrors.tencent.com/... tag
        # and the original swe_builder_eval_*:latest image on disk.
        if client and cleanup_local:
            images_to_cleanup = []
            if target_tag:
                images_to_cleanup.append(target_tag)
            images_to_cleanup.append(image_name)
            _cleanup_local_image(client, images_to_cleanup)
            logger.debug(f'[UPLOAD] Cleaned up local images: {", ".join(images_to_cleanup)}')
        if client:
            client.close()


async def save_upload_metadata(
    metadata_list: list[dict[str, Any]], output_dir: str, exp_name: str = 'swe_builder'
) -> str:
    """
    Save upload metadata to JSONL file.

    Args:
        metadata_list: List of metadata dictionaries
        output_dir: Output directory for metadata
        exp_name: Experiment name for filename

    Returns:
        Path to saved metadata file
    """
    import json
    import uuid

    if not metadata_list:
        logger.warning('[UPLOAD] No metadata to save')
        return ''

    # Generate unique filename with timestamp
    timestamp = int(time.time() * 1_000_000)
    unique_uuid = uuid.uuid4()
    custom_uuid = uuid.UUID(int=(timestamp << 64) | unique_uuid.int)

    # Create nexus_metadata directory
    nexus_dir = os.path.join(output_dir, 'nexus_metadata')
    os.makedirs(nexus_dir, exist_ok=True)

    # Save metadata
    metadata_path = os.path.join(nexus_dir, f'{exp_name}_{str(custom_uuid)}.jsonl')

    with open(metadata_path, 'w', encoding='utf-8') as f:
        for metadata in metadata_list:
            f.write(json.dumps(metadata, ensure_ascii=False) + '\n')

    logger.info(f'[UPLOAD] Saved {len(metadata_list)} metadata entries to: {metadata_path}')

    return metadata_path
