"""Image registry and resolution for code_mt runtime switching.

Contains the canonical list of known base images and
the mirror URL resolution logic. Extracted from swe_setup_agent/tools.py
to avoid coupling code_mt to the setup agent.
"""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)

KNOWN_IMAGES: set[str] = {
    # Ubuntu (base)
    'ubuntu-git:20.04',
    'ubuntu-git:22.04',
    # Python (slim)
    'python-git:2.7-slim',
    'python-git:3.5-slim',
    'python-git:3.6-slim',
    'python-git:3.7-slim',
    'python-git:3.8-slim',
    'python-git:3.9-slim',
    'python-git:3.10-slim',
    'python-git:3.11-slim',
    'python-git:3.12-slim',
    'python-git:3.13-slim',
    # Python (bullseye)
    'python-git:3.6-bullseye',
    'python-git:3.7-bullseye',
    'python-git:3.8-bullseye',
    'python-git:3.9-bullseye',
    'python-git:3.10-bullseye',
    'python-git:3.11-bullseye',
    'python-git:3.12-bullseye',
    'python-git:3.13-bullseye',
    # Ruby (slim)
    'ruby-git:2.3-slim',
    'ruby-git:2.4-slim',
    'ruby-git:2.5-slim',
    'ruby-git:2.6-slim',
    'ruby-git:2.7-slim',
    'ruby-git:3.0-slim',
    'ruby-git:3.1-slim',
    'ruby-git:3.2-slim',
    'ruby-git:3.3-slim',
    'ruby-git:3.4-slim',
    # Ruby (bullseye)
    'ruby-git:2.7-bullseye',
    'ruby-git:3.1-bullseye',
    'ruby-git:3.2-bullseye',
    'ruby-git:3.3-bullseye',
    # Node
    'node-git:6-stretch',
    'node-git:8-stretch',
    'node-git:10-buster',
    'node-git:12-bullseye',
    'node-git:14-bullseye',
    'node-git:16-bullseye',
    'node-git:18-bullseye',
    'node-git:20-bullseye',
    'node-git:22-bullseye',
    'node-git:20-slim',
    'node-git:22-bookworm',
    # Go
    'golang-local:1.11',
    'golang-local:1.12',
    'golang-local:1.13',
    'golang-local:1.14',
    'golang-local:1.15',
    'golang-local:1.16',
    'golang-local:1.17',
    'golang-local:1.18',
    'golang-local:1.19',
    'golang-local:1.20',
    'golang-local:1.21',
    'golang-local:1.22',
    'golang-local:1.23',
    'golang-local:1.24',
    'golang-local:1.25',
    'golang-local:1.26',
    # C# / .NET
    'dotnet-git:2.1',
    'dotnet-git:core-2.2',
    'dotnet-git:3.1',
    'dotnet-git:5.0',
    'dotnet-git:6.0',
    'dotnet-git:7.0',
    'dotnet-git:8.0',
    'dotnet-git:9.0',
    'dotnet-git:10.0',
    # C / C++ / Debian / ROS
    'debian-git:bullseye-slim',
    'debian-git:bookworm-slim',
    'ubuntu-cpp:20.04',
    'ubuntu-cpp:22.04',
    'ros-git:noetic',
    'ros-git:humble',
    'ros-git:foxy',
    # PHP
    'php-git:5.6-cli',
    'php-git:7.0-cli',
    'php-git:7.1-cli',
    'php-git:7.2-cli',
    'php-git:7.3-cli',
    'php-git:7.4-cli',
    'php-git:8.0-cli',
    'php-git:8.1-cli',
    'php-git:8.2-cli',
    'php-git:8.3-cli',
    'php-git:8.4-cli',
    # Java
    'java-git:8',
    'java-git:11',
    'java-git:17',
    'java-git:21',
    # Rust
    'rust-git:1.60-slim',
    'rust-git:1.65-slim',
    'rust-git:1.70-slim',
    'rust-git:1.73-slim',
    'rust-git:1.75-slim',
    'rust-git:1.77-slim',
    'rust-git:1.80-slim',
    'rust-git:1.83-slim',
    'rust-git:1.85-slim',
    'rust-git:latest',
}

DEFAULT_MIRROR_REGISTRY = 'mirrors.tencent.com/hunyuan_yanguan/t3build_repolaunch_midtrain'
_MARKER = '.swe_builder_eval_base_'


def validate_image(image: str) -> bool:
    """Check if a short image name exists in the known registry."""
    return image in KNOWN_IMAGES


def resolve_mirror_image(new_image: str, current_base_image: str) -> str:
    """Translate a short image name to its mirror registry URL.

    If new_image is already a full URL (contains hostname with dots), returns as-is.
    Otherwise derives the registry prefix from current_base_image or uses the default.
    """
    parts = new_image.split('/')
    if len(parts) >= 2 and '.' in parts[0]:
        return new_image

    if _MARKER in current_base_image:
        registry_prefix = current_base_image[: current_base_image.index(_MARKER)]
    else:
        registry_prefix = DEFAULT_MIRROR_REGISTRY

    safe = new_image.lower()
    for ch in ('-', ':', '/', '.'):
        safe = safe.replace(ch, '_')
    resolved = f'{registry_prefix}{_MARKER}{safe}:latest'
    logger.debug('Resolved image: %r → %r', new_image, resolved)
    return resolved


def image_alias(full_image: str) -> str:
    """Extract short alias from a full mirror image URL.

    E.g. 'mirrors.tencent.com/.../swe_builder_eval_base_golang_local_1_21:latest'
         → 'golang-local:1.21'

    Returns the original string if it doesn't match the mirror pattern.
    """
    if _MARKER not in full_image:
        return full_image

    suffix = full_image[full_image.index(_MARKER) + len(_MARKER) :]
    suffix = suffix.removesuffix(':latest')

    for known in KNOWN_IMAGES:
        safe = known.lower()
        for ch in ('-', ':', '/', '.'):
            safe = safe.replace(ch, '_')
        if safe == suffix:
            return known

    return full_image
