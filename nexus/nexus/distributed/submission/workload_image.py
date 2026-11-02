from __future__ import annotations

import re


_IMAGE_VERSION_PATTERN = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}')
_DIGEST_PINNED_IMAGE_PATTERN = re.compile(r'(?P<repository>.+)@sha256:[0-9a-f]{64}')


def validate_workload_image_version(value: str) -> str:
    if value.lower() == 'latest' or _IMAGE_VERSION_PATTERN.fullmatch(value) is None:
        raise ValueError('image_version must be an immutable Docker tag, not latest or a full image reference')
    return value


def workload_image_repository(base_image: str, *, require_digest: bool = False) -> str:
    if (
        not isinstance(base_image, str)
        or not base_image
        or base_image != base_image.strip()
        or '://' in base_image
        or any(character.isspace() for character in base_image)
    ):
        raise ValueError('workload base image is invalid')

    digest_match = _DIGEST_PINNED_IMAGE_PATTERN.fullmatch(base_image)
    if require_digest and digest_match is None:
        raise ValueError('workload base image must be digest-pinned')
    if '@' in base_image and digest_match is None:
        raise ValueError('workload base image is invalid')

    repository = digest_match.group('repository') if digest_match is not None else base_image
    final_colon = repository.rfind(':')
    final_slash = repository.rfind('/')
    if final_colon > final_slash:
        repository = repository[:final_colon]
    if not repository or repository.endswith('/') or '@' in repository:
        raise ValueError('workload base image is invalid')
    return repository


def resolve_workload_image(base_image: str, image_version: str, *, require_digest: bool = False) -> str:
    repository = workload_image_repository(base_image, require_digest=require_digest)
    return f'{repository}:{validate_workload_image_version(image_version)}'


__all__ = [
    'resolve_workload_image',
    'validate_workload_image_version',
    'workload_image_repository',
]
