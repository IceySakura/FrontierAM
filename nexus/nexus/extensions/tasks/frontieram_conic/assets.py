"""Sealed conic task package, separate from the exponential numerical contract."""

from __future__ import annotations

import json

from pathlib import Path

from nexus.extensions.tasks.frontieram_offline.assets import AssetDriftError, LockedAssets, file_sha256


IMAGE_ID = 'sha256:a0d0870fae06d94f210cee1493b4a0c6cb5825c644ca9f81f28830be33008724'


class ConicAssets:
    def __init__(self, source_root, profile_path, profile_sha256):
        self.common = LockedAssets(source_root)
        self.source_root = self.common.source_root
        self.profile_path = Path(profile_path).resolve(strict=True)
        self.profile_sha256 = profile_sha256
        if file_sha256(self.profile_path) != profile_sha256:
            raise AssetDriftError('Conic profile hash mismatch')
        self.profile = json.loads(self.profile_path.read_text())
        p = self.profile
        if (p['task_version'], p['scoring_version'], p['scoring_mode'], p['image_id']) != (
            '0.3.0',
            'success-v2',
            'pilot',
            IMAGE_ID,
        ):
            raise AssetDriftError('Unsupported conic pilot contract')
        ids = p['case_ids']
        if len(ids) != 20 or len(set(ids)) != 20 or p['repeats'] != 3:
            raise AssetDriftError('Expected twenty unique cases and three repeats')
        self.task_root = Path(p['task_root']).resolve(strict=True)
        self.public_tar = Path(p['public_tar']).resolve(strict=True)
        self.image_id = IMAGE_ID
        self.verify()

    def verify(self):
        self.common.verify()
        if file_sha256(self.profile_path) != self.profile_sha256:
            raise AssetDriftError('Conic profile changed')
        if file_sha256(self.public_tar) != self.profile['public_tar_sha256']:
            raise AssetDriftError('Conic public package changed')
        return {'profile_sha256': self.profile_sha256, 'public_tar_sha256': self.profile['public_tar_sha256']}

    def verify_private_data(self):
        self.verify()
        for relative, expected in self.profile['locked_files'].items():
            path = self.task_root / relative
            if not path.resolve().is_relative_to(self.task_root) or path.is_symlink() or file_sha256(path) != expected:
                raise AssetDriftError(f'Conic task file changed: {relative}')
        return self.profile['locked_files']

    @property
    def controller(self):
        return self.common.controller

    @property
    def docker_mcp(self):
        return self.common.docker_mcp

    @property
    def codex_path(self):
        return self.common.codex_path

    def provenance(self):
        return {
            **self.verify(),
            'task_version': '0.3.0',
            'scoring_version': 'success-v2',
            'archive_sha256': self.profile['archive_sha256'],
            'image_id': self.image_id,
            'subset_cases': 20,
            'repeats': 3,
            'formal_score': False,
            'score_status': 'pilot',
            'public_only_during_development': True,
        }
