"""Hash-locked local numerical assets; no model or campaign execution on import."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
import uuid

from pathlib import Path
from types import ModuleType
from typing import Any


IMAGE_ID = 'sha256:f203391e0381bdca377548961998bbbcac9bfd2212cfd00ffb77c4b31ed230f2'
CODEX_SHA256 = '58a91f6ea19bea477ae3a2ec48d619f0039afe75d3476260747e11d201fa7cea'
ASSET_SHA256 = {
    'campaign_runner_v2.py': '9021159821e24c4449b74403ffa9e11d58040e56251eb338cb0f292a2c4ba5d8',
    'harness_resilient/docker_mcp.py': 'd75a609874647c183b66c1e1222b3453afd2356257cd3553b50eff6ef1641106',
    'public_pilot_v2/public_pilot.tar': '931d071f291720b5dc0f8fc1eb400becc53e7d039e2dd8ecdfc9e7a2043537bb',
    'public_pilot_v2/task/PUBLIC_MANIFEST.json': '0013e554392033e4bc879df809f8dfaab1f422240b0304550ed31eac7442fb4a',
    'image_lock.json': '8f8c369c53697ec1fcbf490b5c1fe84506a07b225256e4a2fa7b82d36a98897f',
    'private_pilot/dataset_v1/freeze.json': '3d42dbc64de7e87032d166e8eff9efac03f7b000a86c30cbdd24173e0c63c1da',
    'private_pilot/dataset_v1/manifest.json': '958fb28f19e22f129553c66025b03573727aea499652f6b39954297015713b0f',
    'private_pilot/reference_calibration/report.json': 'fd317984133031a38b469e6ffb31e9736979e943b4c2ace2af40dd10e21a1d41',
    'evaluator/run_docker.py': 'a78f21731932aa82b82f3ac4966b029c340f215f5a75f22075cd9154b69ef36a',
    'evaluator/supervisor.py': '62c3722738fe48199381b74d42ef5063876e390f0a7e094df18716e7a6e820ea',
    '../public/tools/scoring.py': '182386804a8ca159d780e33dca2ed6348922f27efc3b43043befb0bf993ab0ad',
    'harness_recovery/metadata/retry_policies.json': 'e145b2e37bb4a82222b4199cceef9b4f1547be770b1bb0180cae39eb08d3e461',
    'harness_recovery/metadata/claude-opus-5-5.protocol.json': '039167b5a234429a78b8472041342dd037db8f50e38b12ddd8495b92c0973600',
    'harness_recovery/metadata/deepseek-v4-pro.protocol.json': '6365c7197e7a50a649de19791618bf3ce7ce685b4e37921ce71dd1731a95c099',
    'harness_recovery/metadata/hy4-preview.protocol.json': '58bd0cc6d732479e8182238e478cffdd6a3a627a02966b139b97608c3f260153',
    'harness_recovery/metadata/minimax-m3.1-flash.protocol.json': 'e50af5316b1054746736e0b607eb4d9565cc6127f1d6ac3be8a72d3bf2832d18',
    'harness_recovery/metadata/step-5-preview.protocol.json': 'b6fdd8c4565cd8ad080256e7578d500725b03d689a94f7d80a44d519c5dc0ddb',
    'harness_recovery/metadata/claude-opus-5-5.detail.json': '4d523494acd57c42ac55317ae6a96904be1abb2cc34056fcc9fbf2e4b48181e7',
    'harness_recovery/metadata/deepseek-v4-pro.detail.json': '07bd50794f49b665c80f6ff8acffc6bb05164226fe58e7190f73fb6de15f6a56',
    'harness_recovery/metadata/hy4-preview.detail.json': '8dd477661a868e595e065dcf7861b8a597a6c57d7213c4d05f1a9fe43b1090af',
    'harness_recovery/metadata/minimax-m3.1-flash.detail.json': 'ebbaf3fbc255ec5223593011970974f0159e28f4970b5baf2dd88731f439ea79',
    'harness_recovery/metadata/step-5-preview.detail.json': 'ad2496cc865374bf69c857f2e20313a56e439a8aaab2b0ed64182e09ab13aa4e',
}


def file_sha256(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


class AssetDriftError(RuntimeError):
    """The local copy no longer matches the pre-existing numerical contract."""


class LockedAssets:
    """Load only reviewed, immutable primitives from an existing docker_eval tree."""

    def __init__(self, source_root: str | Path) -> None:
        self.source_root = Path(source_root).expanduser().resolve(strict=True)
        self.image_id = IMAGE_ID
        self.public_tar = self.source_root / 'public_pilot_v2/public_pilot.tar'
        self.manifest = self.source_root / 'private_pilot/dataset_v1/manifest.json'
        self.baseline = self.source_root / 'private_pilot/reference_calibration/report.json'
        self._modules: dict[str, ModuleType] = {}
        self._module_lock = threading.RLock()
        self.verify()

    def verify(self) -> dict[str, str]:
        """Check fixed code, public package and numerical manifest locks before use."""
        for relative, expected in ASSET_SHA256.items():
            path = self.source_root / relative
            if not path.is_file() or path.is_symlink() or file_sha256(path) != expected:
                raise AssetDriftError(f'Frozen asset changed or missing: {relative}')
        image_lock = json.loads((self.source_root / 'image_lock.json').read_text())
        if image_lock.get('image_id') != self.image_id:
            raise AssetDriftError('Frozen image ID changed')
        return dict(ASSET_SHA256)

    def verify_private_data(self) -> dict[str, str]:
        """Validate the sealed dataset without copying it into an agent workspace."""
        self.verify()
        root = self.manifest.parent
        freeze = json.loads((root / 'freeze.json').read_text())
        for relative, expected in freeze['files'].items():
            path = root / relative
            if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
                raise AssetDriftError(f'Frozen dataset asset changed: {relative}')
        return dict(freeze['files'])

    def _load(self, relative: str) -> ModuleType:
        with self._module_lock:
            return self._load_locked(relative)

    def _load_locked(self, relative: str) -> ModuleType:
        if relative not in ASSET_SHA256:
            raise AssetDriftError('Unregistered module requested')
        path = self.source_root / relative
        source_bytes = path.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != ASSET_SHA256[relative]:
            raise AssetDriftError(f'Frozen module changed: {relative}')
        if relative not in self._modules:
            identity = hashlib.sha256(str(path).encode()).hexdigest()[:20]
            name = f'_nexus_frontieram_asset_{identity}_{uuid.uuid4().hex}'
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise AssetDriftError('Unable to load the reviewed asset')
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                # Compile the exact checked bytes without writing __pycache__
                # into the preserved legacy asset tree.
                exec(compile(source_bytes, str(path), 'exec'), module.__dict__)
            except BaseException:
                sys.modules.pop(name, None)
                raise
            self._modules[relative] = module
        return self._modules[relative]

    @property
    def controller(self) -> ModuleType:
        """Trusted Docker/build/export/grading primitives; never call its main()."""
        return self._load('campaign_runner_v2.py')

    @property
    def docker_mcp(self) -> ModuleType:
        return self._load('harness_resilient/docker_mcp.py')

    @property
    def scoring(self) -> ModuleType:
        return self._load('../public/tools/scoring.py')

    @property
    def codex_path(self) -> Path:
        path = self.source_root / 'runtime/bin/codex-frozen'
        if not path.is_file() or file_sha256(path) != CODEX_SHA256:
            raise AssetDriftError('Frozen Codex binary changed or missing')
        return path.resolve()

    def provenance(self) -> dict[str, Any]:
        return {
            'source_root': str(self.source_root),
            'image_id': self.image_id,
            'locked_files': self.verify(),
            'legacy_model_bridge_used': False,
            'legacy_campaign_runner_started': False,
        }
