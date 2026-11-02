"""Tests for venv management models."""

from datetime import datetime, timezone

from nexus.runtime.server.venv.models import ArtifactSpec, VenvSpec, VenvStatus, WorkerStatus


class TestArtifactSpec:
    def test_push_mode(self):
        spec = ArtifactSpec(name='my-pkg', whl_data='base64data', whl_sha256='abc123')
        assert spec.name == 'my-pkg'
        assert spec.whl_data == 'base64data'
        assert spec.whl_url is None

    def test_pull_mode(self):
        spec = ArtifactSpec(name='my-pkg', whl_url='https://example.com/pkg.whl')
        assert spec.whl_url == 'https://example.com/pkg.whl'
        assert spec.whl_data is None

    def test_extras(self):
        spec = ArtifactSpec(name='nexus-agent', extras=['browser', 'terminal'])
        assert spec.extras == ['browser', 'terminal']


class TestVenvSpec:
    def test_defaults(self):
        spec = VenvSpec()
        assert spec.python_version == '3.12'
        assert spec.artifacts == []
        assert spec.extra_requirements == []
        assert spec.env_vars == {}

    def test_full(self):
        spec = VenvSpec(
            python_version='3.12',
            artifacts=[ArtifactSpec(name='my-pkg', whl_data='data')],
            extra_requirements=['numpy>=1.24'],
            env_vars={'MY_VAR': 'value'},
        )
        assert len(spec.artifacts) == 1
        assert spec.extra_requirements == ['numpy>=1.24']

    def test_serialization_roundtrip(self):
        spec = VenvSpec(
            artifacts=[ArtifactSpec(name='pkg', whl_url='http://x.com/a.whl')],
            extra_requirements=['foo'],
        )
        data = spec.model_dump()
        spec2 = VenvSpec(**data)
        assert spec2.artifacts[0].name == 'pkg'


class TestVenvStatus:
    def test_ready(self):
        status = VenvStatus(
            venv_id='abc',
            state='ready',
            python_version='3.11',
            artifacts=['my-pkg'],
            created_at=datetime.now(timezone.utc),
        )
        assert status.state == 'ready'
        assert status.error is None

    def test_failed(self):
        status = VenvStatus(
            venv_id='abc',
            state='failed',
            python_version='3.11',
            created_at=datetime.now(timezone.utc),
            error='uv venv failed',
        )
        assert status.state == 'failed'
        assert 'uv' in status.error


class TestWorkerStatus:
    def test_basic(self):
        status = WorkerStatus(
            venv_id='abc',
            worker_id='worker-abc',
            state='running',
            pid=12345,
        )
        assert status.state == 'running'
        assert status.pid == 12345
