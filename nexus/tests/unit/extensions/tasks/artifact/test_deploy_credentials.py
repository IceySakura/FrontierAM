from __future__ import annotations

from nexus.extensions.tasks.artifact.agent_setup import cos_io
from nexus.extensions.tasks.artifact.judges.capabilities import deploy


async def test_render_deployment_requires_an_explicit_gongfeng_token(monkeypatch) -> None:
    monkeypatch.delenv('GONGFENG_TOKEN', raising=False)
    monkeypatch.delenv('GONGFENG_RUNTIME_TOKEN', raising=False)
    monkeypatch.setattr(
        cos_io,
        'upload_archive_b64',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('must fail before upload')),
    )

    result = await deploy.prepare_sandbox_context('archive', instance_id='case')

    assert result.sandbox_url is None
    assert result.deploy_error == ('infra: GONGFENG_TOKEN or GONGFENG_RUNTIME_TOKEN is required for render deployment')


async def test_render_deployment_accepts_the_runtime_token_alias(monkeypatch) -> None:
    monkeypatch.delenv('GONGFENG_TOKEN', raising=False)
    monkeypatch.setenv('GONGFENG_RUNTIME_TOKEN', 'runtime-token')
    monkeypatch.setattr(cos_io, 'upload_archive_b64', lambda *_args, **_kwargs: 'cos://bucket/archive.tar.gz')
    submitted: dict = {}

    def fake_render_http(method, _url, body=None, timeout=90.0):
        del timeout
        if method == 'POST':
            submitted.update(body or {})
            return 202, {'task_id': 'render-task'}
        return 200, {'status': 'success', 'success': True, 'frontend_url': 'http://frontend.test'}

    monkeypatch.setattr(deploy, '_render_http', fake_render_http)

    result = await deploy.prepare_sandbox_context('archive', instance_id='case')

    assert result.deploy_error is None
    assert result.sandbox_url == 'http://frontend.test'
    assert submitted['gongfeng_token'] == 'runtime-token'
