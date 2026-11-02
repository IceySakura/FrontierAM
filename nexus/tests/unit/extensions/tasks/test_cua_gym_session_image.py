from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(('exists', 'force'), [(True, False), (False, False), (True, True)])
def test_cua_gym_image_build_resolves_base_and_skips_existing(monkeypatch, exists, force):
    from nexus.extensions.tasks.cua_gym import cua_gym_session_image as image

    calls = []
    dockerfiles = []

    def docker(command, **kwargs):
        calls.append((command, kwargs))
        if command.startswith('build '):
            context = Path(command.split()[-1])
            dockerfiles.append((context / 'Dockerfile').read_text())
            assert (context / 'reward_judge.py').is_file()
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=0 if exists else 1)

    monkeypatch.setattr(image, '_docker', docker)
    monkeypatch.delenv(image.CUA_GYM_SESSION_IMAGE_FORCE_BUILD_ENV, raising=False)
    monkeypatch.setenv('NEXUS_UBUNTU_SESSION_IMAGE', 'fixture/base:latest')
    assert image.ensure_cua_gym_session_image('fixture/cua:test', force=force) == 'fixture/cua:test'
    assert len(calls) == (1 if exists and not force else 2)
    if dockerfiles:
        assert dockerfiles[0].startswith('FROM fixture/base:latest\n')


def test_cua_gym_session_dockerfile_extends_shared_gui_image_with_cua_dependencies():
    from nexus.extensions.tasks.cua_gym.cua_gym_session_image import build_cua_gym_session_dockerfile

    dockerfile = build_cua_gym_session_dockerfile(
        'nexus-ubuntu-session:latest',
        include_hub=True,
        hub_apps=['instagram_mock', 'gmail_mock'],
    )

    assert dockerfile.startswith('FROM nexus-ubuntu-session:latest\n')
    assert 'NEXUS_CUA_GYM_HUB_ROOT=/opt/cua-gym-hub' in dockerfile
    assert 'PIP_BREAK_SYSTEM_PACKAGES=1' in dockerfile
    assert 'PIP_NO_INDEX=1' in dockerfile
    assert 'odfpy' in dockerfile
    assert 'openpyxl' in dockerfile
    assert 'pyautogui' in dockerfile
    assert 'python-docx' in dockerfile
    assert 'python-pptx' in dockerfile
    assert 'openai' in dockerfile
    assert 'setup_22.x' in dockerfile
    assert 'nodejs' in dockerfile
    assert 'npm --version' in dockerfile
    assert 'python3-tk' in dockerfile
    assert 'COPY --chown=user:user reward_judge.py /tmp/reward_judge.py' in dockerfile
    assert 'COPY --chown=user:user hub /opt/cua-gym-hub' in dockerfile
    assert '/opt/cua-gym-hub/websites/instagram_mock' in dockerfile
    assert '/opt/cua-gym-hub/websites/gmail_mock' in dockerfile
    assert 'npm install --include=dev --legacy-peer-deps --no-audit --no-fund' in dockerfile
    assert 'npm run build' in dockerfile


def test_cua_gym_session_image_default_hub_apps_cover_train_mock_apps():
    from nexus.extensions.tasks.cua_gym.hub_manifest import DEFAULT_CUA_GYM_HUB_APPS

    assert 'instagram_mock' in DEFAULT_CUA_GYM_HUB_APPS
    assert 'github_mock' in DEFAULT_CUA_GYM_HUB_APPS
    assert 'trello_mock' in DEFAULT_CUA_GYM_HUB_APPS


def test_cua_gym_session_dockerfile_patches_known_hub_icon_aliases():
    from nexus.extensions.tasks.cua_gym.cua_gym_session_image import build_cua_gym_session_dockerfile
    from nexus.extensions.tasks.cua_gym.hub_manifest import HUB_BUILD_PATCHES

    dockerfile = build_cua_gym_session_dockerfile(
        'nexus-ubuntu-session:latest',
        include_hub=True,
        hub_apps=['github_mock', 'trello_mock'],
    )

    assert any(patch.app_name == 'github_mock' and 'XitHub' in patch.reason for patch in HUB_BUILD_PATCHES)
    assert any(patch.app_name == 'trello_mock' and 'Xrello' in patch.reason for patch in HUB_BUILD_PATCHES)
    assert 'Github as XitHub' in dockerfile
    assert 'Trello as Xrello' in dockerfile


def test_cua_gym_session_image_resolves_env_override(monkeypatch):
    from nexus.extensions.tasks.cua_gym.cua_gym_session_image import (
        DEFAULT_CUA_GYM_SESSION_IMAGE,
        resolve_cua_gym_session_image,
    )

    assert DEFAULT_CUA_GYM_SESSION_IMAGE == 'mirrors.tencent.com/hunyuan_yanguan/nexus-cua-gym-session:latest'
    assert resolve_cua_gym_session_image() == DEFAULT_CUA_GYM_SESSION_IMAGE

    monkeypatch.setenv('NEXUS_CUA_GYM_SESSION_IMAGE', 'custom/cua:dev')

    assert resolve_cua_gym_session_image() == 'custom/cua:dev'


def test_cua_gym_design_doc_marks_session_image_build_as_maintainer_path():
    doc = Path('docs/developing/cua-gym-task-design.md').read_text(encoding='utf-8')

    assert '训练集验证和示例配置使用发布镜像' in doc
    assert '构建入口只用于镜像维护' in doc
    assert 'nexus-cua-gym-session:dev-integration' in doc
