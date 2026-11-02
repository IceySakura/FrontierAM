import pytest

from pydantic import ValidationError

from nexus.extensions.agents.local_one_agent.context import (
    LocalContext,
    LocalEnv,
    LocalPermission,
    LocalRuntime,
    LocalWorkspace,
    PermissionEngineConfig,
    UserDecisionRule,
    resolve_workspace_path,
)


class TestLocalWorkspace:
    def test_valid_workspace(self):
        ws = LocalWorkspace(path='/workspace/project', scope='conversation')
        assert ws.path == '/workspace/project'
        assert ws.scope == 'conversation'

    def test_invalid_scope_rejected(self):
        with pytest.raises(ValidationError):
            LocalWorkspace(path='/workspace', scope='invalid')


class TestLocalPermission:
    def test_all_levels(self):
        for level in ('readonly', 'basic'):
            perm = LocalPermission(level=level)
            assert perm.level == level

    def test_full_level_removed(self):
        with pytest.raises(ValidationError):
            LocalPermission(level='full')

    def test_invalid_level_rejected(self):
        with pytest.raises(ValidationError):
            LocalPermission(level='admin')

    def test_previous_level_default_none(self):
        perm = LocalPermission(level='basic')
        assert perm.previous_level is None

    def test_previous_level_records_switch(self):
        perm = LocalPermission(level='basic', previous_level='readonly')
        assert perm.previous_level == 'readonly'


class TestLocalRuntime:
    def test_macos_runtime(self):
        rt = LocalRuntime(os='macos', arch='arm64', shell='zsh', default_browser='Safari')
        assert rt.os == 'macos'
        assert rt.default_browser == 'Safari'

    def test_windows_runtime(self):
        rt = LocalRuntime(os='windows', arch='x86_64', shell='powershell')
        assert rt.os == 'windows'
        assert rt.default_browser is None

    def test_linux_os_rejected(self):
        # linux 不是用户可选 OS（仅作训练容器基底模拟 macos）
        with pytest.raises(ValidationError):
            LocalRuntime(os='linux', arch='x86_64', shell='bash')

    def test_invalid_os_rejected(self):
        with pytest.raises(ValidationError):
            LocalRuntime(os='freebsd', arch='x86_64', shell='bash')


class TestLocalEnv:
    def test_default_is_cold(self):
        env = LocalEnv()
        assert env.mode == 'cold'
        assert env.python_requirements is None
        assert env.node_manifest is None

    def test_warm_with_manifests(self):
        env = LocalEnv(mode='warm', python_requirements='requirements.txt', node_manifest='package.json')
        assert env.mode == 'warm'
        assert env.python_requirements == 'requirements.txt'
        assert env.node_manifest == 'package.json'

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValidationError):
            LocalEnv(mode='hot')

    def test_context_defaults_env_cold(self):
        ctx = LocalContext(
            workspace=LocalWorkspace(path='/workspace', scope='conversation'),
            permission=LocalPermission(level='basic'),
            runtime=LocalRuntime(os='macos', arch='arm64', shell='zsh'),
        )
        assert ctx.env.mode == 'cold'


class TestPermissionEngineConfig:
    def test_with_user_decisions(self):
        cfg = PermissionEngineConfig(
            user_decisions=[
                UserDecisionRule(pattern='pip install', decision='confirmed'),
                UserDecisionRule(pattern='*', decision='rejected'),
            ]
        )
        assert len(cfg.user_decisions) == 2
        assert cfg.user_decisions[0].decision == 'confirmed'

    def test_empty_decisions(self):
        cfg = PermissionEngineConfig(user_decisions=[])
        assert cfg.user_decisions == []

    def test_invalid_decision_rejected(self):
        with pytest.raises(ValidationError):
            UserDecisionRule(pattern='*', decision='maybe')


class TestLocalContext:
    def test_full_context(self):
        ctx = LocalContext(
            workspace=LocalWorkspace(path='/workspace/project', scope='conversation'),
            permission=LocalPermission(level='basic', previous_level='readonly'),
            runtime=LocalRuntime(os='macos', arch='arm64', shell='zsh'),
            permission_engine=PermissionEngineConfig(
                user_decisions=[UserDecisionRule(pattern='*', decision='confirmed')]
            ),
            skill_path='/data/skills',
        )
        assert ctx.workspace.path == '/workspace/project'
        assert ctx.permission.level == 'basic'
        assert ctx.permission.previous_level == 'readonly'
        assert ctx.runtime.os == 'macos'
        assert ctx.skill_path == '/data/skills'
        assert len(ctx.permission_engine.user_decisions) == 1

    def test_workspace_required(self):
        with pytest.raises(ValidationError):
            LocalContext(
                permission=LocalPermission(level='basic'),
                runtime=LocalRuntime(os='macos', arch='arm64', shell='zsh'),
            )

    def test_permission_required(self):
        with pytest.raises(ValidationError):
            LocalContext(
                workspace=LocalWorkspace(path='/workspace', scope='conversation'),
                runtime=LocalRuntime(os='macos', arch='x86_64', shell='zsh'),
            )

    def test_runtime_required(self):
        with pytest.raises(ValidationError):
            LocalContext(
                workspace=LocalWorkspace(path='/workspace', scope='conversation'),
                permission=LocalPermission(level='basic'),
            )


class TestResolveWorkspacePath:
    def test_prefers_local_context_path(self):
        task_config = {
            'local_context': {'workspace': {'path': '/workspace/project'}},
            'workspace_dir': '/legacy',
        }
        assert resolve_workspace_path(task_config) == '/workspace/project'

    def test_falls_back_to_legacy_workspace_dir(self):
        assert resolve_workspace_path({'workspace_dir': '/legacy'}) == '/legacy'

    def test_default_workspace(self):
        assert resolve_workspace_path({}) == '/workspace'
