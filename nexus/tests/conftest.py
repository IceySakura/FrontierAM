"""Shared pytest fixtures and configuration."""

import logging

from pathlib import Path

import pytest


@pytest.fixture
def register_test_component(request: pytest.FixtureRequest):
    """Remove only the components explicitly registered by the current test."""

    def register(registry, name):
        def decorate(obj):
            if name in registry:
                raise ValueError(f'Temporary component name is already registered: {name}')
            registered = registry.register(name)(obj)
            entry = registry.get_entry(name)

            def cleanup():
                if registry._registry.get(name) is entry:
                    del registry._registry[name]

            request.addfinalizer(cleanup)
            return registered

        return decorate

    return register


@pytest.fixture(autouse=True)
def _isolate_root_logger():
    """Undo global logging changes made by a test.

    ``nexus.utils.log.setup_logging`` (called by ``nexus.cli.main.main``) wipes the root
    logger's handlers and installs a ``sys.stdout`` StreamHandler. Without this fixture the
    handler survives the test that triggered it and later tests see log lines mixed into
    their captured stdout, which breaks anything parsing that stdout as JSON.
    """
    root_logger = logging.getLogger()
    saved_handlers = root_logger.handlers[:]
    saved_level = root_logger.level
    try:
        yield
    finally:
        for handler in root_logger.handlers[:]:
            if handler not in saved_handlers:
                root_logger.removeHandler(handler)
        for handler in saved_handlers:
            if handler not in root_logger.handlers:
                root_logger.addHandler(handler)
        root_logger.setLevel(saved_level)


@pytest.fixture
def isolated_shell_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``HOME`` at an empty profile for tests that spawn real shell commands.

    ``run_command`` sets ``BASH_ENV=~/.bashrc`` to emulate a login shell, so every spawned
    bash sources the *developer's* profile. A profile that runs ``conda init`` (or any other
    non-trivial setup) adds close to a second to each command start-up, which silently eats
    the budget of tests asserting on sub-second command deadlines: they then pass or fail
    depending on whose machine runs them.
    """
    home = tmp_path_factory.mktemp('shell-home')
    (home / '.bashrc').touch()
    monkeypatch.setenv('HOME', str(home))
    return home


@pytest.fixture
def tmp_path_factory_scoped(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a session-scoped temporary directory."""
    return tmp_path_factory.mktemp('test_data')


@pytest.fixture
def sample_data() -> dict:
    """Provide sample test data."""
    return {
        'name': 'test',
        'value': 42,
    }


@pytest.fixture
def otel_hyway_instrumentor(monkeypatch):
    from nexus.utils.otel.hyway import NexusHywayInstrumentor

    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    instrumentor = NexusHywayInstrumentor()
    instrumentor.instrument()
    try:
        yield instrumentor
    finally:
        instrumentor.uninstrument()
