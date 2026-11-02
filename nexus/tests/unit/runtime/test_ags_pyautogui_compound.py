from unittest.mock import AsyncMock

from nexus.runtime.providers.ags.osworld import AGSOSWorldRuntime


async def test_controller_prefix_allows_compound_first_statement(monkeypatch):
    post = AsyncMock(return_value=({'returncode': 0, 'output': ''}, '', 200))
    monkeypatch.setattr(AGSOSWorldRuntime, '_post_osworld_json', post)
    # No instance initialization/network is needed to exercise payload construction.
    runtime = object.__new__(AGSOSWorldRuntime)
    code = 'def paste():\n    pass\npaste()\n'
    await runtime._run_python_controller_code(code)
    source = post.call_args.args[1]['command'][2]
    compile(source, '<ags-controller>', 'exec')
    assert source.endswith(code)
