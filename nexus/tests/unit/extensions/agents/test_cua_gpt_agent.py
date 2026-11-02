"""Discovery contract for the shared GPT CUA agent."""

from nexus.extensions.agents.cua.cua_gpt_agent import agent as cua_agent


def test_gpt_agent_is_discovered_only_through_shared_cua_module() -> None:
    from nexus.core.module_meta import discover_all_modules

    modules = [
        module
        for module in discover_all_modules(include_compiled_manifest=False)
        if 'cua_gpt_agent' in module.objects.get('agents', {})
    ]
    assert [module.name for module in modules] == ['cua']
    assert modules[0].objects['agents']['cua_gpt_agent'] == cua_agent.__name__
