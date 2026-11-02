from nexus.extensions.projects.code_mt.agent.agent import ch_code_mt_agent, code_mt_agent


def test_ch_code_mt_registers_a_distinct_policy_scoped_agent_name() -> None:
    assert code_mt_agent.nexus_meta['name'] == 'code_mt'
    assert ch_code_mt_agent.nexus_meta['name'] == 'ch_code_mt'
    assert code_mt_agent is not ch_code_mt_agent
