"""Tests for shared generator helpers (env reading, cache_task_id injection)."""

import json
import re

import pytest

from nexus.extensions.tasks.hitl.utils.generator_util import (
    EnvValueError,
    _build_worker_config,
    _env_bool,
    _env_choice,
    _env_int,
    _read_user_memory,
    inject_cache_task_ids,
    make_cache_task_id,
    worker_usage_profile,
)


# -- reading the environment ----------------------------------------------
#
# A launch script is a shell script, so a variable's value is whatever someone
# typed. These cover the three ways a typo used to take effect silently, each one
# measured against the generator before the change.


def test_an_unset_variable_takes_the_default(monkeypatch):
    monkeypatch.delenv('X', raising=False)

    assert _env_int('X', default=7) == 7
    assert _env_bool('X', default=True) is True
    assert _env_choice('X', 'mean', ('mean', 'unit_test')) == 'mean'


@pytest.mark.parametrize('raw', ['3', ' 3 ', '-1', '0'])
def test_an_integer_is_read_and_trimmed(monkeypatch, raw):
    monkeypatch.setenv('X', raw)

    assert _env_int('X', default=99) == int(raw.strip())


def test_a_non_integer_is_refused_rather_than_read_as_the_default(monkeypatch):
    """``SWE_LIMIT=three`` used to read as 0, which the caller turns into "no
    limit" -- a three-instance smoke run became the whole 4996-row dataset."""
    monkeypatch.setenv('SWE_LIMIT', 'three')

    with pytest.raises(EnvValueError, match="SWE_LIMIT='three' is not an integer"):
        _env_int('SWE_LIMIT')


@pytest.mark.parametrize('raw', ['1', 'true', 'TRUE', 'yes', 'on'])
def test_the_true_spellings(monkeypatch, raw):
    monkeypatch.setenv('X', raw)

    assert _env_bool('X') is True


@pytest.mark.parametrize('raw', ['0', 'false', 'FALSE', 'no', 'off'])
def test_the_false_spellings(monkeypatch, raw):
    monkeypatch.setenv('X', raw)

    assert _env_bool('X', default=True) is False


def test_a_misspelled_boolean_is_refused_rather_than_read_as_false(monkeypatch):
    """``HITL_SKIP_EVAL=ture`` used to read as False and run the evaluation."""
    monkeypatch.setenv('HITL_SKIP_EVAL', 'ture')

    with pytest.raises(EnvValueError, match='is not a boolean'):
        _env_bool('HITL_SKIP_EVAL')


def test_a_choice_outside_the_set_is_refused_and_the_set_is_named(monkeypatch):
    """``HITL_SCORE_POLICY=medain`` reached an ``else`` branch and scored by the
    mean, so the run finished and reported a number under a policy nobody chose."""
    monkeypatch.setenv('HITL_SCORE_POLICY', 'medain')

    with pytest.raises(EnvValueError, match=r"not one of \['mean', 'patch_rubrics', 'unit_test'\]"):
        _env_choice('HITL_SCORE_POLICY', 'mean', ('unit_test', 'patch_rubrics', 'mean'))


def test_a_value_in_the_set_is_returned(monkeypatch):
    monkeypatch.setenv('X', 'patch_rubrics')

    assert _env_choice('X', 'mean', ('unit_test', 'patch_rubrics', 'mean')) == 'patch_rubrics'


# -- cache_task_id injection ----------------------------------------------


def _clients():
    return {
        'worker': {'config': {'client_args': {'api_key': 'app:key?provider=x'}}},
        'judge': {'config': {'client_args': {'api_key': 'app:key'}}},
        'hy3': {'config': {'client_args': {'api_key': ''}}},
    }


def _ctid(client):
    m = re.search(r'cache_task_id=(\w+)', client['config']['client_args']['api_key'])
    return m.group(1) if m else None


def test_make_cache_task_id_is_32_md5():
    tid = make_cache_task_id()
    assert len(tid) == 32 and re.fullmatch(r'[0-9a-f]{32}', tid)
    assert make_cache_task_id() != make_cache_task_id()  # 唯一


def test_inject_appends_with_correct_separator():
    out = inject_cache_task_ids(_clients(), roles=None)
    assert '?provider=x&cache_task_id=' in out['worker']['config']['client_args']['api_key']  # 已有?→&
    assert 'app:key?cache_task_id=' in out['judge']['config']['client_args']['api_key']  # 无?→?


def test_each_client_gets_distinct_id():
    out = inject_cache_task_ids(_clients(), roles=None)
    assert _ctid(out['worker']) != _ctid(out['judge'])


def test_empty_api_key_skipped():
    out = inject_cache_task_ids(_clients(), roles=None)
    assert out['hy3']['config']['client_args']['api_key'] == ''  # HY3 空 key 不动


def test_does_not_mutate_input():
    clients = _clients()
    inject_cache_task_ids(clients, roles=None)
    assert clients['worker']['config']['client_args']['api_key'] == 'app:key?provider=x'


def test_roles_filter():
    out = inject_cache_task_ids(_clients(), roles={'worker'})
    assert _ctid(out['worker']) is not None
    assert _ctid(out['judge']) is None  # 不在 roles → 不注入


def test_idempotent_no_double_append():
    out1 = inject_cache_task_ids(_clients(), roles={'worker'})
    out2 = inject_cache_task_ids(out1, roles={'worker'})
    # 已含 cache_task_id → 不重复追加
    assert out2['worker']['config']['client_args']['api_key'].count('cache_task_id=') == 1


# -- the worker's user memory file -----------------------------------------
#
# The path may be anywhere: a run points at whichever preferences file it wants,
# inside the repo or not. What must not happen is a typo turning into "this run
# has no preferences" with nothing said.


def test_a_memory_file_is_read_from_wherever_it_is(monkeypatch, tmp_path):
    memory = tmp_path / 'prefs.md'
    memory.write_text('## How I like to work\n\nBe honest about what you verified.\n')
    monkeypatch.setenv('HITL_WORKER_MEMORY_FILE', str(memory))

    assert _read_user_memory().startswith('## How I like to work')


def test_a_memory_path_with_a_tilde_is_expanded(monkeypatch, tmp_path):
    """A run pointing at ``~/prefs.md`` is the ordinary case for a shared box."""
    monkeypatch.setenv('HOME', str(tmp_path))
    (tmp_path / 'prefs.md').write_text('from home\n')
    monkeypatch.setenv('HITL_WORKER_MEMORY_FILE', '~/prefs.md')

    assert _read_user_memory() == 'from home\n'


def test_an_unset_memory_file_means_no_memory(monkeypatch):
    monkeypatch.delenv('HITL_WORKER_MEMORY_FILE', raising=False)

    assert _read_user_memory() == ''


def test_an_empty_memory_file_means_no_memory(monkeypatch, tmp_path):
    """An all-blank file is one nobody finished writing, not a request for none."""
    blank = tmp_path / 'blank.md'
    blank.write_text('   \n\n')
    monkeypatch.setenv('HITL_WORKER_MEMORY_FILE', str(blank))

    assert _read_user_memory() == ''


def test_a_memory_path_that_names_no_file_is_refused(monkeypatch, tmp_path):
    """Returning '' would run the whole batch with no preferences and say nothing."""
    missing = tmp_path / 'typo.md'
    monkeypatch.setenv('HITL_WORKER_MEMORY_FILE', str(missing))

    with pytest.raises(ValueError, match=str(missing)):
        _read_user_memory()


def test_the_memory_reaches_the_worker_config(monkeypatch, tmp_path):
    """The channel the managers read: worker_config['user_memory']."""
    memory = tmp_path / 'prefs.md'
    memory.write_text('be honest\n')
    monkeypatch.setenv('HITL_WORKER_MEMORY_FILE', str(memory))
    for key, value in (
        ('HITL_WORKER_API_KEY', 'k'),
        ('HITL_WORKER_BASE_URL', 'http://x/v1'),
        ('HITL_WORKER_MODEL', 'm'),
    ):
        monkeypatch.setenv(key, value)

    assert _build_worker_config('default', session_id='s')['user_memory'] == 'be honest\n'


def test_a_run_without_a_memory_file_carries_no_such_key(monkeypatch):
    """The managers key off presence, so an absent file must leave the key out."""
    monkeypatch.delenv('HITL_WORKER_MEMORY_FILE', raising=False)
    for key, value in (
        ('HITL_WORKER_API_KEY', 'k'),
        ('HITL_WORKER_BASE_URL', 'http://x/v1'),
        ('HITL_WORKER_MODEL', 'm'),
    ):
        monkeypatch.setenv(key, value)

    assert 'user_memory' not in _build_worker_config('default', session_id='s')


def test_every_model_profile_states_which_thinking_shape_it_wants() -> None:
    """Left out, a profile takes the proxy's own default rather than a stated choice.

    The two shapes are not interchangeable: `bedrock-claude` keeps the signature a
    Claude thinking block carries between turns, and `hunyuan` has nowhere to put
    one. Getting it wrong drops signatures, which upstream then rejects.
    """
    from nexus.extensions.tasks.hitl.model import known_profiles
    from nexus.extensions.tasks.hitl.utils.generator_util.claude_code import _PROFILE_THINKING_PRESET

    unstated = sorted(set(known_profiles()) - set(_PROFILE_THINKING_PRESET))

    assert not unstated, f'no thinking shape stated for: {unstated}'


def test_the_stated_shapes_are_ones_the_proxy_knows() -> None:
    """The table is hitl's choice per profile; the shapes themselves are the proxy's."""
    from nexus.extensions.tasks.hitl.utils.generator_util.claude_code import _PROFILE_THINKING_PRESET
    from nexus.runtime.server.services.anthropic_proxy.models import THINKING_PRESETS

    unknown = sorted(set(_PROFILE_THINKING_PRESET.values()) - set(THINKING_PRESETS))

    assert not unknown, f'not a preset the proxy defines: {unknown}'


def test_the_basic_openai_profile_reads_usage_as_the_default_one() -> None:
    """`openai_basic` is the one to name for a plain OpenAI-compatible model.

    `naci_glm` differs only in the provenance stamped on the export; a change that
    made them read different counters would be worth noticing here.
    """
    from nexus.extensions.tasks.hitl.model import resolve_profile
    from nexus.extensions.tasks.hitl.trace.usage.profiles import spec_for

    basic = resolve_profile('openai_basic')()
    glm = resolve_profile('naci_glm')()

    assert basic.usage_profile == 'default'
    assert basic.think_request_args == glm.think_request_args == {}
    assert spec_for(basic.usage_profile) == spec_for(glm.usage_profile)


# -- the usage profile is the run's, not the sandbox's --------------------


def test_the_worker_config_does_not_carry_the_usage_profile() -> None:
    """It is read by ``hitl_usage`` on the host, so it does not belong in the dict
    the worker manager passes into the sandbox: both backends had to whitelist it
    out (``pty_claude_code/runtime_adapter.py``, ``print_claude_code/backend.py``).
    """
    assert 'usage_profile' not in _build_worker_config('bedrock_claude', session_id='s')


@pytest.mark.parametrize(
    ('profile', 'expected'),
    [
        ('bedrock_claude', 'bedrock_claude'),
        ('gcp_claude', 'bedrock_claude'),
        ('naci_glm', 'naci_glm'),
        ('openai_basic', 'default'),
        ('hy3', 'default'),
        ('gpt_responses', 'default'),
        ('default', 'default'),
    ],
)
def test_the_usage_profile_follows_from_the_model_profile(profile: str, expected: str) -> None:
    """Vertex and Bedrock report cache creation separately from cache reads, and
    ``default`` does not look for it, so naming the wrong one reads that counter as
    zero on a run that paid for it.
    """
    assert worker_usage_profile(profile) == expected


def test_every_known_profile_resolves_to_a_usable_usage_profile() -> None:
    """A profile added without a usage spec would export unparsed counters."""
    from nexus.extensions.tasks.hitl.model import known_profiles
    from nexus.extensions.tasks.hitl.trace.usage.profiles import spec_for

    for name in known_profiles():
        assert spec_for(worker_usage_profile(name)) is not None, name


@pytest.mark.parametrize('profile', ['bedrock_claude', 'gcp_claude'])
def test_turning_thinking_off_does_not_ask_for_a_summarized_block(profile: str) -> None:
    """``display`` is only legal while thinking is on; with it, the request is rejected.

    Both Anthropic-protocol profiles build the block from one helper for this
    reason. Sending the pair is not a degraded request, it is a 400, so the
    thinking_type=disabled path would be unusable rather than merely odd.
    """
    from nexus.extensions.tasks.hitl.model import resolve_profile

    thinking = resolve_profile(profile)(thinking_type='disabled').think_request_args['thinking']

    assert thinking == {'type': 'disabled'}


@pytest.mark.parametrize('profile', ['bedrock_claude', 'gcp_claude'])
def test_thinking_on_still_asks_for_a_summarized_block(profile: str) -> None:
    """Without it the thinking content does not survive the round trip."""
    from nexus.extensions.tasks.hitl.model import resolve_profile

    thinking = resolve_profile(profile)(thinking_type='adaptive').think_request_args['thinking']

    assert thinking == {'type': 'adaptive', 'display': 'summarized'}


def test_effort_and_thinking_are_set_independently() -> None:
    """A run asking for low effort without thinking has to be able to say so.

    ``HITL_WORKER_THINKING_ENABLED=0`` is the other knob and a different request:
    it swaps in the profile's no-think args, which for Claude are empty, dropping
    the effort along with the thinking block.
    """
    from nexus.extensions.tasks.hitl.model import resolve_profile

    built = resolve_profile('bedrock_claude')(thinking_type='disabled', reasoning_effort='high')

    assert built.think_request_args['output_config'] == {'effort': 'high'}
    assert built.think_request_args['thinking'] == {'type': 'disabled'}


# -- the Responses profile ------------------------------------------------


def test_the_responses_no_think_args_repeat_the_whole_request() -> None:
    """The no-think args replace the request args rather than merging into them
    (``openai_responses.py`` ``_request_args_override``), so a set holding only the
    effort would send Claude Code's sidecar calls with ``store`` back at the
    upstream default of on. This gateway is stateless, and a stored turn is answered
    with ``rs_ not found`` on the next one.
    """
    from nexus.extensions.tasks.hitl.model import resolve_profile

    built = resolve_profile('gpt_responses')()

    assert built.no_think_request_args['reasoning'] == {
        'effort': 'none',
        'summary': 'detailed',
        'mode': 'standard',
    }
    for key in ('store', 'max_output_tokens', 'include'):
        assert built.no_think_request_args[key] == built.think_request_args[key], key
    assert built.no_think_request_args['reasoning']['mode'] == built.think_request_args['reasoning']['mode']


def test_the_responses_client_is_given_no_litellm_only_args() -> None:
    """``openai_responses`` hands every client_arg it does not claim to
    ``openai.AsyncOpenAI``, which raises on one it does not know. Measured:
    ``AsyncOpenAI.__init__() got an unexpected keyword argument 'num_retries'``.
    """
    from nexus.extensions.tasks.hitl.model import apply_protocol_to_client, build_llm_client, resolve_profile

    protocol = resolve_profile('gpt_responses')()
    client = build_llm_client(api_key='k', base_url='http://e/openai/v1', model='m', client_name=protocol.client_name)
    apply_protocol_to_client(client, protocol)

    assert client['name'] == 'openai_responses'
    assert 'num_retries' not in client['config']['client_args']
    assert 'log_llm_usage' not in client['config']['client_args']


def test_the_responses_reasoning_mode_comes_from_the_environment(monkeypatch) -> None:
    """``reasoning.mode`` picks the serving path; ``standard`` is not ``pro``."""
    from nexus.extensions.tasks.hitl.model import resolve_profile
    from nexus.extensions.tasks.hitl.utils.generator_util.worker import _worker_profile_kwargs

    monkeypatch.setenv('HITL_WORKER_REASONING_MODE', 'pro')
    built = resolve_profile('gpt_responses')(**_worker_profile_kwargs())

    assert built.think_request_args['reasoning']['mode'] == 'pro'


def test_an_empty_reasoning_mode_omits_the_field() -> None:
    """A gateway that does not know the field rejects it, rather than ignoring it."""
    from nexus.extensions.tasks.hitl.model import resolve_profile

    built = resolve_profile('gpt_responses')(reasoning_mode='')

    assert 'mode' not in built.think_request_args['reasoning']
    assert 'mode' not in built.no_think_request_args['reasoning']



# -- the auto-compact window is one value, named in two layers ------------


def test_the_generator_and_the_specs_name_the_same_auto_compact_window(monkeypatch) -> None:
    """Both layers carry a default for it, and the generator's is the one that reaches
    the worker: it always writes the key, so the spec's own default is only ever the
    fallback for a config built elsewhere. A disagreement changes when Claude Code
    compacts, and nothing reports it."""
    from nexus.extensions.tasks.hitl.actor.print_claude_code.spec import (
        DEFAULT_AUTO_COMPACT_WINDOW as PRINT_DEFAULT,
    )
    from nexus.extensions.tasks.hitl.actor.pty_claude_code.spec import (
        DEFAULT_AUTO_COMPACT_WINDOW as PTY_DEFAULT,
    )
    from nexus.extensions.tasks.hitl.utils.generator_util import _build_worker_config

    monkeypatch.delenv('HITL_CC_AUTO_COMPACT_WINDOW', raising=False)
    built = _build_worker_config('bedrock_claude')

    assert PRINT_DEFAULT == PTY_DEFAULT
    assert str(built['auto_compact_window']) == PRINT_DEFAULT


def test_the_window_the_worker_is_given_is_the_one_the_generator_built(monkeypatch) -> None:
    """``worker_env`` takes it from the config rather than the host environment."""
    from nexus.extensions.tasks.hitl.actor.print_claude_code.spec import worker_env
    from nexus.extensions.tasks.hitl.utils.generator_util import _build_worker_config

    monkeypatch.setenv('HITL_WORKER_CONTEXT_WINDOW', '123456')
    monkeypatch.delenv('CLAUDE_CODE_AUTO_COMPACT_WINDOW', raising=False)

    env = worker_env(_build_worker_config('bedrock_claude'))

    assert env['CLAUDE_CODE_AUTO_COMPACT_WINDOW'] == '123456'


# -- the Codex worker config ----------------------------------------------


def test_a_catalog_name_resolves_against_the_assets_dir(monkeypatch) -> None:
    """A run names the file, not its path: the assets dir travels with the wheel."""
    from nexus.extensions.tasks.hitl.utils.generator_util import _build_codex_worker_config
    from nexus.extensions.tasks.hitl.utils.generator_util.codex import CATALOG_DIR

    monkeypatch.setenv('HITL_CODEX_MODEL_CATALOG', 'gpt-5.6-sol')
    monkeypatch.setenv('HITL_CODEX_MODEL', 'gpt-5.6-sol')
    monkeypatch.delenv('HITL_CODEX_RESPONSES_LITE', raising=False)

    catalog = _build_codex_worker_config('gpt_responses')['model_catalog_json']

    assert (CATALOG_DIR / 'gpt-5.6-sol.json').is_file()
    assert [entry['slug'] for entry in catalog['models']] == ['gpt-5.6-sol']


def test_a_catalog_name_that_names_no_file_is_refused(monkeypatch) -> None:
    """Returning None would run the whole batch under the bundled catalog, whose
    gpt-5.6 entry leaves the worker with no tools, and say nothing."""
    from nexus.extensions.tasks.hitl.utils.generator_util import _build_codex_worker_config

    monkeypatch.setenv('HITL_CODEX_MODEL_CATALOG', 'gpt-5.6-typo')

    with pytest.raises(ValueError, match='HITL_CODEX_MODEL_CATALOG'):
        _build_codex_worker_config('gpt_responses')


def test_an_unnamed_catalog_leaves_the_bundled_one_in_effect(monkeypatch) -> None:
    from nexus.extensions.tasks.hitl.utils.generator_util import _build_codex_worker_config

    monkeypatch.delenv('HITL_CODEX_MODEL_CATALOG', raising=False)

    assert 'model_catalog_json' not in _build_codex_worker_config('gpt_responses')


@pytest.mark.parametrize(('raw', 'expected'), [('0', False), ('1', True)])
def test_the_responses_lite_override_reaches_every_entry(monkeypatch, raw, expected) -> None:
    """The A/B knob for it: the file's own value is the default, and this replaces it."""
    from nexus.extensions.tasks.hitl.utils.generator_util import _build_codex_worker_config

    monkeypatch.setenv('HITL_CODEX_MODEL_CATALOG', 'gpt-5.6-sol')
    monkeypatch.setenv('HITL_CODEX_RESPONSES_LITE', raw)

    catalog = _build_codex_worker_config('gpt_responses')['model_catalog_json']

    assert [entry['use_responses_lite'] for entry in catalog['models']] == [expected]


@pytest.mark.parametrize('version', ['disabled', 'v1', 'v2'])
def test_the_multi_agent_version_reaches_every_entry(monkeypatch, version) -> None:
    """Codex takes these three and drops anything else, so the file's own value is
    replaced rather than merged."""
    from nexus.extensions.tasks.hitl.utils.generator_util import _build_codex_worker_config

    monkeypatch.setenv('HITL_CODEX_MODEL_CATALOG', 'gpt-5.6-sol')
    monkeypatch.setenv('HITL_CODEX_MULTI_AGENT', version)

    catalog = _build_codex_worker_config('gpt_responses')['model_catalog_json']

    assert [entry['multi_agent_version'] for entry in catalog['models']] == [version]


def test_no_multi_agent_version_named_disables_it(monkeypatch) -> None:
    """The catalogs in the assets dir carry v2, which spends 3387 prompt bytes on the
    agent tools and their two developer messages. A run asks for them by name."""
    from nexus.extensions.tasks.hitl.utils.generator_util import _build_codex_worker_config
    from nexus.extensions.tasks.hitl.utils.generator_util.codex import CATALOG_DIR

    monkeypatch.setenv('HITL_CODEX_MODEL_CATALOG', 'gpt-5.6-sol')
    monkeypatch.delenv('HITL_CODEX_MULTI_AGENT', raising=False)

    catalog = _build_codex_worker_config('gpt_responses')['model_catalog_json']

    assert json.loads((CATALOG_DIR / 'gpt-5.6-sol.json').read_text())['models'][0]['multi_agent_version'] == 'v2'
    assert [entry['multi_agent_version'] for entry in catalog['models']] == ['disabled']


def test_an_unknown_multi_agent_version_is_refused(monkeypatch) -> None:
    """Codex parses an unknown value back as null, which silently leaves the agent
    tools in whatever state its own default decides."""
    from nexus.extensions.tasks.hitl.utils.generator_util import _build_codex_worker_config

    monkeypatch.setenv('HITL_CODEX_MODEL_CATALOG', 'gpt-5.6-sol')
    monkeypatch.setenv('HITL_CODEX_MULTI_AGENT', 'off')

    with pytest.raises(ValueError, match='HITL_CODEX_MULTI_AGENT'):
        _build_codex_worker_config('gpt_responses')


def test_neither_user_input_tool_is_offered_by_default(monkeypatch) -> None:
    """Both are answerable only by a human at a terminal, and the async one fails
    silently: Codex answers ``{"accepted":true}`` and emits the question as an
    ``agent_message``, which reaches the simulator as speech with no question tag.

    They leave the wire in two different places, so one knob sets both: the async one is
    a catalog entry, the synchronous one a key the backend writes into ``config.toml``.
    """
    from nexus.extensions.tasks.hitl.utils.generator_util import _build_codex_worker_config
    from nexus.extensions.tasks.hitl.utils.generator_util.codex import ASYNC_USER_INPUT_TOOL, CATALOG_DIR

    monkeypatch.setenv('HITL_CODEX_MODEL_CATALOG', 'gpt-6.1-sol')
    monkeypatch.delenv('HITL_CODEX_USER_INPUT_TOOLS', raising=False)

    config = _build_codex_worker_config('gpt_responses')

    on_disk = json.loads((CATALOG_DIR / 'gpt-6.1-sol.json').read_text())
    assert ASYNC_USER_INPUT_TOOL in on_disk['models'][0]['experimental_supported_tools']
    assert [entry['experimental_supported_tools'] for entry in config['model_catalog_json']['models']] == [['clock']]
    assert config['user_input_tools'] is False


def test_a_run_can_ask_for_the_user_input_tools(monkeypatch) -> None:
    """Asking for them leaves the catalog entry as the file has it."""
    from nexus.extensions.tasks.hitl.utils.generator_util import _build_codex_worker_config
    from nexus.extensions.tasks.hitl.utils.generator_util.codex import ASYNC_USER_INPUT_TOOL

    monkeypatch.setenv('HITL_CODEX_MODEL_CATALOG', 'gpt-6.1-sol')
    monkeypatch.setenv('HITL_CODEX_USER_INPUT_TOOLS', '1')

    config = _build_codex_worker_config('gpt_responses')

    assert ASYNC_USER_INPUT_TOOL in config['model_catalog_json']['models'][0]['experimental_supported_tools']
    assert config['user_input_tools'] is True
