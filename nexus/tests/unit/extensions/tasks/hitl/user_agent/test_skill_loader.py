"""Loading skills from directories, and what the simulator sees of them.

The cases here are the ones that would otherwise surface as a skill the
simulator can read in its catalogue but cannot invoke, or as files it is told
about that never arrived. Each check runs on the host before a sandbox exists,
so a misconfigured skill fails at assembly rather than on some later turn.
"""

import re

from pathlib import Path

import pytest

from nexus.extensions.tasks.hitl.user_agent.skills import load_user_skills
from nexus.extensions.tasks.hitl.user_agent.skills.loader import (
    MAX_DESCRIPTION_LEN,
    MAX_NAME_LEN,
    SANDBOX_ROOT,
    SkillError,
    catalogue,
    load_skills,
    read_skill,
    sandbox_dir_for,
    split_frontmatter,
    upload_requests,
)


def write_skill(
    parent: Path,
    dirname: str,
    *,
    name: str | None = None,
    description: str = 'what it does',
    body: str = 'do the thing',
    extra: dict[str, str] | None = None,
) -> Path:
    directory = parent / dirname
    directory.mkdir(parents=True, exist_ok=True)
    front = f'---\nname: {dirname if name is None else name}\ndescription: {description}\n---\n'
    (directory / 'SKILL.md').write_text(f'{front}\n{body}\n', encoding='utf-8')
    for relative, content in (extra or {}).items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
    return directory


def test_frontmatter_is_split_off_so_the_body_carries_no_yaml(tmp_path: Path) -> None:
    directory = write_skill(tmp_path, 'hunter', description='find bugs', body='# Hunter\n\nlook closely')

    skill = read_skill(directory)

    assert skill.description == 'find bugs'
    assert skill.body.startswith('# Hunter')
    assert 'description:' not in skill.body


def test_split_frontmatter_leaves_text_without_a_block_alone() -> None:
    fields, body = split_frontmatter('# no frontmatter\n\nbody')

    assert fields == {}
    assert body == '# no frontmatter\n\nbody'


def test_body_keeps_a_bare_triple_dash_that_follows_the_block() -> None:
    text = '---\nname: n\ndescription: d\n---\nfirst\n\n---\n\nsecond\n'

    fields, body = split_frontmatter(text)

    assert fields['name'] == 'n'
    assert 'second' in body
    assert body.count('---') == 1


def test_a_folded_description_reads_as_the_text_it_folds() -> None:
    """The real shape this stopped being hand-parsed for.

    ``hytuner/.cursor/skills/ray-remote-debug`` writes its description as a YAML
    folded scalar. Read line by line, ``description`` is the marker ``>-`` and the
    catalogue offers that as the skill's one-line summary.
    """
    text = '---\nname: n\ndescription: >-\n  first half,\n  second half\n---\nbody\n'

    fields, body = split_frontmatter(text)

    assert fields['description'] == 'first half, second half'
    assert body.strip() == 'body'


def test_a_quoted_description_keeps_the_quotes_it_escaped() -> None:
    """``deployment/build_for_ags/skills/xlsx`` quotes inside a quoted scalar."""
    text = '---\nname: n\ndescription: "she said \\"go\\" and left"\n---\nbody\n'

    fields, _ = split_frontmatter(text)

    assert fields['description'] == 'she said "go" and left'


def test_a_multiline_description_arrives_as_one_line(tmp_path: Path) -> None:
    """The catalogue is one line per skill, so a wrapped description must fold."""
    directory = tmp_path / 'wrapped'
    directory.mkdir()
    (directory / 'SKILL.md').write_text(
        '---\nname: wrapped\ndescription: |\n  one\n  two\n---\nbody\n', encoding='utf-8'
    )

    skill = read_skill(directory)

    assert skill.description == 'one two'
    assert '\n' not in skill.description


def test_frontmatter_that_is_not_a_mapping_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / 'listy'
    directory.mkdir()
    (directory / 'SKILL.md').write_text('---\n- a\n- b\n---\nbody\n', encoding='utf-8')

    with pytest.raises(SkillError, match='YAML mapping'):
        read_skill(directory)


def test_invalid_yaml_names_the_manifest(tmp_path: Path) -> None:
    directory = tmp_path / 'broken'
    directory.mkdir()
    (directory / 'SKILL.md').write_text('---\nname: n\ndescription: a: b: c\n---\nbody\n', encoding='utf-8')

    with pytest.raises(SkillError, match='not valid YAML'):
        read_skill(directory)


@pytest.mark.parametrize(('field', 'limit'), [('name', MAX_NAME_LEN), ('description', MAX_DESCRIPTION_LEN)])
def test_the_upstream_length_limits_apply(tmp_path: Path, field: str, limit: int) -> None:
    """A skill this framework accepts must be one every other consumer accepts."""
    directory = write_skill(tmp_path, 'long', **{field: 'x' * (limit + 1)})

    with pytest.raises(SkillError, match=f'exceeds {limit} characters'):
        read_skill(directory)


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        ({'name': ''}, 'missing field `name`'),
        ({'description': ''}, 'missing field `description`'),
        ({'name': 'other'}, 'does not match directory'),
        ({'body': ''}, 'body is empty'),
    ],
)
def test_a_skill_the_simulator_could_not_invoke_fails_at_load(tmp_path: Path, kwargs: dict, expected: str) -> None:
    directory = write_skill(tmp_path, 'target', **kwargs)

    with pytest.raises(SkillError, match=re.escape(expected)):
        read_skill(directory)


def test_a_directory_without_a_manifest_is_not_a_skill(tmp_path: Path) -> None:
    (tmp_path / 'bare').mkdir()

    with pytest.raises(SkillError, match='no SKILL.md'):
        read_skill(tmp_path / 'bare')


def test_scanning_skips_directories_that_hold_no_manifest(tmp_path: Path) -> None:
    write_skill(tmp_path, 'real')
    (tmp_path / 'notes').mkdir()
    (tmp_path / 'notes' / 'readme.txt').write_text('x', encoding='utf-8')

    assert sorted(load_skills(external_dirs=[tmp_path])) == ['real']


def test_files_list_excludes_the_manifest_whose_content_is_already_the_body(tmp_path: Path) -> None:
    directory = write_skill(tmp_path, 'browser', extra={'setup.sh': 'echo hi', 'fixtures/page.html': '<html>'})

    skill = read_skill(directory)

    assert skill.files == ('fixtures/page.html', 'setup.sh')
    assert 'SKILL.md' not in skill.files


def test_render_tells_the_simulator_where_the_files_are(tmp_path: Path) -> None:
    directory = write_skill(tmp_path, 'browser', extra={'setup.sh': 'echo hi'})

    rendered = read_skill(directory).render()

    assert '<skill name="browser">' in rendered
    assert f'<skill_files dir="{SANDBOX_ROOT}/browser">' in rendered
    assert 'setup.sh' in rendered


def test_render_omits_the_files_block_when_a_skill_is_text_only(tmp_path: Path) -> None:
    directory = write_skill(tmp_path, 'advice')

    rendered = read_skill(directory).render()

    assert 'skill_files' not in rendered


def test_external_may_replace_a_builtin_of_the_same_name(tmp_path: Path) -> None:
    builtin = tmp_path / 'builtin'
    external = tmp_path / 'external'
    write_skill(builtin, 'hunter', description='builtin version')
    write_skill(external, 'hunter', description='external version')

    skills = load_skills(builtin_dir=builtin, external_dirs=[external])

    assert skills['hunter'].description == 'external version'
    assert skills['hunter'].origin == 'external'


def test_two_external_directories_offering_one_name_is_a_configuration_error(tmp_path: Path) -> None:
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_skill(first, 'hunter')
    write_skill(second, 'hunter')

    with pytest.raises(SkillError, match='two external directories'):
        load_skills(external_dirs=[first, second])


def test_a_missing_external_directory_is_not_silently_skipped(tmp_path: Path) -> None:
    with pytest.raises(SkillError, match='not found'):
        load_skills(external_dirs=[tmp_path / 'absent'])


def test_catalogue_lists_names_and_descriptions_without_bodies(tmp_path: Path) -> None:
    write_skill(tmp_path, 'beta', description='second', body='BETA BODY')
    write_skill(tmp_path, 'alpha', description='first', body='ALPHA BODY')

    menu = catalogue(load_skills(external_dirs=[tmp_path]))

    assert menu == '- alpha: first\n- beta: second'
    assert 'BODY' not in menu


def test_catalogue_of_nothing_is_empty() -> None:
    assert catalogue({}) == ''


def test_sandbox_root_is_absolute_because_upload_file_does_not_expand_tilde() -> None:
    assert SANDBOX_ROOT.startswith('/')

    with pytest.raises(SkillError, match='must be absolute'):
        sandbox_dir_for('hunter', root='~/.hitl_user/skills')


def test_upload_requests_place_every_file_under_the_sandbox_directory(tmp_path: Path) -> None:
    write_skill(tmp_path, 'browser', extra={'setup.sh': 'echo hi', 'fixtures/page.html': '<html>'})
    skills = load_skills(external_dirs=[tmp_path])

    items = upload_requests(skills.values())

    paths = sorted(item.path for item in items)
    assert paths == [
        f'{SANDBOX_ROOT}/browser/SKILL.md',
        f'{SANDBOX_ROOT}/browser/fixtures/page.html',
        f'{SANDBOX_ROOT}/browser/setup.sh',
    ]
    assert all(item.path.startswith('/') for item in items)


def test_upload_requests_send_text_as_text_and_binaries_as_base64(tmp_path: Path) -> None:
    directory = write_skill(tmp_path, 'browser')
    (directory / 'logo.png').write_bytes(b'\x89PNG\r\n\x1a\n\xff\xfe')
    skills = load_skills(external_dirs=[tmp_path])

    encodings = {Path(item.path).name: item.encoding for item in upload_requests(skills.values())}

    assert encodings['SKILL.md'] == 'utf8'
    assert encodings['logo.png'] == 'base64'


# --- the skills that ship ---------------------------------------------------


def test_every_shipped_skill_loads() -> None:
    """A built-in that fails to load fails every run, not one configuration."""
    skills = load_user_skills()

    assert sorted(skills) == ['verify-tests']
    for skill in skills.values():
        assert skill.origin == 'builtin'


def test_a_shipped_skill_names_no_command_to_copy() -> None:
    """A skill says what is worth doing; the simulator decides how.

    Two costs measured on the version that did carry commands. The dataset spans
    11 languages and Python is 21% of it, so an invocation written out serves one
    slice and misleads the rest. And a command block is copied rather than read,
    which turns one way of doing a thing into the only way.
    """
    for name, skill in load_user_skills().items():
        assert '```' not in skill.body, f'{name} carries a command block'


@pytest.mark.parametrize('destructive', ['git stash', 'git reset', 'git checkout', 'git clean', 'git commit'])
def test_no_shipped_skill_names_a_command_that_moves_the_worker_s_work(destructive: str) -> None:
    """The worker's uncommitted work is the only copy of the run's answer.

    Measured: after ``git stash`` with no ``pop``, the extraction the task runs
    (``git add -A`` then ``git diff <base>``) reports only the untracked files --
    every tracked edit sits in the stash and is absent from the patch. Reaching
    the pre-handover state is a reasonable thing for a skill to suggest, so this
    holds the line at the forms of it that destroy the run.
    """
    for name, skill in load_user_skills().items():
        assert destructive not in skill.body, f'{name} names {destructive!r}'
