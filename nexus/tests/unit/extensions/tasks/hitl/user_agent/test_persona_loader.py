"""Reading a persona from a file, and what a malformed one says.

A persona is data -- strings and tuples of strings -- so a file carries the whole of
one and adding a persona is not a code change. The frontmatter list fields are where
that goes wrong: an unquoted ``Label: action`` parses as a YAML mapping, not a string,
and a dict reaching the prompt would be a persona nobody wrote.

These cover the shape a file may take, the defaults that let a bare prompt be a
persona, and each refusal that keeps a silent misparse from reaching a run.
"""

from pathlib import Path

import pytest

from nexus.extensions.tasks.hitl.user_agent.personas import get_persona, registered_personas
from nexus.extensions.tasks.hitl.user_agent.personas.loader import (
    BASE_STEM,
    BUILTIN_DIR,
    load_personas,
    read_persona,
)
from nexus.extensions.tasks.hitl.user_agent.personas.registry import DEFAULT_TOOLS, PersonaError


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / f'{name}.md'
    path.write_text(text, encoding='utf-8')
    return path


# -- what a file may leave out --------------------------------------------


def test_a_file_holding_only_a_prompt_is_a_persona(tmp_path: Path) -> None:
    """The name comes from the filename and every other field has a default, so the
    smallest persona anyone can write is one paragraph.
    """
    path = write(tmp_path, 'terse', 'You want the change and you do not explain twice.\n')

    persona = read_persona(path)

    assert persona.name == 'terse'
    assert persona.tools == DEFAULT_TOOLS
    assert persona.recalls == ()
    assert persona.prompt.startswith('You want the change')


def test_empty_frontmatter_is_the_same_as_none(tmp_path: Path) -> None:
    path = write(tmp_path, 'terse', '---\n---\n\nA prompt.\n')

    assert read_persona(path).prompt == 'A prompt.'


def test_the_declared_fields_reach_the_persona(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        'reviewer',
        '---\n'
        'tools: [send_message, accept]\n'
        'recalls:\n'
        '  - "Asking: one thing at a time."\n'
        '---\n'
        '\n'
        'You review.\n',
    )

    persona = read_persona(path)

    assert persona.tools == ('send_message', 'accept')
    assert persona.recalls == ('Asking: one thing at a time.',)


def test_a_quoted_entry_keeps_its_colon(tmp_path: Path) -> None:
    """The built-ins are written this way, and the colon is part of the sentence."""
    path = write(tmp_path, 'p', '---\nrecalls:\n  - "Accepting: use the thing yourself first."\n---\n\nBody.\n')

    assert read_persona(path).recalls == ('Accepting: use the thing yourself first.',)


# -- the refusals ----------------------------------------------------------


def test_an_unquoted_colon_says_to_quote_it(tmp_path: Path) -> None:
    """The failure this loader exists to catch: YAML reads it as a mapping."""
    path = write(tmp_path, 'p', '---\nrecalls:\n  - Asking: one thing at a time\n---\n\nBody.\n')

    with pytest.raises(PersonaError, match='unquoted colon'):
        read_persona(path)


def test_a_list_field_given_one_string_is_refused(tmp_path: Path) -> None:
    """``recalls: a line`` would otherwise become one character per entry."""
    path = write(tmp_path, 'p', '---\nrecalls: just one line\n---\n\nBody.\n')

    with pytest.raises(PersonaError, match='must be a list'):
        read_persona(path)


def test_an_unknown_frontmatter_key_is_refused_by_name(tmp_path: Path) -> None:
    """A misspelt key would otherwise be a field silently left at its default."""
    path = write(tmp_path, 'p', '---\nrecal1s:\n  - "x: y"\n---\n\nBody.\n')

    with pytest.raises(PersonaError, match='recal1s'):
        read_persona(path)


def test_a_name_disagreeing_with_the_filename_is_refused(tmp_path: Path) -> None:
    """The name is how a run asks for it; two answers hide which one wins."""
    path = write(tmp_path, 'p', '---\nname: other\n---\n\nBody.\n')

    with pytest.raises(PersonaError, match='does not match'):
        read_persona(path)


def test_a_file_with_no_prompt_is_refused(tmp_path: Path) -> None:
    """The body is the prompt, so an empty one is a persona that says nothing."""
    path = write(tmp_path, 'p', '---\ntools: [accept]\n---\n\n\n')

    with pytest.raises(PersonaError, match='empty'):
        read_persona(path)


def test_broken_yaml_names_the_file(tmp_path: Path) -> None:
    path = write(tmp_path, 'p', '---\nrecalls: [unclosed\n---\n\nBody.\n')

    with pytest.raises(PersonaError, match='not valid YAML'):
        read_persona(path)


def test_a_missing_directory_is_refused_rather_than_skipped(tmp_path: Path) -> None:
    """A typo in the configured path would otherwise run with the built-ins only."""
    with pytest.raises(PersonaError, match='not found'):
        load_personas(builtin_dir=None, external_dirs=[tmp_path / 'nope'])


def test_one_name_from_two_directories_is_refused(tmp_path: Path) -> None:
    """Neither is preferable to the other, so picking one would be arbitrary."""
    first = tmp_path / 'a'
    second = tmp_path / 'b'
    for parent in (first, second):
        parent.mkdir()
        write(parent, 'same', 'A prompt.\n')

    with pytest.raises(PersonaError, match='defined twice'):
        load_personas(builtin_dir=None, external_dirs=[first, second])


# -- built-ins and replacement --------------------------------------------


def test_the_built_ins_are_files_now(tmp_path: Path) -> None:
    """Both are read from disk, so nothing about them needs a module."""
    del tmp_path
    names = sorted(persona.name for persona in load_personas().values())

    assert names == ['blank', 'default']
    assert not sorted(BUILTIN_DIR.glob('*.py'))


# -- a persona written as a directory --------------------------------------


def build_directory(parent: Path, name: str, base: str, **variants: str) -> Path:
    """A persona directory: its base file, and one file per language beside it."""
    directory = parent / name
    directory.mkdir()
    (directory / f'{BASE_STEM}.md').write_text(base, encoding='utf-8')
    for code, text in variants.items():
        (directory / f'{BASE_STEM}.{code}.md').write_text(text, encoding='utf-8')
    return directory


def test_a_directory_is_a_persona_named_after_itself(tmp_path: Path) -> None:
    directory = build_directory(tmp_path, 'terse', 'You want the change.\n')

    persona = read_persona(directory)

    assert persona.name == 'terse'
    assert persona.prompt == 'You want the change.'
    assert persona.variants == {}


def test_a_language_file_reaches_the_prompt_only_for_its_own_language(tmp_path: Path) -> None:
    """A variant read into the persona and never appended would be a file nobody sees."""
    directory = build_directory(tmp_path, 'terse', 'You want the change.\n', zh='说话简短。\n')

    persona = read_persona(directory)

    assert persona.variants == {'zh': '说话简短。'}
    assert '说话简短。' in persona.for_language('zh').prompt
    assert persona.for_language('en') is persona


def test_a_language_file_is_added_to_the_base_rather_than_replacing_it(tmp_path: Path) -> None:
    directory = build_directory(tmp_path, 'terse', 'You want the change.\n', en='Say it once.\n')

    prompt = read_persona(directory).for_language('en').prompt

    assert 'You want the change.' in prompt
    assert 'Say it once.' in prompt


def test_a_directory_with_no_base_file_is_refused(tmp_path: Path) -> None:
    """Otherwise it would load as a persona with nothing in it."""
    directory = tmp_path / 'terse'
    directory.mkdir()
    (directory / f'{BASE_STEM}.zh.md').write_text('说话简短。\n', encoding='utf-8')

    with pytest.raises(PersonaError, match=BASE_STEM):
        read_persona(directory)


def test_an_empty_language_file_is_refused(tmp_path: Path) -> None:
    directory = build_directory(tmp_path, 'terse', 'You want the change.\n', zh='   \n')

    with pytest.raises(PersonaError, match='nothing in it'):
        read_persona(directory)


def test_a_directory_and_a_file_load_side_by_side(tmp_path: Path) -> None:
    build_directory(tmp_path, 'folded', 'A prompt.\n')
    write(tmp_path, 'flat', 'Another prompt.\n')

    personas = load_personas(builtin_dir=None, external_dirs=[tmp_path])

    assert sorted(personas) == ['flat', 'folded']


def test_a_subdirectory_that_is_not_a_persona_is_passed_over(tmp_path: Path) -> None:
    """A directory holding personas also holds ``__pycache__`` and whatever else. Every
    one of those reaching ``read_persona`` fails the whole load.
    """
    write(tmp_path, 'mine', 'A prompt.\n')
    for name in ('__pycache__', 'fixtures'):
        (tmp_path / name).mkdir()
    (tmp_path / '__pycache__' / 'x.pyc').write_bytes(b'\0')

    assert sorted(load_personas(builtin_dir=None, external_dirs=[tmp_path])) == ['mine']


def test_the_built_in_default_is_a_directory(tmp_path: Path) -> None:
    """Where its per-language text goes when it is written."""
    del tmp_path

    assert (BUILTIN_DIR / 'default' / f'{BASE_STEM}.md').is_file()


def test_the_default_persona_still_carries_what_the_run_needs() -> None:
    """Its declarations drive the tool set and the reminders, not just the prompt."""
    persona = get_persona('default')

    assert persona.tools == DEFAULT_TOOLS
    assert persona.recalls


def test_a_persona_that_minds_nothing_stays_that_way() -> None:
    persona = get_persona('blank')

    assert persona.reads_memory is False


def test_an_external_file_replaces_a_built_in_of_the_same_name(tmp_path: Path) -> None:
    """How a run words the default persona for its own topic."""
    write(tmp_path, 'default', 'A different person entirely.\n')

    personas = load_personas(external_dirs=[tmp_path])

    assert personas['default'].prompt == 'A different person entirely.'
    assert 'blank' in personas, 'the other built-in is untouched'


def test_the_registry_reports_what_a_directory_added(tmp_path: Path) -> None:
    """The generator checks the configured name against this, before the batch runs."""
    from nexus.extensions.tasks.hitl.user_agent.personas import load_persona_dirs

    write(tmp_path, 'probe_only_here', 'A prompt.\n')

    added = load_persona_dirs([tmp_path])

    assert added == ('probe_only_here',)
    assert 'probe_only_here' in registered_personas()
