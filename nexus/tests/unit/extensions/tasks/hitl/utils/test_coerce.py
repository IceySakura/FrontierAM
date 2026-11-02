"""Tests for reading typed values out of untyped session config."""

import pytest

from nexus.extensions.tasks.hitl.utils import coerce


# --- optional_dict ---------------------------------------------------------


def test_optional_dict_stringifies_keys():
    assert coerce.optional_dict({1: 'a', 'b': 2}) == {'1': 'a', 'b': 2}


@pytest.mark.parametrize('value', [None, 'x', 5, [], ('a', 'b')])
def test_optional_dict_returns_none_for_non_mappings(value):
    assert coerce.optional_dict(value) is None


def test_optional_dict_accepts_any_mapping():
    from types import MappingProxyType

    assert coerce.optional_dict(MappingProxyType({'a': 1})) == {'a': 1}


def test_optional_dict_copies():
    source = {'a': 1}
    result = coerce.optional_dict(source)
    assert result is not None
    result['b'] = 2
    assert source == {'a': 1}


# --- optional_string / optional_bool ---------------------------------------


def test_optional_string_passes_strings_through_including_empty():
    assert coerce.optional_string('x') == 'x'
    assert coerce.optional_string('') == ''
    assert coerce.optional_string(5) is None
    assert coerce.optional_string(None) is None


def test_optional_bool_accepts_only_real_bools():
    assert coerce.optional_bool(True, default=False) is True
    assert coerce.optional_bool(False, default=True) is False
    # A config meaning to switch something on has to say so unambiguously.
    assert coerce.optional_bool('true', default=False) is False
    assert coerce.optional_bool(1, default=False) is False
    assert coerce.optional_bool(None, default=True) is True


# --- optional_int ----------------------------------------------------------


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (7, 7),
        (-3, -3),
        (0, 0),
        ('7', 7),
        ('  7  ', 7),
        ('-2', -2),
        ('', None),
        ('   ', None),
        ('x', None),
        ('7.5', None),
        (7.5, None),
        (None, None),
        ([], None),
    ],
)
def test_optional_int(value, expected):
    assert coerce.optional_int(value) == expected


def test_optional_int_rejects_bools_despite_the_int_subclass():
    assert coerce.optional_int(True) is None
    assert coerce.optional_int(False) is None


# --- optional_float --------------------------------------------------------


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (7, 7.0),
        (7.5, 7.5),
        ('7.5', 7.5),
        ('  2 ', 2.0),
        ('x', 99.0),
        ('', 99.0),
        (None, 99.0),
        (True, 99.0),
        ([], 99.0),
    ],
)
def test_optional_float(value, expected):
    assert coerce.optional_float(value, default=99.0) == expected


# --- optional_str_list / string_map ---------------------------------------


def test_optional_str_list_stringifies_items():
    assert coerce.optional_str_list(['a', 1, True]) == ['a', '1', 'True']
    assert coerce.optional_str_list([]) == []


@pytest.mark.parametrize('value', [None, 'abc', {'a': 1}, 5])
def test_optional_str_list_returns_none_for_non_lists(value):
    # A bare string is not a list of arguments; splitting it would be a guess.
    assert coerce.optional_str_list(value) is None


def test_string_map_stringifies_both_sides():
    assert coerce.string_map({'A': 1, 2: True}) == {'A': '1', '2': 'True'}


@pytest.mark.parametrize('value', [None, 'x', 5, []])
def test_string_map_is_empty_for_non_mappings(value):
    assert coerce.string_map(value) == {}


# --- pick ------------------------------------------------------------------


def test_pick_keeps_only_whitelisted_keys_that_are_set():
    config = {'a': 1, 'b': None, 'c': 3, 'd': 4}

    assert coerce.pick(config, {'a', 'b', 'c'}) == {'a': 1, 'c': 3}


def test_pick_keeps_falsy_values_that_are_not_none():
    config = {'zero': 0, 'empty': '', 'false': False, 'missing': None}

    assert coerce.pick(config, config) == {'zero': 0, 'empty': '', 'false': False}


def test_pick_tolerates_keys_absent_from_the_config():
    assert coerce.pick({'a': 1}, ['a', 'nope']) == {'a': 1}
