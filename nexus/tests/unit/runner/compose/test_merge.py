from nexus.runner.compose.merge import deep_merge, set_path


def test_deep_merge_replaces_lists_by_default():
    merged = deep_merge({'imports': ['a'], 'nested': {'x': 1}}, {'imports': ['b'], 'nested': {'y': 2}})

    assert merged == {'imports': ['b'], 'nested': {'x': 1, 'y': 2}}


def test_deep_merge_append_unique_list_op():
    merged = deep_merge({'imports': ['a']}, {'imports': ['a', 'b']}, {'imports': 'append_unique'})

    assert merged == {'imports': ['a', 'b']}


def test_set_path_append():
    merged = set_path({'imports': ['a']}, 'imports', 'b', list_op='append')

    assert merged == {'imports': ['a', 'b']}
