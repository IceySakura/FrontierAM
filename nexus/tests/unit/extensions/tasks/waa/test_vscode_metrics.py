from nexus.extensions.tasks.waa.metrics.vscode import check_json_settings


def test_check_json_settings_matches_expected(tmp_path):
    settings = tmp_path / 'settings.json'
    settings.write_text('{"workbench.editor.wrapTabs": true}', encoding='utf-8')

    score = check_json_settings(str(settings), {'expected': {'workbench.editor.wrapTabs': True}})

    assert score == 1.0


def test_check_json_settings_invalid_json_returns_zero(tmp_path):
    settings = tmp_path / 'settings.json'
    settings.write_text('{"workbench.editor.wrapTabs": true}\n{"extra": true}', encoding='utf-8')

    score = check_json_settings(str(settings), {'expected': {'workbench.editor.wrapTabs': True}})

    assert score == 0.0
