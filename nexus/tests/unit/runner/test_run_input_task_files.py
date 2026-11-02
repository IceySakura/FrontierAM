from nexus.runner.single import RunInput


def test_run_input_file_accepts_relative_task_files_directory(tmp_path):
    files = tmp_path / 'files'
    files.mkdir()
    config = tmp_path / 'run.yaml'
    config.write_text(
        """
task_name: identity
task_config: {}
task_files: ./files
""".lstrip(),
        encoding='utf-8',
    )

    run_input = RunInput.from_file(config)

    assert run_input.task_files is not None
    assert run_input.task_files.type == 'directory'
    assert run_input.task_files.path == str(files)
