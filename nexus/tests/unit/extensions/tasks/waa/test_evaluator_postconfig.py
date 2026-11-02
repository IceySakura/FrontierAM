import pytest

from nexus.extensions.tasks.waa.evaluator import evaluate_task


class FakeRuntime:
    def __init__(self, missing_paths=()):
        self.missing_paths = {path.lower() for path in missing_paths}
        self.opened_paths = []

    async def setup_open_file(self, path):
        self.opened_paths.append(path)
        if path.lower() in self.missing_paths:
            raise RuntimeError(f'WAA setup open_file failed: HTTP 404 File not found: {path}')


@pytest.mark.asyncio
async def test_missing_result_file_in_postconfig_scores_zero():
    result_path = r'C:\Users\Docker\Documents\example_count.txt'
    runtime = FakeRuntime(missing_paths={result_path})

    score = await evaluate_task(
        runtime,
        {
            'postconfig': [
                {'type': 'open', 'parameters': {'path': r'C:\Users\Docker\Documents\largefile.txt'}},
                {'type': 'open', 'parameters': {'path': result_path}},
            ],
            'func': 'compare_text_file',
            'result': {'type': 'vm_file', 'path': result_path, 'dest': 'example_count.txt'},
        },
        'DONE',
    )

    assert score == 0.0
    assert runtime.opened_paths == [r'C:\Users\Docker\Documents\largefile.txt', result_path]


@pytest.mark.asyncio
async def test_missing_non_result_file_in_postconfig_still_fails():
    missing_path = r'C:\Users\Docker\Documents\largefile.txt'
    runtime = FakeRuntime(missing_paths={missing_path})

    with pytest.raises(RuntimeError, match='HTTP 404 File not found'):
        await evaluate_task(
            runtime,
            {
                'postconfig': [
                    {'type': 'open', 'parameters': {'path': missing_path}},
                ],
                'func': 'compare_text_file',
                'result': {
                    'type': 'vm_file',
                    'path': r'C:\Users\Docker\Documents\example_count.txt',
                    'dest': 'example_count.txt',
                },
            },
            'DONE',
        )
