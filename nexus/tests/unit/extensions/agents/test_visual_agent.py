"""Tests for the ``visual_agent`` delegation layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pydantic import ValidationError

from nexus.core import ExecutionContext
from nexus.extensions.agents.main.agent import MainAgentInput, MainAgentOutput, MainExitStatus
from nexus.extensions.agents.visual_agent.agent import (
    VisualAgentInput,
    VisualAgentOutput,
    run_visual_agent,
)
from nexus.extensions.agents.visual_agent.resources import images as images_module
from nexus.extensions.agents.visual_agent.tools import DEFAULT_VISUAL_TOOLS
from nexus.runtime.executors.models.file import FileOperationResult, UploadFileResponse


def _media_file(tmp_path: Path, name: str, size: tuple[int, int] | None = None) -> Path:
    path = tmp_path / name
    if size:
        from PIL import Image

        Image.new('RGB', size, 'white').save(path)
    else:
        path.write_bytes(b'\x89PNG\r\n\x1a\nfake' if path.suffix in {'.png', '.jpg'} else b'fixture')
    return path


async def _run_with_task_data(user_prompt, history=None, **config):
    async with ExecutionContext(task_config={'user_prompt': user_prompt, 'history': history}):
        return await run_visual_agent(VisualAgentInput(user_prompt=user_prompt, history=history, **config))


async def test_visual_defaults_reach_main_agent(captured_main_input: list[MainAgentInput]) -> None:
    await _run_with_task_data(user_prompt='what is in this image?')

    (main_input,) = captured_main_input
    assert main_input.tools == list(DEFAULT_VISUAL_TOOLS)
    assert main_input.parallel_tool_calls is True


async def test_explicit_values_override_defaults(
    captured_main_input: list[MainAgentInput],
) -> None:
    await _run_with_task_data(
        user_prompt='hi',
        system_prompt='custom prompt',
        tools=['jupyter_with_image', 'web_search'],
        image_history_limit=2,
        multimodal_tool_output=False,
        tool_error_continue=True,
        max_iterations=3,
        max_tool_response_tokens=4096,
        warmup_runtime=True,
        openai_client_key='vision',
    )

    (main_input,) = captured_main_input
    assert main_input.system_prompt == 'custom prompt'
    assert main_input.tools == ['jupyter_with_image', 'web_search']
    assert main_input.image_history_limit == 2
    assert main_input.multimodal_tool_output is False
    assert main_input.tool_error_continue is True
    assert main_input.max_iterations == 3
    assert main_input.max_tool_response_tokens == 4096
    assert main_input.warmup_runtime is True
    assert main_input.openai_client_key == 'vision'


async def test_empty_tools_list_falls_back_to_default(
    captured_main_input: list[MainAgentInput],
) -> None:
    await _run_with_task_data(user_prompt='hi', tools=[])

    (main_input,) = captured_main_input
    assert main_input.tools == list(DEFAULT_VISUAL_TOOLS)


@pytest.mark.parametrize('position', [None, 'before', 'after'])
@pytest.mark.parametrize('inline', [False, True])
async def test_input_image_path_position_preserves_pairs_and_text(
    position,
    inline,
    tmp_path,
    fake_runtime,
    captured_main_input,
):
    import copy

    path = _media_file(tmp_path, 'first.png', (12, 8))
    uri = 'data:image/png;base64,' + images_module.encode_image(str(path))
    refs = [str(path), uri, str(path)]
    prompt = [{'type': 'text', 'text': 'Start'}]
    for index, ref in enumerate(refs):
        prompt.extend([
            {'type': 'image_url', 'image_url': {'url': ref, 'detail': 'high'}},
            {'type': 'text', 'text': f'After {index}'},
        ])
    original = copy.deepcopy(prompt)
    config = {} if position is None else {'input_image_path_position': position}
    await _run_with_task_data(
        user_prompt=prompt,
        inline_images=inline,
        show_image_dimensions=True,
        **config,
    )
    locations = images_module.upload_locations(list(dict.fromkeys(refs)))
    expected = [prompt[0]]
    for index, ref in enumerate(refs):
        note = {
            'type': 'text',
            'text': f'\n![{images_module._label_for(ref, locations[ref])}]({locations[ref]}) (12x8 px)',
        }
        picture = {'type': 'image_url', 'image_url': {'url': uri, 'detail': 'high'}}
        pair = [picture, note] if position == 'after' else [note, picture]
        expected.extend(pair if inline else [note])
        expected.append({'type': 'text', 'text': f'After {index}'})
    assert captured_main_input[0].user_prompt == expected
    assert prompt == original
    assert len(fake_runtime) == 1
    assert len(fake_runtime[0]) == 2


@pytest.mark.parametrize(('paths_enabled', 'runtime_available'), [(False, True), (True, False)])
async def test_after_position_does_not_create_disabled_or_unavailable_path_notes(
    paths_enabled,
    runtime_available,
    fake_runtime,
    captured_main_input,
):
    uri = 'data:image/png;base64,QUJD'
    prompt = [{'type': 'image_url', 'image_url': {'url': uri}}]
    await _run_with_task_data(
        user_prompt=prompt,
        input_image_paths=paths_enabled,
        upload_images_to_runtime=runtime_available,
        input_image_path_position='after',
    )
    assert captured_main_input[0].user_prompt == prompt


def test_input_image_path_position_rejects_unknown_values():
    with pytest.raises(ValidationError, match='input_image_path_position'):
        VisualAgentInput(user_prompt='test', input_image_path_position='middle')


@pytest.mark.parametrize('inline', [False, True])
@pytest.mark.parametrize('phase', ['acquire', 'upload'])
async def test_upload_failure_stops_before_model(
    captured_main_input, fake_runtime, visual_runtime, monkeypatch, tmp_path, inline, phase
):
    png = _media_file(tmp_path, 'shelf.png')

    target, method = (
        (images_module.require_context(), 'get_runtime') if phase == 'acquire' else (visual_runtime, 'upload_file')
    )
    monkeypatch.setattr(target, method, AsyncMock(side_effect=RuntimeError('sandbox unavailable')))
    with pytest.raises(RuntimeError, match='sandbox unavailable'):
        await _run_with_task_data(
            user_prompt=[{'type': 'image_url', 'image_url': {'url': str(png)}}],
            inline_images=inline,
        )
    assert captured_main_input == []


async def test_main_output_is_forwarded_faithfully(main_delegate) -> None:
    trajectory = [
        {'role': 'system', 'content': 'sys'},
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': 'the answer is 42'},
    ]

    main_delegate.return_value = MainAgentOutput(
        main_trajectory=trajectory,
        exit_status=MainExitStatus.MAX_ITERATIONS.value,
        iterations=7,
        final_answer='the answer is 42',
    )

    output = await _run_with_task_data(user_prompt='q')

    assert isinstance(output, VisualAgentOutput)
    assert output.exit_status == MainExitStatus.MAX_ITERATIONS.value
    assert output.iterations == 7
    assert output.final_answer == 'the answer is 42'
    assert output.main_trajectory == trajectory
    assert output.generated_message_start == 2
    assert output.trajectory_compacted is False
    assert MainExitStatus.termination_reason_for(output.exit_status) == 'MAX_STEPS'


async def test_visual_scoring_boundary_excludes_history_and_preserves_hook_messages(main_delegate):
    from nexus.extensions.tasks.visual.reward import _trajectory_messages

    history = [{'role': 'user', 'content': 'Old question'}, {'role': 'assistant', 'content': 'Old answer'}]
    generated = [
        {'role': 'assistant', 'content': 'Working'},
        {'role': 'user', 'content': 'Hook feedback'},
        {'role': 'assistant', 'content': 'Answer'},
    ]
    main_delegate.return_value = MainAgentOutput(
        main_trajectory=[{'role': 'system'}, *history, {'role': 'user', 'content': 'Question'}, *generated],
        final_answer='Answer',
        exit_status='finished',
    )

    output = await _run_with_task_data(user_prompt='Question', history=history)
    messages, _ = await _trajectory_messages(output, max_tool_images=0, strip_multimodal=False)

    assert output.generated_message_start == len(history) + 2
    assert output.trajectory_compacted is False
    assert messages == generated[:-1]


@pytest.mark.parametrize(('install_during_run', 'compact'), [(False, False), (False, True), (True, True)])
async def test_visual_scoring_detects_compaction_during_this_run(
    main_delegate, monkeypatch, install_during_run, compact
):
    from types import SimpleNamespace

    from nexus.core import require_context
    from nexus.extensions.compaction.compaction_manager import CompactionManager
    from nexus.extensions.tasks.visual.reward import _trajectory_messages

    monkeypatch.setattr(
        'nexus.extensions.compaction.compaction_manager.require_context',
        lambda: SimpleNamespace(openai_client=SimpleNamespace()),
    )

    async def count(messages, tools):
        return len(messages)

    manager = CompactionManager(
        strategy_name='keep_recent', condition_name='always', target_tokens=2, token_count_fn=count
    )
    initial_history = [
        {'role': 'system', 'content': 'System'},
        {'role': 'user', 'content': 'Old question'},
        {'role': 'assistant', 'content': 'Old answer'},
        {'role': 'user', 'content': 'Question'},
    ]
    # A reused manager's earlier compactions must not mark this run as compacted.
    if not install_during_run:
        await manager.maybe_compact(initial_history, tools=[])

    async def delegate(main_input):
        if install_during_run:
            require_context().compaction_manager = manager
        trajectory = initial_history
        if compact:
            trajectory = (await manager.maybe_compact(trajectory, tools=[])).messages
        return MainAgentOutput(
            main_trajectory=[*trajectory, {'role': 'assistant', 'content': 'Answer'}],
            final_answer='Answer',
            exit_status='finished',
        )

    main_delegate.side_effect = delegate
    async with ExecutionContext(compaction_manager=None if install_during_run else manager):
        output = await run_visual_agent(VisualAgentInput(user_prompt='Question', history=initial_history[1:-1]))

    assert output.trajectory_compacted is compact
    if compact:
        assert output.generated_message_start is None
        with pytest.raises(ValueError, match='compacted trajectories'):
            await _trajectory_messages(output, max_tool_images=0, strip_multimodal=False)
    else:
        assert output.generated_message_start == 4
        assert await _trajectory_messages(output, max_tool_images=0, strip_multimodal=False) == ([], 0)


@pytest.mark.parametrize('source_field', ['url', 'file_data'])
async def test_file_part_becomes_a_sandbox_note(
    source_field: str,
    captured_main_input: list[MainAgentInput],
    fake_runtime: list[list[Any]],
    tmp_path: Path,
) -> None:
    pdf = _media_file(tmp_path, 'report.pdf')

    await _run_with_task_data(
        user_prompt=[
            {'type': 'file', 'file': {source_field: str(pdf), 'filename': 'report.pdf'}},
            {'type': 'text', 'text': 'Summarize it.'},
        ]
    )

    (main_input,) = captured_main_input
    sandbox = images_module.upload_locations([str(pdf)])[str(pdf)]
    assert sandbox.endswith('/report.pdf')
    label = 'File report.pdf'
    assert main_input.user_prompt == [
        {'type': 'text', 'text': f'\n[{label} saved at {sandbox}]'},
        {'type': 'text', 'text': 'Summarize it.'},
    ]
    assert all(p['type'] == 'text' for p in main_input.user_prompt)
    (batch,) = fake_runtime
    assert [f.path for f in batch] == [sandbox]


@pytest.mark.parametrize('source_field', ['url', 'file_data'])
async def test_local_file_without_sandbox_stops_before_model(source_field, captured_main_input, no_runtime, tmp_path):
    pdf = _media_file(tmp_path, 'report.pdf')
    with pytest.raises(RuntimeError, match='Local file could not be delivered'):
        await _run_with_task_data(user_prompt=[{'type': 'file', 'file': {source_field: str(pdf)}}])
    assert captured_main_input == []


async def test_file_part_with_a_data_uri_passes_through(
    captured_main_input: list[MainAgentInput],
    no_runtime: None,
) -> None:
    user_prompt: list[dict[str, Any]] = [
        {'type': 'file', 'file': {'filename': 'a.pdf', 'file_data': 'data:application/pdf;base64,QUJD'}},
    ]

    await _run_with_task_data(user_prompt=user_prompt)

    (main_input,) = captured_main_input
    assert main_input.user_prompt == user_prompt


async def test_file_url_wins_and_filename_is_display_only(
    captured_main_input: list[MainAgentInput],
    fake_runtime: list[list[Any]],
    tmp_path: Path,
) -> None:
    import base64

    pdf = _media_file(tmp_path, 'source.pdf')
    label = str(tmp_path / 'display-only.pdf')
    prompt = [{'type': 'file', 'file': {'url': str(pdf), 'filename': label, 'file_data': 'unused'}}]
    await _run_with_task_data(user_prompt=prompt)

    (batch,) = fake_runtime
    (uploaded,) = batch
    assert uploaded.path.endswith('/source.pdf')
    assert base64.b64decode(uploaded.content) == pdf.read_bytes()
    assert captured_main_input[0].user_prompt == [
        {'type': 'text', 'text': f'\n[File {label} saved at {uploaded.path}]'}
    ]
    assert prompt[0]['file']['url'] == str(pdf)


@pytest.mark.parametrize('filename', [None, 'report.pdf'])
async def test_file_url_data_uri_is_adapted_for_backend_without_runtime(
    filename: str | None,
    captured_main_input: list[MainAgentInput],
    no_runtime: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    uri = 'data:application/pdf;base64,QUJD'
    file_obj = {'url': uri}
    expected = {'file_data': uri}
    if filename is not None:
        file_obj['filename'] = filename
        expected['filename'] = filename
    await _run_with_task_data(user_prompt=[{'type': 'file', 'file': file_obj}])

    assert captured_main_input[0].user_prompt == [{'type': 'file', 'file': expected}]
    assert file_obj['url'] == uri
    assert not any('deprecated' in r.message for r in caplog.records)


@pytest.mark.parametrize(
    ('mime', 'suffix'), [('application/pdf', '.pdf'), ('text/csv', '.csv'), ('application/x-custom', '.bin')]
)
async def test_file_url_data_uri_is_uploaded_with_file_extension(
    mime: str,
    suffix: str,
    captured_main_input: list[MainAgentInput],
    fake_runtime: list[list[Any]],
) -> None:
    uri = f'data:{mime};base64,QUJD'
    await _run_with_task_data(user_prompt=[{'type': 'file', 'file': {'url': uri}}])

    (batch,) = fake_runtime
    (uploaded,) = batch
    assert uploaded.path.endswith(f'/file_0{suffix}')
    assert uploaded.content == 'QUJD'
    assert captured_main_input[0].user_prompt == [{'type': 'text', 'text': f'\n[File saved at {uploaded.path}]'}]


@pytest.mark.parametrize('upload_enabled', [False, True])
@pytest.mark.parametrize('inline', [False, True])
async def test_file_url_only_inlines_when_upload_was_not_requested(
    upload_enabled,
    inline,
    captured_main_input,
    fake_runtime,
    monkeypatch,
    tmp_path,
):
    async def fail_upload(*args, **kwargs):
        raise RuntimeError('sandbox unavailable')

    monkeypatch.setattr(images_module, 'upload_images_to_runtime', fail_upload)
    source = 'data:application/pdf;base64,QUJD' if inline else str(_media_file(tmp_path, 'source.pdf'))
    config = {
        'user_prompt': [{'type': 'file', 'file': {'url': source, 'filename': 'Report'}}],
        'upload_images_to_runtime': upload_enabled,
    }
    if upload_enabled or not inline:
        with pytest.raises(RuntimeError):
            await _run_with_task_data(**config)
        assert captured_main_input == []
    else:
        await _run_with_task_data(**config)
        assert captured_main_input[0].user_prompt == [
            {'type': 'file', 'file': {'file_data': source, 'filename': 'Report'}},
        ]
    assert fake_runtime == []


@pytest.mark.parametrize(
    'file_obj',
    [None, {}, {'url': 42}],
)
async def test_invalid_file_url_shape_is_rejected_before_upload(file_obj: Any, fake_runtime: list[list[Any]]) -> None:
    with pytest.raises(ValueError, match=r'file(?:\.url| part)'):
        await images_module.resolve_user_prompt([{'type': 'file', 'file': file_obj}])
    assert fake_runtime == []


@pytest.mark.parametrize('filename', ['  ', 42])
async def test_invalid_file_display_name_is_rejected(filename: Any, fake_runtime: list[list[Any]]) -> None:
    with pytest.raises(ValueError, match=r'file\.filename'):
        await images_module.resolve_user_prompt([
            {'type': 'file', 'file': {'url': 'data:application/pdf;base64,QUJD', 'filename': filename}}
        ])
    assert fake_runtime == []


@pytest.mark.parametrize('url', [None, 'https://example.com/report.pdf'])
async def test_invalid_file_url_does_not_fall_back_to_file_data(
    url: str,
    captured_main_input: list[MainAgentInput],
    fake_runtime: list[list[Any]],
    tmp_path: Path,
) -> None:
    pdf = _media_file(tmp_path, 'fallback.pdf')
    with pytest.raises(ValueError, match='file'):
        await _run_with_task_data(user_prompt=[{'type': 'file', 'file': {'url': url, 'file_data': str(pdf)}}])
    assert fake_runtime == []
    assert captured_main_input == []


async def test_legacy_file_id_still_passes_through(
    captured_main_input: list[MainAgentInput], no_runtime: None, caplog: pytest.LogCaptureFixture
) -> None:
    prompt = [{'type': 'file', 'file': {'file_id': 'file-123'}}]
    await _run_with_task_data(user_prompt=prompt)
    assert captured_main_input[0].user_prompt == prompt
    assert not any('deprecated' in r.message for r in caplog.records)


async def test_repeated_file_urls_keep_order_and_labels_in_mixed_prompt(
    captured_main_input: list[MainAgentInput],
    fake_runtime: list[list[Any]],
    tmp_path: Path,
) -> None:
    pdf = _media_file(tmp_path, 'source.pdf')
    png = _media_file(tmp_path, 'chart.png')
    video = tmp_path / 'clip.mp4'
    video.write_bytes(b'video')
    await _run_with_task_data(
        user_prompt=[
            {'type': 'text', 'text': 'Compare'},
            {'type': 'file', 'file': {'url': str(pdf), 'filename': 'First'}},
            {'type': 'image_url', 'image_url': {'url': str(png)}},
            {'type': 'video_url', 'video_url': {'url': str(video)}},
            {'type': 'file', 'file': {'url': str(pdf), 'filename': 'Again'}},
            {'type': 'text', 'text': 'Explain'},
        ]
    )
    locations = images_module.upload_locations([str(png), str(video), str(pdf)])
    parts = captured_main_input[0].user_prompt
    assert [p['type'] for p in parts] == ['text', 'text', 'text', 'image_url', 'text', 'text', 'text']
    assert [p['text'] for p in parts if p['type'] == 'text'] == [
        'Compare',
        f'\n[File First saved at {locations[str(pdf)]}]',
        f'\n![chart]({locations[str(png)]})',
        f'\n[Video saved at {locations[str(video)]}]',
        f'\n[File Again saved at {locations[str(pdf)]}]',
        'Explain',
    ]
    (batch,) = fake_runtime
    assert len(batch) == 3


async def test_image_and_file_reference_share_one_upload(
    captured_main_input: list[MainAgentInput], fake_runtime: list[list[Any]], tmp_path: Path
) -> None:
    png = _media_file(tmp_path, 'chart.png')
    await _run_with_task_data(
        user_prompt=[
            {'type': 'file', 'file': {'url': str(png)}},
            {'type': 'image_url', 'image_url': {'url': str(png)}},
        ]
    )
    (batch,) = fake_runtime
    assert len(batch) == 1
    assert captured_main_input[0].user_prompt[0]['text'] == f'\n[File saved at {batch[0].path}]'


async def test_dimensions_also_annotate_a_file_part(
    captured_main_input: list[MainAgentInput],
    fake_runtime: list[list[Any]],
    tmp_path: Path,
) -> None:
    png = _media_file(tmp_path, 'board.png', (376, 595))

    await _run_with_task_data(
        user_prompt=[{'type': 'file', 'file': {'file_data': str(png)}}], show_image_dimensions=True
    )

    (main_input,) = captured_main_input
    sandbox = images_module.upload_locations([str(png)])[str(png)]
    assert main_input.user_prompt == [{'type': 'text', 'text': f'\n[File saved at {sandbox} (376x595 px)]'}]


def test_dimensions_match_raw_sandbox_image_with_exif(tmp_path: Path) -> None:
    from PIL import Image

    img = Image.new('RGB', (400, 100), 'white')
    exif = img.getexif()
    exif[274] = 6  # rotate 90° CW on display
    path = tmp_path / 'rotated.jpg'
    img.save(path, exif=exif)

    assert images_module.image_dimensions(str(path)) == (400, 100)


@pytest.mark.parametrize('names', [('shelf.png', 'chart.png'), ('image.jpg', 'image.jpg', 'image.jpg')])
def test_upload_destinations_are_stable_and_disambiguate_names(tmp_path, names):
    sources = []
    for index, name in enumerate(names):
        parent = tmp_path / str(index)
        parent.mkdir()
        sources.append(str(_media_file(parent, name)))
    locations = images_module.upload_locations(sources)
    assert locations == images_module.upload_locations(sources)
    assert len(set(locations.values())) == len(sources)
    assert len({Path(p).parent for p in locations.values()}) == 1
    expected = ['image.jpg', 'image-2.jpg', 'image-3.jpg'] if names[0] == 'image.jpg' else list(names)
    assert [Path(p).name for p in locations.values()] == expected
    separate = [images_module.upload_locations([source])[source] for source in sources]
    assert len(set(separate)) == len(sources)
    assert [Path(p).name for p in separate] == list(names)


@pytest.mark.parametrize('scope', ['../../etc', '/abs', '..', '', 'a/b'])
def test_scope_cannot_escape_the_upload_dir(scope: str) -> None:
    resolved = images_module._scoped_dir(scope)

    assert resolved.startswith(images_module.RUNTIME_UPLOAD_DIR + '/')
    relative = resolved[len(images_module.RUNTIME_UPLOAD_DIR) + 1 :]
    assert '/' not in relative
    assert relative not in {'.', '..'}
    assert Path(resolved).resolve().parent == Path(images_module.RUNTIME_UPLOAD_DIR).resolve()


def test_encode_image_refuses_an_unreadable_string() -> None:
    with pytest.raises(ValueError, match='wrap raw base64'):
        images_module.encode_image('hvtb-img://deadbeef')


async def test_data_uri_image_is_uploaded_and_named(
    captured_main_input: list[MainAgentInput],
    fake_runtime: list[list[Any]],
) -> None:
    import base64 as b64

    payload = b64.b64encode(b'\x89PNG\r\n\x1a\nfake').decode()
    ref = f'data:image/png;base64,{payload}'

    await _run_with_task_data(user_prompt=[{'type': 'image_url', 'image_url': {'url': ref}}])

    sandbox = images_module.upload_locations([ref])[ref]
    assert sandbox.endswith('/image_0.png')
    (batch,) = fake_runtime
    assert [f.path for f in batch] == [sandbox]
    (main_input,) = captured_main_input
    assert main_input.user_prompt[0] == {'type': 'text', 'text': f'\n![image_0]({sandbox})'}


def test_synthetic_names_follow_the_declared_mime() -> None:
    jpeg = 'data:image/jpeg;base64,AAAA'
    webm = 'data:video/webm;base64,AAAA'

    assert images_module.upload_locations([jpeg])[jpeg].endswith('/image_0.jpg')
    assert images_module.upload_locations([webm], kinds={webm: 'video'})[webm].endswith('/video_0.webm')


def test_dimensions_work_from_a_data_uri(tmp_path: Path) -> None:
    import base64 as b64

    from PIL import Image

    path = tmp_path / 'x.png'
    Image.new('RGB', (21, 12), 'white').save(path)
    ref = 'data:image/png;base64,' + b64.b64encode(path.read_bytes()).decode()

    assert images_module.image_dimensions(ref) == (21, 12)


async def test_video_part_is_uploaded_and_noted(
    captured_main_input: list[MainAgentInput],
    fake_runtime: list[list[Any]],
    tmp_path: Path,
) -> None:
    clip = _media_file(tmp_path, 'scene.mp4')

    await _run_with_task_data(
        user_prompt=[
            {'type': 'video_url', 'video_url': {'url': str(clip)}},
            {'type': 'text', 'text': 'What happens?'},
        ]
    )

    sandbox = images_module.upload_locations([str(clip)])[str(clip)]
    assert sandbox.endswith('/scene.mp4')
    (batch,) = fake_runtime
    assert [f.path for f in batch] == [sandbox]
    (main_input,) = captured_main_input
    assert main_input.user_prompt == [
        {'type': 'text', 'text': f'\n[Video saved at {sandbox}]'},
        {'type': 'text', 'text': 'What happens?'},
    ]


async def test_video_without_sandbox_stops_before_model(captured_main_input, no_runtime, tmp_path):
    clip = _media_file(tmp_path, 'scene.mp4')
    with pytest.raises(RuntimeError, match='Video could not be delivered'):
        await _run_with_task_data(user_prompt=[{'type': 'video_url', 'video_url': {'url': str(clip)}}])
    assert captured_main_input == []


async def test_remote_video_url_is_refused_with_a_useful_message(
    captured_main_input: list[MainAgentInput],
    fake_runtime: list[list[Any]],
) -> None:
    user_prompt: list[dict[str, Any]] = [
        {'type': 'video_url', 'video_url': {'url': 'https://example.com/a.mp4'}},
    ]

    with pytest.raises(ValueError, match='cannot deliver a remote video URL'):
        await _run_with_task_data(user_prompt=user_prompt)

    assert fake_runtime == []


async def test_runner_forwards_inherited_main_agent_fields(captured_main_input):
    from nexus.runner.single import RunInput, run

    result = await run(
        RunInput(
            task_name='visual',
            task_config={'user_prompt': 'test'},
            agent_name='visual_agent',
            agent_config={
                'tool_search': {'enabled': True},
                'wrap_failed_tool_response_as_tool_error': True,
                'oom_runtime_recovery': True,
                'openai_client_key': 'vision',
            },
        )
    )
    assert result.success, result.error
    (value,) = captured_main_input
    assert value.tool_search.enabled
    assert value.wrap_failed_tool_response_as_tool_error
    assert value.oom_runtime_recovery
    assert value.openai_client_key == 'vision'


@pytest.mark.parametrize('reference', ['A' * 5000, '/missing/image.png'])
async def test_invalid_image_source_fails_with_actionable_message(reference, captured_main_input):
    assert images_module.classify_ref(reference) == images_module.REF_OPAQUE
    with pytest.raises(ValueError, match='wrap raw base64'):
        await _run_with_task_data(user_prompt=[{'type': 'image_url', 'image_url': {'url': reference}}])
    assert captured_main_input == []


@pytest.mark.parametrize('result', [[], [FileOperationResult(path='image', success=False)]])
async def test_upload_result_must_confirm_every_file(
    result, fake_runtime, visual_runtime, tmp_path, captured_main_input
):
    visual_runtime.upload_response = UploadFileResponse(results=result)
    png = _media_file(tmp_path, 'image.png')
    with pytest.raises(RuntimeError, match='incomplete results'):
        await _run_with_task_data(user_prompt=[{'type': 'image_url', 'image_url': {'url': str(png)}}])
    assert captured_main_input == []


async def test_hidden_image_cannot_be_silently_dropped(captured_main_input):
    with pytest.raises(RuntimeError, match='Image is unavailable'):
        await _run_with_task_data(
            user_prompt=[{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,QUJD'}}],
            inline_images=False,
            upload_images_to_runtime=False,
        )
    assert captured_main_input == []


def test_local_media_and_data_uri_have_bounded_reads(tmp_path, monkeypatch):
    from nexus.extensions.agents.visual_agent.resources import media

    path = tmp_path / 'large.png'
    path.write_bytes(b'12345')
    monkeypatch.setattr(media, 'MAX_MEDIA_BYTES', 4)
    monkeypatch.setattr(images_module, 'MAX_MEDIA_BYTES', 4)
    for read in (images_module.read_source, images_module.encode_image):
        with pytest.raises(ValueError, match='64 MiB'):
            read(str(path))
        with pytest.raises(ValueError, match='64 MiB'):
            read('data:image/png;base64,MTIzNDU=')


async def test_pdf_size_hint_does_not_invoke_image_decoder(monkeypatch, fake_runtime, tmp_path):
    def unexpected(source):
        raise AssertionError('PDF must not be passed to the image decoder')

    monkeypatch.setattr(images_module, 'image_dimensions', unexpected)
    result = await images_module.resolve_user_prompt(
        [{'type': 'file', 'file': {'url': str(_media_file(tmp_path, 'document.pdf'))}}],
        show_image_dimensions=True,
    )
    assert 'saved at' in result[0]['text']


@pytest.mark.parametrize('source_kind', ['local', 'data_uri', 'remote'])
@pytest.mark.parametrize('field', ['image_url', 'image'])
async def test_image_alias_reaches_main_agent_in_standard_shape(source_kind, field, tmp_path, captured_main_input):
    import base64
    import copy

    path = _media_file(tmp_path, 'image.png')
    ref = {
        'local': str(path),
        'data_uri': 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode(),
        'remote': 'https://example.com/image.png',
    }[source_kind]
    prompt = [{'type': 'text', 'text': 'before'}, {'type': 'image', field: {'url': ref, 'detail': 'high'}}]
    before = copy.deepcopy(prompt)
    await _run_with_task_data(user_prompt=prompt, upload_images_to_runtime=False)
    parts = captured_main_input[0].user_prompt
    assert parts[0] == prompt[0]
    assert parts[1]['type'] == 'image_url'
    assert parts[1]['image_url']['detail'] == 'high'
    assert 'image' not in parts[1]
    if source_kind == 'remote':
        assert parts[1]['image_url']['url'] == ref
    else:
        assert images_module.read_source(parts[1]['image_url']['url']) == path.read_bytes()
    assert prompt == before


@pytest.mark.parametrize('field', ['url', 'file_data'])
@pytest.mark.parametrize(
    'source', ['', None, 'data:application/pdf;base64,', 'data:application/pdf;base64,invalid!', '/missing/file.pdf']
)
async def test_invalid_inline_file_never_reaches_model(field, source, captured_main_input):
    with pytest.raises(ValueError, match='file|base64'):
        await _run_with_task_data(
            user_prompt=[{'type': 'file', 'file': {field: source}}], upload_images_to_runtime=False
        )
    assert captured_main_input == []


@pytest.mark.parametrize('field', ['url', 'file_data'])
async def test_inline_file_size_limit_applies_without_upload(field, monkeypatch, captured_main_input):
    monkeypatch.setattr(images_module, 'MAX_MEDIA_BYTES', 4)
    with pytest.raises(ValueError, match='64 MiB'):
        await _run_with_task_data(
            user_prompt=[{'type': 'file', 'file': {field: 'data:application/pdf;base64,MTIzNDU='}}],
            upload_images_to_runtime=False,
        )
    assert captured_main_input == []


async def test_video_alias_uploads_and_becomes_a_note(tmp_path, fake_runtime, captured_main_input):
    source = _media_file(tmp_path, 'video.mp4')
    await _run_with_task_data(user_prompt=[{'type': 'video', 'video': str(source)}])
    assert len(fake_runtime) == 1
    assert captured_main_input[0].user_prompt == [
        {'type': 'text', 'text': f'\n[Video saved at {fake_runtime[0][0].path}]'}
    ]


@pytest.mark.parametrize('part', [{'type': 'image_url', 'image_url': {}}, {'type': 'video_url', 'video_url': {}}])
async def test_malformed_media_fails_with_clear_error(part, captured_main_input):
    with pytest.raises(ValueError, match='requires a non-empty'):
        await _run_with_task_data(user_prompt=[part])
    assert captured_main_input == []


def test_visual_gateway_policy_builds_and_recognizes_main_agent_metadata():
    from nexus.core.agent import get_agent_gateway_scaffold_policy
    from nexus.gateway.scaffold_policy import RequestEvent, build_scaffold_policy
    from nexus.gateway.trajectory_assembler import TrajectoryAssembler

    policy_id = get_agent_gateway_scaffold_policy('visual_agent')
    assert policy_id == 'nexus_main_agent'
    assert build_scaffold_policy(policy_id).policy_id == policy_id
    assembler = TrajectoryAssembler(scaffold_policy=policy_id)
    context = assembler.extract_context(
        [{'role': 'user', 'content': 'Resume'}],
        request_meta={
            'declared_chain': {'chain_id': 'main', 'kind': 'main'},
            'invocation_kind': 'compaction_resume',
        },
    )
    assert context.context_key == ('nexus_main_agent', 'main', False)
    assert context.declared_claim.chain_id == 'main'
    facts = build_scaffold_policy(policy_id).inspect_event(
        RequestEvent(
            messages=({'role': 'user', 'content': 'Resume'},),
            tools=None,
            request_meta={'invocation_kind': 'compaction_resume'},
        )
    )
    assert [boundary.rule_name for boundary in facts.boundary_facts] == ['nexus_compaction_resume_explicit']
