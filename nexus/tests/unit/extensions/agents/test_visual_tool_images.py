"""Agent-local image size hints and composition with user iteration hooks."""

import asyncio
import base64
import copy
import io

import pytest

from PIL import Image

from nexus.core import ExecutionContext
from nexus.extensions.agents.main.agent import MainAgentOutput, PostIterationHookContext, PostIterationHookResult
from nexus.extensions.agents.visual_agent import agent
from nexus.extensions.agents.visual_agent.resources import tool_images


def image_part(fmt='PNG', size=(32, 18)):
    output = io.BytesIO()
    Image.new('RGB', size).save(output, format=fmt)
    return {
        'type': 'image_url',
        'image_url': {'url': f'data:image/{fmt.lower()};base64,' + base64.b64encode(output.getvalue()).decode()},
    }


def hook_context():
    return PostIterationHookContext(
        iteration=1,
        max_iterations=4,
        history=(),
        tool_messages=(
            {
                'role': 'tool',
                'tool_call_id': 'image-call',
                'content': [
                    {'type': 'text', 'text': '![crop](/tmp/crop.png)'},
                    image_part(),
                ],
            },
        ),
    )


@pytest.mark.parametrize('fmt', ['PNG', 'JPEG', 'GIF', 'BMP', 'WEBP'])
async def test_size_hint_preserves_image_bytes_and_does_not_mutate_tool_results(fmt):
    context = hook_context()
    context.tool_messages[0]['content'][1] = image_part(fmt)
    before = copy.deepcopy(context.tool_messages)
    result = await tool_images.annotate_tool_images(None, context)
    assert context.tool_messages == before
    assert result.tool_messages[0]['content'][0]['text'] == '![crop](/tmp/crop.png) (32x18 px)'
    assert result.tool_messages[0]['content'][1] == before[0]['content'][1]
    assert tool_images._annotate_content(result.tool_messages[0]['content']) == result.tool_messages[0]['content']


def test_unannotated_image_gets_hint_without_changing_caption_or_invalid_images():
    valid = image_part()
    invalid = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,broken'}}
    remote = {'type': 'image_url', 'image_url': {'url': 'https://example.com/image.png'}}
    caption = {'type': 'text', 'text': 'Two output images'}
    assert tool_images._annotate_content([caption, valid, invalid, remote]) == [
        caption,
        {'type': 'text', 'text': 'Image dimensions: (32x18 px)'},
        valid,
        invalid,
        remote,
    ]


@pytest.mark.parametrize('replace_messages', [False, True])
async def test_user_hook_receives_hints_and_preserves_its_output(monkeypatch, replace_messages):
    async def user_hook(ctx, context):
        assert context.tool_messages[0]['content'][0]['text'].endswith('(32x18 px)')
        return PostIterationHookResult(
            tool_messages=[{'role': 'tool', 'tool_call_id': 'image-call', 'content': 'custom'}]
            if replace_messages
            else None,
            extra_messages=[{'role': 'user', 'content': 'Finish now'}],
            next_completion_kwargs={'tool_choice': 'none'},
        )

    monkeypatch.setattr(tool_images, 'import_from_path', lambda path: user_hook)
    with tool_images.image_hook_context('custom_hook'):
        result = await tool_images.annotate_tool_images(None, hook_context())
    assert result.extra_messages == [{'role': 'user', 'content': 'Finish now'}]
    assert result.next_completion_kwargs == {'tool_choice': 'none'}
    if replace_messages:
        assert result.tool_messages[0]['content'] == 'custom'
    else:
        assert result.tool_messages[0]['content'][0]['text'].endswith('(32x18 px)')


async def test_concurrent_agents_keep_user_hooks_separate_and_reset_after_run(monkeypatch):
    def load(name):
        async def user_hook(ctx, context):
            await asyncio.sleep(0)
            return PostIterationHookResult(extra_messages=[{'role': 'user', 'content': name}])

        return user_hook

    monkeypatch.setattr(tool_images, 'import_from_path', load)

    async def one(name):
        with tool_images.image_hook_context(name):
            await asyncio.sleep(0)
            return await tool_images.annotate_tool_images(None, hook_context())

    first, second = await asyncio.gather(one('first'), one('second'))
    assert first.extra_messages[0]['content'] == 'first'
    assert second.extra_messages[0]['content'] == 'second'
    assert (await tool_images.annotate_tool_images(None, hook_context())).extra_messages is None


async def test_failing_user_hook_does_not_remove_image_hints(monkeypatch):
    async def fail(ctx, context):
        raise ValueError('custom hook failed')

    monkeypatch.setattr(tool_images, 'import_from_path', lambda path: fail)
    with tool_images.image_hook_context('failing_hook'):
        result = await tool_images.annotate_tool_images(None, hook_context())
    assert result.tool_messages[0]['content'][0]['text'].endswith('(32x18 px)')


@pytest.mark.parametrize('enabled', [False, True])
async def test_visual_agent_installs_image_hook_only_when_requested(monkeypatch, enabled):
    async def delegate(main_input):
        assert main_input.post_iteration_hook == (tool_images.HOOK_PATH if enabled else None)
        if enabled:
            hook = tool_images.import_from_path(main_input.post_iteration_hook)
            result = await hook(None, hook_context())
            assert result.tool_messages[0]['content'][0]['text'].endswith('(32x18 px)')
        return MainAgentOutput(main_trajectory=[], exit_status='finished', final_answer='ok')

    monkeypatch.setattr(agent, 'run_main_agent', delegate)
    async with ExecutionContext() as ctx:
        await agent.run_visual_agent(agent.VisualAgentInput(user_prompt='hello', show_image_dimensions=enabled))
        assert ctx.tool_configs == {}


@pytest.mark.parametrize('prefix', [b'', b'\xff\xff\xd0', b'\xff\x01', b'\xff\xe0\x00\x02'])
def test_jpeg_dimensions_accept_standalone_markers_and_exact_boundary(prefix):
    # SOF bytes end exactly after width; no trailing scan bytes are needed.
    raw = b'\xff\xd8' + prefix + b'\xff\xc0\x00\x07\x08\x00\x12\x00\x20'
    assert tool_images.pixel_size(raw) == (32, 18)


@pytest.mark.parametrize('prefix', [b'\xff\xda', b'\xff\xd9', b'\xff\x00', b'\xff\xe0\x00\x01'])
def test_jpeg_scan_and_invalid_markers_are_not_parsed_as_dimensions(prefix):
    raw = b'\xff\xd8' + prefix + b'\xff\xc0\x00\x07\x08\x00\x12\x00\x20'
    assert tool_images.pixel_size(raw) is None


@pytest.mark.parametrize('options', [{'lossless': True}, {'icc_profile': b'x' * 64000}])
def test_webp_variants_have_dimensions(options):
    output = io.BytesIO()
    Image.new('RGB', (32, 18)).save(output, format='WEBP', **options)
    assert tool_images.pixel_size(output.getvalue()) == (32, 18)


def test_image_size_only_decodes_a_bounded_header(monkeypatch, tmp_path):
    from nexus.extensions.agents.visual_agent.resources import images, media

    output = io.BytesIO()
    Image.new('RGB', (32, 18)).save(output, format='JPEG', icc_profile=b'x' * 64000)
    raw = output.getvalue() + b'x' * 2_000_000
    path = tmp_path / 'large.jpg'
    path.write_bytes(raw)
    part = {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(raw).decode()}}
    decode = base64.b64decode
    lengths = []

    def bounded(payload, **kwargs):
        lengths.append(len(payload))
        return decode(payload, **kwargs)

    monkeypatch.setattr(media.base64, 'b64decode', bounded)
    assert '32x18 px' in tool_images._annotate_content([part])[0]['text']
    assert images.image_dimensions(part['image_url']['url']) == (32, 18)
    assert images.image_dimensions(str(path)) == (32, 18)
    assert lengths
    assert max(lengths) <= (media.IMAGE_HEADER_BYTES // 3) * 4


def test_media_read_preserves_io_error_cause(monkeypatch, tmp_path):
    from pathlib import Path

    from nexus.extensions.agents.visual_agent.resources.media import read_media_file

    def fail(path):
        raise OSError(5, 'Input/output error')

    monkeypatch.setattr(Path, 'stat', fail)
    with pytest.raises(ValueError, match='readable local file') as caught:
        read_media_file(tmp_path / 'image.png')
    assert isinstance(caught.value.__cause__, OSError)
    assert caught.value.__cause__.errno == 5
