"""What survives validation of a video content part on its way into an agent."""

import pytest

from nexus.extensions.agents.main.agent import MainAgentInput


def video_part(**overrides):
    """Build a prompt through the real agent input, and hand back the part that came out."""
    part = {'type': 'video', 'video': '/data/a.mp4', **overrides}
    return MainAgentInput(tools=[], user_prompt=[part]).user_prompt[0]


class TestExtraKeysSurvive:
    """Pydantic drops undeclared TypedDict keys by default, and ``single.py`` validates here."""

    @pytest.mark.parametrize(
        ('key', 'value'),
        [
            # Dropping this one is silent: hytuner falls back to sampling by ``fps`` instead.
            ('nframes', 16),
            # The Omni path reads these for its resolution cap and its clip window.
            ('max_pixels', 1_000_000),
            ('video_start', 1.5),
            ('video_end', 9.0),
        ],
    )
    def test_a_sampling_knob_reaches_the_agent(self, key, value):
        assert video_part(**{key: value})[key] == value

    def test_the_declared_keys_still_come_through(self):
        part = video_part(fps=2.0, max_frames=64, min_frames=8)
        assert (part['fps'], part['max_frames'], part['min_frames']) == (2.0, 64, 8)


class TestVideoSource:
    @pytest.mark.parametrize(
        'source',
        [
            '/data/a.mp4',
            'file:///data/a.mp4',
            'data:video/mp4;base64,AAAA',
            # hytuner's fetch_video also takes a video as its already-extracted frames.
            ['/frames/0.png', '/frames/1.png'],
        ],
    )
    def test_every_form_the_decoder_accepts_validates(self, source):
        assert video_part(video=source)['video'] == source

    @pytest.mark.parametrize('part', [{'type': 'video'}, {'type': 'video', 'video': 7}])
    def test_a_part_without_a_usable_video_is_still_rejected(self, part):
        # Allowing extras must not turn the union into the bare ``dict`` it replaced.
        with pytest.raises(ValueError, match='user_prompt'):
            MainAgentInput(tools=[], user_prompt=[part])


class TestTheRestOfTheUnion:
    def test_an_openai_part_is_unaffected(self):
        assert MainAgentInput(tools=[], user_prompt=[{'type': 'text', 'text': 'hi'}]).user_prompt[0]['text'] == 'hi'

    def test_a_plain_string_prompt_is_unaffected(self):
        assert MainAgentInput(tools=[], user_prompt='hi').user_prompt == 'hi'
