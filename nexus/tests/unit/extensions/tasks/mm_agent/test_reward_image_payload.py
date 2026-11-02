"""Image normalisation for mm_agent reward scoring."""

import pytest

from nexus.extensions.tasks.mm_agent.task import _build_reward_images, _to_reward_image_url


@pytest.fixture
def image_file(tmp_path):
    """Write a tiny file under ``name`` and return its path."""

    def _write(name: str) -> str:
        path = tmp_path / name
        path.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 16)
        return str(path)

    return _write


@pytest.mark.parametrize(
    ('filename', 'expected_mime'),
    [
        ('a.png', 'image/png'),
        ('b.jpg', 'image/jpeg'),
        ('c.jpeg', 'image/jpeg'),
        ('d.webp', 'image/webp'),
        ('e.gif', 'image/gif'),
    ],
)
def test_local_image_declares_the_mime_its_suffix_implies(image_file, filename, expected_mime):
    """The reward platform is a separate consumer from vLLM.

    vLLM sniffs the decoded bytes, so the policy side gets away with a
    hardcoded ``image/jpeg``; the platform may instead trust the declared
    type, where a mislabelled PNG scores as a broken image.
    """
    assert _to_reward_image_url(image_file(filename)).startswith(f'data:{expected_mime};base64,')


def test_unknown_suffix_keeps_the_historical_jpeg_default(image_file):
    # Better a known-accepted type than one the platform may reject outright.
    assert _to_reward_image_url(image_file('screenshot')).startswith('data:image/jpeg;base64,')


@pytest.mark.parametrize(
    'image_source',
    [
        'data:image/png;base64,AAAA',
        'https://example.com/a.png',
        'http://example.com/b.jpg',
    ],
)
def test_already_fetchable_sources_pass_through_untouched(image_source):
    assert _to_reward_image_url(image_source) == image_source


def test_unreadable_image_is_dropped_rather_than_sinking_the_whole_sample(image_file, caplog):
    # Raising here would sentinel the sample to -100 and mask it out of the
    # gradient over one bad file, discarding an otherwise valid rollout.
    usable = image_file('a.png')

    reward_images = _build_reward_images([usable, '/nonexistent/x.png', 'https://example.com/c.png'])

    assert reward_images == [_to_reward_image_url(usable), 'https://example.com/c.png']
    assert 'Skipping image for reward scoring' in caplog.text


def test_no_images_yields_no_payload():
    assert _build_reward_images(None) == []
    assert _build_reward_images([]) == []
