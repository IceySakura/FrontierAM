import asyncio

import pytest

from arena.netproxy import Refused, refusal, resolve


@pytest.mark.parametrize('host, reason', [
    ('export.arxiv.org', None),
    ('api.openalex.org', None),
    ('www.bing.com', None),
    ('leanprover-community.github.io', None),
    ('noa.com', None),
    ('github.com', 'github_or_mirror'),
    ('raw.githubusercontent.com', 'github_or_mirror'),
    ('IceySakura.github.io', 'github_or_mirror'),
    ('kgithub.com', 'github_or_mirror'),
    ('mirror.ghproxy.com', 'github_or_mirror'),
    ('archive.softwareheritage.org', 'mirror'),
    ('files.pythonhosted.org', 'package_index'),
    ('pypi.org', 'package_index'),
    ('api.openai.com', 'language_model'),
    ('generativelanguage.googleapis.com', 'language_model'),
    ('git.woa.com', 'internal'),
    ('10.1.2.3', 'internal_address'),
    ('100.64.0.1', 'internal_address'),
    ('127.0.0.1', 'internal_address'),
    ('8.8.8.8', None),
])
def test_refusal(host, reason):
    assert refusal(host) == reason


def test_public_name_resolving_to_loopback_is_refused():
    with pytest.raises(Refused, match='internal_address|unresolved'):
        asyncio.run(resolve('localtest.me', 443))
