"""The natural language a run writes in, and whether a message is in it.

Not the programming language of the work, which ``recipe/swe_chain_v2/task.py`` carries
as ``instance_language``. This is the language of the prose a user sends a worker.

A language is told apart by its marker, the characters no other language writes. A
language written in the Latin alphabet has no marker of its own and is recognised by
every marker being absent, so two of them cannot be told apart here.
"""

import re

from dataclasses import dataclass


HAN = re.compile(r'[一-鿿㐀-䶿豈-﫿]')

LETTER = re.compile(r'[A-Za-z]')

# A marked character carries about as much as this many letters, which is what makes
# the two counts comparable.
LETTERS_PER_MARK = 2.5

MARKED_SHARE = 0.20
"""How much of a text has to be one marker before the text counts as that language."""

_FENCE = re.compile(r'```.*?```', re.S)
_SPAN = re.compile(r'`[^`\n]*`')


def prose(text: str) -> str:
    """``text`` with fenced blocks and inline spans replaced by a space.

    Identifiers, paths and pasted output sit in those, and they are written in whatever
    the code is written in rather than in the language of the prose around them.
    """
    return _SPAN.sub(' ', _FENCE.sub(' ', text))


@dataclass(frozen=True)
class NaturalLanguage:
    """One language a run may write in."""

    code: str
    """How a run records and looks this language up. Names a persona variant file."""

    name: str
    """What the model is told to write in."""

    marker: re.Pattern[str] | None
    """The characters no other language writes. None for a language in the Latin
    alphabet, which is recognised by every other marker being absent."""

    def share(self, text: str) -> float:
        """How much of ``text`` is this language's marker, between 0 and 1.

        0 for a language with no marker.
        """
        if self.marker is None:
            return 0.0
        marked = len(self.marker.findall(text))
        if not marked:
            return 0.0
        return marked / (marked + len(LETTER.findall(text)) / LETTERS_PER_MARK)

    def accepts(self, message: str) -> bool:
        """Whether ``message`` is prose in this language.

        It may carry no other language's marker, and it must carry this one's own where
        this language has one.
        """
        body = prose(message)
        for other in LANGUAGES:
            if other is not self and other.marker is not None and other.marker.search(body):
                return False
        return self.marker is None or bool(self.marker.search(body))


ENGLISH = NaturalLanguage(code='en', name='English', marker=None)
CHINESE = NaturalLanguage(code='zh', name='中文', marker=HAN)

LANGUAGES: tuple[NaturalLanguage, ...] = (ENGLISH, CHINESE)
"""Every language a run may be read as. Adding one is a line here."""

DEFAULT_LANGUAGE = ENGLISH
"""What a text with no marker in it is taken to be."""


def detect(text: str) -> NaturalLanguage:
    """The language ``text`` is written in, or :data:`DEFAULT_LANGUAGE`."""
    body = prose(text)
    found = DEFAULT_LANGUAGE
    highest = MARKED_SHARE
    for language in LANGUAGES:
        share = language.share(body)
        if share >= highest:
            found, highest = language, share
    return found


__all__ = [
    'CHINESE',
    'DEFAULT_LANGUAGE',
    'ENGLISH',
    'LANGUAGES',
    'MARKED_SHARE',
    'NaturalLanguage',
    'detect',
    'prose',
]
