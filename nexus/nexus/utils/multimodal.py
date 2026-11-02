"""Layout rules shared by the actor prompt and the reward payload.

Both sides have to place the same pictures against the same question text. They
used to do it independently and drifted: the actor honoured the corpus markers
while the reward payload sent them as literal text. One definition here keeps
the judge reading the question the policy actually answered.
"""

from collections.abc import Sequence
from typing import Any


# Corpus-side marker for "the image belongs at this point in the question".
IMAGE_PLACEHOLDER = '<image>'


def place_images_in_question(
    question: str,
    image_parts: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Lay out the question text and the images relative to each other.

    Each ``<image>`` marker is replaced positionally by its image, following
    ``projects/mnist``. Dropping the markers instead would splice the
    surrounding clauses together and change what the question asks.

    A question with no marker keeps every image up front, then the question
    verbatim. MathVision's ``<image1>`` prose references rely on that, and they
    are not this marker.
    """
    if IMAGE_PLACEHOLDER not in question:
        # An image-only row carries no question text. Appending the empty string
        # would put a blank text part on the wire, which some vision endpoints
        # reject.
        text_parts = [{'type': 'text', 'text': question}] if question else []
        return [*image_parts, *text_parts]

    parts: list[dict[str, Any]] = []
    segments = question.split(IMAGE_PLACEHOLDER)
    for index, segment in enumerate(segments):
        if segment.strip():
            parts.append({'type': 'text', 'text': segment.strip()})
        if index < len(segments) - 1 and index < len(image_parts):
            parts.append(image_parts[index])
    # Markers and images need not agree in number; trailing images still go out
    # so the reader never silently loses one it was supposed to see.
    parts.extend(image_parts[len(segments) - 1 :])
    return parts
