"""Worker-model profile definitions.

Importing this package registers every profile (one file per model family).
Add a model by dropping a new module here that calls
:func:`..base.register`, then importing it below.
"""

from nexus.extensions.tasks.hitl.model.profiles import (  # noqa: F401
    claude,
    default,
    gcp_claude,
    glm,
    gpt_responses,
    hy3,
    openai_basic,
)
