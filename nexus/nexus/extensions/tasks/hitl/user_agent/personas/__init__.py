"""Personas: who the user is, the tools it holds, and what it is reminded of.

One module per persona, each registering itself on import.
"""

from nexus.extensions.tasks.hitl.user_agent.personas.registry import (
    DEFAULT_TOOLS,
    Persona,
    PersonaError,
    get_persona,
    load_persona_dirs,
    register_persona,
    registered_personas,
)


__all__ = [
    'DEFAULT_TOOLS',
    'Persona',
    'PersonaError',
    'get_persona',
    'load_persona_dirs',
    'register_persona',
    'registered_personas',
]
