"""OSWorld evaluator getters."""

from __future__ import annotations

# Import submodules to trigger decorator registration.
import nexus.extensions.tasks.os_world.getters.calc  # noqa: F401
import nexus.extensions.tasks.os_world.getters.chrome  # noqa: F401
import nexus.extensions.tasks.os_world.getters.file  # noqa: F401
import nexus.extensions.tasks.os_world.getters.general  # noqa: F401
import nexus.extensions.tasks.os_world.getters.gimp  # noqa: F401
import nexus.extensions.tasks.os_world.getters.impress  # noqa: F401
import nexus.extensions.tasks.os_world.getters.info  # noqa: F401
import nexus.extensions.tasks.os_world.getters.misc  # noqa: F401
import nexus.extensions.tasks.os_world.getters.replay  # noqa: F401
import nexus.extensions.tasks.os_world.getters.vlc  # noqa: F401
import nexus.extensions.tasks.os_world.getters.vscode  # noqa: F401

from nexus.extensions.tasks.os_world.evaluation.registry import getter_registry


GETTER_REGISTRY = getter_registry
