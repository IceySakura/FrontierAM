"""WAA evaluation getters.

Getters retrieve the current state from the Windows VM for evaluation.
Each getter has the unified async signature:

    async def get_xxx(runtime, config: dict) -> Any

Registration happens via @getter_registry.register() decorators in each submodule.
"""

# Import all submodules to trigger decorator registration
import nexus.extensions.tasks.waa.getters.chrome  # noqa: F401
import nexus.extensions.tasks.waa.getters.edge  # noqa: F401
import nexus.extensions.tasks.waa.getters.file  # noqa: F401
import nexus.extensions.tasks.waa.getters.fileexplorer  # noqa: F401
import nexus.extensions.tasks.waa.getters.general  # noqa: F401
import nexus.extensions.tasks.waa.getters.info  # noqa: F401
import nexus.extensions.tasks.waa.getters.microsoftpaint  # noqa: F401
import nexus.extensions.tasks.waa.getters.misc  # noqa: F401
import nexus.extensions.tasks.waa.getters.msedge  # noqa: F401
import nexus.extensions.tasks.waa.getters.settings  # noqa: F401
import nexus.extensions.tasks.waa.getters.vlc  # noqa: F401
import nexus.extensions.tasks.waa.getters.windows_clock  # noqa: F401

from nexus.extensions.tasks.waa.registry import getter_registry


GETTER_REGISTRY = getter_registry
