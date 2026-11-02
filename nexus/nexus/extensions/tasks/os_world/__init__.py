"""OSWorld task integration for Linux GUI automation."""

from nexus.core import Registry


Registry.import_module('nexus.extensions.tools.desktop.tools')
Registry.import_module('nexus.extensions.tasks.os_world.execution.tools')
Registry.import_module('nexus.extensions.tasks.os_world.execution.runtime')
Registry.import_module('nexus.extensions.tasks.os_world.execution.claude_code')
Registry.import_module('nexus.extensions.tasks.os_world.execution.codex_profile')
Registry.import_module('nexus.extensions.tasks.os_world.task')
