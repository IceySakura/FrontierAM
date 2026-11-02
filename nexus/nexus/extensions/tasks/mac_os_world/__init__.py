"""macOSWorld task integration for macOS GUI automation via DevCloud."""

from nexus.core import Registry


Registry.import_module('nexus.extensions.tools.desktop.tools')
Registry.import_module('nexus.extensions.tasks.os_world.execution.tools')
Registry.import_module('nexus.extensions.tasks.mac_os_world.task')
