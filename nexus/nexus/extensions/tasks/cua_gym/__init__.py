"""CUA-Gym task integration for Linux GUI automation."""

from nexus.core import Registry


Registry.import_module('nexus.extensions.tools.desktop.tools')
Registry.import_module('nexus.extensions.tasks.os_world.execution.tools')
Registry.import_module('nexus.extensions.tasks.cua_gym.task')
