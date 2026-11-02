"""Android World task integration for mobile GUI automation."""

from nexus.core import Registry


Registry.import_module('nexus.extensions.tools.mobile.tools')
Registry.import_module('nexus.extensions.tasks.android_world.task')
