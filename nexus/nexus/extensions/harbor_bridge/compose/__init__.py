"""Native Docker Compose task contract for Harbor Bridge."""

from nexus.extensions.harbor_bridge.compose.admission import compile_compose_plan, compose_identity_paths
from nexus.extensions.harbor_bridge.compose.errors import (
    ComposeAdmissionError,
    ComposeConfigError,
    ComposeDefinitionError,
)
from nexus.extensions.harbor_bridge.compose.models import (
    ComposePlan,
    ComposeServiceSummary,
    ComposeTaskConfig,
)
from nexus.extensions.harbor_bridge.compose.parser import (
    LoadedComposeDefinition,
    compose_config_from_task,
    load_compose_definition,
)
from nexus.extensions.harbor_bridge.compose.runtime import topology_spec_from_plan


__all__ = [
    'ComposeAdmissionError',
    'ComposeConfigError',
    'ComposeDefinitionError',
    'ComposePlan',
    'ComposeServiceSummary',
    'ComposeTaskConfig',
    'LoadedComposeDefinition',
    'compile_compose_plan',
    'compose_config_from_task',
    'compose_identity_paths',
    'load_compose_definition',
    'topology_spec_from_plan',
]
