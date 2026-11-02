"""WAA evaluation metrics.

Metrics compare the actual state (from getter) against the expected state
and return a float score (typically 0.0 or 1.0).

Registration happens via @metric_registry.register() decorators in each submodule.
"""

# Import all submodules to trigger decorator registration
import nexus.extensions.tasks.waa.metrics.chrome  # noqa: F401
import nexus.extensions.tasks.waa.metrics.docs  # noqa: F401
import nexus.extensions.tasks.waa.metrics.edge  # noqa: F401
import nexus.extensions.tasks.waa.metrics.general  # noqa: F401
import nexus.extensions.tasks.waa.metrics.slides  # noqa: F401
import nexus.extensions.tasks.waa.metrics.table  # noqa: F401
import nexus.extensions.tasks.waa.metrics.vlc  # noqa: F401
import nexus.extensions.tasks.waa.metrics.vscode  # noqa: F401

from nexus.extensions.tasks.waa.registry import metric_registry


METRIC_REGISTRY = metric_registry
