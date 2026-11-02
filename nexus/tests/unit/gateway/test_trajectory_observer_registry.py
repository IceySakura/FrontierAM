import uuid

import pytest

from nexus.core.registry import RegistryError
from nexus.gateway.observers import TrajectoryObserver, trajectory_observer_registry


def test_registry_requires_trajectory_observer_subclass() -> None:
    observer_name = f'test_structural_observer_{uuid.uuid4().hex}'

    class StructuralObserver:
        def on_update(self, _update) -> None:
            pass

        def on_final(self, _trajectory) -> None:
            pass

        def close(self) -> None:
            pass

    with pytest.raises(RegistryError, match='must be a subclass of TrajectoryObserver'):
        trajectory_observer_registry.register(observer_name)(StructuralObserver)


def test_registry_accepts_trajectory_observer_subclass() -> None:
    observer_name = f'test_trajectory_observer_{uuid.uuid4().hex}'

    @trajectory_observer_registry.register(observer_name)
    class RegisteredObserver(TrajectoryObserver):
        pass

    assert trajectory_observer_registry.get(observer_name) is RegisteredObserver
