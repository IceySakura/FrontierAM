from __future__ import annotations

from nexus.distributed.artifacts import gateway as artifact_gateway_module
from nexus.distributed.artifacts import maintenance as maintenance_module
from nexus.distributed.artifacts import store as store_module
from nexus.distributed.artifacts.ports import (
    AllocationArtifactReader,
    BatchArtifactClient,
    RunResultPublisher,
    RunResultReader,
)
from nexus.distributed.artifacts.store import ArtifactStore, CredentialProvider
from nexus.distributed.batch import control as batch_control_module
from nexus.distributed.batch import lifecycle as batch_lifecycle_module
from nexus.distributed.batch.ports import BatchControl, BatchControlServicePort
from nexus.distributed.config import DistributedConfig
from nexus.distributed.config.views import (
    ArtifactConfigView,
    ArtifactGatewayConfigView,
    BatchControlConfigView,
    BatchLifecycleConfigView,
    BatchRepositoryConfigView,
    DependencyBuildConfigView,
    EnvironmentConfigView,
    InstanceManagerConfigView,
    InternalTokenConfigView,
    ReconcilerConfigView,
    SchedulerConfigView,
    artifact_config_view,
    batch_lifecycle_config_view,
    dependency_build_config_view,
    environment_config_view,
    instance_manager_config_view,
    reconciler_config_view,
    scheduler_config_view,
)
from nexus.distributed.dependency_build import controller as dependency_build_module
from nexus.distributed.environment import allocator as environment_allocator_module
from nexus.distributed.environment import controller as environment_controller_module
from nexus.distributed.environment import kubernetes as kubernetes_module
from nexus.distributed.environment import providers as provider_module
from nexus.distributed.environment import tkex as tkex_module
from nexus.distributed.environment.ports import (
    AllocationWorkloadClient,
    AllocationWorkloadClientResolver,
    DependencyBuildWorkloadClient,
    EnvironmentAllocator,
    ResolvedAllocationWorkloadClient,
    ResolvedWorkloadClientPort,
    WorkloadClient,
    WorkloadClientResolverPort,
)
from nexus.distributed.execution.drivers import nexus_batch, nexus_live, nexus_single
from nexus.distributed.execution.executors import asyncio_executor, thread_executor
from nexus.distributed.execution.executors.base import (
    ExecutionDriver,
    ExecutionInfoDriver,
    ExecutionReporter,
    Executor,
    LiveExecutionDriver,
    RuntimeEndpointDriver,
    SseEventStreamingDriver,
)
from nexus.distributed.instance_manager import reporting as manager_reporting_module
from nexus.distributed.instance_manager.loops import InstanceManagerLoopPort
from nexus.distributed.instance_manager.reporting import ManagerServicePort
from nexus.distributed.instance_manager.service import InstanceManagerService
from nexus.distributed.observability.metrics import ControlPlaneMetrics, OpenTelemetryControlPlaneMetrics
from nexus.distributed.ports import AsyncCloseable, ReconcileController


def test_core_adapters_explicitly_declare_their_ports() -> None:
    assert BatchControl in batch_control_module.BatchControlService.__mro__
    assert BatchControl in manager_reporting_module.ManagerBatchControl.__mro__
    assert ArtifactStore in store_module.LocalArtifactStore.__mro__
    assert ArtifactStore in store_module.GenericRepositoryArtifactStore.__mro__
    assert EnvironmentAllocator in environment_allocator_module.PredefinedEnvironmentAllocator.__mro__
    assert EnvironmentAllocator in environment_allocator_module.StatefulEnvironmentAllocator.__mro__
    assert ExecutionReporter in manager_reporting_module.ManagerExecutionReporter.__mro__


def test_artifact_adapters_explicitly_declare_their_ports() -> None:
    assert AsyncCloseable in artifact_gateway_module.ApplicationGatewayArtifactTransport.__mro__
    assert AllocationArtifactReader in artifact_gateway_module.ArtifactContentReader.__mro__
    assert BatchArtifactClient in artifact_gateway_module.TaskListArtifactGateway.__mro__
    assert RunResultPublisher in artifact_gateway_module.RunResultArtifactGateway.__mro__
    assert RunResultReader in artifact_gateway_module.RunResultArtifactGateway.__mro__
    assert CredentialProvider in store_module.CredentialFiles.__mro__


def test_control_plane_adapters_explicitly_declare_their_ports() -> None:
    assert ReconcileController in environment_controller_module.EnvironmentController.__mro__
    assert ReconcileController in dependency_build_module.DependencyBuildController.__mro__
    assert maintenance_module.ArtifactMaintenance in maintenance_module.ArtifactMaintenanceReconciler.__mro__
    assert ReconcileController in maintenance_module.ArtifactMaintenanceReconciler.__mro__
    assert batch_lifecycle_module.BatchLifecycle in batch_lifecycle_module.BatchLifecycleReconciler.__mro__
    assert ReconcileController in batch_lifecycle_module.BatchLifecycleReconciler.__mro__
    assert ControlPlaneMetrics in OpenTelemetryControlPlaneMetrics.__mro__


def test_workload_adapters_explicitly_declare_their_ports() -> None:
    for client_type in (kubernetes_module.KubernetesWorkloadClient, tkex_module.TkexWorkloadClient):
        assert WorkloadClient in client_type.__mro__
        assert AllocationWorkloadClient in client_type.__mro__
        assert DependencyBuildWorkloadClient in client_type.__mro__
    assert ResolvedWorkloadClientPort in provider_module.ResolvedWorkloadClient.__mro__
    assert ResolvedAllocationWorkloadClient in provider_module.ResolvedWorkloadClient.__mro__
    assert WorkloadClientResolverPort in provider_module.WorkloadClientResolver.__mro__
    assert AllocationWorkloadClientResolver in provider_module.WorkloadClientResolver.__mro__


def test_execution_adapters_explicitly_declare_their_ports() -> None:
    assert ExecutionDriver in nexus_single.NexusSingleDriver.__mro__
    assert ExecutionDriver in nexus_batch.NexusBatchDriver.__mro__
    assert ExecutionDriver in nexus_live.NexusLiveDriver.__mro__
    assert LiveExecutionDriver in nexus_live.NexusLiveDriver.__mro__
    assert SseEventStreamingDriver in nexus_live.NexusLiveDriver.__mro__
    assert RuntimeEndpointDriver in nexus_live.NexusLiveDriver.__mro__
    assert ExecutionInfoDriver in nexus_live.NexusLiveDriver.__mro__
    assert Executor in asyncio_executor.AsyncioExecutor.__mro__
    assert Executor in thread_executor.ThreadExecutor.__mro__
    assert BatchControlServicePort in batch_control_module.BatchControlService.__mro__
    assert InstanceManagerLoopPort in InstanceManagerService.__mro__
    assert ManagerServicePort in InstanceManagerService.__mro__


def test_config_projection_types_explicitly_declare_their_views() -> None:
    config = DistributedConfig()
    scheduler = scheduler_config_view(config)
    reconciler = reconciler_config_view(config)
    artifact = artifact_config_view(config)
    environment = environment_config_view(config)
    dependency_build = dependency_build_config_view(config)
    instance_manager = instance_manager_config_view(config)
    batch_lifecycle = batch_lifecycle_config_view(config)
    projections = (
        (scheduler, SchedulerConfigView),
        (reconciler, ReconcilerConfigView),
        (artifact, ArtifactConfigView),
        (environment, EnvironmentConfigView),
        (dependency_build, DependencyBuildConfigView),
        (instance_manager, InstanceManagerConfigView),
        (batch_lifecycle, BatchLifecycleConfigView),
    )
    for projection, view_type in projections:
        assert view_type in type(projection).__mro__

    assert InternalTokenConfigView in type(scheduler).__mro__
    assert ArtifactGatewayConfigView in type(artifact).__mro__
    manager_mro = type(instance_manager).__mro__
    assert InternalTokenConfigView in manager_mro
    assert ArtifactGatewayConfigView in manager_mro
    assert BatchRepositoryConfigView in manager_mro
    assert BatchControlConfigView in manager_mro
    lifecycle_mro = type(batch_lifecycle).__mro__
    assert InternalTokenConfigView in lifecycle_mro
    assert ArtifactGatewayConfigView in lifecycle_mro
    assert BatchRepositoryConfigView in lifecycle_mro


def test_long_lived_objects_use_service_and_reconciler_names() -> None:
    assert maintenance_module.ArtifactMaintenanceReconciler.__name__ == 'ArtifactMaintenanceReconciler'
    assert batch_control_module.BatchControlService.__name__ == 'BatchControlService'
    assert not hasattr(maintenance_module, 'ArtifactMaintenanceService')
    assert not hasattr(batch_control_module, 'BatchControlClient')
