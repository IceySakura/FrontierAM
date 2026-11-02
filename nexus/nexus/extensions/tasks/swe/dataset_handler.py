from abc import ABC, abstractmethod
from pathlib import Path
from types import ModuleType
from typing import Any

from nexus.extensions.tasks.swe.constants import (
    DATASET_ABAKA,
    DATASET_MULTI_SWE_BENCH,
    DATASET_SCALE_SWE,
    DATASET_SWE_ANY,
    DATASET_SWE_BENCH,
    DATASET_SWE_BENCH_MULTILINGUAL,
    DATASET_SWE_BENCH_PLUS_PLUS,
    DATASET_SWE_BENCH_PRO,
    DATASET_SWE_BENCH_TURING,
    DATASET_SWE_COSMOS,
    DATASET_SWE_ENV,
    DATASET_SWE_GYM,
    DATASET_SWE_REBENCH,
    DATASET_SWE_REBENCH_V2,
    DATASET_SWE_SMITH,
    DATASET_T3BUILD_REPOLAUNCH_MIDTRAIN,
    DATASET_T3BUILD_SWEBUILDER_MIDTRAIN,
    DATASET_T3BUILD_SWE_BENCH_LIVE,
    SWEbenchInstance,
)
from nexus.runtime import Runtime


class DatasetHandler(ABC):
    """Abstract base class for dataset handlers.

    Subclasses must implement the abstract methods. Other methods have default
    implementations that can be overridden if needed.
    """

    @abstractmethod
    def get_eval_script(self, instance: SWEbenchInstance) -> str:
        """Generate the evaluation script for the instance."""
        ...

    @abstractmethod
    def get_eval_report_map(
        self,
        instance: SWEbenchInstance,
        prediction: dict[str, str],
        test_log_path: str,
        include_tests_status: bool,
    ) -> dict[str, Any]:
        """Get the evaluation report map."""
        ...

    @abstractmethod
    def get_problem_statement(self, instance: SWEbenchInstance) -> str:
        """Get the problem statement for the instance."""
        ...

    @abstractmethod
    def get_repo_language(self, instance: SWEbenchInstance) -> str | None:
        """Get the repository language."""
        ...

    async def infer_runtime_init(
        self, instance: SWEbenchInstance, runtime: Runtime, *, skip_git_truncate: bool = False, **kwargs: Any
    ) -> None:
        """Initialize runtime for inference.

        Default implementation truncates git history at base_commit to prevent
        the model from peeking at future changes via git diff/log/reflog.

        Subclasses can override this for dataset-specific initialization,
        and should call run_git_truncate at the end.

        Args:
            instance: The SWE-bench instance being processed.
            runtime: The runtime environment to initialize.
            skip_git_truncate: If True, skip git history truncation.
        """
        if skip_git_truncate:
            return

        from nexus.extensions.tasks.swe.infer import run_git_truncate

        base_commit = instance.get('base_commit', '')
        if not base_commit:
            base_commit = instance.get('base', {}).get('sha', '')

        working_dir = instance.get('workdir', '/testbed')
        if kwargs.get('remove_git_dir', False):
            from nexus.extensions.tasks.swe.infer import remove_git_dir

            await remove_git_dir(runtime, working_dir)
            return

        await run_git_truncate(runtime, base_commit, working_dir)

    async def eval_runtime_init(self, instance: SWEbenchInstance, runtime: Runtime) -> None:
        """Initialize runtime for evaluation. No-op by default."""
        return

    def eval_write_test_output(self, instance: SWEbenchInstance, path: Path, stdout: str | None) -> None:
        """Write test output to file. Default implementation writes stdout to path."""
        path.write_text(stdout or '', encoding='utf-8')


class ModuleDatasetHandler(DatasetHandler):
    """Adapter class that wraps a module with dataset handler functions.

    This allows existing module-based implementations to be used with the
    DatasetHandler interface.
    """

    def __init__(self, module: ModuleType) -> None:
        self._module = module

    def get_eval_script(self, instance: SWEbenchInstance) -> str:
        return self._module.get_eval_script(instance)

    def get_eval_report_map(
        self,
        instance: SWEbenchInstance,
        prediction: dict[str, str],
        test_log_path: str,
        include_tests_status: bool,
    ) -> dict[str, Any]:
        return self._module.get_eval_report_map(instance, prediction, test_log_path, include_tests_status)

    def get_problem_statement(self, instance: SWEbenchInstance) -> str:
        return self._module.get_problem_statement(instance)

    def get_repo_language(self, instance: SWEbenchInstance) -> str | None:
        return self._module.get_repo_language(instance)

    async def infer_runtime_init(
        self, instance: SWEbenchInstance, runtime: Runtime, *, skip_git_truncate: bool = False, **kwargs: Any
    ) -> None:
        if hasattr(self._module, 'infer_runtime_init'):
            return await self._module.infer_runtime_init(
                instance, runtime, skip_git_truncate=skip_git_truncate, **kwargs
            )
        return await super().infer_runtime_init(instance, runtime, skip_git_truncate=skip_git_truncate, **kwargs)

    async def eval_runtime_init(self, instance: SWEbenchInstance, runtime: Runtime) -> None:
        if hasattr(self._module, 'eval_runtime_init'):
            return await self._module.eval_runtime_init(instance, runtime)
        return await super().eval_runtime_init(instance, runtime)

    def eval_write_test_output(self, instance: SWEbenchInstance, path: Path, stdout: str | None) -> None:
        if hasattr(self._module, 'eval_write_test_output'):
            return self._module.eval_write_test_output(instance, path, stdout)
        return super().eval_write_test_output(instance, path, stdout)


# Lazy import to avoid circular dependencies
def _get_registry() -> dict[str, DatasetHandler]:
    from nexus.extensions.tasks.swe import (
        abaka,
        multi_swe_bench,
        scale_swe,
        swe_any,
        swe_bench,
        swe_bench_multilingual,
        swe_bench_plus_plus,
        swe_bench_pro,
        swe_bench_turing,
        swe_cosmos,
        swe_env,
        swe_gym,
        swe_rebench,
        swe_rebench_v2,
        swe_smith,
        t3build_repolaunch_midtrain,
        t3build_swe_bench_live,
    )
    from nexus.extensions.tasks.swe.t3build_swebuilder_midtrain import dataset_handler as swebuilder_midtrain

    return {
        DATASET_SWE_BENCH: ModuleDatasetHandler(swe_bench),
        DATASET_SWE_GYM: ModuleDatasetHandler(swe_gym),
        DATASET_T3BUILD_SWE_BENCH_LIVE: ModuleDatasetHandler(t3build_swe_bench_live),
        DATASET_SWE_REBENCH: ModuleDatasetHandler(swe_rebench),
        DATASET_SWE_REBENCH_V2: ModuleDatasetHandler(swe_rebench_v2),
        DATASET_SWE_SMITH: ModuleDatasetHandler(swe_smith),
        DATASET_SWE_COSMOS: ModuleDatasetHandler(swe_cosmos),
        DATASET_SWE_ANY: ModuleDatasetHandler(swe_any),
        DATASET_SWE_ENV: ModuleDatasetHandler(swe_env),
        DATASET_SWE_BENCH_TURING: ModuleDatasetHandler(swe_bench_turing),
        DATASET_SWE_BENCH_PRO: ModuleDatasetHandler(swe_bench_pro),
        DATASET_MULTI_SWE_BENCH: ModuleDatasetHandler(multi_swe_bench),
        DATASET_SWE_BENCH_PLUS_PLUS: ModuleDatasetHandler(swe_bench_plus_plus),
        DATASET_SWE_BENCH_MULTILINGUAL: ModuleDatasetHandler(swe_bench_multilingual),
        DATASET_ABAKA: ModuleDatasetHandler(abaka),
        DATASET_T3BUILD_REPOLAUNCH_MIDTRAIN: ModuleDatasetHandler(t3build_repolaunch_midtrain),
        DATASET_T3BUILD_SWEBUILDER_MIDTRAIN: ModuleDatasetHandler(swebuilder_midtrain),
        DATASET_SCALE_SWE: ModuleDatasetHandler(scale_swe),
    }


def get_handler(dataset_type: str | None) -> DatasetHandler:
    """Get the handler for the given dataset type"""
    registry = _get_registry()
    if dataset_type not in registry:
        raise ValueError(f'Unsupported dataset type: {dataset_type}')
    return registry[dataset_type]
