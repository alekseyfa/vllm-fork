from typing import Callable, Optional, Tuple, Type

import habana_frameworks.torch  # noqa: F401
import torch

from vllm.executor.multiproc_gpu_executor import (
    MultiprocessingGPUExecutor, MultiprocessingGPUExecutorAsync)
from vllm.executor.multiproc_worker_utils import WorkerMonitor
from vllm.logger import init_logger
from vllm.utils import make_async
from vllm.worker.worker_base import WorkerBase

logger = init_logger(__name__)


def close_with_atexit(func):
    func()
    import atexit
    atexit._run_exitfuncs()


class MultiprocessingHPUExecutor(MultiprocessingGPUExecutor):
    """Python multiprocessing-based multi-HPU executor"""

    def _init_executor(self):
        super()._init_executor()
        # NOTE(kzawora): This is nasty and it'd be best if it wasn't needed.
        # Monkey-patch child process shutdown code to execute registered HCCL
        # atexit teardown code - this will prevent deadlocks during shutdown
        if isinstance(self.worker_monitor, WorkerMonitor):
            self.close_process_func = self.worker_monitor.close
            self.worker_monitor.close = close_with_atexit(  # type: ignore[method-assign]
                self.close_process_func)

    def _get_worker_module_and_class(
            self) -> Tuple[str, str, Optional[Callable[[], Type[WorkerBase]]]]:
        worker_class_fn = None
        if self.scheduler_config.is_multi_step:
            module_name = "vllm.worker.multi_step_hpu_worker"
            class_name = "MultiStepHPUWorker"
        elif self.speculative_config is not None:
            module_name = "vllm.spec_decode.spec_decode_worker"
            class_name = "create_spec_worker"
        else:
            module_name = "vllm.worker.hpu_worker"
            class_name = "HPUWorker"
        return (module_name, class_name, worker_class_fn)

    def _check_executor_parameters(self):
        world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size

        hpu_device_count = torch.hpu.device_count()
        assert tensor_parallel_size <= hpu_device_count, (
            f"please set tensor_parallel_size ({tensor_parallel_size}) "
            f"to less than max local hpu count ({hpu_device_count})")

        assert world_size <= hpu_device_count, (
            f"please ensure that world_size ({world_size}) "
            f"is less than than max local hpu count ({hpu_device_count})")


class MultiprocessingHPUExecutorAsync(MultiprocessingHPUExecutor,
                                      MultiprocessingGPUExecutorAsync):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver_exec_model = make_async(self.driver_worker.execute_model)
