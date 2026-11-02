"""Bound executor submissions while retaining only pending operation metadata."""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Condition


class BoundedExecutor:
    def __init__(self, workers: int, name: str):
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name)
        self._max_workers = workers
        self._condition = Condition()
        self._pending_tasks = deque()
        self._inflight_count = 0
        self._dispatching = False
        self._closed = False

    def submit(self, function, *args) -> Future:
        future = Future()
        # Storage cancellation is an explicit drain/cleanup protocol. A caller
        # cannot make a live file operation appear stopped via Future.cancel().
        future.set_running_or_notify_cancel()
        with self._condition:
            closed = self._closed
            if not closed:
                self._pending_tasks.append((future, function, args))
        if closed:
            future.set_exception(RuntimeError('cannot schedule work after shutdown'))
            return future
        self._schedule_dispatch()
        return future

    def _schedule_dispatch(self):
        with self._condition:
            if self._dispatching or not self._pending_tasks or self._inflight_count >= self._max_workers:
                return
            self._dispatching = True
        try:
            self._executor.submit(self._dispatch_pending)
        except RuntimeError as error:
            self._handle_submit_failure(error, [])

    def _dispatch_pending(self):
        with self._condition:
            reserved_tasks = []
            while self._pending_tasks and self._inflight_count < self._max_workers:
                reserved_tasks.append(self._pending_tasks.popleft())
                self._inflight_count += 1
        for index, (result_future, function, args) in enumerate(reserved_tasks):
            try:
                worker_future = self._executor.submit(function, *args)
            except RuntimeError as error:
                self._handle_submit_failure(error, reserved_tasks[index:])
                return
            worker_future.add_done_callback(
                lambda done, result_future=result_future: self._on_task_done(result_future, done)
            )
        with self._condition:
            self._dispatching = False
            self._condition.notify_all()
        self._schedule_dispatch()

    def _on_task_done(self, result_future, worker_future):
        # Completing a Future invokes application callbacks inline. They can
        # enqueue dependent cleanup, so retain this slot until they return.
        try:
            error = worker_future.exception()
            if error is None:
                result_future.set_result(worker_future.result())
            else:
                result_future.set_exception(error)
        finally:
            with self._condition:
                self._inflight_count -= 1
                self._condition.notify_all()
            self._schedule_dispatch()

    def _handle_submit_failure(self, error, unsubmitted_tasks):
        with self._condition:
            self._closed = True
            waiting = [*unsubmitted_tasks, *self._pending_tasks]
            self._pending_tasks.clear()
        try:
            for result_future, _, _ in waiting:
                result_future.set_exception(error)
        finally:
            with self._condition:
                self._inflight_count -= len(unsubmitted_tasks)
                self._dispatching = False
                self._condition.notify_all()

    def wait_for_idle(self):
        with self._condition:
            while self._pending_tasks or self._inflight_count or self._dispatching:
                self._condition.wait()

    def shutdown(self):
        with self._condition:
            while self._pending_tasks or self._inflight_count or self._dispatching:
                self._condition.wait()
            self._closed = True
        self._executor.shutdown(wait=True)
