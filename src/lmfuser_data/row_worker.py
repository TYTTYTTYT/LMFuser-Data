from typing import Generic, TypeVar, Callable
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import os
import logging
import time
import signal
from copy import deepcopy

from torch.multiprocessing import Queue, Process

from .interfaces import Row, RowBase, Index
from .scanners import Scanner
from .data_operators import Index, Reader, DataFlow

logger = logging.getLogger(__name__)


class WorkerInstruct:
    ...


class NextIter(WorkerInstruct):
    ...


class ResetIter(WorkerInstruct):
    ...


class Stop(WorkerInstruct):
    ...


@dataclass(frozen=True)
class Seek(WorkerInstruct):
    epoch: int
    part: int
    row: int


class WorkerResult:
    ...


@dataclass(frozen=True)
class WorkerInfo(WorkerResult):
    is_countable: bool
    total_length: int | None
    pid: int


@dataclass(frozen=True)
class RowResult(WorkerResult, Generic[Row]):
    row: Row

MappedRow = TypeVar("MappedRow", bound=RowBase)
ProcessedRow = TypeVar("ProcessedRow", bound=RowBase)

def _worker_loop(
    path_list: list[str],
    scanner_type: type[Scanner[Row]],
    seed: int,
    shuffle: bool,
    instruct_queue: Queue,
    result_queue: Queue,
    index_queue: Queue,
    index: Index | None = None,
    infinite: bool = False,
    map_fn: Callable[[Row], MappedRow] | None = None,
    flow_fn: Callable[[Iterable[MappedRow]], Iterable[ProcessedRow]] | None = None,
    ignore_error: bool = False,
    qps: float | None = None,
    instruct_timeout: float | None = 600.0,
):
    reader = Reader(scanner_type, path_list, seed, shuffle, index, infinite)
    data_flow = DataFlow(reader, map_fn, flow_fn, ignore_error)
    stream = iter(data_flow)

    if data_flow.is_countable():
        length = len(data_flow)
    else:
        length = None
    result_queue.put(WorkerInfo(data_flow.is_countable(), length, os.getpid()))

    if qps is not None:
        assert qps > 0, "qps must be greater than 0"
        last_time = time.time()
        min_interval = 1 / qps

    while True:
        try:
            instruct = instruct_queue.get(timeout=instruct_timeout)
        except Exception as e:
            logger.critical(f"Worker {os.getpid()} timed out waiting for instruction: {e}")
            return

        if isinstance(instruct, ResetIter):
            reader.seek(Index(0, 0, 0))
            stream = iter(data_flow)
        elif isinstance(instruct, Stop):
            return
        elif isinstance(instruct, NextIter):
            while True:
                try:
                    if qps is not None:
                        current_time = time.time()
                        if current_time - last_time < min_interval: # type: ignore
                            time.sleep(min_interval - (current_time - last_time)) # type: ignore
                        last_time = time.time()
                    row = next(stream)
                    if not isinstance(row, Exception):
                        result_queue.put(RowResult[ProcessedRow](row))
                        index_queue.put(deepcopy(reader.index))
                        break
                    else:
                        continue
                except StopIteration:
                    stream = iter(data_flow)
        elif isinstance(instruct, Seek):
            reader.seek(Index(instruct.epoch, instruct.part, instruct.row))
            stream = iter(data_flow)
            while True:
                try:
                    if qps is not None:
                        current_time = time.time()
                        if current_time - last_time < min_interval: # type: ignore
                            time.sleep(min_interval - (current_time - last_time)) # type: ignore
                        last_time = time.time()
                    row = next(stream)
                    if not isinstance(row, Exception):
                        result_queue.put(RowResult[ProcessedRow](row))
                        index_queue.put(deepcopy(reader.index))
                        break
                    else:
                        continue
                except StopIteration:
                    stream = iter(data_flow)
        else:
            logger.warning(f"Worker {os.getpid()} received unknown instruction: {instruct}")
            raise ValueError(f"Unknown instruction: {instruct}")

def force_kill(pid: int):
    """Force kill a process by PID."""
    try:
        os.kill(pid, signal.SIGKILL)  # POSIX only
        print(f"Killed process {pid} with SIGKILL")
    except AttributeError:
        # On Windows, no SIGKILL constant — fallback to TerminateProcess
        import psutil
        p = psutil.Process(pid)
        p.kill()
        print(f"Killed process {pid} using psutil")
    except ProcessLookupError:
        print(f"Process {pid} not found")
    except PermissionError:
        print(f"No permission to kill {pid}")


class RowWorker(Generic[ProcessedRow]):
    def __init__(
        self,
        path_list: list[str],
        scanner_type: type[Scanner[Row]],
        seed: int,
        shuffle: bool,
        pre_fetch_factor: int = 0,
        index: Index | None = None,
        infinite: bool = False,
        map_fn: Callable[[Row], MappedRow] | None = None,
        flow_fn: Callable[[Iterable[MappedRow]], Iterable[ProcessedRow]] | None = None,
        ignore_error: bool = False,
        qps: float | None = None,
        instruct_timeout: float | None = 600.0,
        worker_timeout: float | None = 600.0,
        restart_cnt: int | None = None,
    ) -> None:
        self.path_list = path_list
        self.scanner_type = scanner_type
        self.seed = seed
        self.shuffle = shuffle

        assert pre_fetch_factor >= 0, "pre_fetch_factor must be greater than 0"
        self.pre_fetch_factor = pre_fetch_factor

        self.index = index if index is not None else Index(0, 0, 0)
        self.infinite = infinite
        self.map_fn = map_fn
        self.flow_fn = flow_fn
        self.ignore_error = ignore_error
        self.qps = qps
        self.instruct_timeout = instruct_timeout
        self.worker_timeout = worker_timeout
        self.restart_cnt = restart_cnt

        self._start_worker()

        self.last_epoch_row: ProcessedRow | None = None
        self.last_epoch_idx: Index | None = None

    def _start_worker(self) -> None:
        self.instruct_queue = Queue()
        self.result_queue = Queue()
        self.index_queue = Queue()

        self.queue_size = 0

        self.worker = Process(
            target=_worker_loop,
            args=(
                self.path_list,
                self.scanner_type,
                self.seed,
                self.shuffle,
                self.instruct_queue,
                self.result_queue,
                self.index_queue,
                self.index,
                self.infinite,
                self.map_fn,
                self.flow_fn,
                self.ignore_error,
                self.qps,
                self.instruct_timeout,
            ),
            daemon=True
        )
        self.worker.start()
        try:
            worker_info = self.result_queue.get(timeout=self.worker_timeout)
            assert isinstance(worker_info, WorkerInfo)
            self.worker_info = worker_info
        except TimeoutError as e:
            logger.error(f"Worker timed out waiting for worker info: {e}")
            raise e

    def check_alive(self) -> bool:
        return self.worker.is_alive()

    def _try_start(self) -> None:
        if not self.check_alive():
            self._start_worker()

    def seek(self, index: Index) -> ProcessedRow:
        self._try_start()
        try:
            self.instruct_queue.put(Seek(index.epoch, index.part, index.row), timeout=self.worker_timeout)
        except TimeoutError as e:
            logger.error(f"Worker timed out waiting for seek instruction: {index}")
            raise e
        self.queue_size += 1

        item = None
        while self.queue_size > 0:
            try:
                item = self.result_queue.get(timeout=self.worker_timeout)
                index = self.index_queue.get(timeout=self.worker_timeout)
                self.queue_size -= 1
            except TimeoutError as e:
                logger.error(f"Worker timed out waiting for index: {index}")
                raise e

        self.index = index

        assert isinstance(item, RowResult)
        return item.row

    def reset_iter(self) -> None:
        self._try_start()
        self.index = Index(0, 0, 0)
        self.last_epoch_idx = None
        self.last_epoch_row = None
        while self.queue_size > 0:
            try:
                self.result_queue.get(timeout=self.worker_timeout)
                self.index_queue.get(timeout=self.worker_timeout)
                self.queue_size -= 1
            except TimeoutError as e:
                logger.error(f"Worker timed out waiting for clearing the result queue.")
                raise e
        try:
            self.instruct_queue.put(ResetIter(), timeout=self.worker_timeout)
        except TimeoutError as e:
            logger.error(f"Worker timed out waiting for reset iter instruction: {e}")
            raise e

    def __iter__(self) -> Iterator[ProcessedRow]:
        self._try_start()
        cur_epoch = self.index.epoch
        while self.queue_size < self.pre_fetch_factor:
            try:
                self.instruct_queue.put(NextIter(), timeout=self.worker_timeout)
                self.queue_size += 1
            except TimeoutError as e:
                logger.error(f"Worker timed out waiting for next iter instruction: {e}")
                raise e

        if self.last_epoch_row is not None:
            assert self.last_epoch_idx is not None
            self.last_epoch_idx = None
            row = self.last_epoch_row
            self.last_epoch_row = None
            yield row

        while True:

            try:
                self.instruct_queue.put(NextIter(), timeout=self.worker_timeout)
                self.queue_size += 1
            except TimeoutError as e:
                logger.error(f"Worker timed out waiting for next iter instruction: {e}")
                raise e

            try:
                item = self.result_queue.get(timeout=self.worker_timeout)
                index = self.index_queue.get(timeout=self.worker_timeout)
                self.queue_size -= 1
            except TimeoutError as e:
                logger.error(f"Worker timed out waiting for the next iteration.")
                raise e

            assert isinstance(index, Index)
            if index.epoch == cur_epoch:
                self.index = index
                assert isinstance(item, RowResult)
                yield item.row
            else:
                self.last_epoch_row = item.row
                self.last_epoch_idx = index
                self.index = self.last_epoch_idx
                break

    def __del__(self):
        """Destructor: ensure cleanup even if user forgets."""
        try:
            if self.worker.is_alive():
                self.worker.terminate()
                self.worker.join(timeout=30.0)
        except Exception as e:
            # Avoid throwing exceptions in __del__
            logger.critical(f"[Worker] Cleanup error: {e}")
            force_kill(self.worker_info.pid)
