from typing import Callable, Generic
from collections.abc import Iterable, Iterator
import os
import logging

import requests

from .row_worker import RowWorker
from .interfaces import Row, Index
from .data_operators import MappedRow, ProcessedRow
from .scanners import Scanner
from .utils import split_list

logger = logging.getLogger(__name__)


class DataDistributor(Generic[ProcessedRow]):
    def __init__(
        self,
        path: str | os.PathLike,
        scanner_type: type[Scanner[Row]],
        seed: int,
        shuffle: bool,
        pre_fetch_factor: int = 0,
        indexes: list[Index] | None = None,
        infinite: bool = False,
        map_fn: Callable[[Row], MappedRow] | None = None,
        flow_fn: Callable[[Iterable[MappedRow]], Iterable[ProcessedRow]] | None = None,
        ignore_error: bool = False,
        qps: float | None = None,
        instruct_timeout: float | None = 600.0,
        worker_timeout: float | None = 600.0,
        restart_cnt: int | None = None,
        num_workers: int = 1,
        num_distributors: int = 1,
        distributor_idx: int = 0
    ) -> None:
        assert num_distributors > 0, "num_distributors must be greater than 0"
        assert distributor_idx < num_distributors, "shard_idx must be less than num_shards"
        assert num_workers > 0, "num_workers must be greater than 0"

        self.path = path
        self.scanner_type = scanner_type
        self.seed = seed
        self.shuffle = shuffle
        self.pre_fetch_factor = pre_fetch_factor
        self.indexes = indexes
        self.infinite = infinite
        self.map_fn = map_fn
        self.flow_fn = flow_fn
        self.ignore_error = ignore_error
        self.qps = qps
        self.instruct_timeout = instruct_timeout
        self.worker_timeout = worker_timeout
        self.restart_cnt = restart_cnt
        self.num_workers = num_workers
        self.num_distributors = num_distributors
        self.distributor_idx = distributor_idx

        if self.indexes is not None:
            assert len(self.indexes) == self.num_workers, "indexes must have the same length as num_workers"

        self.last_epoch_row: ProcessedRow | None = None

        self.init_workers()

    def init_path_lists(self) -> None:
        if str(self.path).startswith('https://'):
            resp = requests.get(str(self.path))
            resp.raise_for_status()
            content = resp.content
            assert isinstance(content, bytes)
            lines = [line for line in content.decode('utf-8').splitlines()]
        else:
            with open(self.path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

        lines = [line.strip() for line in lines if len(line.strip()) > 0]

        num_shards = self.num_distributors * self.num_workers
        if len(lines) < num_shards:
            num_workers = len(lines) // self.num_distributors
            if num_workers == 0:
                raise ValueError(f"Can't distirbute {len(lines)} shard of data to {self.num_distributors} distributors")
            self.num_workers = num_workers
            logger.critical((f'Number of lines ({len(lines)}) is less than the total num_workers ({num_shards}). ',
                            f'Change num_workers to {num_workers}.'))
            if self.indexes is not None:
                self.indexes = self.indexes[:num_workers]

        path_lists = split_list(lines, self.num_workers * self.num_distributors)
        self.path_lists = path_lists[
            self.distributor_idx * self.num_workers:(self.distributor_idx + 1) * self.num_workers
        ]

    def _try_init_path_lists(self) -> None:
        if not hasattr(self, 'path_lists'):
            self.init_path_lists()

    def init_workers(self) -> None:
        self._try_init_path_lists()
        logger.info(f'Begin initialize {self.num_workers} workers.')
        self.workers = [
            RowWorker(
                path_list=self.path_lists[worker_idx],
                scanner_type=self.scanner_type,
                seed=self.seed,
                shuffle=self.shuffle,
                pre_fetch_factor=self.pre_fetch_factor,
                index=self.indexes[worker_idx] if self.indexes is not None else None,
                infinite=self.infinite,
                map_fn=self.map_fn,
                flow_fn=self.flow_fn,
                ignore_error=self.ignore_error,
                qps=self.qps,
                instruct_timeout=self.instruct_timeout,
                worker_timeout=self.worker_timeout,
                restart_cnt=self.restart_cnt,
            )
            for worker_idx in range(self.num_workers)
        ]
        logger.info(f"Initialized {self.num_workers} workers.")

        self.worker_pointer = 0

    def _try_init_workers(self) -> None:
        if not hasattr(self, 'workers'):
            self.init_workers()

    @property
    def epoch(self) -> int:
        epoches = [worker.index.epoch for worker in self.workers]
        return max(epoches)

    def __iter__(self) -> Iterator[ProcessedRow]:
        iters = [iter(worker) for worker in self.workers]
        current_epoch = self.epoch
        while True:
            if self.last_epoch_row is not None:
                yield self.last_epoch_row
                self.last_epoch_row = None
            for worker_pointer in list(range(self.worker_pointer, len(self.workers))):
                self.worker_pointer = worker_pointer
                try:
                    row = next(iters[worker_pointer])
                except StopIteration:
                    iters[worker_pointer] = iter(self.workers[worker_pointer])
                    row = next(iters[worker_pointer])

                if self.epoch > current_epoch:
                    self.last_epoch_row = row
                    self.worker_pointer += 1
                    if self.worker_pointer >= len(self.workers):
                        self.worker_pointer = 0
                    return

                yield row
            self.worker_pointer = 0

    def reset_iter(self) -> None:
        self.last_epoch_row = None
        self.worker_pointer = 0
        self._try_init_workers()
        for worker in self.workers:
            worker.reset_iter()
