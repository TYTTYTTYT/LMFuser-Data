from typing import Callable, TypeVar, Hashable, Any
from collections.abc import Iterator, Sequence, Mapping, Iterable
from random import Random
import logging
import os

import torch
from torch import Tensor
import numpy as np

from .data_operators import ProcessedRow, MappedRow
from .data_distributor import DataDistributor
from .interfaces import Batch, Row, Index
from .utils import mix_iterables
from .scanners import Scanner

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

def _try_stack(value_list: list[T]) -> Tensor | list[T]:
    if isinstance(value_list[0], Tensor):
        try:
            return torch.stack(value_list) # type: ignore
        except Exception:
            return value_list
    elif isinstance(value_list[0], np.ndarray):
        try:
            return torch.tensor(np.array(value_list))
        except Exception:
            return value_list
    else:
        return value_list

def _collate_fn(rows: Sequence[Row]) -> Batch[MappedRow]:
    """
    Default collate function that returns a Batch from a list of Rows.
    This can be overridden by the user to provide custom collation logic.
    """
    # Initialize a dictionary to hold lists for each key in the batch
    batch: Mapping[Hashable, list[Any]] = {}

    for acc, row in enumerate(rows):
        for key, value in row.items():
            if key not in batch:
                batch[key] = []
            if len(batch[key]) < acc:
                for _ in range(acc - len(batch[key])):
                    batch[key].append(None)
            batch[key].append(value)

    result: Batch[MappedRow] = {}
    for key, v_lst in batch.items():
        result[key] = _try_stack(v_lst)

    return result


class DataLoader:
    def __init__(
        self,
        batch_size: int,
        path_list: list[str | os.PathLike],
        scanner_type: type[Scanner[Row]],
        seed: int,
        shuffle: bool,
        pre_fetch_factor: int = 0,
        indexes: list[list[Index]] | None = None,
        infinite: bool = False,
        map_fn: Callable[[Row], MappedRow] | None = None,
        flow_fn: Callable[[Iterable[MappedRow]], Iterable[ProcessedRow]] | None = None,
        ignore_error: bool = False,
        qps: float | None = None,
        instruct_timeout: float | None = 600.0,
        worker_timeout: float | None = 600.0,
        restart_cnt: int | None = None,
        num_workers: int = 1,
        num_ranks: int = 1,
        rank_idx: int = 0,
        batch_map_fn: Callable[[Batch[T]], Batch[R]] | None = None,
        distributor_weights: list[float] | None = None,
    ) -> None:
        self.distributors = [
            DataDistributor(
                path=path,
                scanner_type=scanner_type,
                seed=seed,
                shuffle=shuffle,
                pre_fetch_factor=pre_fetch_factor,
                indexes=idx_list,
                infinite=infinite,
                map_fn=map_fn,
                flow_fn=flow_fn,
                ignore_error=ignore_error,
                qps=qps,
                instruct_timeout=instruct_timeout,
                worker_timeout=worker_timeout,
                restart_cnt=restart_cnt,
                num_workers=num_workers,
                num_distributors=num_ranks,
                distributor_idx=rank_idx,
            ) for path, idx_list in zip(path_list, indexes)
        ] if indexes is not None else [
            DataDistributor(
                path=path,
                scanner_type=scanner_type,
                seed=seed,
                shuffle=shuffle,
                pre_fetch_factor=pre_fetch_factor,
                indexes=None,
                infinite=infinite,
                map_fn=map_fn,
                flow_fn=flow_fn,
                ignore_error=ignore_error,
                qps=qps,
                instruct_timeout=instruct_timeout,
                worker_timeout=worker_timeout,
                restart_cnt=restart_cnt,
                num_workers=num_workers,
                num_distributors=num_ranks,
                distributor_idx=rank_idx,
            ) for path in path_list
        ]

        self.batch_size = batch_size
        self.batch_map_fn = batch_map_fn
        self.seed = seed
        self.ignore_error = ignore_error

        if distributor_weights is None:
            self.distributor_weights = [1.0] * len(self.distributors)
        else:
            assert len(self.distributors) == len(self.distributor_weights), \
                "The number of distributors and weights must be the same."
            self.distributor_weights = distributor_weights

        self.distributor_index = 0
        self.rng = Random(seed)
        self.current_batch = []

    @property
    def epoch(self) -> int:
        return max([distributor.epoch for distributor in self.distributors])

    def __iter__(self) -> Iterator[Batch]:
        current_epoch = self.epoch
        stream = mix_iterables(
            self.distributors,
            self.distributor_weights,
            rng=self.rng
        )

        while True:
            if len(self.current_batch) == self.batch_size:
                try:
                    batch = _collate_fn(self.current_batch)
                    if self.batch_map_fn is not None:
                        batch = self.batch_map_fn(batch)
                    yield batch
                except Exception as e:
                    logger.warning(f'Error in processing a batch: {e}')
                    if not self.ignore_error:
                        raise e
                self.current_batch = []

            row = next(stream)
            self.current_batch.append(row)
            if self.epoch > current_epoch:
                break
