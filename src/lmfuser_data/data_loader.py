from typing import Callable, TypeVar, Hashable, Any
from collections.abc import Iterator, Sequence, Mapping, Iterable
from random import Random
import logging
import os

import torch
from torch import Tensor
import numpy as np
from cloudpickle import pickle
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader as Loader

from .data_distributor import DataDistributor
from .interfaces import Batch, Row, Index
from .utils import mix_iterables
from .scanners import Scanner
from .data_operators import CombinedReader

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

def _collate_fn(rows: Sequence[Row]) -> Batch:
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

    result: Batch = {}
    for key, v_lst in batch.items():
        result[key] = _try_stack(v_lst)

    return result


class DataLoader:
    def __init__(
        self,
        batch_size: int,
        path_list: Sequence[str | os.PathLike],
        scanner_type: type[Scanner] | str,
        seed: int,
        shuffle: bool,
        pre_fetch_factor: int = 0,
        indexes: list[list[Index]] | None = None,
        infinite: bool = False,
        map_fn: Callable[[Row], Row] | None = None,
        flow_fn: Callable[[Iterable[Row]], Iterable[Row]] | None = None,
        ignore_error: bool = False,
        qps: float | None = None,
        instruct_timeout: float | None = 600.0,
        worker_timeout: float | None = 600.0,
        restart_cnt: int | None = None,
        num_workers: int = 1,
        num_ranks: int = 1,
        rank_idx: int = 0,
        batch_map_fn: Callable[[Batch], Batch] | None = None,
        distributor_weights: list[float] | None = None,
    ) -> None:
        if isinstance(scanner_type, str):
            scanner_type = Scanner.get_subclass(scanner_type)
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
            assert len(self.distributors) == len(distributor_weights), \
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

    def current_index(self) -> list[list[Index]]:
        return [[worker.index for worker in distriubtor.workers] for distriubtor in self.distributors]

    def states(self) -> bytes:
        return pickle.dumps({
            'batch_size': self.batch_size,
            'path_list': self.distributors[0].path,
            'scanner_type': self.distributors[0].scanner_type,
            'seed': self.seed,
            'shuffle': self.distributors[0].shuffle,
            'pre_fetch_factor': self.distributors[0].pre_fetch_factor,
            'indexes': self.current_index(),
            'infinite': self.distributors[0].infinite,
            'map_fn': self.distributors[0].map_fn,
            'flow_fn': self.distributors[0].flow_fn,
            'ignore_error': self.ignore_error,
            'qps': self.distributors[0].qps,
            'instruct_timeout': self.distributors[0].instruct_timeout,
            'worker_timeout': self.distributors[0].worker_timeout,
            'restart_cnt': self.distributors[0].restart_cnt,
            'num_workers': self.distributors[0].num_workers,
            'num_ranks': self.distributors[0].num_distributors,
            'rank_idx': self.distributors[0].distributor_idx,
            'batch_map_fn': self.batch_map_fn,
            'distributor_weights': self.distributor_weights,
        })

    @classmethod
    def from_states(cls, states: bytes) -> 'DataLoader':
        states = pickle.loads(states)
        assert isinstance(states, dict)
        return DataLoader(**states) # type: ignore


class PyTorchDataLoader:
    def __init__(
        self,
        batch_size: int,
        path_list: Sequence[str | os.PathLike],
        scanner_type: type[Scanner] | str,
        seed: int,
        shuffle: bool,
        pre_fetch_factor: int = 0,
        num_workers: int = 1,
        num_ranks: int = 1,
        rank_idx: int = 0,
        collate_fn: Callable[[list[Row]], Batch] | None = None,
        drop_last: bool = False
    ) -> None:
        self.batch_size = batch_size
        self.path_list = path_list
        if isinstance(scanner_type, str):
            self.scanner_type = Scanner.get_subclass(scanner_type)
        else:
            self.scanner_type = scanner_type
        self.seed = seed
        self.shuffle = shuffle
        self.pre_fetch_factor = pre_fetch_factor
        self.num_workers = num_workers
        self.num_ranks = num_ranks
        self.rank_idx = rank_idx
        self.collate_fn = collate_fn
        self.drop_last = drop_last

        self.dataset = CombinedReader(scanner_type=self.scanner_type, path_list=self.path_list) # type: ignore
        if num_ranks > 1:
            sampler = DistributedSampler(
                self.dataset,  # type: ignore
                num_replicas=num_ranks, 
                rank=rank_idx, 
                shuffle=shuffle, 
                seed=seed, 
                drop_last=drop_last
            )
            dataloader = Loader(
                dataset=self.dataset, # type: ignore
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=drop_last,
                prefetch_factor=pre_fetch_factor,
                collate_fn=collate_fn,
                generator=torch.Generator().manual_seed(seed)
            )
        else:
            dataloader = Loader(
                dataset=self.dataset, # type: ignore
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=drop_last,
                prefetch_factor=pre_fetch_factor,
                collate_fn=collate_fn,
                generator=torch.Generator().manual_seed(seed)
            )

        self.dataloader = dataloader
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __iter__(self) -> Iterator[Batch]:
        def it_wrap() -> Iterator[Batch]:
            it = iter(self.dataloader)
            for batch in it:
                yield batch
            self._epoch += 1
        return it_wrap()

    def __len__(self) -> int:
        return self.dataloader.__len__()
