"""Batch-level data loading over shared memory.

Motivation (measured on a 4xH100 pixel-LM pretrain): the row-level pipeline
(`DataLoader` + per-source `RowWorker`s) parallelizes row processing but leaves
a single-threaded consumer per rank doing per-row queue round-trips,
deserialization, weighted mixing and collation — at large row payloads the
main process saturates one core (~300MB/s of pickle traffic) while the GPUs
starve.

`BatchDataLoader` moves the whole assembly into worker processes:

  * each worker owns ALL sources (a slice of every source's shards), applies
    the per-source ``map_fn``/``flow_fn`` chain (identical semantics to the
    row-level pipeline), samples sources by weight with its own RNG, collates
    complete batches, and writes the tensors directly into a shared-memory
    slot;
  * the control queues carry only a slot index plus a tiny per-batch spec —
    tensor payloads never go through pickle;
  * the consumer copies tensors out of the slot (a single memcpy), releases
    the slot, and yields a regular CPU `Batch` — so the training loop needs
    no changes at all.

Semantics preserved from the row-level pipeline:
  * per-source ``map_fn`` (1-to-1) and ``flow_fn`` (many-to-many) run inside
    the worker, before mixing — exactly as before;
  * source mixing follows ``weights`` in expectation (each worker samples
    independently with a seeded RNG);
  * ``ignore_error``: error sentinels emitted by ``DataFlow`` are dropped in
    the worker;
  * non-tensor batch fields (e.g. lists of strings from a custom collate) are
    passed through the control queue — small, off the hot path.

Intended for infinite/streaming TRAIN loads (`stop_by: step`). Exact-pass
eval/test loads should keep using the row-level or PyTorch loaders.
"""
from typing import Any, Callable
from collections.abc import Iterable, Iterator, Sequence
from random import Random
from multiprocessing import shared_memory
import multiprocessing as mp
import os
import logging
import queue as queue_mod

import numpy as np
import torch
import requests
from torch.multiprocessing import Process, Queue

from .interfaces import Batch, Row
from .scanners import Scanner
from .data_operators import ShardedReader, DataFlow
from .data_loader import _collate_fn
from .utils import split_list

logger = logging.getLogger(__name__)

_ALIGN = 64  # byte alignment for tensor regions inside a slot


def _read_shard_lines(path: str | os.PathLike) -> list[str]:
    """One line per shard, from a local file or an https:// url (same contract
    as ``DataDistributor.init_path_lists``)."""
    if str(path).startswith('https://'):
        resp = requests.get(str(path))
        resp.raise_for_status()
        lines = resp.content.decode('utf-8').splitlines()
    else:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    return [ln.strip() for ln in lines if len(ln.strip()) > 0]


def _to_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return None


def _batch_worker_loop(
    worker_id: int,
    source_shards: list[list[str]],       # per source: this worker's shard slice
    scanner_type: type[Scanner],
    weights: list[float],
    batch_size: int,
    seed: int,
    shuffle: bool,
    map_fn: Callable[[Row], Row] | None,
    flow_fn: Callable[[Iterable[Row]], Iterable[Row]] | None,
    collate_fn: Callable[[list[Row]], Batch] | None,
    batch_map_fn: Callable[[Batch], Batch] | None,
    ignore_error: bool,
    shm_name: str,
    slot_nbytes: int,
    free_q: Queue,
    ready_q: Queue,
) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        rng = Random((seed * 1_000_003 + worker_id * 7919) & 0x7FFFFFFF)
        epochs = [0] * len(source_shards)

        def stream(src_idx: int) -> Iterator[Row]:
            reader = ShardedReader(
                scanner_type, source_shards[src_idx],
                seed + 31 * worker_id + src_idx, shuffle, None, True,  # infinite
            )
            flow = DataFlow(reader, map_fn, flow_fn, ignore_error)
            while True:
                for row in iter(flow):
                    if isinstance(row, Exception):
                        continue  # error sentinel: drop (ignore_error semantics)
                    epochs[src_idx] = reader.index.epoch
                    yield row
                # DataFlow exhausted (finite reader edge case): loop forever
                logger.info(f'[batch-worker {worker_id}] source {src_idx} restarting stream')

        streams = [stream(i) for i in range(len(source_shards))]
        src_ids = list(range(len(streams)))

        while True:
            rows: list[Row] = []
            while len(rows) < batch_size:
                i = rng.choices(src_ids, weights=weights, k=1)[0]
                rows.append(next(streams[i]))

            batch = (collate_fn or _collate_fn)(rows)
            if batch_map_fn is not None:
                batch = batch_map_fn(batch)

            specs: dict[Any, tuple[str, tuple[int, ...], int]] = {}
            objects: dict[Any, Any] = {}
            arrays: list[tuple[Any, np.ndarray]] = []
            cursor = 0
            for key, value in batch.items():
                arr = _to_numpy(value)
                if arr is None:
                    objects[key] = value
                    continue
                start = (cursor + _ALIGN - 1) // _ALIGN * _ALIGN
                if start + arr.nbytes > slot_nbytes:
                    raise RuntimeError(
                        f'batch does not fit the shared-memory slot: key {key!r} needs '
                        f'{arr.nbytes}B at offset {start} of {slot_nbytes}B. '
                        f'Raise batch_slot_mb.'
                    )
                specs[key] = (str(arr.dtype), tuple(arr.shape), start)
                arrays.append((key, arr))
                cursor = start + arr.nbytes

            slot = free_q.get()  # blocks when the ring is full (backpressure)
            base = slot * slot_nbytes
            for key, arr in arrays:
                _, _, off = specs[key]
                dst = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf, offset=base + off)
                np.copyto(dst, arr)
            ready_q.put((slot, specs, objects, min(epochs) if epochs else 0))
    except Exception:
        logger.exception(f'[batch-worker {worker_id}] fatal error, exiting')
        raise
    finally:
        shm.close()


class BatchDataLoader:
    """Drop-in alternative to the sharded ``DataLoader`` for streaming train
    loads. Same constructor vocabulary plus ``num_workers`` (TOTAL batch
    workers for this rank — not per source), ``queue_depth`` and ``slot_mb``.

    Yields regular CPU batches; expose ``epoch`` like the other loaders.
    """

    def __init__(
        self,
        batch_size: int,
        path_list: Sequence[str | os.PathLike],
        scanner_type: type[Scanner] | str,
        seed: int,
        shuffle: bool,
        distributor_weights: list[float] | None = None,
        map_fn: Callable[[Row], Row] | None = None,
        flow_fn: Callable[[Iterable[Row]], Iterable[Row]] | None = None,
        collate_fn: Callable[[list[Row]], Batch] | None = None,
        batch_map_fn: Callable[[Batch], Batch] | None = None,
        ignore_error: bool = False,
        num_workers: int = 4,
        queue_depth: int = 4,
        slot_mb: int = 128,
        num_ranks: int = 1,
        rank_idx: int = 0,
        worker_timeout: float | None = 600.0,
    ) -> None:
        if isinstance(scanner_type, str):
            scanner_type = Scanner.get_subclass(scanner_type)
        assert num_workers > 0 and queue_depth > 0 and slot_mb > 0

        weights = distributor_weights or [1.0] * len(path_list)
        assert len(weights) == len(path_list), \
            'path_list and distributor_weights must have the same length'

        # ---- partition every source's shards across (ranks x workers) ----
        num_consumers = num_ranks * num_workers
        per_worker: list[list[list[str]]] = [[] for _ in range(num_workers)]
        for path in path_list:
            lines = _read_shard_lines(path)
            if len(lines) >= num_consumers:
                parts = split_list(lines, num_consumers)
                for w in range(num_workers):
                    per_worker[w].append(parts[rank_idx * num_workers + w])
            else:
                # fewer shards than consumers: assign round-robin (a shard may
                # be read by several workers, with different seeds/orders)
                logger.warning(
                    f'source {path} has {len(lines)} shards < {num_consumers} '
                    f'consumers; assigning shards round-robin (duplicated reads).'
                )
                for w in range(num_workers):
                    consumer = rank_idx * num_workers + w
                    per_worker[w].append([lines[consumer % len(lines)]])

        # ---- shared-memory ring ----
        self.slot_nbytes = slot_mb * 1024 * 1024
        self.num_slots = max(queue_depth, 2)
        self.shm = shared_memory.SharedMemory(
            create=True, size=self.num_slots * self.slot_nbytes)
        self._owns_shm = True
        self.free_q: Queue = Queue()
        self.ready_q: Queue = Queue()
        for s in range(self.num_slots):
            self.free_q.put(s)

        self.worker_timeout = worker_timeout
        self.batch_size = batch_size
        self._epoch = 0

        self.workers = [
            Process(
                target=_batch_worker_loop,
                args=(
                    w, per_worker[w], scanner_type, weights, batch_size,
                    seed, shuffle, map_fn, flow_fn, collate_fn, batch_map_fn,
                    ignore_error, self.shm.name, self.slot_nbytes,
                    self.free_q, self.ready_q,
                ),
                daemon=True,
            )
            for w in range(num_workers)
        ]
        for p in self.workers:
            p.start()
        logger.info(
            f'BatchDataLoader: {num_workers} batch workers, '
            f'{self.num_slots} x {slot_mb}MB shm slots, '
            f'{len(path_list)} sources (rank {rank_idx}/{num_ranks})'
        )

    @property
    def epoch(self) -> int:
        return self._epoch

    def __iter__(self) -> Iterator[Batch]:
        while True:
            try:
                slot, specs, objects, ep = self.ready_q.get(timeout=self.worker_timeout)
            except queue_mod.Empty:
                if not any(p.is_alive() for p in self.workers):
                    raise RuntimeError('all batch workers died — see worker logs')
                raise TimeoutError(
                    f'no batch arrived within {self.worker_timeout}s '
                    f'({sum(p.is_alive() for p in self.workers)} workers alive)'
                )
            base = slot * self.slot_nbytes
            batch: Batch = {}
            for key, (dtype, shape, off) in specs.items():
                src = np.ndarray(shape, dtype=np.dtype(dtype), buffer=self.shm.buf, offset=base + off)
                batch[key] = torch.from_numpy(src.copy())   # one memcpy; slot freed below
            batch.update(objects)
            self.free_q.put(slot)
            self._epoch = max(self._epoch, ep)
            yield batch

    def close(self) -> None:
        for p in getattr(self, 'workers', []):
            if p.is_alive():
                p.terminate()
        for p in getattr(self, 'workers', []):
            p.join(timeout=10.0)
        if getattr(self, '_owns_shm', False):
            try:
                self.shm.close()
                self.shm.unlink()
            except Exception:
                pass
            self._owns_shm = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
