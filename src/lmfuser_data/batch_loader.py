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
import pickle

import numpy as np
import torch
import requests

# (connect, read) seconds. Without a timeout a stalled HTTP fetch — no RST,
# no data — hangs the worker forever: @retry never engages because nothing
# raises, and the livelock guard never runs because the iterator never
# returns. The read budget is per socket read, so large shards are fine.
_HTTP_TIMEOUT = (10, 120)
from torch.multiprocessing import Process, Queue

from .interfaces import Batch, Row
from .scanners import Scanner
from .data_operators import ResumableShardReader, DataFlow
from .data_loader import _collate_fn
from .utils import split_list, slowest_epoch

logger = logging.getLogger(__name__)

_ALIGN = 64  # byte alignment for tensor regions inside a slot


def _read_shard_lines(path: str | os.PathLike) -> list[str]:
    """One line per shard, from a local file or an https:// url (same contract
    as ``DataDistributor.init_path_lists``)."""
    if str(path).startswith('https://'):
        resp = requests.get(str(path), timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        lines = resp.content.decode('utf-8').splitlines()
    else:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    return [ln.strip() for ln in lines if len(ln.strip()) > 0]


def rows_consumed(cursor: list[int]) -> tuple:
    """How far a cursor has actually read, for comparing two cursors on the
    same shard.

    `(epoch, next_row)` is not that: a shard whose epoch was bumped because it
    could not be READ sits at `[epoch + 1, 0]` and would beat a sibling that
    legitimately read most of the epoch. When the row count is known, total
    rows is the honest measure; without it (a 2-element cursor from 0.3.0)
    fall back to the pair, which is right whenever the epochs match.
    """
    epoch, next_row = cursor[0], cursor[1]
    known_n = cursor[2] if len(cursor) > 2 else -1
    if known_n >= 0:
        return (epoch * known_n + next_row,)
    return (epoch, next_row)


def merge_cursors(into: dict[str, list[int]], cursors: dict[str, list[int]]) -> None:
    """Merge one shard-cursor table into another, keeping the furthest.

    Consumers own disjoint shards EXCEPT on the round-robin fallback (a source
    with fewer shards than consumers), where several play the same shard.
    Last-writer-wins there kept whichever consumer happened to be enumerated
    last rather than the one that had read furthest — measured rewinding 25
    rows. Exported so the runner merging across RANKS uses the same rule as
    the loader merging across workers; they used to disagree.
    """
    for url, cur in cursors.items():
        have = into.get(url)
        if have is None or rows_consumed(cur) > rows_consumed(have):
            into[url] = list(cur)


def _to_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return None


def _batch_worker_loop(
    worker_id: int,
    source_shards: list[list[str]],       # per source: this worker's shard slice
    source_states: list[dict[str, list[int]] | None],  # per source: resume cursors
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

        # row permutations are seeded by shard identity (row_seed = the LOADER
        # seed, identical for every worker) so the cursors reported below stay
        # valid for any future rank x worker re-slicing; only the play ORDER
        # of a slice is worker-local
        readers = [
            ResumableShardReader(
                scanner_type, source_shards[i],
                row_seed=seed,
                order_seed=seed + 31 * worker_id + i,
                shuffle=shuffle,
                state=source_states[i],
            )
            for i in range(len(source_shards))
        ]

        def stream(src_idx: int) -> Iterator[Row]:
            flow = DataFlow(readers[src_idx], map_fn, flow_fn, ignore_error)
            while True:
                for row in iter(flow):
                    if isinstance(row, Exception):
                        continue  # error sentinel: drop (ignore_error semantics)
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
            try:
                base = slot * slot_nbytes
                for key, arr in arrays:
                    _, _, off = specs[key]
                    dst = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf, offset=base + off)
                    np.copyto(dst, arr)
                epoch = min(r.epoch for r in readers) if readers else 0
                # cursor snapshot rides along with every batch (a few KB); the
                # consumer keeps the latest per worker — saving that reproduces
                # the stream up to this batch (in-flight reads are skipped, i.e.
                # a resume loses at most the buffered windows, never repeats)
                cursors = [r.snapshot() for r in readers]
                # mp.Queue pickles on a background feeder thread, so an
                # unpicklable payload raises THERE: put() returns normally,
                # nothing is delivered, and the slot is lost while the worker
                # stays alive — the ring drains until every worker blocks and
                # the consumer reports a timeout with "N workers alive".
                # Moving the call inside this try does not help; the payload
                # has to be checked before it is handed over. Only `objects`
                # can hold arbitrary values (tensors travel through shm), and
                # it is small, so the probe is cheap.
                if objects:
                    try:
                        pickle.dumps(objects)
                    except Exception as e:
                        raise RuntimeError(
                            f'batch field(s) {sorted(objects)} cannot be sent to '
                            f'the consumer: {e}. Return tensors or plain '
                            f'picklable values from collate_fn.'
                        ) from e
                ready_q.put((slot, specs, objects, epoch, worker_id, cursors))
            except BaseException:
                # a slot acquired and never handed on is lost for the process
                # lifetime; enough of those and every worker blocks on free_q
                free_q.put(slot)
                raise
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
        resume_state: dict[str, dict[str, list[int]]] | None = None,
    ) -> None:
        """``resume_state``: the table returned by :meth:`state_dict` (possibly
        merged across ranks) — ``{source_key: {shard_url: [epoch, next_row]}}``.
        Cursors are keyed by shard identity, so the saved table remains valid
        for ANY rank/worker count: shards are re-sliced first, then each worker
        picks up the cursors of the shards it now owns."""
        if isinstance(scanner_type, str):
            scanner_type = Scanner.get_subclass(scanner_type)
        assert num_workers > 0 and queue_depth > 0 and slot_mb > 0

        weights = distributor_weights or [1.0] * len(path_list)
        assert len(weights) == len(path_list), \
            'path_list and distributor_weights must have the same length'

        # ---- partition every source's shards across (ranks x workers) ----
        num_consumers = num_ranks * num_workers
        self.source_keys = [str(p) for p in path_list]
        per_worker: list[list[list[str]]] = [[] for _ in range(num_workers)]
        for path in path_list:
            lines = _read_shard_lines(path)
            if len(lines) >= num_consumers:
                table = (resume_state or {}).get(str(path))
                if table:
                    # deal shards in progress order (fresh first) with a
                    # stride, so every consumer gets a fair mix of fresh and
                    # already-consumed shards — a contiguous split could hand
                    # one consumer only exhausted shards, forcing it into the
                    # next epoch (replays) while others still hold fresh data
                    def _progress(u: str) -> tuple:
                        cur = table.get(u, [0, 0])
                        return (cur[0], cur[1], u)   # epoch, rows consumed, url

                    order = sorted(lines, key=_progress)
                    parts = [order[c::num_consumers] for c in range(num_consumers)]
                else:
                    parts = split_list(lines, num_consumers)
                for w in range(num_workers):
                    per_worker[w].append(parts[rank_idx * num_workers + w])
            else:
                # fewer shards than consumers: assign round-robin (a shard may
                # be read by several workers, with different seeds/orders; the
                # duplicate cursors collapse to one entry, so a resume of such
                # a source is approximate)
                logger.warning(
                    f'source {path} has {len(lines)} shards < {num_consumers} '
                    f'consumers; assigning shards round-robin (duplicated reads).'
                )
                for w in range(num_workers):
                    consumer = rank_idx * num_workers + w
                    per_worker[w].append([lines[consumer % len(lines)]])

        # ---- per-worker resume cursors: subset of the table per owned shard ----
        per_worker_states: list[list[dict[str, list[int]] | None]] = []
        for w in range(num_workers):
            states: list[dict[str, list[int]] | None] = []
            for s, key in enumerate(self.source_keys):
                table = (resume_state or {}).get(key)
                if table is None:
                    states.append(None)
                else:
                    states.append({u: table[u] for u in per_worker[w][s] if u in table})
            per_worker_states.append(states)
        if resume_state:
            n_cursors = sum(len(t) for t in resume_state.values())
            logger.info(f'BatchDataLoader: resuming from {n_cursors} shard cursors')

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
        # Latest epoch reported by each worker. The loader's epoch is the
        # SLOWEST of these — see slowest_epoch. A worker that has not reported
        # yet counts as 0, which is exactly right: it has not finished its
        # first pass.
        self._worker_epochs: dict[int, int] = {w: 0 for w in range(num_workers)}
        # liveness polling cadence on the happy path (is_alive() is a cheap
        # waitpid; once per ring-depth of batches is plenty)
        self._check_every = max(num_workers, 1)
        self._since_check = 0
        self._closed = False

        # latest cursor snapshot per worker (payload of the last CONSUMED
        # batch) — the basis of state_dict()
        self._worker_cursors: dict[int, list[dict[str, list[int]]]] = {}
        self._initial_states = per_worker_states  # fallback before first batch

        self.workers = [
            Process(
                target=_batch_worker_loop,
                args=(
                    w, per_worker[w], per_worker_states[w], scanner_type,
                    weights, batch_size,
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

    def _check_workers(self) -> None:
        """A dead worker takes its shard slice with it: the survivors keep the
        queue full, so training continues on a silently shrunken corpus with a
        skewed source mixture. Surface it instead."""
        if getattr(self, '_closed', False):
            return          # we sent those SIGTERMs ourselves
        dead = [i for i, p in enumerate(self.workers) if not p.is_alive()]
        if dead:
            raise RuntimeError(
                f'batch worker(s) {dead} died (exit codes '
                f'{[self.workers[i].exitcode for i in dead]}) — their shards would '
                f'never be read again; see the worker traceback in the logs'
            )

    def __iter__(self) -> Iterator[Batch]:
        while True:
            # Poll in slices rather than one long blocking wait: dead workers
            # are only noticed on the happy path or on a timeout, so with the
            # generous worker_timeout a streaming source needs (an hour, in
            # the pretrain configs) a source that dies takes that long to be
            # reported — as a timeout, which points at the wrong thing.
            deadline = self.worker_timeout
            got = None
            while deadline is None or deadline > 0:
                slice_s = 15.0 if deadline is None else min(15.0, deadline)
                try:
                    got = self.ready_q.get(timeout=slice_s)
                    break
                except queue_mod.Empty:
                    self._check_workers()       # raises if any worker died
                    if deadline is not None:
                        deadline -= slice_s
            if got is None:
                self._check_workers()
                raise TimeoutError(
                    f'no batch arrived within {self.worker_timeout}s '
                    f'({sum(p.is_alive() for p in self.workers)} workers alive)'
                )
            slot, specs, objects, ep, worker_id, cursors = got
            base = slot * self.slot_nbytes
            batch: Batch = {}
            for key, (dtype, shape, off) in specs.items():
                src = np.ndarray(shape, dtype=np.dtype(dtype), buffer=self.shm.buf, offset=base + off)
                batch[key] = torch.from_numpy(src.copy())   # one memcpy; slot freed below
            batch.update(objects)
            self.free_q.put(slot)
            # `max` here let the FASTEST worker declare the epoch over for the
            # whole loader: a 3-row shard beside a 400-row shard rolled the
            # epoch after 4 batches with 0/400 rows of the large shard seen,
            # and reached epoch 20 while the large shard was still untouched.
            # This is the same defect 0.3.13 removed from the other three axes;
            # this fourth one is the loader pretraining actually uses.
            self._worker_epochs[worker_id] = ep
            self._epoch = slowest_epoch(list(self._worker_epochs.values()))
            self._worker_cursors[worker_id] = cursors
            self._since_check += 1
            if self._since_check >= self._check_every:
                self._since_check = 0
                self._check_workers()
            yield batch

    def state_dict(self) -> dict[str, dict[str, list[int]]]:
        """Merged shard-cursor table for this rank:
        ``{source_key: {shard_url: [epoch, next_row]}}``.

        Reflects the stream up to the last batch each worker HANDED OVER —
        which is not the same as the last batch trained on. Rows still inside
        the flow buffers and the shm ring are past these cursors, and so is
        anything a consumer-side prefetcher has pulled but not yet used (the
        runner's device prefetch runs up to three batches ahead). Resuming
        therefore skips a bounded number of rows and never repeats any.

        Merge per-rank tables with :func:`merge_cursors`, which keeps the
        furthest cursor per shard. Ranks own disjoint shards except on the
        round-robin fallback, where several consumers share one and the
        furthest-wins rule decides."""
        table: dict[str, dict[str, list[int]]] = {k: {} for k in self.source_keys}
        for w, states in enumerate(self._initial_states):
            per_source = self._worker_cursors.get(w)
            for s, key in enumerate(self.source_keys):
                merge_cursors(table[key], (states[s] or {}) if per_source is None
                              else per_source[s])
        return table

    def close(self) -> None:
        self._closed = True
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
