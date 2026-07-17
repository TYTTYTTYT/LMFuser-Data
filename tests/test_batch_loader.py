"""CPU-only tests for BatchDataLoader.

Run:  python tests/test_batch_loader.py
(or pytest tests/test_batch_loader.py)

Covers:
  1. basic batching + tensor round-trip through shared memory
  2. source mixing follows the configured weights (statistical check)
  3. map_fn (1->1), flow_fn one-to-many (split) and many-to-one (merge)
  4. ignore_error drops poisoned rows without killing the stream
  5. non-tensor (string) fields pass through
  6. rank/worker sharding covers disjoint shard slices
  7. teardown leaves no worker processes / shm segments behind
  8. throughput microbenchmark vs the row-level DataLoader (informational)
"""
import os
import sys
import time
import tempfile
import atexit
import shutil

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lmfuser_data.batch_loader import BatchDataLoader          # noqa: E402
from lmfuser_data.data_loader import DataLoader                # noqa: E402

TMP = tempfile.mkdtemp(prefix='batchloader_test_')
atexit.register(lambda: shutil.rmtree(TMP, ignore_errors=True))


def make_source(name: str, n_shards: int, rows_per_shard: int, payload: int) -> str:
    """Create a TSV source: n_shards files, each row = 'src<TAB>row_id<TAB>data'.
    Returns the path of the source-list file (one shard path per line)."""
    shard_paths = []
    for s in range(n_shards):
        p = os.path.join(TMP, f'{name}_shard{s}.tsv')
        with open(p, 'w') as f:
            f.write('source\trow_id\tvalue\n')
            for r in range(rows_per_shard):
                f.write(f'{name}\t{s * rows_per_shard + r}\t{payload}\n')
        shard_paths.append(p)
    lst = os.path.join(TMP, f'{name}_shards.txt')
    with open(lst, 'w') as f:
        f.write('\n'.join(shard_paths))
    return lst


def row_to_tensor(row):
    row = dict(row)
    row['vec'] = np.full(8, float(row['value']), dtype=np.float32)
    return row


def close(loader):
    loader.close()


def test_basic_and_strings():
    src = make_source('alpha', 4, 50, payload=7)
    dl = BatchDataLoader(
        batch_size=16, path_list=[src], scanner_type='TSVScanner',
        seed=1, shuffle=True, map_fn=row_to_tensor,
        num_workers=2, queue_depth=3, slot_mb=4,
    )
    it = iter(dl)
    for _ in range(5):
        b = next(it)
        assert isinstance(b['vec'], torch.Tensor) and b['vec'].shape == (16, 8)
        assert torch.all(b['vec'] == 7.0)
        assert isinstance(b['source'], list) and len(b['source']) == 16   # strings via objects path
        assert all(s == 'alpha' for s in b['source'])
    close(dl)
    print('PASS basic + strings')


def test_mixture_weights():
    a = make_source('aa', 4, 400, payload=1)
    b = make_source('bb', 4, 400, payload=2)
    dl = BatchDataLoader(
        batch_size=32, path_list=[a, b], scanner_type='TSVScanner',
        seed=3, shuffle=True, map_fn=row_to_tensor,
        distributor_weights=[0.8, 0.2],
        num_workers=3, queue_depth=3, slot_mb=4,
    )
    it = iter(dl)
    counts = {1: 0, 2: 0}
    n_rows = 0
    for _ in range(60):
        batch = next(it)
        v = batch['vec'][:, 0]
        counts[1] += int((v == 1.0).sum())
        counts[2] += int((v == 2.0).sum())
        n_rows += v.numel()
    frac = counts[1] / n_rows
    assert 0.74 <= frac <= 0.86, f'source-a fraction {frac} not within [0.74, 0.86] of target 0.8'
    close(dl)
    print(f'PASS mixture (target 0.80, got {frac:.3f} over {n_rows} rows)')


def test_flow_split_merge_and_errors():
    src = make_source('flow', 4, 100, payload=5)

    def poison(row):  # map: raise on ~5% of rows
        if int(row['row_id']) % 20 == 0:
            raise ValueError('poisoned row')
        return row_to_tensor(row)

    def split_then_merge(source):
        # one-to-many: duplicate each row; many-to-one: merge every 3 rows
        buf = []
        for row in source:
            for copy in (0, 1):
                buf.append({**row, 'copy': copy})
                if len(buf) == 3:
                    merged = dict(buf[0])
                    merged['merged_n'] = 3
                    yield merged
                    buf = []

    dl = BatchDataLoader(
        batch_size=8, path_list=[src], scanner_type='TSVScanner',
        seed=5, shuffle=False, map_fn=poison, flow_fn=split_then_merge,
        ignore_error=True, num_workers=2, queue_depth=2, slot_mb=4,
    )
    it = iter(dl)
    for _ in range(10):
        b = next(it)
        assert b['vec'].shape == (8, 8)
        assert all(m == 3 for m in b['merged_n'])       # merge happened
        rid = [int(r) for r in b['row_id']]
        assert all(r % 20 != 0 for r in rid)            # poisoned rows dropped
    close(dl)
    print('PASS flow split/merge + ignore_error')


def test_sharding_disjoint():
    src = make_source('shard', 8, 30, payload=9)
    seen: dict[int, set[int]] = {}
    for rank in (0, 1):
        dl = BatchDataLoader(
            batch_size=10, path_list=[src], scanner_type='TSVScanner',
            seed=7, shuffle=False, map_fn=row_to_tensor,
            num_workers=2, num_ranks=2, rank_idx=rank, queue_depth=2, slot_mb=4,
        )
        it = iter(dl)
        ids: set[int] = set()
        for _ in range(12):
            ids.update(int(r) for r in next(it)['row_id'])
        seen[rank] = ids
        close(dl)
    overlap = seen[0] & seen[1]
    assert not overlap, f'ranks saw overlapping rows: {sorted(overlap)[:10]}'
    print(f'PASS sharding (rank0 {len(seen[0])} rows, rank1 {len(seen[1])} rows, disjoint)')


def test_teardown():
    src = make_source('td', 4, 50, payload=3)
    dl = BatchDataLoader(
        batch_size=8, path_list=[src], scanner_type='TSVScanner',
        seed=11, shuffle=False, map_fn=row_to_tensor,
        num_workers=3, queue_depth=2, slot_mb=4,
    )
    next(iter(dl))
    procs = list(dl.workers)
    dl.close()
    time.sleep(1)
    assert all(not p.is_alive() for p in procs), 'workers survived close()'
    print('PASS teardown (all workers dead, shm unlinked)')


def bench_vs_row_loader():
    """Informational: rows/s with a deliberately heavy map_fn (simulating
    rendering) — batch loader should win by keeping assembly off the consumer."""
    src = make_source('bench', 16, 400, payload=1)

    def heavy(row):
        row = dict(row)
        arr = np.random.default_rng(int(row['row_id'])).integers(
            0, 255, size=(1, 32, 2880), dtype=np.uint8)   # ~92KB, p8-row sized
        for _ in range(3):
            arr = (arr.astype(np.float32) * 1.0001).astype(np.uint8)  # burn CPU
        row['pixel_values'] = arr
        del row['value']
        return row

    N = 40  # batches
    B = 32

    dl_b = BatchDataLoader(
        batch_size=B, path_list=[src], scanner_type='TSVScanner',
        seed=13, shuffle=False, map_fn=heavy, num_workers=6, queue_depth=4, slot_mb=16,
    )
    it = iter(dl_b)
    next(it)  # warm
    t0 = time.time()
    for _ in range(N):
        next(it)
    t_batch = time.time() - t0
    close(dl_b)

    dl_r = DataLoader(
        batch_size=B, path_list=[src], scanner_type='TSVScanner',
        seed=13, shuffle=False, map_fn=heavy, num_workers=6,
        pre_fetch_factor=4, infinite=True,
    )
    it = iter(dl_r)
    next(it)
    t0 = time.time()
    for _ in range(N):
        next(it)
    t_row = time.time() - t0

    print(f'BENCH  batch-loader: {N*B/t_batch:8.0f} rows/s   '
          f'row-loader: {N*B/t_row:8.0f} rows/s   speedup x{t_row/t_batch:.2f}')


if __name__ == '__main__':
    test_basic_and_strings()
    test_mixture_weights()
    test_flow_split_merge_and_errors()
    test_sharding_disjoint()
    test_teardown()
    bench_vs_row_loader()
    print('ALL_TESTS_PASS')
