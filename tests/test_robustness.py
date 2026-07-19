"""Failure-mode tests for the streaming pipeline (0.3.1).

Run:  python tests/test_robustness.py

Each case is a silent-corruption mode found in review: the pipeline kept
running and kept producing plausible batches while quietly training on the
wrong data.

Covers:
  1. an unreadable shard is skipped, not re-selected forever (livelock)
  2. a data source where nothing is readable raises instead of spinning
  3. a shard whose row count changed is replayed, not silently skipped
  4. a cursor pointing past the end of a shard is reset, not skipped
  5. a dead batch worker is surfaced instead of silently shrinking the corpus
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lmfuser_data.data_operators import ResumableShardReader  # noqa: E402


class MemScanner:
    """Scanner over an in-memory table registered under a fake URL."""

    TABLES: dict[str, list[dict]] = {}
    OPENS: list[str] = []

    def __init__(self, path: str) -> None:
        MemScanner.OPENS.append(path)
        rows = MemScanner.TABLES[path]
        if rows is None:
            raise OSError(f'cannot read {path}')
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i: int) -> dict:
        return self._rows[i]


def reset(tables: dict) -> None:
    MemScanner.TABLES = tables
    MemScanner.OPENS = []


def test_unreadable_shard_is_skipped() -> None:
    reset({'bad': None, 'good': [{'i': i} for i in range(10)]})
    r = ResumableShardReader(MemScanner, ['bad', 'good'], row_seed=1, order_seed=1)
    got = [next(it)['i'] for it in [iter(r)] * 1 for _ in range(10)]
    assert sorted(got) == list(range(10)), f'healthy shard not fully served: {got}'
    assert 'good' in MemScanner.OPENS, 'the healthy sibling was never opened'
    print(f'PASS 1: unreadable shard skipped, healthy shard served ({len(got)} rows)')


def test_all_unreadable_raises() -> None:
    reset({'bad1': None, 'bad2': None})
    r = ResumableShardReader(MemScanner, ['bad1', 'bad2'], row_seed=1, order_seed=1)
    t0 = time.time()
    try:
        next(iter(r))
    except RuntimeError as e:
        assert 'unusable' in str(e), e
        assert time.time() - t0 < 5, 'took too long — was it spinning?'
        print('PASS 2: all-unreadable source raises promptly instead of livelocking')
        return
    raise AssertionError('expected RuntimeError for a fully unreadable source')


def test_changed_row_count_replays() -> None:
    reset({'s': [{'i': i} for i in range(100)]})
    r1 = ResumableShardReader(MemScanner, ['s'], row_seed=3, order_seed=1)
    it = iter(r1)
    [next(it) for _ in range(80)]
    state = r1.snapshot()
    assert state['s'][2] == 100, f'row count not recorded: {state}'

    reset({'s': [{'i': i} for i in range(50)]})          # shard shrank
    r2 = ResumableShardReader(MemScanner, ['s'], row_seed=3, order_seed=1, state=state)
    it2 = iter(r2)
    got = [next(it2)['i'] for _ in range(50)]
    assert sorted(got) == list(range(50)), \
        f'shrunken shard was skipped instead of replayed ({len(got)} rows)'
    print('PASS 3: changed row count -> shard replayed, nothing silently skipped')


def test_stale_cursor_past_end() -> None:
    reset({'s': [{'i': i} for i in range(10)]})
    r = ResumableShardReader(MemScanner, ['s'], row_seed=1, order_seed=1,
                             state={'s': [0, 999, -1]})
    got = [next(iter(r))['i'] for _ in range(1)]
    assert got, 'cursor past the end produced nothing'
    print('PASS 4: cursor past end -> reset and served, not skipped')


def test_dead_worker_is_surfaced() -> None:
    import csv
    import tempfile
    from lmfuser_data.batch_loader import BatchDataLoader

    tmp = tempfile.mkdtemp(prefix='deadworker_')
    shards = []
    for s in range(8):
        p = os.path.join(tmp, f'{s}.csv')
        with open(p, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['id'])
            for i in range(200):
                w.writerow([s * 200 + i])
        shards.append(p)
    src = os.path.join(tmp, 'src.txt')
    with open(src, 'w') as fh:
        fh.write('\n'.join(shards))

    loader = BatchDataLoader(
        batch_size=4, path_list=[src], scanner_type='CSVScanner', seed=5,
        shuffle=True, num_workers=2, queue_depth=2, slot_mb=1, worker_timeout=30.0,
    )
    it = iter(loader)
    next(it)                                    # warm both workers
    loader.workers[0].kill()
    loader.workers[0].join(timeout=5)

    try:
        for _ in range(200):                    # survivor keeps the queue full
            next(it)
    except RuntimeError as e:
        assert 'died' in str(e), e
        print('PASS 5: dead worker surfaced instead of silently halving the corpus')
        loader.close()
        return
    loader.close()
    raise AssertionError('a dead worker went undetected for 200 batches')


if __name__ == '__main__':
    test_unreadable_shard_is_skipped()
    test_all_unreadable_raises()
    test_changed_row_count_replays()
    test_stale_cursor_past_end()
    test_dead_worker_is_surfaced()
    print('ALL PASS')
