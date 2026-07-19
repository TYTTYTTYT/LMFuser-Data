"""Failure-mode tests for the streaming pipeline (0.3.2).

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

from lmfuser_data.data_operators import (  # noqa: E402
    ResumableShardReader, DataFlow, UnusableDataSource,
)


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


def test_boundary_cursor_resume() -> None:
    """A cursor at [epoch, n, n] — a shard consumed to its last row — is the
    single most common snapshot (the row index advances before the yield, so
    any snapshot taken while paused on the final row lands there). Resuming
    from it must roll the shard into its next epoch, NOT be mistaken for a
    shard that cannot produce."""
    reset({'s': [{'i': i} for i in range(10)]})
    r = ResumableShardReader(MemScanner, ['s'], row_seed=2, order_seed=1,
                             state={'s': [0, 10, 10]})          # boundary cursor
    it = iter(r)
    got = [next(it)['i'] for _ in range(10)]
    assert sorted(got) == list(range(10)), f'epoch 1 not served: {got}'
    assert r.snapshot()['s'][0] == 1, 'shard did not roll into the next epoch'
    print('PASS 6: boundary cursor rolls to the next epoch instead of raising')


def test_boundary_cursor_resume_through_loader() -> None:
    """The same thing end to end: save mid-stream (the table will contain
    boundary cursors), resume, and keep pulling long enough for the liveness
    check to fire. A worker that raised on a boundary cursor used to die here
    and take the whole run with it."""
    import csv
    import tempfile
    from lmfuser_data.batch_loader import BatchDataLoader

    tmp = tempfile.mkdtemp(prefix='boundary_')
    shards = []
    for s in range(8):
        p = os.path.join(tmp, f'{s}.csv')
        with open(p, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['id'])
            for i in range(40):
                w.writerow([s * 40 + i])
        shards.append(p)
    src = os.path.join(tmp, 'src.txt')
    with open(src, 'w') as fh:
        fh.write('\n'.join(shards))

    common = dict(batch_size=4, path_list=[src], scanner_type='CSVScanner',
                  seed=7, shuffle=True, num_workers=8, queue_depth=2,
                  slot_mb=1, worker_timeout=30.0)

    failures = []
    for consumed in (5, 7, 9, 10, 15):
        a = BatchDataLoader(**common)
        it = iter(a)
        for _ in range(consumed):
            next(it)
        state = a.state_dict()
        a.close()

        boundary = [u for tab in state.values() for u, c in tab.items()
                    if len(c) > 2 and c[1] == c[2] and c[2] > 0]
        b = BatchDataLoader(resume_state=state, **common)
        itb = iter(b)
        try:
            for _ in range(120):        # long enough for _check_workers to fire
                next(itb)
        except Exception as e:
            failures.append((consumed, len(boundary), repr(e)[:80]))
        finally:
            b.close()

    assert not failures, 'resume died on a boundary cursor: ' + repr(failures)
    print('PASS 7: cross-process resume survives boundary cursors at 5 save points')


def test_every_row_failing_raises() -> None:
    """A shard that OPENS fine but whose every row fails must not spin. The
    per-row guard introduced to stop such a shard killing the worker created
    exactly that: 245k failed reads in 2s, no rows, no error, a warning per
    attempt."""
    class RowFailScanner(MemScanner):
        def __getitem__(self, i):
            raise OSError('bad row')

    reset({'x': [{'i': i} for i in range(10)], 'y': [{'i': i} for i in range(10)]})
    r = ResumableShardReader(RowFailScanner, ['x', 'y'], row_seed=1, order_seed=1)
    t0 = time.time()
    try:
        next(iter(r))
    except RuntimeError as e:
        assert 'unusable' in str(e), e
        assert time.time() - t0 < 5, 'spun before raising'
        print('PASS 8: a shard whose every row fails raises instead of spinning')
        return
    raise AssertionError('all-rows-failing produced no error')


def test_empty_shards_alongside_boundary_cursors() -> None:
    """Empty shards must not accumulate towards the livelock guard while
    healthy shards sit at their boundary cursors — that combination killed
    both workers on the first resume."""
    reset({'a': [], 'b': [],
           'c': [{'i': i} for i in range(5)], 'd': [{'i': i} for i in range(5)]})
    r = ResumableShardReader(
        MemScanner, ['a', 'b', 'c', 'd'], row_seed=1, order_seed=3,
        state={'a': [0, 0, 0], 'b': [0, 0, 0], 'c': [0, 5, 5], 'd': [0, 5, 5]},
    )
    it = iter(r)
    got = [next(it)['i'] for _ in range(10)]
    assert len(got) == 10, got
    print('PASS 9: empty shards + boundary cursors keep serving')


def test_partial_breakage_keeps_serving() -> None:
    """Broken shards must never accumulate to a raise while any shard works."""
    reset({'p': None, 'q': None, 'g': [{'i': i} for i in range(4)]})
    r = ResumableShardReader(MemScanner, ['p', 'q', 'g'], row_seed=1, order_seed=1)
    it = iter(r)
    got = [next(it)['i'] for _ in range(12)]
    assert len(got) == 12
    print('PASS 10: two broken shards do not stop the healthy one')


def test_guard_survives_ignore_error() -> None:
    """The guard must not be downgradable to a per-row error sentinel.

    ignore_error exists to skip poisoned ROWS. Routing a source-level failure
    through it meant the sentinel was dropped, a fresh iterator was built over
    the same dead source, and the pipeline spun at ~47k failed opens/second
    reporting nothing — for every config that sets ignore_data_error: true,
    which is all of them."""
    reset({'a': None, 'b': None})
    flow = DataFlow(ResumableShardReader(MemScanner, ['a', 'b'], row_seed=1,
                                         order_seed=1),
                    None, None, True)              # ignore_error=True
    t0 = time.time()
    try:
        while time.time() - t0 < 8:
            for _ in iter(flow):
                pass
    except UnusableDataSource:
        print('PASS 11: the livelock guard survives ignore_error')
        return
    raise AssertionError('guard was swallowed; the pipeline span with no error')


def test_no_false_positive_with_wide_epoch_gap() -> None:
    """Quiet shards are rolled forward one epoch per visit, so a shard lagging
    the rest by a wide gap is re-selected many times before it catches up.
    Counting visits rather than distinct shards read that as a stuck stream
    and killed workers that still had thousands of readable rows."""
    reset({f's{i}': ([] if i < 2 else [{'i': i * 100 + j} for j in range(20)])
           for i in range(8)})
    state = {f's{i}': ([0, 0, 0] if i < 2 else [12, 0, 20]) for i in range(8)}
    r = ResumableShardReader(MemScanner, [f's{i}' for i in range(8)],
                             row_seed=1, order_seed=5, state=state)
    it = iter(r)
    got = [next(it)['i'] for _ in range(60)]
    assert len(got) == 60
    print('PASS 12: an epoch gap of 12 with empty shards does not raise')


def test_dead_source_is_reported_promptly() -> None:
    """A dead source must not hide behind worker_timeout. The consumer only
    checked liveness on the happy path or after the full timeout, so with the
    production worker_timeout of 3600s a source that died at startup looked
    like an hour-long hang and then reported a timeout — pointing at the
    wrong thing entirely."""
    import csv
    import tempfile
    from lmfuser_data.batch_loader import BatchDataLoader

    tmp = tempfile.mkdtemp(prefix='deadsrc_')
    shards = []
    for s in range(2):
        p = os.path.join(tmp, f'{s}.csv')
        with open(p, 'w', newline='') as fh:
            csv.writer(fh).writerow(['id'])          # header only: no rows
        shards.append(p)
    src = os.path.join(tmp, 'src.txt')
    with open(src, 'w') as fh:
        fh.write('\n'.join(shards))

    dl = BatchDataLoader(batch_size=2, path_list=[src], scanner_type='CSVScanner',
                         seed=1, shuffle=True, num_workers=2, queue_depth=2,
                         slot_mb=1, worker_timeout=3600.0)
    t0 = time.time()
    try:
        next(iter(dl))
    except RuntimeError as e:
        took = time.time() - t0
        assert 'died' in str(e), e
        assert took < 120, f'took {took:.0f}s with worker_timeout=3600'
        print(f'PASS 13: dead source reported in {took:.0f}s despite worker_timeout=3600')
        return
    finally:
        dl.close()
    raise AssertionError('a dead source produced no error')


def test_tiny_slice_survives_transient_failures() -> None:
    """A source with fewer shards than consumers gives each worker a single
    shard, where two sweeps means two failed opens — a brief blip would kill
    the run."""
    calls = {'n': 0}

    class FlakyScanner(MemScanner):
        def __init__(self, path):
            calls['n'] += 1
            if calls['n'] <= 3:                      # three transient failures
                raise OSError('transient')
            super().__init__(path)

    reset({'only': [{'i': i} for i in range(5)]})
    r = ResumableShardReader(FlakyScanner, ['only'], row_seed=1, order_seed=1)
    it = iter(r)
    got = [next(it)['i'] for _ in range(5)]
    assert sorted(got) == list(range(5)), got
    print('PASS 14: a one-shard slice survives transient open failures')


def test_unreadable_plus_empty_does_not_spin() -> None:
    """The two silence counters must share one sweep test.

    Unreadable shards were held in a retry set and empty ones in the quiet
    set; the sweep test looked at the quiet set alone and the all-skipped test
    at the retry set alone, so a slice holding one of each satisfied neither —
    2.36M spin iterations in 5 seconds, one core pinned, no rows, no error.
    """
    reset({'U': None, 'E': []})
    r = ResumableShardReader(MemScanner, ['U', 'E'], row_seed=1, order_seed=1)
    t0 = time.time()
    try:
        next(iter(r))
    except RuntimeError as e:
        assert 'unusable' in str(e), e
        assert time.time() - t0 < 5, 'spun before raising'
        print('PASS 15: unreadable + empty in one slice raises instead of spinning')
        return
    raise AssertionError('mixed unreadable/empty source produced no error')


def test_dead_shard_is_not_retried_every_visit() -> None:
    """A dead shard keeps its cursor, so it holds the lowest epoch and would be
    re-selected on every visit — one connection timeout per visit on a remote
    source. It must be retried on a budget, and it must not freeze the reader's
    epoch (an epoch-bounded run would never terminate)."""
    opens = {'n': 0}

    class CountingScanner(MemScanner):
        def __init__(self, path):
            if path == 'U':
                opens['n'] += 1
            super().__init__(path)

    reset({'U': None,
           **{f'g{i}': [{'i': i * 100 + j} for j in range(20)] for i in range(4)}})
    r = ResumableShardReader(CountingScanner, ['U'] + [f'g{i}' for i in range(4)],
                             row_seed=1, order_seed=1)
    it = iter(r)
    rows = [next(it) for _ in range(500)]
    assert len(rows) == 500
    assert opens['n'] <= 2, f'dead shard retried {opens["n"]}x in 500 rows'
    assert r.epoch > 0, 'a dead shard froze the epoch counter at 0'
    print(f'PASS 16: dead shard retried {opens["n"]}x over 500 rows, epoch reached {r.epoch}')


def test_epoch_boundary_row_is_not_lost() -> None:
    """No row may be lost or repeated across epoch boundaries.

    The loader drops the row that already belongs to the next epoch, on the
    understanding that the distributor stashed it and the next epoch's stream
    re-serves it. That understanding rests on `last_epoch_row` being cleared
    AFTER its yield in DataDistributor.__iter__: the loader abandons the
    generator mid-yield, so the clear never runs. Moving the clear before the
    yield — a refactor that reads as equivalent — loses one row per epoch
    silently, which is what this test exists to catch.
    """
    import csv
    import tempfile
    from lmfuser_data.data_loader import DataLoader

    # shapes where the row count divides the batch size, so every epoch emits
    # a whole number of batches and can be checked exactly
    for n_shards, rows_each, batch in ((2, 6, 3), (3, 4, 2), (1, 8, 4)):
        tmp = tempfile.mkdtemp(prefix='epochb_')
        shards = []
        for s in range(n_shards):
            p = os.path.join(tmp, f'{s}.csv')
            with open(p, 'w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(['id'])
                for i in range(rows_each):
                    w.writerow([s * 1000 + i])
            shards.append(p)
        src = os.path.join(tmp, 'src.txt')
        with open(src, 'w') as fh:
            fh.write('\n'.join(shards))

        dl = DataLoader(batch_size=batch, path_list=[src], scanner_type='CSVScanner',
                        seed=1, shuffle=False, pre_fetch_factor=0,
                        instruct_timeout=30, worker_timeout=30)
        total = n_shards * rows_each
        for ep in range(4):
            rows = [int(x) for b in dl for x in b['id']]
            assert len(rows) == len(set(rows)), (
                f'{n_shards}x{rows_each}/b{batch} epoch{ep}: duplicated rows')
            assert len(rows) == total, (
                f'{n_shards}x{rows_each}/b{batch} epoch{ep}: {len(rows)} rows, '
                f'expected {total} — a row was lost at the boundary')
    # and a shape that does NOT divide: the trailing partial batch is carried
    # into the next epoch by design, so check the multiset across epochs
    tmp = tempfile.mkdtemp(prefix='epochb_odd_')
    shards = []
    for s in range(3):
        p = os.path.join(tmp, f'{s}.csv')
        with open(p, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['id'])
            for i in range(5):
                w.writerow([s * 1000 + i])
        shards.append(p)
    src = os.path.join(tmp, 'src.txt')
    with open(src, 'w') as fh:
        fh.write('\n'.join(shards))
    dl = DataLoader(batch_size=2, path_list=[src], scanner_type='CSVScanner',
                    seed=1, shuffle=False, pre_fetch_factor=0,
                    instruct_timeout=30, worker_timeout=30)
    from collections import Counter
    seen: Counter = Counter()
    epochs = 4
    for _ in range(epochs):
        seen.update(int(x) for b in dl for x in b['id'])
    assert len(seen) == 15, f'only {len(seen)} distinct rows ever appeared'
    for row_id, count in seen.items():
        assert epochs - 1 <= count <= epochs, (
            f'row {row_id} seen {count}x over {epochs} epochs — expected '
            f'{epochs} (or {epochs - 1} with one still buffered)')
    print('PASS 17: epoch boundaries lose and duplicate nothing '
          '(3 exact shapes x 4 epochs, plus a non-dividing shape by multiset)')


def test_multi_source_epoch_covers_every_source() -> None:
    """An epoch must not end before the LARGEST source has been round once.

    The loader took max() over its sources, so the smallest one closed the
    epoch for everybody and the largest was never covered — 18 rows across two
    sources delivered 10, 12, 6 and 12 rows over four epochs, some rows seen
    four times and others once. min() means every source gets a full pass;
    small sources repeat within the epoch, which is what the mixing weights
    already imply.
    """
    import csv
    import tempfile
    from collections import Counter
    from lmfuser_data.data_loader import DataLoader

    tmp = tempfile.mkdtemp(prefix='multisrc_')
    srcs = []
    for name, n_shards, n_rows in (('A', 2, 3), ('B', 3, 4)):     # 6 rows vs 12
        shards = []
        for s in range(n_shards):
            p = os.path.join(tmp, f'{name}{s}.csv')
            with open(p, 'w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(['id'])
                for i in range(n_rows):
                    w.writerow([f'{name}{s}-{i}'])
            shards.append(p)
        sp = os.path.join(tmp, f'{name}.txt')
        with open(sp, 'w') as fh:
            fh.write('\n'.join(shards))
        srcs.append(sp)

    dl = DataLoader(batch_size=2, path_list=srcs, scanner_type='CSVScanner',
                    seed=1, shuffle=False, pre_fetch_factor=0,
                    instruct_timeout=30, worker_timeout=30)
    big_seen = Counter()
    for _ in range(3):
        rows = [x for b in dl for x in b['id']]
        big_seen.update(x for x in rows if x.startswith('B'))
    # the large source must be fully covered, not truncated by the small one
    assert len(big_seen) == 12, (
        f'large source covered {len(big_seen)}/12 rows over three epochs — '
        f'the small source is still closing the epoch early')
    assert min(big_seen.values()) >= 2, (
        f'some rows of the large source seen only {min(big_seen.values())}x '
        f'in three epochs: {big_seen}')
    print(f'PASS 18: large source fully covered ({len(big_seen)}/12), '
          f'min exposure {min(big_seen.values())} over 3 epochs')


def test_epoch_covers_every_worker_within_a_source() -> None:
    """The same defect one axis down: shards within ONE source.

    Fixing the source axis alone left this untouched — a 3-row shard and a
    12-row shard behind num_workers=2 delivered the small shard 3x and only
    9/12 of the large one, which is the source-axis bug reproduced verbatim
    one level below it. num_workers > 1 is the common case, so this axis was
    the more live of the two.
    """
    import tempfile
    import csv
    from collections import Counter
    from lmfuser_data.data_loader import DataLoader
    tmp = tempfile.mkdtemp()
    shards = []
    for name, n_rows in (('S', 3), ('L', 12)):
        p = os.path.join(tmp, f'{name}.csv')
        with open(p, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['id'])
            for i in range(n_rows):
                w.writerow([f'{name}-{i}'])
        shards.append(p)
    sp = os.path.join(tmp, 'src.txt')
    with open(sp, 'w') as fh:
        fh.write('\n'.join(shards))

    dl = DataLoader(batch_size=1, path_list=[sp], scanner_type='CSVScanner',
                    seed=1, shuffle=False, pre_fetch_factor=0, num_workers=2,
                    instruct_timeout=30, worker_timeout=30)
    seen = Counter()
    for _ in range(3):
        seen.update(x for b in dl for x in b['id'])
    big = {f'L-{i}' for i in range(12)}
    covered = len(big & set(seen))
    assert covered == 12, (
        f'large shard covered {covered}/12 over three epochs — the small '
        f'shard is still closing the epoch for the whole source')
    print(f'PASS 19: every worker covered within a source ({covered}/12)')


def test_zero_weight_source_does_not_stall_the_epoch() -> None:
    """A weight of 0.0 must not make the epoch unreachable.

    Taking the minimum over sources is right, but a source with weight 0.0 is
    never drawn, so its epoch stays at 0 and the minimum never advances: the
    epoch never ends and `stop_by: epoch` never stops. FloatArg(min_value=0.0)
    makes 0.0 a legal config value, so this is reachable from a YAML.
    """
    import tempfile
    import csv
    from collections import Counter
    from lmfuser_data.data_loader import DataLoader
    tmp = tempfile.mkdtemp()
    srcs = []
    for name in ('A', 'B'):
        p = os.path.join(tmp, f'{name}.csv')
        with open(p, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['id'])
            for i in range(4):
                w.writerow([f'{name}{i}'])
        sp = os.path.join(tmp, f'{name}.txt')
        with open(sp, 'w') as fh:
            fh.write(p)
        srcs.append(sp)

    dl = DataLoader(batch_size=1, path_list=srcs, scanner_type='CSVScanner',
                    seed=1, shuffle=False, pre_fetch_factor=0,
                    distributor_weights=[1.0, 0.0],
                    instruct_timeout=30, worker_timeout=30)
    rows = []
    for b in dl:
        rows.extend(b['id'])
        assert len(rows) <= 40, 'epoch never ended with a zero-weight source'
    assert all(x.startswith('A') for x in rows), \
        f'the zero-weight source was drawn from: {rows}'
    print(f'PASS 20: zero-weight source does not stall the epoch '
          f'({len(rows)} rows, terminated)')


if __name__ == '__main__':
    test_unreadable_shard_is_skipped()
    test_all_unreadable_raises()
    test_changed_row_count_replays()
    test_stale_cursor_past_end()
    test_dead_worker_is_surfaced()
    test_boundary_cursor_resume()
    test_boundary_cursor_resume_through_loader()
    test_every_row_failing_raises()
    test_empty_shards_alongside_boundary_cursors()
    test_partial_breakage_keeps_serving()
    test_guard_survives_ignore_error()
    test_no_false_positive_with_wide_epoch_gap()
    test_dead_source_is_reported_promptly()
    test_tiny_slice_survives_transient_failures()
    test_unreadable_plus_empty_does_not_spin()
    test_dead_shard_is_not_retried_every_visit()
    test_epoch_boundary_row_is_not_lost()
    test_multi_source_epoch_covers_every_source()
    test_epoch_covers_every_worker_within_a_source()
    test_zero_weight_source_does_not_stall_the_epoch()
    print('ALL PASS')
