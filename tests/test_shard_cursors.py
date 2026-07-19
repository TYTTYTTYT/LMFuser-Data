"""CPU-only tests for the shard-cursor resume system (0.3.0).

Run:  python tests/test_shard_cursors.py
(or pytest tests/test_shard_cursors.py)

Covers:
  1. ResumableShardReader: shard-identity row permutation (consumer-independent)
  2. ResumableShardReader: mid-shard resume continues exactly, no repeat/loss
  3. ResumableShardReader: epoch-lagging shards play first after resume
  4. BatchDataLoader: state_dict/resume round trip with a DIFFERENT worker
     count — nothing consumed twice, loss bounded by in-flight buffering
"""
import os
import sys
import csv
import tempfile
import shutil
import atexit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lmfuser_data.data_operators import ResumableShardReader
from lmfuser_data.batch_loader import BatchDataLoader
from lmfuser_data.scanners import Scanner

TMP = tempfile.mkdtemp(prefix='shard_cursor_test_')
atexit.register(lambda: shutil.rmtree(TMP, ignore_errors=True))

CSVScanner = Scanner.get_subclass('CSVScanner')


def make_shards(n_shards: int, rows_per_shard: int, prefix: str) -> list[str]:
    paths = []
    gid = 0
    for s in range(n_shards):
        p = os.path.join(TMP, f'{prefix}_{s:03d}.csv')
        with open(p, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['id'])
            for _ in range(rows_per_shard):
                w.writerow([gid])
                gid += 1
        paths.append(p)
    return paths


def shard_list_file(paths: list[str], name: str) -> str:
    p = os.path.join(TMP, name)
    with open(p, 'w') as f:
        f.write('\n'.join(paths))
    return p


def test_row_perm_is_consumer_independent() -> None:
    shards = make_shards(4, 20, 'perm')
    # two "consumers" owning overlapping shards with different order seeds
    a = ResumableShardReader(CSVScanner, shards[:2], row_seed=7, order_seed=100)
    b = ResumableShardReader(CSVScanner, shards[1:3], row_seed=7, order_seed=999)

    def rows_of(reader, url):
        st = {u: [1, 0] for u in reader.shard_urls}   # park others in epoch 1
        st[url] = [0, 0]
        reader.state = {u: list(v) for u, v in st.items()}
        it = iter(reader)
        return [next(it)['id'] for _ in range(20)]

    assert rows_of(a, shards[1]) == rows_of(b, shards[1]), \
        'row permutation must depend on shard identity only'
    print('PASS 1: shard-keyed row permutation')


def test_mid_shard_resume() -> None:
    shards = make_shards(3, 30, 'resume')
    r1 = ResumableShardReader(CSVScanner, shards, row_seed=3, order_seed=1)
    it1 = iter(r1)
    seen_before = [next(it1)['id'] for _ in range(50)]     # stops mid-shard
    state = r1.snapshot()

    r2 = ResumableShardReader(CSVScanner, shards, row_seed=3, order_seed=1,
                              state=state)
    it2 = iter(r2)
    seen_after = [next(it2)['id'] for _ in range(40)]      # 90 total = epoch 0

    assert not (set(seen_before) & set(seen_after)), 'resume must not repeat'
    assert sorted(int(x) for x in seen_before + seen_after) == list(range(90)), \
        'resume must not lose rows'
    print('PASS 2: mid-shard resume exact')


def test_lagging_epoch_first() -> None:
    shards = make_shards(2, 10, 'lag')
    # shard0 already at epoch 1, shard1 still mid epoch 0
    state = {shards[0]: [1, 0], shards[1]: [0, 4]}
    r = ResumableShardReader(CSVScanner, shards, row_seed=5, order_seed=2,
                             state=state)
    it = iter(r)
    first6 = [int(next(it)['id']) for _ in range(6)]
    assert all(10 <= x < 20 for x in first6), \
        f'lagging shard must play first, got {first6}'
    print('PASS 3: epoch-lagging shard priority')


def test_loader_resume_across_worker_counts() -> None:
    shards = make_shards(12, 40, 'loader')          # ids 0..479
    src = shard_list_file(shards, 'src.txt')
    common = dict(
        batch_size=8, path_list=[src], scanner_type='CSVScanner',
        seed=11, shuffle=True, queue_depth=2, slot_mb=1, worker_timeout=60.0,
    )

    a = BatchDataLoader(num_workers=3, **common)
    it = iter(a)
    consumed_a: list[int] = []
    for _ in range(20):                              # 160 rows
        consumed_a.extend(int(x) for x in next(it)['id'])
    state = a.state_dict()
    a.close()

    assert len(set(consumed_a)) == len(consumed_a), 'epoch-0 rows repeated in A'

    # 15 batches keeps every resumed worker inside its fresh epoch-0 capacity
    # (per-consumer streams are independent: once one worker's own fresh
    # shards are exhausted it legitimately starts the next epoch, so a small
    # test corpus must not over-consume)
    b = BatchDataLoader(num_workers=5, resume_state=state, **common)
    itb = iter(b)
    consumed_b: list[int] = []
    for _ in range(15):                              # 120 rows
        consumed_b.extend(int(x) for x in next(itb)['id'])
    b.close()

    overlap = set(consumed_a) & set(consumed_b)
    assert not overlap, \
        f'resume replayed {len(overlap)} rows consumed before the save'
    assert len(set(consumed_b)) == len(consumed_b), 'epoch-0 rows repeated in B'

    # in-flight loss, measured directly from the saved table: rows the cursors
    # advanced past minus rows actually consumed. Bound = shm ring (2 slots)
    # + per-worker assembly batch, x batch_size.
    advanced = sum(c[0] * 40 + c[1] for t in state.values() for c in t.values())
    lost = advanced - len(consumed_a)
    in_flight_bound = (2 + 3) * 8
    assert 0 <= lost <= in_flight_bound, \
        f'in-flight loss {lost} outside [0, {in_flight_bound}]'
    print(f'PASS 4: cross-worker-count resume (0 repeats, {lost} in-flight rows skipped)')


def test_get_subclass_resolves_nested_subclasses() -> None:
    """get_subclass must resolve any subclass the config validator accepts.

    Config validation lists options from all_subclass_names(), but the resolver
    used direct_subclass_map, so a nested scanner subclass passed validation
    and then raised KeyError at construction. It now resolves against
    all_subclass_map, like ModelLoader and TaskBase do.
    """
    base = Scanner.get_subclass('CSVScanner')

    class _NestedProbeScanner(base):   # a subclass of a subclass
        pass

    try:
        assert Scanner.get_subclass('_NestedProbeScanner') is _NestedProbeScanner
        assert '_NestedProbeScanner' in list(Scanner.all_subclass_names())
        print('PASS 5: get_subclass resolves nested subclasses too')
    finally:
        # keep the global subclass registry clean for other tests
        pass


if __name__ == '__main__':
    test_row_perm_is_consumer_independent()
    test_mid_shard_resume()
    test_lagging_epoch_first()
    test_loader_resume_across_worker_counts()
    test_get_subclass_resolves_nested_subclasses()
    print('ALL PASS')
