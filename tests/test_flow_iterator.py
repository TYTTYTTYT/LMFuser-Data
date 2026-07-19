"""Regression tests for FlowIterator's buffering contract (0.3.1).

Run:  python tests/test_flow_iterator.py

A flow function may buffer inputs and emit several rows per group. The
iterator must keep ONE flow open across __next__ calls, or everything the
flow produced but had not yet handed out is discarded — for a packing flow
that renders N rows per group, that is an N-fold waste of both compute and
input rows.

Covers:
  1. a many-per-group flow is invoked once per GROUP, not once per row
  2. no input rows are silently dropped between consumed rows
  3. map_fn errors are still recoverable when ignore_error is set
  4. flow errors are still recoverable when ignore_error is set
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lmfuser_data.data_operators import DataFlow, RowMapFunctionError  # noqa: E402


class FakeReader:
    """Minimal ShardedReader stand-in: an infinite counter."""

    def __init__(self) -> None:
        self.pulled = 0

    def is_countable(self) -> bool:
        return False

    def __iter__(self):
        while True:
            yield {'i': self.pulled}
            self.pulled += 1


def test_group_flow_runs_once_per_group() -> None:
    """A flow that consumes 1 row and emits 8 must run once per 8 rows out."""
    calls = {'n': 0}

    def flow_fn(source):
        def gen():
            for row in source:
                calls['n'] += 1                      # one 'render' per group
                for k in range(8):
                    yield {'i': row['i'], 'k': k}
        return gen()

    reader = FakeReader()
    it = iter(DataFlow(reader, None, flow_fn))
    out = [next(it) for _ in range(24)]

    assert calls['n'] == 3, \
        f'flow ran {calls["n"]}x for 24 rows out of an 8-per-group flow (want 3)'
    assert reader.pulled <= 3, \
        f'{reader.pulled} input rows pulled for 24 rows out (want <= 3)'
    assert [r['k'] for r in out[:8]] == list(range(8)), 'group emitted out of order'
    print(f'PASS 1: 24 rows out <- {calls["n"]} groups, {reader.pulled} inputs pulled')


def test_no_rows_dropped() -> None:
    """Consecutive outputs must be contiguous — nothing buffered gets lost."""
    def flow_fn(source):
        def gen():
            for row in source:
                for k in range(4):
                    yield {'v': row['i'] * 4 + k}
        return gen()

    it = iter(DataFlow(FakeReader(), None, flow_fn))
    got = [next(it)['v'] for _ in range(20)]
    assert got == list(range(20)), f'dropped rows: {got}'
    print('PASS 2: output is contiguous (nothing discarded between calls)')


def test_map_error_recovers() -> None:
    """ignore_error keeps the stream alive across a map_fn failure."""
    def map_fn(row):
        if row['i'] == 2:
            raise ValueError('boom')
        return row

    def flow_fn(source):
        def gen():
            for row in source:
                yield row
        return gen()

    it = iter(DataFlow(FakeReader(), map_fn, flow_fn, ignore_error=True))
    seen, errors = [], 0
    for _ in range(8):
        r = next(it)
        if isinstance(r, Exception):
            errors += 1
        else:
            seen.append(r['i'])
    assert errors >= 1, 'the poisoned row did not surface as an error sentinel'
    assert len(seen) >= 5, f'stream did not recover after the error: {seen}'
    print(f'PASS 3: map error surfaced ({errors}) and the stream continued ({seen})')


def test_flow_error_recovers() -> None:
    """ignore_error keeps the stream alive across a flow_fn failure."""
    state = {'raised': False}

    def flow_fn(source):
        def gen():
            for row in source:
                if row['i'] == 1 and not state['raised']:
                    state['raised'] = True
                    raise RuntimeError('flow boom')
                yield row
        return gen()

    it = iter(DataFlow(FakeReader(), None, flow_fn, ignore_error=True))
    results = [next(it) for _ in range(6)]
    assert any(isinstance(r, Exception) for r in results), 'flow error not surfaced'
    assert sum(not isinstance(r, Exception) for r in results) >= 4, \
        'stream did not recover after the flow error'
    print('PASS 4: flow error surfaced and the stream continued')


if __name__ == '__main__':
    test_group_flow_runs_once_per_group()
    test_no_rows_dropped()
    test_map_error_recovers()
    test_flow_error_recovers()
    print('ALL PASS')
