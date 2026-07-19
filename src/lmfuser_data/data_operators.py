from typing import Iterable, Iterator, Callable, overload, SupportsIndex
import logging
import random

from .scanners import Scanner
from .interfaces import Row, Index, SubclassTracer

logger = logging.getLogger(__name__)

# Minimum number of quiet shard visits before a source is declared dead.
# Keeps a one-shard slice from dying on two transient failures while
# staying well under a full sweep for the usual fat slices.
_MIN_QUIET_VISITS = 8


class NotCountableError(Exception):
    ...


class ShardedReader:
    def __init__(
        self, 
        scanner_type: type[Scanner], 
        path_list: list[str], 
        seed: int,
        shuffle: bool,
        index: Index | None = None,
        infinite: bool = False
    ) -> None:
        self.scanner_type = scanner_type
        self.path_list = path_list
        self.seed = seed
        self.shuffle = shuffle
        self.index = index if index is not None else Index(0, 0, 0)
        self.infinite = infinite

        self.current_scanner: Scanner | None = None
        self.epoch_path_list = path_list.copy()
        self.idx_list: list[int] | None = None

    def set_scanner(self, index: Index) -> None:
        if index.part >= len(self.path_list):
            raise IndexError(f"Part index {index.part} out of range.")
        if self.shuffle:
            path_list = self.path_list.copy()
            path_list.sort()
            rand = random.Random(':'.join([str(self.seed), str(index.epoch)]))
            rand.shuffle(path_list)
            self.epoch_path_list = path_list
        path = self.epoch_path_list[index.part]
        logger.info(f'Loading data from {path}...')
        self.current_scanner = self.scanner_type(path)
        logger.info(f'Data loaded from {path}, total {len(self.current_scanner)} rows.')

        self.idx_list = list(range(len(self.current_scanner)))
        if self.shuffle:
            rand = random.Random(':'.join([str(self.seed), str(index.epoch), str(index.part)]))
            rand.shuffle(self.idx_list)

        self.index.part = index.part
        self.index.epoch = index.epoch
        self.index.row = index.row

    def _try_set_scanner(self, index: Index) -> None:
        if self.current_scanner is None or self.index.part != index.part or self.index.epoch != index.epoch:
            self.set_scanner(index)

        self.index.part = index.part
        self.index.epoch = index.epoch
        self.index.row = index.row

    def seek(self, index: Index) -> Row:
        self._try_set_scanner(index)
        assert self.current_scanner is not None
        assert self.idx_list is not None
        return self.current_scanner[self.idx_list[index.row]]

    def __iter__(self) -> Iterator[Row]:
        while True:
            self._try_set_scanner(self.index)
            assert self.current_scanner is not None
            assert self.idx_list is not None
            for part_idx in range(self.index.part, len(self.path_list)):
                self._try_set_scanner(Index(self.index.epoch, part_idx, self.index.row))
                for row_idx in range(self.index.row, len(self.current_scanner)):
                    yield self.current_scanner[self.idx_list[row_idx]]
                    self.index.row += 1
                self.index.row = 0

            self._try_set_scanner(Index(self.index.epoch + 1, 0, 0))

            if not self.infinite:
                break

    def is_countable(self) -> bool:
        if len(self.path_list) == 1 and not self.infinite:
            return True
        return False

    def __len__(self) -> int:
        self._try_set_scanner(self.index)
        if not self.is_countable():
            raise NotCountableError("Reader is not countable now.")
        assert self.current_scanner is not None
        return len(self.current_scanner)


class ResumableShardReader:
    """Infinite shard player with a portable, shard-keyed cursor.

    Designed for ``BatchDataLoader`` resume: unlike ``ShardedReader`` (whose
    row permutation is seeded by the *slice position* of a shard and therefore
    tied to one particular consumer), every per-shard row permutation here is
    seeded by the shard's own identity::

        Random(f'{row_seed}:{epoch}:{shard_url}')

    so the state ``{shard_url: [epoch, next_row]}`` means the same thing for
    ANY consumer that owns the shard — worker/rank counts can change between
    save and resume and the cursors still apply after re-slicing.

    Playback order: shards lagging in epoch play first (keeps coverage uniform
    after a resume hands a consumer a mix of half-played and fresh shards);
    within the same epoch the order is a per-epoch shuffle seeded by
    ``order_seed`` (consumer-local, purely cosmetic). A shard resumes from its
    stored ``next_row`` and, when exhausted, moves to ``[epoch + 1, 0]``.
    """

    def __init__(
        self,
        scanner_type: type[Scanner],
        shard_urls: list[str],
        row_seed: int,
        order_seed: int,
        shuffle: bool = True,
        state: dict[str, list[int]] | None = None,
    ) -> None:
        assert len(shard_urls) > 0, 'ResumableShardReader needs at least one shard'
        self.scanner_type = scanner_type
        self.shard_urls = list(shard_urls)
        self.row_seed = row_seed
        self.order_seed = order_seed
        self.shuffle = shuffle
        # cursor: url -> [epoch, next_row, nrows]; next_row indexes the
        # PERMUTED order, nrows is the row count the cursor was taken against
        # (-1 = unknown, e.g. a 2-element cursor from an older release)
        self._problems: dict[str, int] = {}   # shard -> times it has failed
        self.state: dict[str, list[int]] = {}
        for url in self.shard_urls:
            cur = list((state or {}).get(url, [0, 0, -1]))
            while len(cur) < 3:
                cur.append(-1)
            self.state[url] = cur

    @property
    def epoch(self) -> int:
        """Completed epochs = the minimum epoch across owned shards."""
        return min(v[0] for v in self.state.values())

    def _next_shard(self) -> str:
        min_epoch = min(v[0] for v in self.state.values())
        candidates = [u for u in self.shard_urls if self.state[u][0] == min_epoch]
        if self.shuffle:
            order = sorted(candidates)
            random.Random(f'{self.order_seed}:{min_epoch}').shuffle(order)
        else:
            order = candidates
        # prefer a shard already mid-way (there is at most a handful right
        # after a resume) so partial progress is finished off first
        for u in order:
            if self.state[u][1] > 0:
                return u
        return order[0]

    def _row_perm(self, url: str, epoch: int, n: int) -> list[int]:
        idx = list(range(n))
        if self.shuffle:
            random.Random(f'{self.row_seed}:{epoch}:{url}').shuffle(idx)
        return idx

    def __iter__(self) -> Iterator[Row]:
        # Livelock guard. `_next_shard()` is a pure function of the cursor
        # table, so anything that yields nothing while leaving the table
        # unchanged would spin forever at zero throughput.
        #
        # The signal is "how many full sweeps have gone by with no row", not
        # any classification of why a shard was quiet — earlier attempts to
        # classify got it wrong in both directions. A shard sitting at its
        # boundary cursor legitimately yields nothing on the sweep right after
        # a resume, then serves its next epoch; so one silent sweep is normal
        # and TWO is not.
        #
        # A sweep is counted by DISTINCT shards seen, not by visits: quiet
        # shards are rolled forward one epoch at a time, so a shard lagging
        # the others by a wide epoch gap gets re-selected on every visit until
        # it catches up. Counting visits made that look like a stuck stream
        # and killed workers that still had thousands of readable rows.
        quiet_shards: set[str] = set()
        silent_sweeps = 0
        # A slice can be a single shard (a source with fewer shards than
        # consumers), where two sweeps means two consecutive failed opens —
        # a brief network blip would kill the run. Require a floor on the
        # number of quiet visits before declaring the source dead.
        sweeps_needed = max(2, -(-_MIN_QUIET_VISITS // len(self.shard_urls)))
        while True:
            url = self._next_shard()
            # tolerate 2-element cursors (older releases, or state assigned
            # directly rather than through the constructor)
            cur = self.state[url]
            if len(cur) < 3:                  # 2-element cursor: older release
                cur = list(cur) + [-1] * (3 - len(cur))
                self.state[url] = cur
            epoch, start_row, known_n = cur
            try:
                scanner = self.scanner_type(url)
                n = len(scanner)
            except Exception as e:
                self._note_shard_problem(url, f'unreadable: {e}')
                self.state[url] = [epoch + 1, 0, known_n]
                silent_sweeps = self._note_quiet(url, quiet_shards, silent_sweeps, sweeps_needed)
                continue

            if known_n >= 0 and known_n != n:
                logger.warning(
                    f'shard {url} has {n} rows but its cursor was recorded '
                    f'against {known_n}; the row order is derived from the row '
                    f'count, so the cursor no longer maps — replaying this '
                    f'shard from the start of epoch {epoch}'
                )
                start_row = 0
            elif start_row > n:
                logger.warning(
                    f'cursor for {url} points past its {n} rows — replaying '
                    f'from the start of epoch {epoch}'
                )
                start_row = 0

            if n == 0:
                self._note_shard_problem(url, 'empty')
                self.state[url] = [epoch + 1, 0, n]
                silent_sweeps = self._note_quiet(url, quiet_shards, silent_sweeps, sweeps_needed)
                continue

            # A shard whose cursor already sits at its last row is COMPLETE,
            # not barren: it yields nothing now and rolls to the next epoch,
            # where it will serve rows again. Cursors land there routinely —
            # the row index advances before the yield, so any snapshot taken
            # while paused on a shard's final row records exactly this state.
            # Counting it as "no progress" turned an ordinary resume into a
            # worker-killing RuntimeError.
            perm = self._row_perm(url, epoch, n)
            dropped = 0
            last_error: Exception | None = None
            for r in range(start_row, n):
                self.state[url][1] = r + 1
                self.state[url][2] = n
                try:
                    row = scanner[perm[r]]
                except Exception as e:
                    # a shard that opens but fails mid-read (truncated parquet,
                    # corrupted row group) must not kill the worker. Counted
                    # and reported once per shard: a shard where EVERY row
                    # fails would otherwise emit a warning per row — measured
                    # at ~120k lines/second — while making no progress at all.
                    dropped += 1
                    last_error = e
                    continue
                quiet_shards.clear()
                silent_sweeps = 0
                yield row
            if dropped:
                logger.warning(
                    f'dropped {dropped}/{n - start_row} unreadable rows of '
                    f'{url}; last error: {last_error}'
                )
            if dropped == n - start_row and n > start_row:
                # opened fine, produced nothing: still a quiet shard
                silent_sweeps = self._note_quiet(url, quiet_shards, silent_sweeps, sweeps_needed)
            self.state[url] = [epoch + 1, 0, n]

    def _note_shard_problem(self, url: str, why: str) -> None:
        """Log a bad shard on a decaying schedule.

        A failing shard is revisited every sweep, and the failure handler
        advances its epoch, so throttling on (shard, epoch) never suppressed
        anything — the pair could not repeat. Count the failures per shard
        instead and report the 1st, 10th, 100th ... so a persistent problem
        stays visible without turning into a flood.
        """
        n = self._problems.get(url, 0) + 1
        self._problems[url] = n
        if n == 1 or (n < 10 ** 9 and n in (10, 100, 1000, 10000)):
            suffix = '' if n == 1 else f' (x{n})'
        elif n % 10000 == 0:
            suffix = f' (x{n})'
        else:
            return
        logger.warning(f'skipping shard {url} — {why}{suffix}')

    def _note_quiet(self, url: str, quiet_shards: set, silent_sweeps: int,
                    sweeps_needed: int = 2) -> int:
        """Record a shard that produced nothing and raise if the stream is
        genuinely stuck.

        A sweep ends when every shard has been quiet at least once since the
        last row; two sweeps in a row with no row at all means nothing here
        can produce. Repeat visits to the same lagging shard do not count
        twice — that is the epoch-gap false positive."""
        quiet_shards.add(url)
        if quiet_shards >= set(self.shard_urls):
            quiet_shards.clear()
            silent_sweeps += 1
            if silent_sweeps >= sweeps_needed:
                raise UnusableDataSource(
                    f'no row came out of any of the {len(self.shard_urls)} shards '
                    f'in two full sweeps (unreadable, empty, or every row '
                    f'failing) — data source unusable'
                )
        return silent_sweeps

    def is_countable(self) -> bool:
        return False        # infinite by construction

    def snapshot(self) -> dict[str, list[int]]:
        """Copy of the cursor table, safe to pickle/share."""
        return {u: list(v) for u, v in self.state.items()}


class CombinedReader:
    def __init__(self, scanner_type: type[Scanner], path_list: list[str]) -> None:
        self.scanner_type = scanner_type
        self.path_list = path_list

        self.rows: list[Row] = []
        for scanner in [scanner_type(path) for path in path_list]:
            for row in scanner:
                self.rows.append(row)
        
        self.rows.__getitem__

    def __len__(self) -> int:
        return len(self.rows)

    @overload
    def __getitem__(self, i: SupportsIndex, /) -> Row: ...
    @overload
    def __getitem__(self, s: slice, /) -> list[Row]: ...
    def __getitem__(self, key: slice | SupportsIndex, /) -> Row | list[Row]:
        return self.rows[key]


class RowMapFunctionError(Exception):
    ...


class RowFlowFunctionError(Exception):
    ...


class UnusableDataSource(RuntimeError):
    """The source itself cannot serve data — not a bad row.

    ``ignore_error`` exists to skip individual poisoned rows, so every other
    exception is downgraded to a sentinel the worker drops. That is exactly
    wrong for a source-level failure: the sentinel is dropped, a fresh
    iterator is built over the same dead source, and the pipeline spins at
    zero throughput reporting nothing. This one is never downgraded.

    Subclasses RuntimeError so existing handlers still catch it.
    """


class FlowIterator(Iterator[Row | RowMapFunctionError | RowFlowFunctionError]):
    def __init__(
        self, 
        source: Iterator[Row],
        map_fn: Callable[[Row], Row], 
        flow_fn: Callable[[Iterable[Row]], Iterable[Row]],
        allow_error: bool = False
    ) -> None:
        self.source = source
        self.map_fn = map_fn
        self.flow_fn = flow_fn
        self.allow_error = allow_error
        # The flow is opened ONCE and kept: a flow function may buffer input
        # rows and emit several rows per group (packing, windowing, splitting),
        # and rebuilding it per __next__ would throw away everything it had
        # already produced but not yet handed out. Measured on a pixel-LM
        # packing flow that renders 8 windows per call: 8x the rendering work
        # and 8x the rows pulled from the reader, for one row consumed.
        self._flow: Iterator[Row] | None = None

    def _open_flow(self) -> Iterator[Row]:
        def _map_stream(
            source: Iterator[Row], 
            map_fn: Callable[[Row], Row]
        ) -> Iterator[Row]:
            while True:
                try:
                    row = next(source)
                except StopIteration:
                    break
                try:
                    row = map_fn(row)
                    yield row
                except Exception as e:
                    raise RowMapFunctionError(f"Row mapping function failed: {e}")

        return iter(self.flow_fn(_map_stream(self.source, self.map_fn)))

    def __next__(self) -> Row | RowMapFunctionError | RowFlowFunctionError:
        if self._flow is None:
            self._flow = self._open_flow()
        try:
            return next(self._flow)
        except StopIteration:
            # exhaustion is terminal for this iterator: re-opening here would
            # resurrect a flow that emits on empty input (flush/sentinel
            # patterns alternate forever). DataFlow.__iter__ mints a new
            # FlowIterator when the caller genuinely wants another pass.
            raise
        except UnusableDataSource:
            # never downgraded to a sentinel: the caller would drop it, build
            # a fresh iterator over the same dead source, and spin forever
            self._flow = None
            raise
        except RowMapFunctionError as e:
            self._flow = None
            if self.allow_error:
                logger.warning(f'Ignore error in map function: {e}')
                return e
            else:
                raise e
        except Exception as e:
            self._flow = None
            if self.allow_error:
                logger.warning(f'Ignore error in flow function: {e}')
                return RowFlowFunctionError(f"Row flow function failed: {e}")
            else:
                raise e


class DataFlow(
    SubclassTracer, 
    Iterable[Row | RowMapFunctionError | RowFlowFunctionError], 
):
    """
    DataFlow is a class that provides a unified interface for data processing.
    It can be used to map data in a map style or flow style which returns an iterable
    """

    def __init__(
        self,
        reader: ShardedReader,
        map_fn: Callable[[Row], Row] | None = None,
        flow_fn: Callable[[Iterable[Row]], Iterable[Row]] | None = None,
        ignore_error: bool = False
    ) -> None:
        super().__init__()
        self._is_countable = True

        self.reader = reader
        if not self.reader.is_countable():
            self._is_countable = False

        if map_fn is None:
            self.map_fn: Callable[[Row], Row] = lambda x: x
        else:
            self.map_fn = map_fn

        if flow_fn is None:
            self.flow_fn: Callable[[Iterable[Row]], Iterable[Row]] = lambda x: x
        else:
            self.flow_fn = flow_fn
            self._is_countable = False

        self.allow_error = ignore_error
        if self.allow_error:
            self._is_countable = False

    def is_countable(self) -> bool:
        """
        Check if the DataFlow is countable.
        """
        return self._is_countable

    def __iter__(self) -> Iterator[Row | RowMapFunctionError | RowFlowFunctionError]:
        return FlowIterator(
            source=iter(self.reader),
            map_fn=self.map_fn,
            flow_fn=self.flow_fn,
            allow_error=self.allow_error
        )

    def __len__(self) -> int:
        if not self.is_countable():
            raise NotCountableError("DataFlow is not countable now.")
        return len(self.reader)

    def seek(self, index: Index) -> Row | RowMapFunctionError | RowFlowFunctionError:
        if not self.is_countable():
            raise NotCountableError("DataFlow is not countable so cannot get row by index.")
        raw_row = self.reader.seek(index)
        try:
            mapped_row = self.map_fn(raw_row)
        except Exception as e:
            if self.allow_error:
                return RowMapFunctionError(f"Row mapping function failed: {e}")
            else:
                raise RowMapFunctionError(f"Row mapping function failed: {e}")
        try:
            processed_row = self.flow_fn([mapped_row])
        except Exception as e:
            if self.allow_error:
                return RowFlowFunctionError(f"Row flow function failed: {e}")
            else:
                raise RowFlowFunctionError(f"Row flow function failed: {e}")
        return next(iter(processed_row))
