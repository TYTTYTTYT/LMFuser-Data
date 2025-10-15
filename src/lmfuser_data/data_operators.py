from typing import Iterable, Iterator, Callable, Generic, TypeVar
import logging
import random

from .scanners import Scanner
from .interfaces import Row, Index, SubclassTracer, RowBase

logger = logging.getLogger(__name__)


class NotCountableError(Exception):
    ...


class Reader(Generic[Row]):
    def __init__(
        self, 
        scanner_type: type[Scanner[Row]], 
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

        self.current_scanner: Scanner[Row] | None = None
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


class RowMapFunctionError(Exception):
    ...


class RowFlowFunctionError(Exception):
    ...

MappedRow = TypeVar("MappedRow", bound=RowBase)
ProcessedRow = TypeVar("ProcessedRow", bound=RowBase)


class FlowIterator(Iterator[ProcessedRow | RowMapFunctionError | RowFlowFunctionError]):
    def __init__(
        self, 
        source: Iterator[Row],
        map_fn: Callable[[Row], MappedRow], 
        flow_fn: Callable[[Iterable[MappedRow]], Iterable[ProcessedRow]],
        allow_error: bool = False
    ) -> None:
        self.source = source
        self.map_fn = map_fn
        self.flow_fn = flow_fn
        self.allow_error = allow_error

    def __next__(self) -> ProcessedRow | RowMapFunctionError | RowFlowFunctionError:
        def _map_stream(
            source: Iterator[Row], 
            map_fn: Callable[[Row], MappedRow]
        ) -> Iterator[MappedRow]:
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

        it = iter(self.flow_fn(_map_stream(self.source, self.map_fn)))
        try:
            result = next(it)
            return result
        except StopIteration as e:
            raise e
        except RowMapFunctionError as e:
            if self.allow_error:
                logger.warning(f'Ignore error in map function: {e}')
                return e
            else:
                raise e
        except Exception as e:
            if self.allow_error:
                logger.warning(f'Ignore error in flow function: {e}')
                return RowFlowFunctionError(f"Row flow function failed: {e}")
            else:
                raise e


class DataFlow(
    SubclassTracer, 
    Iterable[ProcessedRow | RowMapFunctionError | RowFlowFunctionError], 
):
    """
    DataFlow is a class that provides a unified interface for data processing.
    It can be used to map data in a map style or flow style which returns an iterable
    """

    def __init__(
        self,
        reader: Reader[Row],
        map_fn: Callable[[Row], MappedRow] | None = None,
        flow_fn: Callable[[Iterable[MappedRow]], Iterable[ProcessedRow]] | None = None,
        ignore_error: bool = False
    ) -> None:
        super().__init__()
        self._is_countable = True

        self.reader = reader
        if not self.reader.is_countable():
            self._is_countable = False

        if map_fn is None:
            self.map_fn: Callable[[Row], MappedRow] = lambda x: x # type: ignore
        else:
            self.map_fn = map_fn

        if flow_fn is None:
            self.flow_fn: Callable[[Iterable[MappedRow]], Iterable[ProcessedRow]] = lambda x: x # type: ignore
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

    def __iter__(self) -> Iterator[ProcessedRow | RowMapFunctionError | RowFlowFunctionError]:
        return FlowIterator[ProcessedRow](
            source=iter(self.reader),
            map_fn=self.map_fn,
            flow_fn=self.flow_fn,
            allow_error=self.allow_error
        )

    def __len__(self) -> int:
        if not self.is_countable():
            raise NotCountableError("DataFlow is not countable now.")
        return len(self.reader)

    def seek(self, index: Index) -> ProcessedRow | RowMapFunctionError | RowFlowFunctionError:
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
