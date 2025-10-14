from typing import TypeVar
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from ..interfaces import SubclassTracer, Row

R = TypeVar('R')


class Scanner(ABC, SubclassTracer, Iterable[Row]):
    """
    The scanner interface for individual file reader.
    """

    def __init__(self, path: str, **kwargs) -> None:
        super().__init__()
        self.path = path
        self.kwargs = kwargs

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of items in the scanner.
        This method should be implemented by subclasses.
        """

    @abstractmethod
    def __getitem__(self, idx: int) -> Row:
        """
        Returns the item at the given index.
        This method should be implemented by subclasses.
        """

    @classmethod
    @abstractmethod
    def check_file(cls, path: str) -> bool:
        """
        Checks whether the specified path is compatible with this scanner.
        """

    def __iter__(self) -> Iterator[Row]:
        for idx in range(len(self)):
            yield self[idx]
