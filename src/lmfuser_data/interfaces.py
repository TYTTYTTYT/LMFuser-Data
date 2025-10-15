from typing import TypeVar, Protocol, TypedDict
from dataclasses import dataclass
from collections.abc import Iterable, Hashable, Sequence, Mapping

from torch import Tensor
from numpy import ndarray


class RowBase(TypedDict):
    """
    A row in a batch.
    """

T = TypeVar("T", bound="SubclassTracer")
R = TypeVar("R")

Row = TypeVar("Row", bound=RowBase)
Batch = Mapping[Hashable, Sequence[R] | Tensor | ndarray]


class SubclassTracer:
    """
    Provide methods to get children classes.
    """

    @classmethod
    def all_subclasses(cls: type[T]) -> Iterable[type[T]]:
        """
        Returns an iterable of recursive subclasses.
        """
        def _recursive_subclasses() -> Iterable[type[T]]:
            for child in cls.__subclasses__():
                yield child
                yield from child.__subclasses__()
        return _recursive_subclasses()

    @classmethod
    def direct_subclasses(cls: type[T]) -> Iterable[type[T]]:
        """
        Returns an iterable of dirct subclasses.
        """
        for child in cls.__subclasses__():
            yield child

    @classmethod
    def all_subclass_names(cls: type[T]) -> Iterable[str]:
        """
        Returns an iterable of recursive subclass names.
        """
        for child in cls.all_subclasses():
            yield child.__name__

    @classmethod
    def direct_subclass_names(cls: type[T]) -> Iterable[str]:
        """
        Returns an iterable of direct subclass names.
        """
        for child in cls.direct_subclasses():
            yield child.__name__

    @classmethod
    def all_subclass_map(cls: type[T]) -> dict[str, type[T]]:
        """
        Returns a dictionary mapping subclass names to subclasses.
        """
        name_dict: dict[str, type[T]] = {}
        for child in cls.all_subclasses():
            if child.__name__ in name_dict:
                raise ValueError(f"Duplicate child class name: {child.__name__}")
            name_dict[child.__name__] = child
        return name_dict

    @classmethod
    def direct_subclass_map(cls: type[T]) -> dict[str, type[T]]:
        """
        Returns a dictionary mapping subclass names to direct subclasses.
        """
        name_dict: dict[str, type[T]] = {}
        for child in cls.direct_subclasses():
            if child.__name__ in name_dict:
                raise ValueError(f"Duplicate child class name: {child.__name__}")
            name_dict[child.__name__] = child
        return name_dict

    @classmethod
    def get_subclass(cls: type[T], name: str) -> type[T]:
        """
        Get the subclass according to the name.
        """
        subclasses = cls.direct_subclass_map()
        if name not in subclasses:
            raise KeyError(f"Subclass '{name}' not found in {cls.__name__}.")
        return subclasses[name]


class CouldCountable(Protocol):
    """
    The protocal for countability check.
    """

    def is_countable(self) -> bool: ...


@dataclass
class Index:
    epoch: int = 0
    part: int = 0
    row: int = 0
