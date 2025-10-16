from typing import Any, Callable, TypeVar, cast
from collections.abc import Iterable, Iterator, Sequence
from random import Random
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')

def split_list(data_list: list[T], num_shards: int) -> list[list[T]]:
    """
    Splits a list into a specified number of shards (sublists).

    This function distributes the elements of the input list as evenly as
    possible among the requested number of shards.

    Args:
        data_list (list): The list to be split.
        num_shards (int): The positive integer number of shards to create.

    Returns:
        list: A list of lists, where each inner list is a shard.
              Returns an empty list if the input list is empty.

    Raises:
        ValueError: If num_shards is not a positive integer.
        ValueError: If num_shards is greater than the number of items in the list.
    """
    if not isinstance(num_shards, int) or num_shards <= 0:
        raise ValueError("Number of shards must be a positive integer.")

    if not data_list:
        return []

    list_len = len(data_list)

    if list_len <= 0:
        raise ValueError("List length must be greater than 0.")
    
    if num_shards > list_len:
        raise ValueError("Number of shards cannot be greater than the number of items in the list.")

    shards = []
    start_index = 0

    for i in range(num_shards):
        # Calculate the size of the current shard
        # The base size is the integer division of the total length by the number of shards
        base_size = list_len // num_shards
        # The remainder is the number of shards that will get one extra element
        remainder = list_len % num_shards
        
        # If the current shard index is less than the remainder, it gets an extra element
        shard_size = base_size + (1 if i < remainder else 0)
        
        # Calculate the end index for slicing
        end_index = start_index + shard_size
        
        # Append the sliced shard to the list of shards
        shards.append(data_list[start_index:end_index])
        
        # Update the start index for the next shard
        start_index = end_index
        
    return shards

def mix_iterables(iterables: Sequence[Iterable[T]], weights: list[float], rng: Random) -> Iterator[T]:
    """
    Mixes elements from multiple iterables randomly according to weights.

    Args:
        iterables (list[Iterable[T]]): A list of iterables to mix.
        weights (list[float]): A list of weights corresponding to each iterable.
        rng (Random): A random number generator.

    Yields:
        T: Elements from the input iterables in a mixed order.
    """
    assert len(iterables) == len(weights), "Number of iterables must be equal to number of weights."
    iters = [iter(iterable) for iterable in iterables]
    ids = list(range(len(iters)))

    while True:
        idx = rng.choices(list(ids), weights=weights, k=1)[0]
        try:
            yield next(iters[idx])
        except StopIteration:
            iters[idx] = iter(iterables[idx])
            yield next(iters[idx])

# A TypeVar is used to represent the generic type of the function being decorated.
F = TypeVar('F', bound=Callable[..., Any])

def retry(
    tries: int = 4,
    delay: int | float = 3,
    backoff: int | float = 2,
    logger: logging.Logger | None = None
) -> Callable[[F], F]:
    """
    A retry decorator with exponential backoff.

    :param tries: The maximum number of attempts.
    :param delay: Initial delay between retries in seconds.
    :param backoff: Multiplier for the delay after each failure.
    :param logger: Logger to use for output. If None, prints to console.
    """

    def deco_retry(f: F) -> F:
        @wraps(f)
        def f_retry(*args: Any, **kwargs: Any) -> Any:
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    msg = f"Function '{f.__name__}' failed with exception: {e}. Retrying in {mdelay} seconds..."
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            # Final attempt
            return f(*args, **kwargs)

        # Cast the wrapper function to the same type as the decorated function
        # to help static type checkers understand the code.
        return cast(F, f_retry)

    return deco_retry
