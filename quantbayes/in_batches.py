import asyncio
import inspect
from collections.abc import Generator, Iterable
from functools import wraps
from typing import Any, Callable, Optional


def _get_batches(iterable: Iterable[Any], batch_size: int) -> Generator[list[Any], None, None]:
    """
    Helper function that yields batches (as lists) from an iterable.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield list(batch)
            batch.clear()
    if batch:
        yield list(batch)


def in_batches(batch_size: int, async_mode: bool = False, param: Optional[str] = None) -> Callable:
    """
    A decorator that splits an input iterable (e.g. list of sentences)
    into batches and calls the decorated function once per batch.

    The decorated function is expected to have one parameter that receives the iterable.
    For methods (instance or class), if the first parameter is named "self" or "cls",
    second parameter is assumed to be the input iterable unless overridden the `param` argument.

    If async_mode is True then the wrapper becomes an async function. In that case, if the
    decorated function is a coroutine function it is awaited; otherwise it is run in a thread
    via asyncio.to_thread. All batch calls are scheduled concurrently via asyncio.gather.

    Args:
        batch_size (int): The size of each batch.
        async_mode (bool): If True, the wrapper is asynchronous and uses asyncio.
        param (Optional[str]): Explicitly specify the parameter name that holds the iterable.
                               If not provided, a heuristic based on the function signature is used.

    Raises:
        ValueError: If batch_size is not a positive integer or if the function has no parameters.
        TypeError:  If the input value is not an iterable.
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        if not param_names:
            raise ValueError("The decorated function must have at least one parameter.")

        # Determine which parameter is expected to be the input iterable.
        if param is not None:
            iterable_param = param
        elif len(param_names) == 1:
            iterable_param = param_names[0]
        elif param_names[0] in ("self", "cls"):
            iterable_param = param_names[1]
        else:
            iterable_param = param_names[0]

        if async_mode:

            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                base_args = dict(bound_args.arguments)

                original_iterable = base_args.get(iterable_param)
                if not hasattr(original_iterable, "__iter__"):
                    raise TypeError("The input must be an iterable.")

                tasks = []
                for batch in _get_batches(original_iterable, batch_size):
                    call_args = dict(base_args)
                    call_args[iterable_param] = batch
                    if asyncio.iscoroutinefunction(func):
                        tasks.append(func(**call_args))
                    else:
                        tasks.append(asyncio.to_thread(func, **call_args))
                return await asyncio.gather(*tasks)

            return async_wrapper

        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                base_args = dict(bound_args.arguments)

                original_iterable = base_args.get(iterable_param)
                if not hasattr(original_iterable, "__iter__"):
                    raise TypeError("The input must be an iterable.")

                results = []
                for batch in _get_batches(original_iterable, batch_size):
                    call_args = dict(base_args)
                    call_args[iterable_param] = batch
                    results.append(func(**call_args))
                return results

            return sync_wrapper

    return decorator

