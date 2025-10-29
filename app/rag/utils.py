import functools
import time
from functools import wraps
from pathlib import Path
from typing import Union


def timer(func):
    """Decorator that times a function call and prints the elapsed seconds."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    return wrapper


@functools.lru_cache(maxsize=3)
def read_txtfile(filepath: Union[str, Path]) -> str:
    """Read a UTF-8 text file and return its content, cached by path.

    The Path is normalized from `str` inputs. Raises `FileNotFoundError`
    if the path does not exist.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} does not exist")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    return content
