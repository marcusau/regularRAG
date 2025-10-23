from pathlib import Path
from typing import Union
import time
import functools
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@functools.lru_cache(maxsize=3)
def read_txtfile(filepath:Union[str,Path])->str:
    if isinstance(filepath,str):
        filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} does not exist")
    
    with open(filepath,'r',encoding='utf-8') as f:
        content = f.read()
    
    return content