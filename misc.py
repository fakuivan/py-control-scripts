from typing import Callable, Any, Mapping, TypeVar, Union, Optional, Iterable, Sequence
from nptdms import TdmsFile
import inspect
import numpy as np
from more_itertools import unzip
from functools import partial

def read_tdms(file, path=None):
    tdms = TdmsFile.read(file)
    if path is not None:
        group, channel = path
        return tdms[group][channel][:]
    first_group, = tdms
    first_channel, = tdms[first_group]
    return tdms[first_group][first_channel][:]


R = TypeVar('R')
def factory_partial(
        *factories: Callable[[], Any],
        **kwfactories: Callable[[Union[tuple[()], tuple[Any]]], Any]
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Similarly to how functools.partial binds a function argument
    to a value, this binds a function argument to the result of a
    "factory" function, called each time the bound function is
    called. If a kwarg is given by the caller, the factory function
    for it will recieve that argument in a singleton tuple, instead
    of an empty tuple
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            bound_args = [factory() for factory in factories]
            bound_kwargs = {k: factory(get_in_monad(kwargs, k)) for k, factory \
                in kwfactories.items() if k not in kwargs}
            free_kwargs = {k: v for k, v in kwargs.items() if k not in bound_kwargs}
            return f(*bound_args, *args, **bound_kwargs, **free_kwargs)
        return wrapper
    return decorator

K = TypeVar('K')
V = TypeVar('V')
def get_in_monad(
    mapping: Mapping[K, V],
    key: K,
) -> Union[tuple[()], tuple[V]]:
    return (mapping[key],) if key in mapping else ()

T = TypeVar('T')
def filter_none(seq: Iterable[Optional[T]]) -> Iterable[T]:
    for item in seq:
        if item is not None:
            yield item

def interweave(v1, v2) -> np.ndarray:
    return np.vstack((v1, v2)).reshape((-1,),order='F')

def progpow(base, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    return np.ones_like(arr)*base**(np.cumsum(np.ones_like(arr))-1)

def concat(*arrs) -> np.ndarray:
    return np.concatenate(tuple(map(np.asarray, arrs)))

def insert_at(outer_arr, arr, n) -> np.ndarray:
    outer_arr = np.asarray(outer_arr)
    prev, post = np.split(outer_arr, (n,))
    return concat(prev, arr, post)

def repr_aligned(mapping: Mapping, indent: int=4) -> str:
    min_spacing = max(len(repr(k)) for k in mapping.keys())
    indent_str = " "*indent
    return f"{{{{\n{indent_str}{{}}\n}}}}".format(f"\n{indent_str}".join(
        f"{(repr(k) + ': ').ljust(min_spacing+2)}{v!r}," \
            for k, v in mapping.items()
    ))

def get_kwargs(f):
    sig = inspect.signature(f)
    return {
        k for k, v in sig.parameters.items()
        if v.kind in {
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD
        }
    }

K = TypeVar("K")
V = TypeVar("V")
def dict_partition(
    pred: Callable[[K, V], bool],
    dict_: dict[K, V]
) -> tuple[dict[K, V], dict[K, V]]:
    false = {}
    true = {}
    for k, v in dict_.items():
        if pred(k, v):
            true[k] = v
        else:
            false[k] = v
    return false, true

def map_kwargs(mapf, mapped, f, not_mapped=None):
    if not_mapped is None:
        f_kwargs = get_kwargs(f)
        not_mapped = {k for k in f_kwargs if k not in mapped}
    def wrapper(*args, **kwargs):
        mapped_kwargs, not_mapped_kwargs = dict_partition(
            lambda k, _: k in not_mapped, kwargs
        )
        return f(*args, **not_mapped_kwargs, **mapf(**mapped_kwargs))
    return wrapper

def map_kwargs_dec(mapf, mapped, not_mapped=None):
    return lambda f: map_kwargs(mapf, mapped, f, not_mapped=not_mapped)

def inner_default(kwargs, *chain, **defaults):
    return map_inner(
        lambda obj, at: obj.get(at, {}),
        lambda obj, key, val: obj | {key: val},
        lambda opt: defaults | opt,
        kwargs, *chain)

def map_inner(get, merge, f, obj, *keys):
    if len(keys) == 0:
        return f(obj)
    head, *tail = keys
    next_ = get(obj, head)
    return merge(obj, head, map_inner(get, merge, f, next_, *tail))

def np_map(f, *arrs, depth=1, dtype=None):
    arrs = map(np.asanyarray, arrs)
    f = partial(np_map, f, depth=depth-1, dtype=dtype) if depth > 1 else f
    return np.array([f(*x) for x in unzip(arrs)], dtype=dtype)

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")
def valmap(f: Callable[[V], T], dict_: dict[K, V]) -> dict[K, T]:
    return {k: f(v) for k, v in dict_.items()}

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")
def keymap(f: Callable[[K], T], dict_: dict[K, V]) -> dict[T, V]:
    return {f(k): v for k, v in dict_.items()}

def last_same(arr):
    df, = np.where(np.diff(arr) != 0)
    return None if len(df) < 1 else df[-1]+1

def inverse_arange(arr) -> tuple[float, float, float]:
    arr = np.asanyarray(arr)
    if len(arr) < 2:
        raise ValueError(
            "Array must contain at least two elements")
    diffs = np.diff(arr)
    if not np.allclose(diffs, diffs[0]):
        raise ValueError(
            "All points in array must increas at the same rate")
    return min(arr), max(arr), diffs[0]

K=TypeVar("K")
V=TypeVar("V")
def unpack(dict_: Mapping[K, V], *keys: K) -> V:
    for k in keys:
        yield dict_[k]

T=TypeVar("T")
def default_seq_accesor(default: T, obj: Sequence[T]):
    def get(index) -> T:
        if 0 <= index < len(obj):
            return obj[index]
        return default
    return get