import itertools
from typing import Iterable, NamedTuple, Any, Callable, Optional, Tuple, Type, TypeVar, ParamSpec, Union

P = ParamSpec('P')
Ri = TypeVar('Ri')
Ro = TypeVar('Ro')
class Composed(NamedTuple):
    outer: Callable[[Ri], Ro]
    inner: Callable[P, Ri]
    def __call__(self, *args, **kwargs) -> Ro:
        return self.outer(self.inner(*args, **kwargs))

    def __repr__(self):
        return f"({self.outer!r} . {self.inner!r})"

P = ParamSpec('P')
Ri = TypeVar('Ri')
Ro = TypeVar('Ro')
def compose(outer: Callable[[Ri], Ro], inner: Callable[P, Ri]) -> Callable[P, Ro]:
    return Composed(outer, inner)

class SkippableArg(NamedTuple):
    maybe: Optional[tuple[Any]] = None
    @property
    def skip(self) -> bool:
        return not isinstance(self.maybe, tuple)

    @property
    def value(self):
        if self.skip:
            raise TypeError("Argument has no assigned value")
        value, = self.maybe
        return value

    def __repr__(self) -> str:
        return "?" if self.skip else repr(self.value)

def arg(value):
    return SkippableArg((value,))

T=TypeVar("T")
V=TypeVar("V")
Predicate=Callable[[T], bool]
def replace_where(
    iterable: Iterable[T],
    pred: Predicate[T],
    replacements: Iterable[V]
) -> Iterable[T | V]:
    reps = iter(replacements)
    vals = iter(iterable)
    for val in vals:
        if pred(val):
            try:
                rep = next(reps)
            except StopIteration:
                yield val
                break
            yield rep
        else:
            yield val
    yield from itertools.chain(vals, reps)


class SkipArg:
    pass
S = SkipArg

ArgList_ = tuple[SkippableArg, ...]

class ArgList(NamedTuple):
    args: ArgList_ = ()

    def update_args(self, new_args: "ArgList") -> "ArgList":
        skip = lambda x: x.skip
        return ArgList(tuple(replace_where(
            self.args, skip, new_args.args)))

    def resolve(self):
        for i, arg in enumerate(self.args):
            if arg.skip:
                raise ValueError(
                    f"Placeholder argument {i} in {self.args!r} has no assigned value")
            yield arg.value

    def is_resolved(self):
        return all(not arg.skip for arg in self.args)

    def __add__(self, other: Union[tuple, "ArgList"]) -> "ArgList":
        return self.update_args(
            other if isinstance(other, ArgList) else as_arg_list(other))

    def __radd__(self, other: Union[tuple, "ArgList"]) -> "ArgList":
        other = other if isinstance(other, ArgList) else as_arg_list(other)
        return other.update_args(self)

    def __repr__(self) -> str:
        return f"({', '.join(map(repr, self.args))})"

    def __iter__(self):
        yield from self.resolve()
        
def as_arg_list(args: Tuple[Any]) -> ArgList:
    return ArgList(tuple(map(arg, args)))

def process_args(
    args: Iterable[Union[Type[S], tuple[Any, ...]]]
) -> Iterable[SkippableArg]:
    for arg_ in args:
        if arg_ is S:
            yield SkippableArg()
            continue
        # We're expecting _only_ tuples here, not any derived classes
        if not type(arg_) is tuple:
            raise TypeError(
                "Only tuples are allowed to wrap fixed arguments")
        yield from map(arg, arg_)

R = TypeVar('R')
class Functional(NamedTuple):
    f: Callable[..., R]
    args: ArgList = ArgList()
    kwargs: dict[str, Any] = {}

    def normalized(self) -> "Functional":
        if not isinstance(self.f, Functional):
            return self
        return self.f.normalized().arg_list_partial(self.args, self.kwargs)

    def arg_list_partial(self, args: ArgList, kwargs: dict[str, Any]) -> "Functional":
        return Functional(self.f, self.args + args, self.kwargs | kwargs)

    def P(self, *args, **kwargs) -> "Functional":
        return self.arg_list_partial(as_arg_list(args), kwargs)

    def S(self, *args):
        args_list = ArgList(tuple(process_args(args)))
        return self.arg_list_partial(args_list, {})

    def __call__(self, *args, **kwargs) -> R:
        if len(args) == len(kwargs) == 0:
            return self.f(*self.args, **self.kwargs)
        return self.P(*args, **kwargs)()

    def __matmul__(self, other) -> "Functional":
        if isinstance(other, Functional):
            return Functional(compose(self, other.f), other.args, other.kwargs).normalized()
        return Functional(compose(self, other)).normalized()

    def __rmatmul__(self, other) -> "Functional":
        return Functional(compose(other, self.f), self.args, self.kwargs).normalized()

    def __repr__(self):
        return deindent("""
        Functional(
            f={}
            args={}
            kwargs={}
        )""").format(
            indent(repr(self.f), 4),
            indent(repr_tuple_ml(self.args.args), 4),
            indent(repr(self.kwargs), 4))

def repr_tuple_ml(t: tuple, indent=4, nl="\n"):
    if len(t) < 1:
        return "()"
    spacing = ' '*indent
    return f"({nl}{spacing}{(','+nl+spacing).join(map(repr, t))}{nl})"

def fpartial(f: Callable, *args, **kwargs):
    return Functional(f).P(*args, **kwargs).normalized()

def deindent(string):
    first, *lines = string.splitlines(True)[1:]
    indents = len(first) - len(first.lstrip())
    return first[indents:] + "".join(line[indents:] for line in lines)

def indent(string, depth):
    first, *rest = string.splitlines(True)
    return first + "".join(
        depth*" " + line for line in rest)

