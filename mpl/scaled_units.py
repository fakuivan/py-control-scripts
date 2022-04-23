from utils.units import NumericalBasis, basis
import numpy as np
from matplotlib.units import ConversionInterface, AxisInfo, ConversionError
import matplotlib.units
import matplotlib.axis
import matplotlib.axes
from typing import Callable, NamedTuple, Any
from sympy import Expr, sympify
from ..math import db
from . import latex_notext
one = sympify(1)

"""
First prototype of how using matplotlib.units for numerical units could work.
This solution is very much non optimal. Having to wrap numpy arrays that should
"just work" is not really the idea of using a numeric representations for units.
"""

class WithUnit:
    """
    Container to trigger matplotlib's unit conversion
    
    We can't use NamedTuple here since matplotlib will think
    this is a 1*n array instead of an unsupported object
    """
    def __init__(self, value, expand=False) -> None:
        self.value = value
        self.expand = expand

    def __repr__(self):
        return f"WithUnit({self.value!r}, expand={self.expand})"

class AxisUnit(NamedTuple):
    unit: Expr
    basis: NumericalBasis
    in_db: bool = False

    @classmethod
    def unitary(cls, in_db: bool = False):
        return cls(one, basis(unitary=True), in_db)

    def to_scalar(self, quantity):
        if isinstance(quantity, Expr):
            return self.to_scalar(self.basis[quantity])
        factor = self.basis[self.unit]
        return quantity / factor if not self.in_db else quantity - db(factor)

    def format_label(self, label):
        db_prefix = "$\\mathrm{dB}$ " if self.in_db else ""
        return f"{label} [{db_prefix}${latex_notext(self.unit)}$]" if self.unit != 1 else label


class NumericalUnitsConverter(ConversionInterface):
    @staticmethod
    def axisinfo(unit: AxisUnit, axis: matplotlib.axis.Axis) -> AxisInfo:
        return AxisInfo()

    @staticmethod
    def convert(quantity: Any, unit: AxisUnit, axis: matplotlib.axis.Axis):
        """
        Converts a quantity into a scalar
        
        Given that matplotlib expects us to pass in a sequence of classes and not primitives,
        we pass in a single WithUnit instance that contains the sequence of primitives, convert
        that sequence into scalars and then pop it out of WithUnit.
        ```
        plot([WithUnit([1, 2, 3, 4], expand=True)], [WithUnit([2, 3, 4, 5], expand=True)])
        ```
        instead of:
        ```
        plot(map(WithUnit, [1, 2, 3, 4]), map(WithUnit, [2, 3, 4, 5]))
        ```
        """
        def convert_one(quantity, can_expand=True):
            if isinstance(quantity, WithUnit):
                if not can_expand and quantity.expand:
                    raise ConversionError(
                        "Can't expand WithUnit, passed iterable of WithUnits must have only one element")
                return unit.to_scalar(quantity.value)
            raise ConversionError("Expected a value in a WithUnit instance")

        # matplotlib is all over the place with how arguments are passed down
        # to us, so we have to be a bit more explicit as to how these they should
        # be interpreted, WithUnit.expand is a result of this behaviour.
        if not np.iterable(quantity):
            return convert_one(quantity)
        head, *tail = quantity
        if len(tail) == 0:
            converted = convert_one(head)
            return converted if head.expand else np.array([converted])
        return np.array([convert_one(q, can_expand=False) for q in [head, *tail]])

    @staticmethod
    def default_units(x, axis):
        return AxisUnit.unitary()

matplotlib.units.registry[WithUnit] = NumericalUnitsConverter

def unitize(*arrays):
    return map(lambda x: WithUnit(np.asanyarray(x), expand=True), arrays)

def pprint(arg):
    print(arg)
    return arg

def quantity_kwargs(kwargs, *names, **quantity_defaults):
    kwargs = quantity_defaults | kwargs
    names = [*names, *quantity_defaults.keys()]
    return { k: v if k not in names else WithUnit(v) for k, v in kwargs.items()}

def arg_map(prop: str):
    def get_method(f: Callable[..., tuple[tuple, dict]]) -> Callable[..., Any]:
        def wrapper(self, *args, **kwargs):
            nargs, nkwargs = f(self, *args, **kwargs)
            target_func = getattr(getattr(self, prop), f.__name__)
            return target_func(*nargs, **nkwargs)
        return wrapper
    return get_method

class MeasuredAxes(NamedTuple):
    axes_fmap = arg_map("axes")
    axes: matplotlib.axes.Axes
    units: tuple[AxisUnit, AxisUnit]

    @classmethod
    def embed(cls, axes: matplotlib.axes.Axes, units: tuple[AxisUnit, AxisUnit]):
        ma = cls(axes, units)
        ma.set_units()
        axes.u = ma
        return ma

    def set_units(self):
        self.axes.xaxis.set_units(self.units[0])
        self.axes.yaxis.set_units(self.units[1])

    @axes_fmap
    def plot(self, x, y, *args, **kwargs):
        return (*unitize(x, y), *args), kwargs
    
    @axes_fmap
    def scatter(self, x, y, *args, **kwargs):
        return (*unitize(x, y), *args), kwargs
    
    @axes_fmap
    def bar(self, x, y, *args, **kwargs):
        return (*unitize(x, y), *args), quantity_kwargs(kwargs, "width")
    
    @axes_fmap
    def set_xlabel(self, text, *args, **kwargs):
        return (self.units[0].format_label(text), *args), kwargs
    
    @axes_fmap
    def set_ylabel(self, text, *args, **kwargs):
        return (self.units[1].format_label(text), *args), kwargs

    @axes_fmap
    def stem(self, x, y, *args, **kwargs):
        return (*unitize(x, y), *args), quantity_kwargs(kwargs, bottom=0)

    def twinx(self, y_unit=None, basis=None, in_db=False):
        return self._twin(True, y_unit, basis, in_db)

    def twiny(self, x_unit=None, basis=None, in_db=False):
        return self._twin(False, x_unit, basis, in_db)

    def _twin(self, twinx: bool, unit, basis, in_db):
        ax = self.axes.twinx() if twinx else self.axes.twiny()
        if unit is None:
            return ax
        keep_unit = self.units[0] if twinx else self.units[1]
        basis = keep_unit.basis if basis is None else basis
        new_axis = AxisUnit(unit, basis, in_db=in_db)
        MeasuredAxes.embed(ax, (
            keep_unit, new_axis
                ) if twinx else (
            new_axis, keep_unit))
        return ax


def axes_unit(xunit, yunit, basis, xdb=False, ydb=False):
    return AxisUnit(xunit, basis, in_db=xdb), AxisUnit(yunit, basis, in_db=ydb)
