from .init import latex_notext
from ..misc import filter_none
from ..units import NumericalBasis
from .transformed_tickers import set_transformed_tickers
from matplotlib.transforms import Affine2D
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from sympy import Expr
from ..math import db
from typing import Callable, Union, Optional

def unit_label(unit, label, in_db=False):
    db_prefix = "$\\mathrm{dB}$" if in_db else None
    unit_label = f"${latex_notext(unit)}$" if unit != 1 else None
    full_label = " ".join(filter_none([db_prefix, unit_label]))
    return f"{label} [{full_label}]" if full_label != "" else label

def nu_transform(
    basis: NumericalBasis,
    xunit=1, yunit=1,
    x_in_db=False, y_in_db=False
) -> Affine2D:
    """
    Computes the transform associalted to a plane with the given x and y units
    under a numerical basis
    """
    xscale = basis[xunit]
    yscale = basis[yunit]
    return Affine2D().scale(
        xscale if not x_in_db else 1,
        yscale if not y_in_db else 1
    ).translate(
        db(xscale) if x_in_db else 0,
        db(yscale) if y_in_db else 0
    )

def unit_axis_set_label(set_label: Callable, unit=1, in_db=False) -> Callable:
    def wrapper(label, *args, **kwargs):
        return set_label(unit_label(unit, label, in_db), *args, **kwargs)
    return wrapper

AxisUnitSpec = Union[Expr, tuple[str, Expr]]

def axis_unitspec(spec: AxisUnitSpec) -> tuple[Optional[str], Expr]:
    label, unit = spec if isinstance(spec, tuple) else (None, spec)
    def map_label(f):
        if label is not None:
            f(label)
        return f
    return map_label, unit

def nu_axis(
    axis: Axis,
    basis: NumericalBasis,
    unit: AxisUnitSpec=1,
    in_db=False, is_x=None
) -> Callable:
    ax_name = axis.__name__ if is_x is None else "xaxis" if is_x else "yaxis"
    map_label, unit = axis_unitspec(unit)
    params = {
        "xaxis": (unit, 1, in_db, False),
        "yaxis": (1, unit, False, in_db)
    }.get(ax_name, None)
    if params is None:
        raise TypeError("Axis must either be x or y")
    
    trans = nu_transform(basis, *params)
    set_transformed_tickers(axis, trans)
    set_label = getattr(axis.axes, f"set_{ax_name[0]}label")
    return map_label(unit_axis_set_label(set_label, unit, in_db))

def nu_axes(
    basis: NumericalBasis,
    axes: Axes,
    xunit: AxisUnitSpec=1, yunit: AxisUnitSpec=1,
    x_in_db=False, y_in_db=False
) -> tuple[Callable, Callable]:
    map_xlabel, xunit = axis_unitspec(xunit)
    map_ylabel, yunit = axis_unitspec(yunit)
    trans = nu_transform(basis, xunit, yunit, x_in_db, y_in_db)
    set_transformed_tickers(axes, trans)
    return (
        map_xlabel(unit_axis_set_label(axes.set_xlabel, xunit, x_in_db)),
        map_ylabel(unit_axis_set_label(axes.set_ylabel, yunit, y_in_db))
    )

def nu_twinx_axes(
    basis: NumericalBasis,
    ax1: Axes, ax2: Axes,
    xunit=1, y1unit=1, y2unit=1
) -> tuple[Callable, Callable, Callable]:
    set_xlabel, set_y1label = nu_axes(
        basis, ax1, xunit, y1unit
    )
    set_y2label = nu_axis(
        ax2.yaxis, basis, y2unit, is_x=False
    )
    return set_xlabel, set_y1label, set_y2label

def nu_sharex_axes(
    basis: NumericalBasis,
    ax1: Axes, ax2: Axes,
    xunit=1, y1unit=1, y2unit=1,
    x_in_db=False, y1_in_db=False, y2_in_db=False
) -> tuple[Callable, Callable, Callable]:
    set_xlabel, set_y1label = nu_axes(
        basis, ax1, xunit, y1unit, x_in_db=x_in_db, y_in_db=y1_in_db
    )
    set_y2label = nu_axis(
        ax2.yaxis, basis, y2unit, is_x=False, in_db=y2_in_db
    )
    return set_xlabel, set_y1label, set_y2label
