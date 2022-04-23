from contextlib import contextmanager
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
from ..misc import factory_partial, concat, insert_at
from sympy import Expr, latex
from matplotlib.axis import Axis
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib as mpl
from ..math import is_clockwise
from typing import Any, Callable, TypeVar

def latex_notext(expr: Expr):
    r"""
    Some backends for matplotlib do not support rendering ``\text``
    """
    return latex(expr).replace('\\text', '\\mathrm')

def share_axes_prop_cycle(prop, cycler=plt.rcParams['axes.prop_cycle']):
    props = iter(cycle(cycler.by_key()[prop]))
    return factory_partial(
        **{prop: lambda c: c[0] if len(c) == 1 else next(props)})


def io_axes(
    subplots=plt.subplots,
    rcparams=plt.rcParams
) -> tuple[Figure, tuple[Axes, Axes],
           Callable[[], str], Callable[[float, str], None]]:
    fig, ax_x = subplots()
    ax_y = ax_x.twinx()
    color_iter = iter(cycle(rcparams['axes.prop_cycle'].by_key()["color"]))
    next_color = lambda: next(color_iter)

    def post(gain: float, legend_loc: str=None):
        ax_x.set_ylim(coord/gain for coord in ax_y.get_ylim())
        if legend_loc is not None:
            fig.legend(
                bbox_to_anchor=loc_to_anchor(legend_loc),
                loc=legend_loc,
                bbox_transform=ax_x.transAxes)
    
    return fig, (ax_x, ax_y), next_color, post


def loc_to_anchor(loc: str) -> tuple[int, int]:
    return {
        'lower left': (0, 0),
        'lower right': (1, 0),
        'upper right': (1, 1),
        'upper left': (0, 1)
    }[loc]

def axis_major_minor_grid_on(axis: Axis) -> tuple[bool, bool]:
    return (
        axis._major_tick_kw.get("gridOn", False),
        axis._minor_tick_kw.get("gridOn", False)
    )

def set_datalim(ax: Axes, x=None, y=None):
    x_margin, y_margin = ax.margins()
    def set_lim(setf, margin, interval):
        start, end = interval
        width = end - start
        setf(start-width*margin, end+width*margin)
    
    if x is not None:
        set_lim(ax.set_xlim, x_margin, x)

    if y is not None:
        set_lim(ax.set_ylim, y_margin, y)

@contextmanager
def rcparam(params: dict[str, Any], **kparams: Any):
    all_params = params | kparams
    old_vals = {}
    for param, value in all_params.items():
        old_vals[param] = mpl.rcParams[param]
        mpl.rcParams[param] = value
    try:
        yield
    finally:
        for param, value in old_vals.items():
            mpl.rcParams[param] = value

def fill_outside(x, y, ll, ur, counter_clockwise=None):
    """
    Creates a polygon where x and y form a crevice of an outer
    rectangle with lower left and upper right corners `ll` and `ur`
    respectively. If `counter_clockwise` is `None` then the orientation
    of the outer polygon will be guessed to be the opposite of the
    inner connecting points.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    xmin, ymin = ll
    xmax, ymax = ur
    xmin, ymin = min(xmin, min(x)), min(ymin, min(y))
    xmax, ymax = max(xmax, max(x)), max(ymax, max(y))
    corners = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin],
        [xmin, ymin],
    ])
    lower_left = corners[0]
    # Get closest point to splicing corner
    x_off, y_off = x-lower_left[0], y-lower_left[1]
    closest_n = (x_off**2+y_off**2).argmin()
    # Guess orientation
    p = [x_off[closest_n], y_off[closest_n]]
    try:
        pn = [x_off[closest_n+1], y_off[closest_n+1]]
    except IndexError:
        # wrap around if we're at the end of the array
        pn = [x_off[0], y_off[0]]
    if counter_clockwise is None:
        counter_clockwise = not is_clockwise(*p, *pn)
    corners = corners[::-1] if counter_clockwise else corners
    # Join the arrays
    corners = concat(np.array([[x[closest_n], y[closest_n]]]), corners)
    xs, ys = np.transpose(corners)
    return insert_at(x, xs, closest_n), insert_at(y, ys, closest_n)

def shade_outside(plot, fill, contour, xlim, ylim):
    ll = xlim[0], ylim[0]
    ul = xlim[1], ylim[1]
    return fill(*fill_outside(*contour, ll, ul)), plot(*contour)

T = TypeVar("T")
def no_scaling(ax: Axes, any_collection: T) -> T:
    any_collection.remove()
    ax.relim()
    ax.add_collection(any_collection, autolim=False)
    return any_collection
