from typing import Callable, NamedTuple, Any, Optional
from matplotlib.axes import Axes
import matplotlib as mpl
import control
import control.matlab
import control.timeresp
from matplotlib.figure import Figure
import numpy as np
from utils.misc import np_map
import sympy as sp

from .units.num_bridge import NumericalBasis
from .misc import progpow, inverse_arange
from .ctrlsys import zoh_stairs
from .units import u
from .ratpolys import numden2ratpoly

class FakeFig(NamedTuple):
    ax_mag: Axes
    ax_phase: Axes

    @property
    def axes(self):
        return {0: self.ax_mag, 2: self.ax_phase}

def bode_plot(sys, ax_gain, ax_phase, *args, **kwargs):
    """
    Turns out that despite if you set `sisotool` to `False`, bode_plot
    uses `fig` as long as `sisotool` is in the kwargs
    """
    return control.bode_plot(
        sys, *args, sisotool=False,
        fig=FakeFig(ax_gain, ax_phase), **kwargs)

def sys_deu(sys, basis, uin, uout, u=u):
    sys = sys/basis[uout/uin]
    if sys.dt > 0:
        dt = sys.dt/basis[u.s]
        sys = control.TransferFunction(sys)
        sys.dt = dt
        return sys.minreal()

    s_unit = basis[u.rad/u.s]
    def poly_deu(poly):
        return poly*np.flip(
            progpow(s_unit, poly))
    def lmap2(f, mat):
        return [[f(x) for x in vec] for vec in mat]
    den = lmap2(poly_deu, sys.den)
    num = lmap2(poly_deu, sys.num)

    return control.TransferFunction(num, den).minreal()

def nu_stepinfo(
    basis: NumericalBasis,
    t_units, y_units, stepinfo,
    ampl=1, percent=u.percent
) -> dict[str, Any]:
    in_percent = ["Overshoot", "Undershoot"]
    in_y = ["SettlingMin", "SettlingMax", "Peak", "SteadyStateValue"]
    in_t = ["SettlingTime", "RiseTime", "PeakTime"]
    with_units = {
        prop: basis.to_symb(percent, stepinfo[prop]/100) for prop in in_percent} | {
        prop: basis.to_symb(y_units, stepinfo[prop]*ampl) for prop in in_y } | {
        prop: basis.to_symb(t_units, stepinfo[prop]) for prop in in_t
    }
    # preserve the original order
    return {k: with_units[k] for k in stepinfo.keys()}

def bode_axes(
    subplots=mpl.pyplot.subplots,
    angle_locator=mpl.ticker.MultipleLocator
) -> tuple[
        Figure, tuple[Axes, Axes],
        Callable[[control.TransferFunction], Any],
        Callable[..., Any]
]:
    fig, (ax_gain, ax_phase) = subplots(2, sharex=True)
    ax_gain.set_xscale('log')

    def add_deg_locator(phase_tick=45):
        loc = angle_locator(base=phase_tick)
        loc_minor = angle_locator(base=phase_tick/3)
        ax_phase.yaxis.set_major_locator(loc)
        ax_phase.yaxis.set_minor_locator(loc_minor)
    
    return ( 
        fig, (ax_gain, ax_phase),
        lambda sys, **kwargs: bode_plot(
            sys, ax_gain, ax_phase, grid=False, **kwargs),
        add_deg_locator
    )

def sys2zpk(sys):
    return np_map(control.matlab.tf2zpk,
        sys.num, sys.den, depth=2, dtype=object)

def siso_tf2zpk(sys):
    return control.matlab.tf2zpk(
        *siso_numden(sys))

def siso_numden(sys):
    ((num,),) = sys.num
    ((den,),) = sys.den
    return num, den

def siso_tf2k(sys):
    _, _, k = siso_tf2zpk(sys)
    return k

def expr2sys(sympy_expr: sp.Basic, var: sp.Basic, tf=None
) -> control.TransferFunction:
    if tf is None:
        tf = str(var)
    if isinstance(tf, str):
        tf = control.matlab.tf(tf)
    one = tf*0+1
    return one*sp.lambdify([var], sympy_expr)(tf)

def sys2expr(
    siso_sys: control.TransferFunction,
    var: sp.Basic
) -> sp.Basic:
    num, den = siso_numden(siso_sys)
    return numden2ratpoly(num, den, var)


class TimeSeries(NamedTuple):
    start: float
    end: float
    dt: Optional[float]
    @classmethod
    def from_t(cls, t_vec):
        return cls(*inverse_arange(t_vec))

    def with_dt(self, dt):
        return TimeSeries(self.start, self.end, dt)

    def __call__(self):
        return np.arange(self.start, self.end, self.dt)
        

def step_resp(plot_ref, plot_resp, next_color, ampl=1):
    current_t = None
    def get_t(t, dt, get_dt):
        is_dt = dt != 0
        def set_t(t):
            nonlocal current_t
            current_t = t
            return t
        
        if t is not None:
            set_t(TimeSeries.from_t(t))
            return t, lambda _: None

        if current_t is None:
            return None, lambda t: set_t(TimeSeries.from_t(t))

        if is_dt:
            return current_t.with_dt(dt)(), lambda _: None
        
        if current_t.dt is None:
            set_t(current_t.with_dt(get_dt()))
            return current_t(), None

    def get_ideal_dt(sys):
        _, dt = control.timeresp._ideal_tfinal_and_dt(sys)
        return dt
    
    def plot_step(sys, t=None, color=None, **plot_kwargs):
        t, set_t = get_t(t, sys.dt, lambda: get_ideal_dt(sys))
        t, resp = control.step_response(sys, T=t)
        set_t(t)
        resp = resp * ampl
        is_dt = sys.dt != 0
        tg, respg = (t, resp) if not is_dt else zoh_stairs(t, resp)
        color = next_color() if color is None else color
        return (t, resp), plot_resp(tg, respg, color=color, **plot_kwargs)

    def plot_reference(color=None, **plot_kwargs):
        color = color if color is not None else next_color()
        return plot_ref(
            [current_t.start, current_t.end],
            [ampl, ampl], color=color, **plot_kwargs
        )

    return plot_step, plot_reference