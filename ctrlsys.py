from control import TransferFunction
import control as ctrl
import numpy as np
from .math import closest_arg, reim, with_conj, ilerp
from sympy import Expr
import sympy as sp
import math
import cmath
from .ratpolys import ratpoly_coeffs
from .misc import interweave, concat, map_kwargs_dec, last_same
from .mpl import shade_outside, no_scaling
from typing import Any, Callable, Sequence, TypeVar, Type
from functools import partial
from matplotlib import pyplot as plt
import scipy.optimize

"""
(mostly) library agnostic algorithms useful for designing linear
filters and controllers.
"""

def s2tf(expr: Expr, s: Expr, tf=TransferFunction):
    numer, denom = ratpoly_coeffs(expr, s)
    return tf(
        list(map(float, numer)),
        list(map(float, denom))
        )

def step(step_f, ampl=1, t_offset=0, t=None):
    # assuming all(np.diff(t) == np.diff(t)[0])
    t, response = step_f() if t is None else step_f(T=t)
    closest_t_n = closest_arg(t, t[0] + t_offset)
    if t_offset == 0:
        return t, (np.ones(len(t))*ampl, response*ampl)
    response_len = len(t) - closest_t_n
    off = np.zeros(closest_t_n)
    input = np.concatenate((off, np.ones(response_len)))
    response = np.concatenate((off, response[:response_len]))
    return t, t[closest_t_n], (input*ampl, response*ampl)

def zoh_stairs(t, x):
    return interweave(t, t)[1:], interweave(x, x)[:-1]

def z_eq_damping(damping, n=100):
    theta = np.arccos(damping)
    u = np.cos(theta)-np.sin(theta)*1j
    x = np.linspace(0, np.pi/u.imag, num=n)
    contour = np.exp(u*x)
    re, im = contour.real, contour.imag
    return concat(re, np.flip(re)), concat(im, np.flip(-im))

def mp_to_damping(mp):
    tg = math.pi/math.log(mp)
    return 1/math.sqrt(tg**2+1)

def ts_to_sigma(ts, prop = .02):
    return -math.log(prop)/ts

def z_eq_settling(sigma, dt, n=100):
    r = math.exp(-sigma*dt)
    t = np.linspace(0, 2*np.pi, n)
    cont = np.exp(1j*t)*r
    return cont.real, cont.imag

def shade_matlab_args(
    line_color='black',
    fill_color='yellow',
    fill_alpha=.3,
    linestyle='-.',
    label=None,
    line_params={},
    fill_params={}
) -> tuple[dict[str, Any], dict[str, Any]]:
    line_params = dict(
        color=line_color, linestyle=linestyle, label=label) | line_params
    fill_params = dict(
        color=fill_color, alpha=fill_alpha) | fill_params
    return {"line_params": line_params, "fill_params": fill_params}

@map_kwargs_dec(shade_matlab_args, {"fill_params", "line_params"})
def shade_overshoot_z(ax, mp_max,
                      fill_params={},
                      line_params={}):
    """
    Shades the minimum angle needed for dominant poles outside
    that region to lead to a Mp value lower than mp_max
    """
    return shade_outside(
        lambda x, y: \
            tuple(no_scaling(ax, poly) for poly in ax.plot(x, y, **line_params)),
        lambda x, y: ax.fill(x, y, **fill_params),
        z_eq_damping(mp_to_damping(mp_max)),
        ax.get_xlim(), ax.get_ylim())

@map_kwargs_dec(shade_matlab_args, {"fill_params", "line_params"})
def shade_settling_z(ax, ts_max, dt, prop: float=0.02,
                     fill_params={},
                     line_params={}):
    """
    Shades the sigma_d values that lead to a settling time
    longer than ts_max
    """
    return shade_outside(
        lambda x, y: \
            tuple(no_scaling(ax, poly) for poly in ax.plot(x, y, **line_params)),
        lambda x, y: ax.fill(x, y, **fill_params),
        z_eq_settling(ts_to_sigma(ts_max, prop), dt),
        ax.get_xlim(), ax.get_ylim())

def pole_specs(ts, mp, prop=0.02):
    dp = ts_to_sigma(ts, prop)*(-1+1j*math.pi/math.log(mp))
    return with_conj(dp)

def z_map(s, T):
    return cmath.exp(s*T)


def scatter_points(scatterf):
    def wrapped(cs, *args, **kwargs):
        return scatterf(
        *reim(arr), *args, **kwargs
    ) if len(arr:=np.asarray(cs)) > 0 else None
    return wrapped

def zp_plot(plt=plt, points_zorder=2):
    """
    Returns figure, axes and a set of functions that
    aid on creating a z-plane plot
    """
    fig, ax = plt.subplots()
    scatter = scatter_points(ax.scatter)
    line_zorder = points_zorder - 1
    fill_zorder = line_zorder - 1 
    scatter_poles = partial(scatter, marker="x", zorder=points_zorder)
    scatter_zeros = partial(scatter, marker="o", zorder=points_zorder)
    ax.set_ylabel("$\\Im(\\mathcal{z})$")
    ax.set_xlabel("$\\Re(\\mathcal{z})$")

    def add_ucircle(color='k', linestyle='dotted', **kwargs):
        ax.autoscale(False)
        circ = plt.Circle(
            (0, 0), radius=1, edgecolor=color, zorder=points_zorder-1,
            linestyle=linestyle, facecolor='None', **kwargs)
        return ax.add_patch(circ)

    def shade(curve, **kwargs):
        ax.autoscale(False)
        params = shade_matlab_args(**kwargs)
        line_params = dict(zorder=line_zorder) | params["line_params"]
        fill_params = dict(zorder=fill_zorder) | params["fill_params"]
        return shade_outside(
            partial(ax.plot, **line_params),
            partial(ax.fill, **fill_params),
            curve,
            ax.get_xlim(), ax.get_ylim())

    def add_ts_max(ts_max, dt, prop=.02, n=100, **kwargs):
        return shade(
            z_eq_settling(ts_to_sigma(ts_max, prop=prop), dt, n=n),
            **kwargs)
    
    def add_mp_max(mp, n=100, **kwargs):
        return shade(
            z_eq_damping(mp_to_damping(mp), n=n),
            **kwargs)
    
    return fig, ax, (
        (add_ucircle, add_ts_max, add_mp_max),
        (scatter_poles, scatter_zeros))

def phase_def(p: complex, poles, zeros, arg=np.angle) -> np.ndarray:
    poles, zeros = map(np.asarray, [poles, zeros])
    return sum(arg(p - poles))-sum(arg(p - zeros))
    
def mag_def(p: complex, poles, zeros) -> np.ndarray:
    poles, zeros = map(np.asarray, [poles, zeros])
    return np.prod(abs(p - poles))/np.prod(abs(p - zeros))

def settling_time(t, y, yfinal, prop=.02):
    """
    Computes the interpolated settling time of a signal

    This is useful for optimization algithms that require a
    smooth output space, otherwise since the settling time
    of a discrete signal is also a discrete time step, solvers
    can get stuck thinking they're at a stationary point
    """
    t, y = map(np.asanyarray, [t, y])
    norm = y/yfinal-1
    cond = abs(norm) <= prop
    if not cond[-1]:
        return None
    time_n = last_same(cond)
    return ilerp(
        t[time_n-1], t[time_n],
        norm[time_n-1], norm[time_n],
        prop)

def sys_terr_calc(f, *params, resp=ctrl.step_response):
    def get_params(sys):
        t, r = resp(sys)
        return np.array(
            [float(param-res)
                for param, res in
                zip(params, f(sys, t, r))])
    return get_params

def feedback(direct, sensing):
    return direct/(1+sensing*direct)

def feedback3(plant, controller, sensing):
    return feedback(plant*controller, sensing)


ControllerConfig = Callable[[
    TransferFunction,   # Plant
    TransferFunction,   # Controller
    TransferFunction    # Sensing
], TransferFunction]

def autotune(
    vars: dict[sp.Basic, float],
    f_symb: sp.Basic,
    f_tf: TransferFunction,
    gc: sp.Expr,
    mp: float, ts: float,
    gp_tf: TransferFunction,
    gh_tf: TransferFunction,
    config: ControllerConfig=feedback3,
    solver=scipy.optimize.root,
    as_dict=False,
    return_optimize_result=False,
    **solver_kwargs):
    """
    Tunes a controller with a symbolic representation of `gc`
    to drive the plant `gp_tf` with sensor `gh_tf` for a
    specific settling time and percentage overshoot.
    """
    syms, inits = zip(*vars.items())
    gcf = sp.lambdify([f_symb, *syms], gc)
    
    def get_perf(_, t, r):
        perf = ctrl.matlab.stepinfo(r, T=t)
        ts = settling_time(t, r, perf["SteadyStateValue"])
        mp = perf["Overshoot"]/100
        return mp, ts
    
    calc_params = sys_terr_calc(get_perf, mp, ts)

    def objective(params):
        return calc_params(
            config(gp_tf, gcf(f_tf, *params), gh_tf).minreal())

    res = solver(objective, inits, **solver_kwargs)
    sol = res.x
    with_result = (lambda sol: sol
        ) if not return_optimize_result else (
            lambda sol: (res, sol))
    if not as_dict:
        return with_result(sol)
    
    return with_result(dict(zip(syms, sol)))

def feedforward(plant, controller, sensor=1, sign=-1):
    return (plant+plant*controller)/(1-sign*controller*plant*sensor)

from .ratpolys import ratpoly_inv

T=TypeVar("T")
def numden2delays(
    num: Sequence[T],
    den: Sequence[T]
) -> tuple[dict[int, T], dict[int, T]]:
    """
    For any rational polynomial on the delay operator with
    numerator `num` and denominator `den`, this function returns
    a pair where the elements are mappings of the type delay->gain.
    The first element in the pair corresponds to the mapping for the
    denominator function (the input of the system) and the second
    to the numerator function (the output of the system).
    """
    inum, iden = ratpoly_inv(num, den)
    #inum, iden = num, den
    # inum = [a_n*z**-n, ..., a_2*z**-2, a_1*z**-1, a_0*z**0]
    # inum = [b_n*z**-n, ..., b_2*z**-2, b_1*z**-1, b_0*z**0]
    b0, *b_rest = reversed(iden)
    a_n = reversed(inum)
    return (
        {delay: -gain/b0 for delay, gain in enumerate(b_rest, start=1)},
        {delay: gain/b0 for delay, gain in enumerate(a_n)}
    )

T=TypeVar("T")
def difference_eq(
    num: Sequence[sp.Basic],
    den: Sequence[sp.Basic],
    in_: sp.Function,
    out: sp.Function,
    k: sp.Basic, eq: Type[T]=sp.Eq) -> T:
    """
    For a discrete time transfer function `out/in_` with
    numerator and denominator `num` and `den` respectively in the z
    domain, returns the difference equation expression of
    variable `k`
    """
    outs, ins = numden2delays(num, den)
    return eq(out(k), sum(
        sum(gain*f(k-delay) for delay, gain in maps.items())
        for f, maps in ((out, outs), (in_, ins))
    ))
