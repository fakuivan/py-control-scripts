import numpy as np
import sympy as sp
from functools import partial
from math import log10

def eint(vec):
    dp = np.sqrt(np.add.accumulate(np.asarray(vec)**2))
    return dp/dp[-1]

def closest_arg(arr, value):
    return np.abs(np.asarray(arr) - value).argmin()

def thd(fundamental, harmonics):
    return np.sqrt(np.sum(np.asarray(harmonics)**2))/fundamental

def db(num: float) -> float:
        return 20*log10(num)

def cross2d(x1, y1, x2, y2):
    return x1*y2-x2*y1

def is_clockwise(x1, y1, x2, y2):
    cp = cross2d(x1, y1, x2, y2)
    return cp < 0 if cp != 0 else None

def dist(start, end):
    start, end = map(np.asarray, (start, end))
    return sum((end - start)**2)**.5

def normalize(p):
    p = np.asarray(p)
    return p/np.linalg.norm(p)

def angle_between(pivot, p1, p2):
    pivot, p1, p2 = map(np.asarray, (pivot, p1, p2))
    return np.arccos(np.dot(*[normalize(p - pivot) for p in [p1, p2]]))

def reim(c):
    return c.real, c.imag

def sp_reim(expr):
    return sp.re(expr), sp.im(expr)

def with_conj(c):
    return c, c.real-1j*c.imag

zero = sp.sympify(0)
def rarg(expr: sp.Basic, atan=sp.atan2) -> sp.Basic:
    if sp.ask(sp.Q.positive(expr)):
        return zero
    if sp.ask(sp.Q.negative(expr)):
        return sp.pi
    return atan(sp.im(expr), sp.re(expr))

def arg_ex(expr: sp.Basic, argf=sp.arg) -> sp.Basic:
    this = partial(arg_ex, argf=argf)
    if isinstance(expr, sp.Mul):
        prods = expr.args
        return sp.Add(*(this(prod) for prod in prods))
    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        return exp*this(base)
    return argf(expr)

def lerp(x1, x2, y1, y2, x):
    prop = (x-x1)/(x2-x1)
    return y1+prop*(y2-y1)

def ilerp(x1, x2, y1, y2, y):
    return lerp(y1, y2, x1, x2, y)
