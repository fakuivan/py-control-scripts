from typing import Sequence
import sympy as sp
from typing import Any

def nsolve(
    sys,
    vars: dict[sp.Basic, Any],
    nsolve_=sp.nsolve,
    **params
) -> dict[sp.Basic, Any]:
    syms, inits = zip(*vars.items())
    sol = nsolve_(sys, syms, inits, **params)
    return dict(zip(syms, sol))

def solve(
    sys,
    vars: Sequence[sp.Basic],
    solve_=sp.solve,
    **params
) -> dict[sp.Basic, Any]:
    return tuple(dict(zip(vars, sol))
        for sol in solve_(sys, vars, **params))

