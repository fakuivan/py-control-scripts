#!/usr/bin/env python3.9
from typing import Sequence, TypeVar
from sympy import (
    Symbol, Basic, LC as leading_coeff, roots, together,
    denom, numer, Poly, Expr, Mul, prod, Pow, UnevaluatedExpr as uneval)
from itertools import starmap
from functools import reduce, partial
from .misc import default_seq_accesor

# most of these were taken from an old repo of mine
# https://github.com/fakuivan/facu-ciran/blob/master/utils.py

def f2nd(function: Basic) -> tuple[Basic, Basic]:
    """
    Given an rational expression, returns the numerator
    and denominator respectively
    """
    function = together(function)
    return numer(function), denom(function)

def f2zpk(function: Basic, var: Symbol
) -> tuple[dict[Basic, int], dict[Basic, int], Basic]:
    """
    Given a rational polynomial in bar, returns a tuple where the
    first and the second elements are the dictionaries in the form
    {zero: order}, {root: order} respectively, and the third element
    is the leading coefficient
    """
    numer, denom = f2nd(function)
    numer_lc, denom_lc = map(
        lambda expr: leading_coeff(expr, var),
        (numer, denom))
    # It's not really necesary to devide by the leading
    # coefficient, but what do I know
    return roots(numer/numer_lc, var), \
           roots(denom/denom_lc, var), numer_lc/denom_lc

def ratpoly_coeffs(function: Basic, var: Symbol
) -> tuple[list[Basic], list[Basic]]:
    """
    Given a rational polynomial in var, returns a tuple where the
    elements are lists of coefficients for the numerator and the
    denominator respectively
    """
    numer, denom = f2nd(function)
    return (Poly(numer, var).all_coeffs(),
            Poly(denom, var).all_coeffs())

def ratpoly_coeffs_lc(ratpoly: Expr, var: Expr) -> tuple[Expr, list[Expr], list[Expr]]:
    """
    Given a rational polynomial in var, returns a tuple of:
        - Leading coefficient
        - Numerator/LC
        - Denominator/LC
    """
    numer, denom = ratpoly_coeffs(ratpoly, var)
    lcn, lcd = numer[0], denom[0]
    return lcn/lcd, [coeff/lcn for coeff in numer], [coeff/lcd for coeff in denom]

def poly_from_coeffs(coeffs: Sequence[Expr], var: Expr) -> Expr:
    """
    Returns a polynomial expression given a sequence of
    factors in reverse order (from higer to lower powers)
    """
    exps = reversed(range(len(coeffs)))
    return sum(
        coeff * var**exp for exp, coeff in zip(exps, coeffs)
        )

umul = lambda *prod: Mul(*(p for p in prod if p != 1), evaluate=False)
uinv = lambda base: Pow(base, -1, evaluate=False)
ufrac = lambda numer, denom: umul(numer, uinv(denom))

def ratpoly_coeffs_form(ratpoly: Expr, var: Expr) -> Expr:
    """
    Returns the ratpoly expression formatted in the coefficient form
    """
    lc, numer, denom = ratpoly_coeffs_lc(ratpoly, var)
    nconst = numer[-1]
    return umul(lc*nconst, ufrac(
        poly_from_coeffs([coeff/nconst for coeff in numer], var), poly_from_coeffs(denom, var)))

def roots_to_tconst(roots: dict[Expr, int], var: Expr) -> tuple[Expr, Expr]:
    """
    Formats a set of roots in the form dict[loc, order] to sums of powers   
    """
    def process_root(k_roots, loc_order):
        loc, order = loc_order
        k, roots = k_roots
        if loc == 0:
            return k, roots + [var**order]
        return k*(-loc)**order, roots + [(1-(var*(1/loc).simplify()))**order]
    k_static, roots = reduce(process_root, roots.items(), (1, []))
    return k_static, umul(*roots)

def ratpoly_tconst_form(ratpoly: Expr, var: Expr) -> Expr:
    """
    Returns the ratpoly expression formatted in time constants form
    """
    z, p, k = f2zpk(ratpoly, var)
    n_k, numer = roots_to_tconst(z, var)
    d_k, denom = roots_to_tconst(p, var)
    k_static = k*n_k/d_k
    k_static = k_static if k_static == 1 else uneval(k_static)
    return uneval(umul(k_static, ufrac(numer, denom)))

def ratpoly_zpk_form(ratpoly: Expr, var: Expr) -> Expr:
    """
    Returns the expression formatted in the zeros, poles and gain form
    """
    z, p, k = f2zpk(ratpoly, var)
    numer, denom = (
        prod((var - loc)**ord for loc, ord in roots.items()) for roots in (z, p)
    )
    return k*uneval(numer/denom)

def numden2ratpoly(num, den, var):
    return poly_eval(num, var)/poly_eval(den, var)

def poly_eval(p: Sequence, var):
    return sum(starmap(
        lambda n, a: a*var**n,
        zip(
            reversed(range(len(p))),
            p
    )))

T=TypeVar("T")
def ratpoly_inv(num: Sequence[T], den: Sequence[T]
) -> tuple[Sequence[T], Sequence[T]]:
    """
    For a rational polynomial in `x` with coefficients `num` and `den`,
    returns a tuple with the numerator and denominator coefficients
    for that same rational polynomial in `1/x`.
    """
    zero_default = partial(default_seq_accesor, 0)
    get_num, get_den = map(
        zero_default, [num, den])
    max_ord = max(len(num), len(den))
    return (
        [get_num(len(num)-(k+1)) for k in range(max_ord)],
        [get_den(len(den)-(k+1)) for k in range(max_ord)])