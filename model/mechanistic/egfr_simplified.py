"""
Simplified EGFR → RAS → RAF → MEK → ERK cascade with negative feedback (NFB).

Five active-form states: RAS_s, RAF_s, MEK_s, NFB_s, ERK_s.

Activation is pseudo-first-order (linear in upstream signal — inactive pools
assumed large / approximately constant, absorbed into rate constants).
Deactivation retains Michaelis-Menten saturation.
"""
from sympy import Eq, Derivative, Symbol
from sympy.abc import t

from model.mechanistic.mechanistic_model import EquationDescription


PARAM_NAMES = [
    "k12", "k21", "K21",
    "k34", "knfb", "k43", "K43",
    "k56", "k65", "K65",
    "k78", "k87", "K87",
    "f12", "f21", "F21",
]

NODE_NAMES = ["RAS", "RAF", "MEK", "NFB", "ERK"]

STATE_NAMES = [f"{node}_s" for node in NODE_NAMES]


def model_eqs(params, states) -> EquationDescription:
    symbols_dict = {}
    for p in params:
        symbols_dict[p] = Symbol(p)
    for s in states:
        symbols_dict[s] = Symbol(s)
    symbols_dict['t'] = t
    symbols_dict['light'] = Symbol('light')

    s = symbols_dict

    equations = [
        # d(RAS_s)/dt = light * k12 - k21 * RAS_s/(K21 + RAS_s)
        Eq(Derivative(s['RAS_s'], t),
           s['light'] * s['k12']
           - s['k21'] * (s['RAS_s'] / (s['K21'] + s['RAS_s']))),

        # d(RAF_s)/dt = k34 * RAS_s - (knfb*NFB_s + k43) * RAF_s/(K43 + RAF_s)
        Eq(Derivative(s['RAF_s'], t),
           s['k34'] * s['RAS_s']
           - (s['knfb'] * s['NFB_s'] + s['k43']) * (s['RAF_s'] / (s['K43'] + s['RAF_s']))),

        # d(MEK_s)/dt = k56 * RAF_s - k65 * MEK_s/(K65 + MEK_s)
        Eq(Derivative(s['MEK_s'], t),
           s['k56'] * s['RAF_s']
           - s['k65'] * (s['MEK_s'] / (s['K65'] + s['MEK_s']))),

        # d(NFB_s)/dt = f12 * ERK_s - f21 * NFB_s/(F21 + NFB_s)
        Eq(Derivative(s['NFB_s'], t),
           s['f12'] * s['ERK_s']
           - s['f21'] * (s['NFB_s'] / (s['F21'] + s['NFB_s']))),

        # d(ERK_s)/dt = k78 * MEK_s - k87 * ERK_s/(K87 + ERK_s)
        Eq(Derivative(s['ERK_s'], t),
           s['k78'] * s['MEK_s']
           - s['k87'] * (s['ERK_s'] / (s['K87'] + s['ERK_s']))),
    ]

    return {'base_equations': equations, 'symbols': symbols_dict, 'equations': equations}


if __name__ == "__main__":
    import numpy as np
    from model.mechanistic.mechanistic_model import Model

    def const_light(t_val, t_args=None):
        return 1.0

    # Test 1: construction
    m = Model(
        name='egfr_smoke',
        states=STATE_NAMES,
        parameters=PARAM_NAMES,
        model_definition=model_eqs,
        t_dep='light',
        t_func=const_light,
    )
    assert len(m.eqs) == 5
    print(f"[PASS] construction — {len(m.eqs)} equations, {len(PARAM_NAMES)} params, {len(STATE_NAMES)} states")

    # Test 2: numerical system is callable and finite
    system = m.make_numerical()
    y0 = [0.05] * 5
    params_vec = np.ones(len(PARAM_NAMES))
    dy = system(0.0, y0, params_vec, const_light)
    assert len(dy) == 5
    assert all(np.isfinite(dy))
    print(f"[PASS] make_numerical — dy = {[f'{v:.4f}' for v in dy]}")

    # Test 3: RAS activation = light * k12 when RAS_s ≈ 0
    # With all params=1, light=1: d(RAS_s)/dt = 1*1 - 1*(0.05/(1+0.05)) ≈ 0.952
    assert dy[0] > 0.9
    print(f"[PASS] RAS activation rate sanity — dy[0] = {dy[0]:.4f}")

    # Test 4: integration doesn't blow up
    from scipy.integrate import solve_ivp
    sol = solve_ivp(
        lambda t, y: system(t, y, params_vec, const_light),
        [0, 10], y0, method='LSODA', rtol=1e-8
    )
    assert sol.success, f"ODE solve failed: {sol.message}"
    assert sol.y.shape[0] == 5
    assert all(np.isfinite(sol.y[:, -1]))
    print(f"[PASS] solve_ivp — final states: {[f'{v:.4f}' for v in sol.y[:, -1]]}")

    # Test 5: no light → everything decays to zero
    def no_light(t_val, t_args=None):
        return 0.0

    sol_dark = solve_ivp(
        lambda t, y: system(t, y, params_vec, no_light),
        [0, 50], [1.0] * 5, method='LSODA', rtol=1e-8
    )
    assert sol_dark.success
    assert all(v < 0.01 for v in sol_dark.y[:, -1]), f"Expected decay, got {sol_dark.y[:, -1]}"
    print(f"[PASS] dark decay — final states: {[f'{v:.4f}' for v in sol_dark.y[:, -1]]}")

    print("\nAll tests passed.")
