"""ODE dynamics for RL gymnasium environments.

Each function follows the solve_ivp signature: f(t, state, *args) -> d(state)/dt
Limits are placed on state variables to avoid solution divergence.
"""

import numpy as np

# Default EGFR parameters (fitted values from egfr_params.json)
EGFR_DEFAULT_PARAMS = {
    "Km": 0.251, "k12": 0.355, "k21": 0.112,
    "k34": 0.708, "knfb": 2.239, "k43": 0.032,
    "k56": 2.239, "k65": 0.316,
    "k78": 1.585, "k87": 0.141,
    "f12": 3.162, "f21": 0.794,
}

EGFR_STATE_NAMES = ["RAS", "RAF", "MEK", "NFB", "ERK"]


def mgm(t, state, K):
    """Moore-Greitzer Model: surge/stall dynamics of a gas turbine engine.

    2D system where x1 is mass flow and x2 is pressure increase.
    Control input K acts on pressure dynamics.

    Based on: "Output-Feedback Control of Nonlinear Systems using Control
    Contraction Metrics and Convex Optimization" by Manchester and Slotine.
    """
    x1, x2 = state
    x1 = np.clip(x1, -20.0, 20.0)
    x2 = np.clip(x2, -20.0, 20.0)
    dx1 = -x2 - 1.5 * x1**2 - 0.5 * x1**3
    dx2 = x1 + K
    return np.array([dx1, dx2])


def van_der_pol(t, state, K, mu=1.0):
    """Van Der Pol oscillator: classic nonlinear oscillating system.

    2D system with parameter mu controlling nonlinearity strength.
    """
    x1, x2 = state
    x1 = np.clip(x1, -20.0, 20.0)
    x2 = np.clip(x2, -20.0, 20.0)
    dx1 = x2
    dx2 = mu * (1 - x1**2) * x2 - x1 + K
    return np.array([dx1, dx2])


def lorenz(t, state, K, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Lorenz attractor: chaotic 3D system sensitive to initial conditions.

    Control input K acts on the first state variable.
    """
    x1, x2, x3 = state
    x1 = np.clip(x1, -50.0, 50.0)
    x2 = np.clip(x2, -50.0, 50.0)
    x3 = np.clip(x3, -50.0, 50.0)
    dx1 = sigma * (x2 - x1) + K
    dx2 = x1 * (rho - x3) - x2
    dx3 = x1 * x2 - beta * x3
    return np.array([dx1, dx2, dx3])


def make_egfr(params=None):
    """Create an EGFR cascade dynamics function for use with ODEEnv.

    Returns a function with signature f(t, state, light) -> dstate/dt
    compatible with solve_ivp and ODEEnv.

    5 states: RAS_s, RAF_s, MEK_s, NFB_s, ERK_s (all in [0, 1])
    Control input: light intensity (non-negative)

    Args:
        params: dict of kinetic parameters. Uses fitted defaults if None.
    """
    p = {**EGFR_DEFAULT_PARAMS, **(params or {})}
    Km = p["Km"]
    k12, k21 = p["k12"], p["k21"]
    k34, knfb, k43 = p["k34"], p["knfb"], p["k43"]
    k56, k65 = p["k56"], p["k65"]
    k78, k87 = p["k78"], p["k87"]
    f12, f21 = p["f12"], p["f21"]

    def egfr(t, state, light):
        RAS, RAF, MEK, NFB, ERK = np.clip(state, 1e-3, 1.0)
        light = max(light, 0.0)

        dRAS = light * k12 * (1 - RAS) - k21 * RAS / (Km + RAS)
        dRAF = k34 * RAS * (1 - RAF) - (knfb * NFB + k43) * RAF / (Km + RAF)
        dMEK = k56 * RAF * (1 - MEK) - k65 * MEK / (Km + MEK)
        dNFB = f12 * ERK * (1 - NFB) - f21 * NFB / (Km + NFB)
        dERK = k78 * MEK * (1 - ERK) - k87 * ERK / (Km + ERK)

        return np.array([dRAS, dRAF, dMEK, dNFB, dERK])

    return egfr
