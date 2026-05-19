"""
ryu_erk.py — Deterministic ERK pathway simulator (Ryu et al. 2015 / Rochat 2024)
with a thin Gymnasium wrapper for RL pre-training.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.integrate
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Default parameters — Rochat 2024 Appendix A.3
# ---------------------------------------------------------------------------

PARAMS: dict[str, float] = {
    # Receptor
    "k1": 0.5,   "kd1": 0.5,   "Km1_star": 0.85,
    # Ras
    "k2": 40.0,  "Km2": 1.0,   "kd2": 3.75,  "Km2_star": 1.0,  "GAP": 1.0,
    # Raf
    "k3": 10.0,  "Km3": 1.0,   "kd3": 3.75,  "Km3_star": 1.0,  "P3": 1.0,
    # MEK
    "k4": 2.0,   "Km4": 1.0,   "kd4": 0.5,   "Km4_star": 1.0,  "P4": 1.0,
    # ERK
    "k5": 2.0,   "Km5": 1.0,   "kd5": 0.25,  "Km5_star": 0.1,  "P5": 1.0,
    # NFB
    "k6": 0.0286,"Km6": 0.01,  "kd6": 0.0057,"Km6_star": 0.5,  "P6": 1.0,
    # PFB
    "k7": 0.1,   "Km7": 0.1,   "kd7": 0.005, "Km7_star": 0.1,  "P7": 1.0,
    # Feedback constants
    "kPFB_NGF": 0.75, "kPFB_EGF": 0.0,
    "KPFB": 0.01, "KNFB": 0.05,
}


# ---------------------------------------------------------------------------
# Signal builders
# ---------------------------------------------------------------------------

def sustained_signal(amplitude: float) -> Callable[[float], float]:
    """Constant growth factor signal."""
    return lambda t: amplitude


def pulse_train(
    amplitude: float,
    pulse_width: float,
    period: float,
    t_start: float = 0.0,
) -> Callable[[float], float]:
    """Square-wave pulses of given width and period, starting at t_start.

    Uses exact modular arithmetic — safe for the integrator's arbitrary t queries.
    """
    def signal(t: float) -> float:
        if t < t_start:
            return 0.0
        return amplitude if (t - t_start) % period < pulse_width else 0.0
    return signal


# ---------------------------------------------------------------------------
# ODE model
# ---------------------------------------------------------------------------

class RyuERKModel:
    """Deterministic Ryu et al. 2015 ERK pathway ODE."""

    STATE_NAMES = ["R", "Ras", "Raf", "MEK", "ERK", "NFB", "PFB"]

    def __init__(
        self,
        growth_factor: str = "NGF",
        params: dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            growth_factor: "NGF" or "EGF". Determines kPFB (PFB is NGF-specific).
            params: Optional dict merged into defaults (any key from PARAMS).
        """
        if growth_factor not in {"NGF", "EGF"}:
            raise ValueError(f"growth_factor must be 'NGF' or 'EGF', got {growth_factor!r}")
        self.growth_factor = growth_factor
        self.p: dict[str, float] = {**PARAMS, **(params or {})}
        self.kPFB: float = self.p["kPFB_NGF"] if growth_factor == "NGF" else self.p["kPFB_EGF"]

    def rhs(self, t: float, x: np.ndarray, signal_fn: Callable[[float], float]) -> np.ndarray:
        """Return dx/dt. signal_fn(t) gives growth factor concentration at time t."""
        p = self.p
        x1, x2, x3, x4, x5, x6, x7 = x
        sig = signal_fn(t)

        dx1 = p["k1"] * (1.0 - x1) * sig - p["kd1"] * x1

        dx2 = (p["k2"] * x1 * (1.0 - x2) / (p["Km2"] + (1.0 - x2))
               - p["kd2"] * p["GAP"] * x2 / (p["Km2_star"] + x2))

        dx3 = (p["k3"] * x2 * (1.0 - x3) / (p["Km3"] + (1.0 - x3))
               * p["KNFB"]**2 / (p["KNFB"]**2 + x6**2)
               - p["kd3"] * p["P3"] * x3 / (p["Km3_star"] + x3)
               + self.kPFB * x7 * (1.0 - x3) / (p["KPFB"] + (1.0 - x3)))

        dx4 = (p["k4"] * x3 * (1.0 - x4) / (p["Km4"] + (1.0 - x4))
               - p["kd4"] * p["P4"] * x4 / (p["Km4_star"] + x4))

        dx5 = (p["k5"] * x4 * (1.0 - x5) / (p["Km5"] + (1.0 - x5))
               - p["kd5"] * p["P5"] * x5 / (p["Km5_star"] + x5))

        dx6 = (p["k6"] * x5 * (1.0 - x6) / (p["Km6"] + (1.0 - x6))
               * x1**2 / (p["Km1_star"]**2 + x1**2)
               - p["kd6"] * p["P6"] * x6 / (p["Km6_star"] + x6))

        dx7 = (p["k7"] * x5 * (1.0 - x7) / (p["Km7"] + (1.0 - x7)) * x1
               - p["kd7"] * p["P7"] * x7 / (p["Km7_star"] + x7))

        return np.array([dx1, dx2, dx3, dx4, dx5, dx6, dx7])

    def simulate(
        self,
        signal_fn: Callable[[float], float],
        t_span: tuple[float, float] = (0.0, 100.0),
        dt: float = 1.0,
        x0: np.ndarray | None = None,
        method: str = "LSODA",
    ) -> dict:
        """Integrate and return trajectory.

        Returns:
            dict with keys:
              't': (T,) array of evaluation times
              'x': (T, 7) state trajectory
              'signal': (T,) signal values at eval times
        """
        if x0 is None:
            x0 = np.zeros(7)

        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

        sol = scipy.integrate.solve_ivp(
            fun=lambda t, x: self.rhs(t, x, signal_fn),
            t_span=(t_span[0], t_eval[-1]),
            y0=x0,
            method=method,
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9,
        )

        return {
            "t": sol.t,
            "x": sol.y.T,           # (T, 7)
            "signal": np.array([signal_fn(ti) for ti in sol.t]),
        }


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class RyuERKEnv(gym.Env):
    """Single-cell ERK environment for RL pre-training.

    Observation: ERK scalar (partial_obs=True) or full 7-state vector.
    Action: growth factor in [0, max_amplitude] (Box) or binary {0,1} (Discrete).
    Episode: fixed-length horizon × dt minutes.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        growth_factor: str = "NGF",
        horizon: int = 100,
        dt: float = 1.0,
        max_amplitude: float = 1.0,
        partial_obs: bool = True,
        discrete_actions: bool = False,
    ) -> None:
        super().__init__()
        self.model = RyuERKModel(growth_factor=growth_factor)
        self.horizon = horizon
        self.dt = dt
        self.max_amplitude = max_amplitude
        self.partial_obs = partial_obs

        obs_dim = 1 if partial_obs else 7
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        if discrete_actions:
            self.action_space = spaces.Discrete(2)
            self._to_amplitude: Callable = lambda a: float(a) * max_amplitude
        else:
            self.action_space = spaces.Box(
                low=0.0, high=max_amplitude, shape=(1,), dtype=np.float32
            )
            self._to_amplitude = lambda a: float(a[0])

        self._state = np.zeros(7)
        self._t: float = 0.0
        self._step_count: int = 0

    def _observe(self) -> np.ndarray:
        if self.partial_obs:
            return np.array([self._state[4]], dtype=np.float32)
        return self._state.astype(np.float32)

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._state = np.zeros(7)
        self._t = 0.0
        self._step_count = 0
        return self._observe(), {}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Integrate one dt step forward with a constant signal equal to action."""
        amplitude = np.clip(self._to_amplitude(action), 0.0, self.max_amplitude)
        result = self.model.simulate(
            signal_fn=sustained_signal(amplitude),
            t_span=(self._t, self._t + self.dt),
            dt=self.dt,
            x0=self._state,
        )
        self._state = result["x"][-1]
        self._t += self.dt
        self._step_count += 1

        erk = float(self._state[4])
        # TODO: replace with task-specific target ERK trajectory reward
        reward = -((erk - 0.5) ** 2)

        terminated = False
        truncated = self._step_count >= self.horizon
        return self._observe(), reward, terminated, truncated, {"erk": erk}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. NGF pulse train — Rochat Figure 16-style (3 min on / 17 min off)
    model_ngf = RyuERKModel(growth_factor="NGF")
    sig_pulse = pulse_train(amplitude=1.0, pulse_width=3.0, period=20.0)
    res = model_ngf.simulate(sig_pulse, t_span=(0.0, 100.0), dt=1.0)

    fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex=True)
    axes_flat = axes.flatten()
    for i, name in enumerate(RyuERKModel.STATE_NAMES):
        axes_flat[i].plot(res["t"], res["x"][:, i])
        axes_flat[i].set_title(name)
        axes_flat[i].set_ylim(-0.05, 1.05)
    axes_flat[7].plot(res["t"], res["signal"], color="gray")
    axes_flat[7].set_title("Signal")
    axes_flat[7].set_ylim(-0.05, 1.2)
    for ax in axes_flat[6:]:
        ax.set_xlabel("Time (min)")
    fig.suptitle("NGF pulse train (3 min on / 17 min off, period 20 min)")
    fig.tight_layout()
    fig.savefig("demo_pulse_train.png", dpi=120)
    plt.close(fig)
    print("Saved demo_pulse_train.png")

    # 2. Sustained NGF vs sustained EGF — ERK qualitative signature
    model_egf = RyuERKModel(growth_factor="EGF")
    sig_sus = sustained_signal(1.0)
    res_ngf = model_ngf.simulate(sig_sus, t_span=(0.0, 100.0), dt=1.0)
    res_egf = model_egf.simulate(sig_sus, t_span=(0.0, 100.0), dt=1.0)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(res_ngf["t"], res_ngf["x"][:, 4], label="NGF (sustained ERK)")
    ax2.plot(res_egf["t"], res_egf["x"][:, 4], label="EGF (transient ERK)")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("ERK activation")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.set_title("Sustained stimulation: NGF vs EGF")
    fig2.tight_layout()
    fig2.savefig("demo_ngf_vs_egf.png", dpi=120)
    plt.close(fig2)
    print("Saved demo_ngf_vs_egf.png")

    # 3. Gym env smoke test
    env = RyuERKEnv(growth_factor="NGF")
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print(f"Final ERK after 50 random steps: {info['erk']:.4f}")
