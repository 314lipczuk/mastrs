"""Custom Gymnasium environment for controlling ODE systems with RL.

Based on: "Reinforcement Learning for Physics: ODEs and Hyperparameter Tuning"
by Robert Etter (Towards Data Science, Oct 2024).

Usage:
    import gymnasium as gym
    from model.rl.ode_env import ODEEnv
    from model.rl.dynamics import mgm

    env = ODEEnv(dynamics_fn=mgm, n_states=2)
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
"""

from typing import Optional, Callable

import math
import numpy as np
from scipy.integrate import solve_ivp

import gymnasium as gym
from gymnasium import spaces


class ODEEnv(gym.Env):
    """Gymnasium environment that wraps an ODE system for RL control.

    The agent observes the full state vector and applies a scalar control
    input at each step. The ODE is integrated over a short time horizon
    using solve_ivp. The reward drives the state toward the origin.

    Args:
        dynamics_fn: ODE right-hand side with signature f(t, state, K) -> dstate/dt.
        n_states: Dimensionality of the state vector.
        state_bound: Symmetric bound on the observation space.
        action_bound: Symmetric bound on the action (control input).
        dt: Integration time step per environment step.
        min_init_dist: Minimum distance from origin for initial state.
        max_init_range: Range for sampling initial state.
        reward_states: Indices of state variables to include in reward.
            Defaults to all states. E.g. [0] to only penalize first state.
        max_steps: Maximum steps per episode (0 = no limit).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dynamics_fn: Callable,
        n_states: int = 2,
        state_bound: float = 30.0,
        action_bound: float = 10.0,
        dt: float = 0.05,
        min_init_dist: float = 2.5,
        max_init_range: float = 8.0,
        reward_states: Optional[list] = None,
        max_steps: int = 0,
        render_mode=None,
    ):
        super().__init__()
        self.dynamics_fn = dynamics_fn
        self.n_states = n_states
        self.dt = dt
        self.min_init_dist = min_init_dist
        self.max_init_range = max_init_range
        self.reward_states = reward_states
        self.max_steps = max_steps
        self._step_count = 0

        self.observation_space = spaces.Box(
            low=-state_bound, high=state_bound, shape=(n_states,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-action_bound, high=action_bound, shape=(1,), dtype=np.float64
        )

        self.state = np.zeros(n_states)

    def _get_obs(self):
        return self.state.copy()

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        # Sample initial state with minimum distance from origin
        while True:
            self.state = self.np_random.uniform(
                -self.max_init_range, self.max_init_range, size=(self.n_states,)
            )
            if np.linalg.norm(self.state) >= self.min_init_dist:
                break

        return self._get_obs(), {}

    def step(self, action):
        u = action.item()
        self._step_count += 1

        result = solve_ivp(
            self.dynamics_fn, (0, self.dt), self.state, args=[u]
        )

        self.state = result.y[:, -1].copy()

        # Reward: negative L2 norm of selected state variables
        if self.reward_states is not None:
            reward_vec = self.state[self.reward_states]
        else:
            reward_vec = self.state
        reward = -math.sqrt(np.sum(reward_vec**2))

        truncated = (
            self.max_steps > 0 and self._step_count >= self.max_steps
        )
        done = False

        return self._get_obs(), reward, done, truncated, {}
