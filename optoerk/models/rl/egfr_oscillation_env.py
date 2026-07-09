"""Gymnasium environment for driving EGFR cascade into sustained oscillations.

The agent controls light intensity at each step. The reward encourages
oscillatory behavior in ERK (or other target states) by measuring variance
over a sliding window — high variance means the signal is moving, not stuck
at a fixed point.

Usage:
    from optoerk.models.rl.egfr_oscillation_env import EGFROscillationEnv

    env = EGFROscillationEnv()
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
"""

from typing import Optional
from collections import deque

import numpy as np
from scipy.integrate import solve_ivp

import gymnasium as gym
from gymnasium import spaces

from optoerk.models.rl.dynamics import make_egfr, EGFR_STATE_NAMES


class EGFROscillationEnv(gym.Env):
    """RL environment that rewards sustained oscillation in the EGFR cascade.

    Observation: [RAS, RAF, MEK, NFB, ERK] — all in [0, 1]
    Action: light intensity (scalar, non-negative)

    The reward uses FFT to measure how well the target state matches a
    desired oscillation frequency and amplitude:
        1. frequency_reward: power at the target frequency bin relative to
           total power (rewards clean oscillation at the right frequency)
        2. amplitude_reward: closeness of the FFT-derived amplitude to the
           target amplitude (penalizes being too small or too large)
        3. flatness_penalty: negative reward when ERK isn't moving

    Args:
        params: EGFR kinetic parameters dict (uses defaults if None).
        dt: ODE integration timestep per env step (seconds).
        max_steps: Episode length.
        target_state: Index of state to oscillate (default 4 = ERK).
        target_freq: Desired oscillation frequency in Hz (default 1/70).
        target_amplitude: Desired peak-to-peak amplitude (default 0.6).
        window_size: Number of past steps for computing oscillation reward.
            Should be >= 2 full cycles at target_freq.
        max_light: Upper bound on light intensity action.
        obs_stack: Number of consecutive states to stack in the observation.
            1 = current state only (default). Higher values give the agent
            temporal context (e.g., 4 means the agent sees the last 4 states).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        params=None,
        dt: float = 0.5,
        max_steps: int = 800,
        target_state: int = 4,  # ERK
        target_freq: float = 1.0 / 70.0,
        target_amplitude: float = 0.6,
        window_size: int = 280,
        max_light: float = 5.0,
        obs_stack: int = 1,
        render_mode=None,
    ):
        super().__init__()
        self.dynamics_fn = make_egfr(params)
        self.n_states = 5
        self.dt = dt
        self.max_steps = max_steps
        self.target_state = target_state
        self.target_freq = target_freq
        self.target_amplitude = target_amplitude
        self.window_size = window_size
        self.obs_stack = obs_stack
        self._step_count = 0

        # Precompute FFT frequency bins and find the one closest to target
        self._fft_freqs = np.fft.rfftfreq(window_size, d=dt)
        self._target_bin = np.argmin(np.abs(self._fft_freqs - target_freq))

        # Observation: obs_stack consecutive state snapshots concatenated
        # e.g. obs_stack=4 → shape (20,) = [state_t, state_t-1, state_t-2, state_t-3]
        obs_size = self.n_states * self.obs_stack
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float64
        )
        # Light is non-negative
        self.action_space = spaces.Box(
            low=0.0, high=max_light, shape=(1,), dtype=np.float64
        )

        self.state = np.zeros(self.n_states)
        self._history = deque(maxlen=window_size)
        # Ring buffer for stacking: oldest first, newest last
        self._obs_buffer = deque(maxlen=obs_stack)

    def _get_obs(self):
        # Pad with copies of current state if buffer isn't full yet
        # (at episode start, all frames are the initial state)
        frames = list(self._obs_buffer)
        while len(frames) < self.obs_stack:
            frames.insert(0, self.state.copy())
        # Most recent state first: [state_t, state_t-1, ..., state_t-k]
        return np.concatenate(frames[::-1])

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._history.clear()
        self._obs_buffer.clear()

        # Start near resting state with small random perturbation
        self.state = self.np_random.uniform(0.01, 0.15, size=(self.n_states,))
        self._obs_buffer.append(self.state.copy())
        return self._get_obs(), {}

    def step(self, action):
        light = float(np.clip(action.item(), 0.0, self.action_space.high[0]))
        self._step_count += 1

        result = solve_ivp(
            self.dynamics_fn, (0, self.dt), self.state,
            args=[light], method="LSODA", rtol=1e-6,
        )

        self.state = np.clip(result.y[:, -1], 1e-3, 1.0)
        self._obs_buffer.append(self.state.copy())
        self._history.append(self.state[self.target_state])

        reward = self._compute_reward()

        truncated = self._step_count >= self.max_steps
        done = False

        return self._get_obs(), reward, done, truncated, {}

    def _compute_reward(self):
        # No reward until we have a full window (need enough data for FFT)
        if len(self._history) < self.window_size:
            return 0.0

        window = np.array(self._history)

        # Penalize flatness — if ERK barely moves, no point doing FFT
        peak_to_peak = np.max(window) - np.min(window)
        if peak_to_peak < 0.05:
            return -1.0

        # FFT of the de-meaned signal
        centered = window - np.mean(window)
        fft_mag = np.abs(np.fft.rfft(centered))
        power = fft_mag ** 2

        # 1. Frequency reward: fraction of total power at the target bin
        #    1.0 = perfect sine at target freq, 0.0 = power elsewhere
        total_power = np.sum(power) + 1e-10
        freq_reward = power[self._target_bin] / total_power

        # 2. Amplitude reward: how close the oscillation amplitude is to target
        #    FFT magnitude → peak-to-peak amplitude: 2 * |FFT[k]| / N
        measured_amplitude = 2.0 * fft_mag[self._target_bin] / len(window)
        amp_error = abs(measured_amplitude - self.target_amplitude)
        # Gaussian-shaped reward: peaks at 1.0 when error=0, decays with error
        amp_reward = np.exp(-10.0 * amp_error ** 2)

        # Combined: both terms in [0, 1], weighted equally
        reward = 5.0 * freq_reward + 5.0 * amp_reward

        return reward
