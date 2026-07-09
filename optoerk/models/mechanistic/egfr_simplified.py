"""
Simplified EGFR → RAS → RAF → MEK → ERK cascade with negative feedback (NFB).

Five active-form states: RAS_s, RAF_s, MEK_s, NFB_s, ERK_s.
Total protein is conserved per node (Active + Inactive = Total, normalised to 1),
so activation is proportional to the inactive pool (1 - Active).
Deactivation retains Michaelis-Menten saturation with a single shared Km.
"""
from sympy import Eq, Derivative, Symbol, log
from sympy.abc import t

from optoerk.models.mechanistic.mechanistic_model import EquationDescription


PARAM_NAMES = [
    "Km",
    "k12", "k21",
    "k34", "knfb", "k43",
    "k56", "k65",
    "k78", "k87",
    "f12", "f21",
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
        # d(RAS_s)/dt = light * k12 * (1 - RAS_s) - k21 * RAS_s/(Km + RAS_s)
        Eq(Derivative(s['RAS_s'], t),
           s['light'] * s['k12'] * (1 - s['RAS_s'])
           - s['k21'] * (s['RAS_s'] / (s['Km'] + s['RAS_s']))),

        # d(RAF_s)/dt = k34 * RAS_s * (1 - RAF_s) - (knfb*NFB_s + k43) * RAF_s/(Km + RAF_s)
        Eq(Derivative(s['RAF_s'], t),
           s['k34'] * s['RAS_s'] * (1 - s['RAF_s'])
           - (s['knfb'] * s['NFB_s'] + s['k43']) * (s['RAF_s'] / (s['Km'] + s['RAF_s']))),

        # d(MEK_s)/dt = k56 * RAF_s * (1 - MEK_s) - k65 * MEK_s/(Km + MEK_s)
        Eq(Derivative(s['MEK_s'], t),
           s['k56'] * s['RAF_s'] * (1 - s['MEK_s'])
           - s['k65'] * (s['MEK_s'] / (s['Km'] + s['MEK_s']))),

        # d(NFB_s)/dt = f12 * ERK_s * (1 - NFB_s) - f21 * NFB_s/(Km + NFB_s)
        Eq(Derivative(s['NFB_s'], t),
           s['f12'] * s['ERK_s'] * (1 - s['NFB_s'])
           - s['f21'] * (s['NFB_s'] / (s['Km'] + s['NFB_s']))),

        # d(ERK_s)/dt = k78 * MEK_s * (1 - ERK_s) - k87 * ERK_s/(Km + ERK_s)
        Eq(Derivative(s['ERK_s'], t),
           s['k78'] * s['MEK_s'] * (1 - s['ERK_s'])
           - s['k87'] * (s['ERK_s'] / (s['Km'] + s['ERK_s']))),
    ]

    return {'base_equations': equations, 'symbols': symbols_dict, 'equations': equations}


if __name__ == "__main__":
    import numpy as np
    from optoerk.models.mechanistic.mechanistic_model import Model

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

    # Test 3: RAS activation with totals
    # With all params=1, light=1, RAS_s=0.05:
    # d(RAS_s)/dt = 1*1*(1-0.05) - 1*(0.05/(1+0.05)) ≈ 0.902
    assert dy[0] > 0.8
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



# ------------------------------
# Generating synthetic data for the light function
# ------------------------------


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d


# --- helpers ---

def _pulses_to_signal(pulses: list, times: np.ndarray, dt: float) -> np.ndarray:
    signal = np.zeros(len(times))
    for p in pulses:
        on_idx  = int(p["t_on"]  / dt)
        off_idx = int(p["t_off"] / dt)
        signal[on_idx:off_idx] = p["amplitude"]
    return signal

def _signal_to_pulses_fragmented(
    signal: np.ndarray,
    times: np.ndarray,
    dt: float,
    rng: np.random.Generator,
    min_duration: float = 0.5,
    amp_range: tuple = (0.1, 5.0),
) -> list:
    """
    Fragment a continuous signal into short pulses that sample the envelope.
    Each pulse duration is exactly min_duration; gap between pulses is randomly sampled.
    Amplitude of each pulse is read from the signal at that timepoint.
    Pulses with amplitude below threshold are skipped.
    """
    amp_threshold = amp_range[0] * 0.5
    pulse_samples = max(1, int(min_duration / dt))

    pulses = []
    i = 0
    while i < len(signal):
        amp = float(signal[i])
        if amp > amp_threshold:
            t_on  = round(float(times[i]), 2)
            t_off = round(float(times[min(i + pulse_samples - 1, len(times) - 1)]), 2)
            pulses.append({
                "t_on":      t_on,
                "t_off":     t_off,
                "amplitude": round(amp, 2),
            })
            gap_samples = int(rng.uniform(0.1, 1.5) / dt)
            i += pulse_samples + gap_samples
        else:
            i += 1

    return pulses


def _sample_amplitude(rng, amp_lo, amp_hi, a=3.0, b=0.35):
    """
    Beta(a, b) mapped onto [amp_lo, amp_hi].
    Default a=b=2 gives a symmetric hill (mid-amplitudes more likely).
    Tweak a, b per generator for different biases:
      a=0.5, b=0.5  -> U-shaped (prefer extremes)
      a=2,   b=5    -> skewed low
      a=5,   b=2    -> skewed high
    """
    return amp_lo + (amp_hi - amp_lo) * rng.beta(a, b)


# --- generators ---

def generate_stochastic_pulses(
    t_max: float = 100.0,
    t_cutoff: float = 70.0,
    dt: float = 0.1,
    amplitude_range: tuple = (0.1, 5.0),
    rng: np.random.Generator = None,
) -> dict:
    """At each timepoint, randomly decide if a pulse starts (Poisson-like)."""
    if rng is None:
        rng = np.random.default_rng()

    threshold   = rng.uniform(0.05, 0.4)
    min_gap     = rng.uniform(0.5, 5.0)
    amp_lo, amp_hi = amplitude_range
    # skew toward mid-to-high amplitudes
    beta_a, beta_b = rng.uniform(1.5, 4.0), rng.uniform(1.5, 4.0)

    times = np.arange(0, t_max, dt)
    cutoff_idx = int(t_cutoff / dt)

    pulses = []
    i = 0
    while i < cutoff_idx:
        if rng.uniform() < threshold * dt:
            t_on     = times[i]
            duration = rng.uniform(0.5, 10.0)
            t_off    = min(t_on + duration, t_cutoff)
            amplitude = round(float(_sample_amplitude(rng, amp_lo, amp_hi, beta_a, beta_b)), 2)
            pulses.append({"t_on": round(t_on, 2), "t_off": round(t_off, 2), "amplitude": amplitude})
            gap = rng.uniform(min_gap, min_gap + 3.0)
            i += int((duration + gap) / dt)
        else:
            i += 1

    return {"t_max": t_max, "pulses": pulses}


def generate_sequential_pulses(
    t_max: float = 100.0,
    t_cutoff: float = 70.0,
    t_start_min: float = 1.0,
    amplitude_range: tuple = (0.1, 5.0),
    rng: np.random.Generator = None,
    **kwargs,
) -> dict:
    """Sequential pulse placement with random duration / amplitude / gap."""
    if rng is None:
        rng = np.random.default_rng()

    min_duration = rng.uniform(0.5, 3.0)
    max_duration = rng.uniform(min_duration, 15.0)
    min_gap      = rng.uniform(0.5, 5.0)
    max_pulses   = rng.integers(1, 6)
    amp_lo, amp_hi = amplitude_range
    # U-shaped: more likely to be clearly on or clearly off
    beta_a, beta_b = 1.6, 0.6

    pulses = []
    t_cursor = t_start_min

    for _ in range(max_pulses):
        remaining = t_cutoff - t_cursor
        if remaining < min_duration:
            break
        duration  = rng.uniform(min_duration, min(max_duration, remaining))
        #amplitude = round(float(_sample_amplitude(rng, amp_lo, amp_hi, beta_a, beta_b)), 2)
        t_on      = rng.uniform(t_cursor, t_cutoff - duration)
        t_off     = t_on + duration
        amplitude = round(float(_sample_amplitude(rng, amp_lo, amp_hi, beta_a, beta_b)), 2)
        pulses.append({"t_on": round(t_on, 2), "t_off": round(t_off, 2), "amplitude": amplitude})
        t_cursor  = t_off + rng.uniform(min_gap, min_gap + 3.0)

    return {"t_max": t_max, "pulses": pulses}


def generate_functional_pulses(
    t_max: float = 100.0,
    t_cutoff: float = 70.0,
    dt: float = 0.1,
    amplitude_range: tuple = (0.1, 3.5),
    rng: np.random.Generator = None,
) -> dict:
    """Stitch random functional segments (sin, cos, linear, square) over [0, t_cutoff]."""
    if rng is None:
        rng = np.random.default_rng()

    times = np.arange(0, t_max, dt)
    cutoff_idx = int(t_cutoff / dt)
    signal = np.zeros(len(times))
    amp_lo, amp_hi = amplitude_range
    # skew high — functional signals tend to look flat at low amplitude
    beta_a, beta_b = 3.0, 1.5

    def _sin(t, freq, amp):    return amp * np.sin(2 * np.pi * freq * t)
    def _cos(t, freq, amp):    return amp * np.cos(2 * np.pi * freq * t)
    def _square(t, freq, amp): return amp * np.sign(np.sin(2 * np.pi * freq * t))
    def _linear(t, _, amp):    # ramp up/down; ignores freq
        mid = t[-1] / 2 if t[-1] > 0 else 1
        return amp * (1 - np.abs(t - mid) / mid)

    func_pool = [_sin, _cos, _linear, _square]

    t_cursor = 0.0
    freq = rng.uniform(0.01, 0.5)
    while t_cursor < t_cutoff:
        seg_duration  = rng.uniform(10.0, 60.0)
        t_end         = min(t_cursor + seg_duration, t_cutoff)
        seg_start_idx = int(t_cursor / dt)
        seg_end_idx   = int(t_end   / dt)
        t_seg         = times[seg_start_idx:seg_end_idx] - t_cursor

        freq =+ rng.normal(0, 0.2)
        amp  = _sample_amplitude(rng, amp_lo, amp_hi, beta_a, beta_b)
        
        fn   = rng.choice(func_pool)

        seg_signal = fn(t_seg, freq, amp)
        seg_signal = np.clip(seg_signal, 0, amp_hi)
        signal[seg_start_idx:seg_end_idx] = seg_signal
        t_cursor = t_end

    pulses = _signal_to_pulses_fragmented(signal[:cutoff_idx], times[:cutoff_idx], dt, rng, amp_range=amplitude_range)
    return {"t_max": t_max, "pulses": pulses}


def generate_smoothed_pulses(
    t_max: float = 100.0,
    t_cutoff: float = 70.0,
    dt: float = 0.1,
    amplitude_range: tuple = (0.1, 5.0),
    rng: np.random.Generator = None,
    base_generator=None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng()
    if base_generator is None:
        base_generator = rng.choice([
            generate_stochastic_pulses,
            generate_sequential_pulses,
            generate_functional_pulses,
        ])

    base   = base_generator(t_max=t_max, t_cutoff=t_cutoff, dt=dt,
                            amplitude_range=amplitude_range, rng=rng)
    times  = np.arange(0, t_max, dt)
    signal = _pulses_to_signal(base["pulses"], times, dt)

    sigma    = rng.uniform(1.0, 20.0)
    smoothed = gaussian_filter1d(signal, sigma=sigma)
    smoothed = np.clip(smoothed, 0, amplitude_range[1])

    cutoff_idx = int(t_cutoff / dt)

    # fragment the smoothed envelope into short discrete pulses
    pulses = _signal_to_pulses_fragmented(
        smoothed[:cutoff_idx], times[:cutoff_idx], dt, rng,
        min_duration=0.5, amp_range=amplitude_range,
    )
    return {"t_max": t_max, "pulses": pulses}


# --- plot ---

def plot_signal_grid(t_max: float = 100.0, dt: float = 0.1, seed: int = 44):
    rng   = np.random.default_rng(seed)
    times = np.arange(0, t_max, dt)

    generators = [
        ("stochastic", generate_stochastic_pulses),
        ("sequential", generate_sequential_pulses),
        ("functional", generate_functional_pulses),
        ("smoothed",   generate_smoothed_pulses),
    ]
    colors = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD"]

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#FAFAFA")
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.3)

    for row, (label, gen_fn) in enumerate(generators):
        for col in range(4):
            ax     = fig.add_subplot(gs[row, col])
            result = gen_fn(t_max=t_max, rng=rng)
            signal = _pulses_to_signal(result["pulses"], times, dt)

            ax.fill_between(times, signal, alpha=0.25, color=colors[row])
            ax.plot(times, signal, lw=1.2, color=colors[row])
            ax.axvline(70, color="#888", lw=0.8, ls="--", alpha=0.5)

            ax.set_xlim(0, t_max)
            ax.set_ylim(-0.2, 5.5)
            ax.set_yticks([0, 2.5, 5])
            ax.tick_params(labelsize=7)
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#CCCCCC")

            if col == 0:
                ax.set_ylabel(label, fontsize=9, fontweight="bold", color=colors[row])
            if row == 0:
                ax.set_title(f"sample {col + 1}", fontsize=9, color="#555")
            if row == 3:
                ax.set_xlabel("time (s)", fontsize=7)

    fig.suptitle("Signal generator samples", fontsize=13, y=1.01, color="#222")
    plt.tight_layout()
    plt.savefig("signal_grid.png", dpi=150, bbox_inches="tight")
    plt.show()