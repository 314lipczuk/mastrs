"""
Generate synthetic EGFR cascade data:
  1. Sample 10000 light stimulation patterns (25% each generator, shuffled)
  2. Simulate ODE trajectories for each pattern
  3. Save to parquet
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

from optoerk.models.mechanistic.mechanistic_model import Model
from optoerk.models.mechanistic.egfr_simplified import (
    model_eqs, PARAM_NAMES, STATE_NAMES, NODE_NAMES,
    generate_stochastic_pulses, generate_sequential_pulses,
    generate_functional_pulses, generate_smoothed_pulses,
    _pulses_to_signal,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DRY_RUN = False

N_TOTAL = 10000
N_PER_GENERATOR = N_TOTAL // 4
T_MAX = 100.0
N_TIMEPOINTS = 100 # output resolution for trajectories
SEED = 42
PARAMS_PATH = "egfr_params.json"
OUTPUT_PATH = "synthetic_EGFR_data.parquet"

GENERATORS = [
    ("stochastic", generate_stochastic_pulses),
    ("sequential", generate_sequential_pulses),
    ("functional", generate_functional_pulses),
    ("smoothed",   generate_smoothed_pulses),
]

# ---------------------------------------------------------------------------
# Setup model + params
# ---------------------------------------------------------------------------

def pattern_light_func(t_val, t_args=None):
    """Evaluate a pulse-list light pattern at time t_val."""
    pulses = t_args["pulses"]
    return sum(p["amplitude"] for p in pulses if p["t_on"] <= t_val <= p["t_off"])


def build_system():
    m = Model(
        name="egfr_synth",
        states=STATE_NAMES,
        parameters=PARAM_NAMES,
        model_definition=model_eqs,
        t_dep="light",
        t_func=pattern_light_func,
    )
    return m.make_numerical()


def load_params(path):
    with open(path) as f:
        pdict = json.load(f)
    return np.array([pdict[p] for p in PARAM_NAMES])


# ---------------------------------------------------------------------------
# 1. Generate light patterns
# ---------------------------------------------------------------------------

def generate_all_patterns(n_total, seed):
    rng = np.random.default_rng(seed)
    n_per = n_total // 4
    remainder = n_total - 4 * n_per

    patterns = []
    for gen_name, gen_fn in GENERATORS:
        count = n_per + (1 if len(patterns) < remainder * (n_per + 1) else 0)
        # just use n_per, handle remainder below
        for _ in range(n_per):
            pat = gen_fn(t_max=T_MAX, rng=rng)
            pat["generator"] = gen_name
            patterns.append(pat)

    # fill remainder from random generators
    for _ in range(remainder):
        gen_name, gen_fn = GENERATORS[rng.integers(0, 4)]
        pat = gen_fn(t_max=T_MAX, rng=rng)
        pat["generator"] = gen_name
        patterns.append(pat)

    # shuffle
    rng.shuffle(patterns)
    return patterns


# ---------------------------------------------------------------------------
# 2. Simulate ODE for each pattern
# ---------------------------------------------------------------------------

def simulate_pattern(system, params_vec, pattern, times):
    y0 = np.zeros(len(STATE_NAMES))
    t_args = {"pulses": pattern["pulses"]}

    sol = solve_ivp(
        lambda t, y: system(t, y, params_vec, pattern_light_func, t_args),
        [times[0], times[-1]],
        y0,
        t_eval=times,
        method="LSODA",
        rtol=1e-8,
    )
    return sol


def build_dataframe(patterns, system, params_vec):
    times = np.linspace(0, T_MAX, N_TIMEPOINTS)
    rows = []
    n_failed = 0

    for i, pat in enumerate(tqdm(patterns, desc="Simulating")):
        sol = simulate_pattern(system, params_vec, pat, times)

        if not sol.success:
            n_failed += 1
            continue

        # light signal at each timepoint
        light_vals = np.array([
            pattern_light_func(tv, {"pulses": pat["pulses"]}) for tv in times
        ])

        row = {
            "pattern_id": i,
            "generator": pat["generator"],
            "pulses_json": json.dumps(pat["pulses"]),
            "t_max": pat["t_max"],
            "times": times.tolist(),
            "light": light_vals.tolist(),
        }
        for si, state in enumerate(STATE_NAMES):
            row[state] = sol.y[si].tolist()

        rows.append(row)

    if n_failed:
        print(f"WARNING: {n_failed}/{len(patterns)} simulations failed")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Plotting for qualitative check
# ---------------------------------------------------------------------------

COLORS = {"RAS": "#e41a1c", "RAF": "#377eb8", "MEK": "#4daf4a", "NFB": "#984ea3", "ERK": "#ff7f00"}


def plot_sample_trajectories(df, n=6, seed=0):
    """Sample n trajectories from the dataframe and plot them."""
    rng = np.random.default_rng(seed)
    sample = df.sample(n=min(n, len(df)), random_state=int(rng.integers(1e9)))

    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), squeeze=False)

    for ax_row, (_, row) in zip(axes, sample.iterrows()):
        ax = ax_row[0]
        times = np.array(row["times"])
        light = np.array(row["light"])

        # light on twin axis
        ax_light = ax.twinx()
        ax_light.fill_between(times, light, alpha=0.15, color="gold")
        ax_light.set_ylabel("Light", color="orange")
        max_light = max(light) if max(light) > 0 else 1
        ax_light.set_ylim(-0.05, max_light * 4)
        ax_light.tick_params(axis="y", labelcolor="orange")

        # state trajectories
        for si, node in enumerate(NODE_NAMES):
            state = STATE_NAMES[si]
            y = np.array(row[state])
            label = node
            if node == "NFB":
                nfb_max = np.max(y)
                if nfb_max > 1e-12:
                    y = y / nfb_max
                label = "NFB (norm)"
            ax.plot(times, y, label=label, color=COLORS[node], linewidth=1.5)

        ax.set_title(f"pattern {row['pattern_id']}  [{row['generator']}]", fontweight="bold")
        ax.set_ylabel("Active fraction")
        ax.set_ylim(bottom=-0.02)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1][0].set_xlabel("Time")
    plt.tight_layout()
    #plt.savefig("synthetic_sample_trajectories.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plotted {len(sample)} sample trajectories")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n = 5 if DRY_RUN else N_TOTAL
    print(f"{'DRY RUN' if DRY_RUN else 'FULL RUN'}: generating {n} patterns")

    patterns = generate_all_patterns(n, seed=SEED)
    print(f"Generated {len(patterns)} light patterns")

    system = build_system()
    params_vec = load_params(PARAMS_PATH)
    print(f"Params: {dict(zip(PARAM_NAMES, params_vec))}")

    df = build_dataframe(patterns, system, params_vec)
    print(f"DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")

    out = OUTPUT_PATH if not DRY_RUN else "synthetic_EGFR_data_dry.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved to {out}")

    plot_sample_trajectories(df, n=min(5, len(df)))
