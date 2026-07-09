"""Train SAC to drive EGFR cascade into sustained ERK oscillations."""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

from optoerk.models.rl.egfr_oscillation_env import EGFROscillationEnv
from optoerk.models.rl.dynamics import EGFR_STATE_NAMES


OBS_STACK = 4


def make_env():
    return EGFROscillationEnv(
        dt=0.5,
        max_steps=400,
        target_state=4,          # ERK
        target_freq=1.0 / 30.0,  # one cycle per 30s
        target_amplitude=0.6,    # peak-to-peak
        window_size=120,         # 60s = 2 full cycles
        max_light=5.0,
        obs_stack=OBS_STACK,
    )


def train(total_timesteps=100_000, test_run=False):
    if test_run:
        total_timesteps = 2_000

    env = make_env()
    eval_env = make_env()

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=256 if not test_run else 64,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="results/rl/egfr_best",
        log_path="results/rl/egfr_logs",
        eval_freq=1_000 if test_run else 10_000,
        n_eval_episodes=2 if test_run else 5,
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save("results/rl/egfr_sac_final")
    return model


def evaluate(model, n_episodes=3, max_steps=400):
    env = make_env()
    fig, axes = plt.subplots(n_episodes, 1, figsize=(14, 4 * n_episodes))
    if n_episodes == 1:
        axes = [axes]

    for ep, ax in enumerate(axes):
        obs, _ = env.reset(seed=ep + 10)
        # With stacking, current state is the first n_states elements
        history = {name: [obs[i]] for i, name in enumerate(EGFR_STATE_NAMES)}
        light_hist = []

        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            for i, name in enumerate(EGFR_STATE_NAMES):
                history[name].append(obs[i])
            light_hist.append(action[0])
            if done or truncated:
                break

        t = np.arange(len(history["ERK"])) * env.dt

        ax_light = ax.twinx()
        ax_light.fill_between(
            t[1:], light_hist, alpha=0.15, color="gold", label="light"
        )
        ax_light.set_ylabel("Light", color="goldenrod")
        ax_light.set_ylim(0, 6)

        for name in EGFR_STATE_NAMES:
            lw = 2.0 if name == "ERK" else 0.8
            alpha = 1.0 if name == "ERK" else 0.5
            ax.plot(t, history[name], label=name, linewidth=lw, alpha=alpha)

        ax.set_ylabel(f"Episode {ep}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper left", fontsize=8, ncol=5)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("SAC-driven EGFR Oscillation")
    fig.tight_layout()
    fig.savefig("results/rl/egfr_oscillation_trajectories.png", dpi=150)
    plt.show()
    print("Saved to results/rl/egfr_oscillation_trajectories.png")


if __name__ == "__main__":
    import sys
    test_run = "--test" in sys.argv
    model = train(test_run=test_run)
    evaluate(model, n_episodes=1 if test_run else 3)
