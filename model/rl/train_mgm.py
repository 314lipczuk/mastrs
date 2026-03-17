"""Proof-of-concept: train SAC on Moore-Greitzer Model using Stable Baselines3."""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

from model.rl.ode_env import ODEEnv
from model.rl.dynamics import mgm


def make_env():
    return ODEEnv(
        dynamics_fn=mgm,
        n_states=2,
        reward_states=[0],  # article only penalizes mass flow (x1)
        max_steps=200,
    )


def train(total_timesteps=50_000):
    env = make_env()
    eval_env = make_env()

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="results/rl/mgm_best",
        log_path="results/rl/mgm_logs",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save("results/rl/mgm_sac_final")
    return model


def evaluate(model, n_episodes=5, max_steps=200):
    env = make_env()
    fig, axes = plt.subplots(n_episodes, 1, figsize=(10, 3 * n_episodes), sharex=True)
    if n_episodes == 1:
        axes = [axes]

    for ep, ax in enumerate(axes):
        obs, _ = env.reset(seed=ep)
        x1_hist, x2_hist, action_hist = [obs[0]], [obs[1]], []

        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            x1_hist.append(obs[0])
            x2_hist.append(obs[1])
            action_hist.append(action[0])
            if done or truncated:
                break

        steps = range(len(x1_hist))
        ax.plot(steps, x1_hist, label="x1 (mass flow)")
        ax.plot(steps, x2_hist, label="x2 (pressure)")
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax.set_ylabel(f"Episode {ep}")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Step")
    fig.suptitle("SAC Controller on Moore-Greitzer Model")
    fig.tight_layout()
    fig.savefig("results/rl/mgm_sac_trajectories.png", dpi=150)
    plt.show()
    print("Saved to results/rl/mgm_sac_trajectories.png")


if __name__ == "__main__":
    model = train()
    evaluate(model)
