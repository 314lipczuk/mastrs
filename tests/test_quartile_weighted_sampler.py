"""Smoke test for QuartileWeightedSampler.

Builds the sampler from a synthetic resp_std distribution, iterates one
epoch, and asserts the realized per-bin frequencies fall within 2% of
the configured weights. Protects against off-by-one errors in per-bin
sample-count rounding.

Run: `uv run python tests/test_quartile_weighted_sampler.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.lstm_seq2scal_mdn_q4weighted import (
    QuartileWeightedSampler,
    TrainingConfig,
)


def main():
    rng = np.random.default_rng(0)
    n_windows = 10_000
    resp_stds = np.abs(rng.standard_normal(n_windows) * 0.1).astype(np.float32)

    weights = (0.10, 0.20, 0.30, 0.40)
    batch_size = 64
    sampler = QuartileWeightedSampler(
        resp_stds,
        batch_size=batch_size,
        weights=weights,
    )

    print(
        f"edges={sampler.edges.round(4).tolist()} "
        f"bin_sizes={[len(b) for b in sampler.bin_indices]} "
        f"weights={list(sampler.weights)} "
        f"samples_per_bin={sampler.samples_per_bin} "
        f"batches/epoch={len(sampler)}"
    )

    counts = np.zeros(4, dtype=np.int64)
    total_drawn = 0
    for batch in sampler:
        for idx in batch:
            for k in range(4):
                if idx in set(sampler.bin_indices[k].tolist()):
                    counts[k] += 1
                    break
            total_drawn += 1

    total = total_drawn
    realized = counts / total
    print(f"realized_freq={realized.round(4).tolist()}  (total={total})")

    tol = 0.02
    for k, (w, r) in enumerate(zip(weights, realized)):
        assert abs(r - w) < tol, (
            f"Q{k+1}: realized {r:.4f} drifted from configured {w:.4f} "
            f"by more than {tol}"
        )

    expected = np.array(sampler.samples_per_bin) / total
    assert np.allclose(expected.sum(), 1.0)

    print("OK — realized per-bin frequencies within 2% of weights.")

    # CLI-string parsing path: pydantic must coerce a JSON-ish string from
    # mo.cli_args() into list[float]. This is the failure mode that bit
    # the first cluster run.
    tc = TrainingConfig(
        sampler_type="quartile_weighted",
        quartile_weights="[0.10, 0.20, 0.30, 0.40]",
    )
    assert tc.quartile_weights == [0.10, 0.20, 0.30, 0.40], tc.quartile_weights

    tc2 = TrainingConfig(
        sampler_type="quartile_weighted",
        quartile_weights="0.05,0.15,0.30,0.50",
    )
    assert tc2.quartile_weights == [0.05, 0.15, 0.30, 0.50], tc2.quartile_weights

    print("OK — TrainingConfig parses CLI-string quartile_weights.")


if __name__ == "__main__":
    main()
