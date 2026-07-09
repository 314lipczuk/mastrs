"""Block-bootstrap baseline prepending.

With a history window of length ``H``, the first ``H`` frames of every trajectory
can never be a forecast target — so models never learn the stimulation *onset*
and videos start mid-trajectory. The first ~10 frames of each trajectory are the
unperturbed baseline. We synthesise ``H`` baseline-like frames (with zero light)
and prepend them, so the first real frame becomes predictable from a full history.

Synthesis is a (non-wrapping) circular block bootstrap over *time indices* of the
observed baseline segment: we sample contiguous index blocks and let the caller
gather every channel by the same indices, preserving cross-channel structure at
each frame. The prepended stimulation is always zero (baseline = no light).

Used by both pipelines (TCN and seq2seq/seq2scal) so training, eval, and video
stay identical.
"""
from __future__ import annotations

import numpy as np

DEFAULT_N_BASELINE = 10
DEFAULT_BLOCK_LEN = 5


def bootstrap_baseline_indices(
    n_baseline: int,
    n_prepend: int,
    block_len: int = DEFAULT_BLOCK_LEN,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return ``n_prepend`` indices into ``[0, n_baseline)`` via block bootstrap.

    Contiguous blocks of length ``min(block_len, n_baseline)`` are sampled from
    non-wrapping start positions and concatenated, preserving local
    autocorrelation of the baseline segment. The caller gathers channel values
    at the returned indices.
    """
    if n_baseline <= 0:
        raise ValueError("n_baseline must be > 0")
    if n_prepend <= 0:
        return np.empty(0, dtype=int)
    rng = np.random.default_rng() if rng is None else rng
    bl = min(block_len, n_baseline)
    max_start = n_baseline - bl
    out = np.empty(n_prepend, dtype=int)
    i = 0
    while i < n_prepend:
        start = int(rng.integers(0, max_start + 1))
        for k in range(bl):
            if i >= n_prepend:
                break
            out[i] = start + k
            i += 1
    return out


def prepend_channels(
    channels: list[np.ndarray],
    n_prepend: int,
    *,
    zero_channels: set[int] | None = None,
    n_baseline: int = DEFAULT_N_BASELINE,
    block_len: int = DEFAULT_BLOCK_LEN,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Prepend ``n_prepend`` baseline-like frames to a list of 1-D channel arrays.

    All channels share the same bootstrapped time indices (so per-frame
    cross-channel structure is preserved), except those in ``zero_channels``
    (e.g. the stimulation/light channel), which are prepended with zeros.

    Each array in ``channels`` must be 1-D of the same length ``T``. Returns new
    arrays of length ``T + n_prepend``.
    """
    if n_prepend <= 0:
        return [np.asarray(c) for c in channels]
    zero_channels = zero_channels or set()
    T = len(channels[0])
    nb = min(n_baseline, T)
    idx = bootstrap_baseline_indices(nb, n_prepend, block_len=block_len, rng=rng)
    out = []
    for ci, arr in enumerate(channels):
        arr = np.asarray(arr)
        if ci in zero_channels:
            pre = np.zeros(n_prepend, dtype=arr.dtype)
        else:
            pre = arr[:nb][idx].astype(arr.dtype)
        out.append(np.concatenate([pre, arr]))
    return out
