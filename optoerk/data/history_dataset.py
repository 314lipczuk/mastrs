"""Dataset for the full-history (long-gap) model.

Pieces, in order of construction:
  * deterministic per-cell train/val/test split (stratified by condition);
  * frozen train-population standardization stats (live-safe: a constant affine
    map, no per-trajectory or future stats);
  * (added next) sliding sample generation with self-concat random-break
    augmentation and the T-F context cap.

Input channels per frame = ``CHANNELS`` = [cnr] + HISTORY_FEATURES. cnr is also
the MDN target, kept in its native cnr units (baseline-normalized
`cnr_median_norm` or raw `cnr_median`, per the training ``cnr_mode``); the
encoder-input cnr is standardized like the other channels. The frozen z-score
stats are per-mode (`stats_path_for_mode`), since the cnr mean/std differ.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from optoerk.data.baseline_prepend import bootstrap_baseline_indices, prepend_channels
from optoerk.data.history_data import CNR_MODES, HISTORY_FEATURES

CHANNELS = ["cnr", *HISTORY_FEATURES]  # ["cnr", "u_t", "fov_density", "n_cells_200px", "optortk_expr"]


def stats_path_for_mode(cnr_mode: str = "norm") -> Path:
    """Frozen-stats file for a given cnr_mode.

    ``norm`` keeps the historical name (``history_norm_stats.json``) so existing
    bundles' fallback path is unchanged; other modes get a suffixed file
    (``history_norm_stats_raw.json``). The mean/std for the cnr channel differ by
    mode, so loading the wrong file silently mis-scales — the serving loader keys
    this off the model's own ``cnr_mode`` to prevent exactly that.
    """
    if cnr_mode not in CNR_MODES:
        raise ValueError(f"cnr_mode must be one of {CNR_MODES}, got {cnr_mode!r}")
    suffix = "" if cnr_mode == "norm" else f"_{cnr_mode}"
    return Path(__file__).resolve().parent / f"history_norm_stats{suffix}.json"


def make_split(
    conditions: np.ndarray,
    *,
    seed: int = 0,
    frac=(0.8, 0.1, 0.1),
) -> dict[str, np.ndarray]:
    """Deterministic per-cell split, stratified within each condition.

    Returns dict with keys 'train'/'val'/'test' -> int64 arrays of cell indices.
    Stratifying by condition keeps every protocol represented in each split.
    """
    assert abs(sum(frac) - 1.0) < 1e-9, "fracs must sum to 1"
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for cond in sorted(set(conditions.tolist())):
        idx = np.where(conditions == cond)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(frac[0] * n))
        n_va = int(round(frac[1] * n))
        train.append(idx[:n_tr])
        val.append(idx[n_tr:n_tr + n_va])
        test.append(idx[n_tr + n_va:])
    return {
        "train": np.concatenate(train).astype(np.int64),
        "val": np.concatenate(val).astype(np.int64),
        "test": np.concatenate(test).astype(np.int64),
    }


@dataclass
class NormStats:
    """Frozen per-channel standardization constants (train population)."""
    channels: list[str]
    mean: list[float]
    std: list[float]

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.mean, dtype=np.float32),
            np.asarray(self.std, dtype=np.float32),
        )

    def save(self, path: "str | Path | None" = None, *, cnr_mode: str = "norm") -> None:
        Path(path or stats_path_for_mode(cnr_mode)).write_text(
            json.dumps(self.__dict__, indent=2)
        )

    @classmethod
    def load(cls, path: "str | Path | None" = None, *, cnr_mode: str = "norm") -> "NormStats":
        return cls(**json.loads(Path(path or stats_path_for_mode(cnr_mode)).read_text()))


def compute_norm_stats(
    cnr: np.ndarray,
    feats: np.ndarray,
    train_idx: np.ndarray,
    *,
    std_floor: float = 1e-6,
) -> NormStats:
    """Per-channel mean/std over all frames of the TRAIN cells only.

    ``cnr`` : object array of (T,) ; ``feats`` : object array of (K, T) with
    rows ordered as HISTORY_FEATURES. Channel order = CHANNELS.
    """
    cnr_vals = np.concatenate([np.asarray(cnr[i], np.float64) for i in train_idx])
    feat_vals = [
        np.concatenate([np.asarray(feats[i], np.float64)[k] for i in train_idx])
        for k in range(len(HISTORY_FEATURES))
    ]
    cols = [cnr_vals, *feat_vals]
    mean = [float(c.mean()) for c in cols]
    std = [float(max(c.std(), std_floor)) for c in cols]
    return NormStats(channels=list(CHANNELS), mean=mean, std=std)


# Channel index of the fluence (u_t) within CHANNELS — the only future/decoder
# input (commanded; future crowding is unknown live).
FLU_IDX = CHANNELS.index("u_t")
CNR_IDX = CHANNELS.index("cnr")


class HistoryDataset(Dataset):
    """Sliding full-history samples with self-concat-random-break augmentation.

    Each item: a causal context window of input channels (length ≤ T-F, ending
    at the prediction point), the future commanded fluence over the horizon, and
    the target CNR over the horizon (in cnr units, unstandardized).

    Self-concat: with prob ``p_concat`` build ``[traj][break g][traj]`` and
    predict inside copy-2 at original index τ; the context is the last ``T-F``
    frames before the prediction point, which excludes copy-1's occurrence of
    the target (break ``g`` is the safety margin). When the prefix has scrolled
    out (large τ or g) the window is naturally shorter — the model has then
    "forgotten" exp1, which is the intended long-break behavior.
    """

    def __init__(
        self,
        cnr: np.ndarray,
        feats: np.ndarray,
        indices: np.ndarray,
        stats: NormStats,
        *,
        F: int = 10,
        t_min: int = 10,
        p_concat: float = 0.5,
        break_min: int = 0,
        break_max: int = 60,
        break_block_len: int = 5,
        n_baseline: int = 10,
        prepend_baseline: bool = False,
        prepend_len: int = 30,
        seed: int = 0,
    ):
        self.cnr = cnr
        self.feats = feats
        self.idx = np.asarray(indices, dtype=np.int64)
        self.F = F
        self.t_min = t_min
        self.p_concat = p_concat
        self.break_min = break_min
        self.break_max = break_max
        self.break_block_len = break_block_len
        self.n_baseline = n_baseline
        self.prepend_baseline = prepend_baseline
        self.prepend_len = prepend_len
        self.seed = seed
        # Persistent generator: advances across epochs so the same item yields
        # different (t, concat, break) draws each pass. Re-seed per DataLoader
        # worker via reseed() in a worker_init_fn to avoid duplicate streams.
        self.rng = np.random.default_rng(seed)
        m, s = stats.as_arrays()
        self.mean = m  # (C,)
        self.std = s
        # Fail loudly if the stats don't match the data's channel count (the
        # classic stale-frozen-stats mismatch) instead of a cryptic broadcast error.
        _ndata = 1 + int(np.asarray(feats[0]).shape[0]) if len(feats) else len(m)
        if len(m) != _ndata:
            raise ValueError(
                f"norm stats have {len(m)} channels but data has {_ndata} "
                f"(CHANNELS={CHANNELS}); the stats are stale for the current feature set"
            )

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.idx)

    def _channels(self, cell: int) -> np.ndarray:
        """Raw per-frame channel matrix (T, C) = [cnr, u_t, fov_density, n_cells_200px].

        When ``prepend_baseline`` is set, ``prepend_len`` block-bootstrapped
        baseline-like frames are prepended (light channel ``FLU_IDX`` zeroed, all
        others gathered from the same baseline indices), so the stimulation onset
        becomes predictable from a synthesised resting history.
        """
        c = np.asarray(self.cnr[cell], np.float32)
        f = np.asarray(self.feats[cell], np.float32)  # (K, T)
        X = np.concatenate([c[None, :], f], axis=0).T  # (T, C)
        if self.prepend_baseline and self.prepend_len > 0:
            cols = prepend_channels(
                [X[:, j] for j in range(X.shape[1])],
                self.prepend_len,
                zero_channels={FLU_IDX},
                n_baseline=self.n_baseline,
                block_len=self.break_block_len,
                rng=self.rng,
            )
            X = np.stack(cols, axis=1)  # (T + prepend_len, C)
        return X

    def __getitem__(self, i: int) -> dict:
        cell = int(self.idx[i])
        rng = self.rng
        X = self._channels(cell)  # (T, C) raw
        T, C = X.shape
        F, t_min = self.F, self.t_min

        if T >= F + t_min + 2 and rng.random() < self.p_concat:
            g = int(rng.integers(self.break_min, self.break_max + 1))
            tau = int(rng.integers(0, T - F + 1))           # target start in copy-2
            # Break = a realistic rest period: block-bootstrap the cell's own
            # baseline frames across all channels (preserves resting noise /
            # crowding), with the light channel zeroed. Same machinery as
            # `prepend_baseline`.
            if g > 0:
                nb = min(self.n_baseline, T)
                bidx = bootstrap_baseline_indices(nb, g, self.break_block_len, rng)
                brk = X[bidx].copy()                         # (g, C) baseline-like
                brk[:, FLU_IDX] = 0.0                        # no light during rest
            else:
                brk = np.zeros((0, C), np.float32)
            # S = copy1 ++ break(g) ++ copy2; predict at copy-2 index tau.
            S = np.concatenate([X, brk, X], axis=0)
            t = T + g + tau                                  # absolute prediction point
            ctx = S[t - (T - F) : t]                         # last T-F frames; excludes copy-1's [tau:tau+F]
            fut_flu = X[tau : tau + F, FLU_IDX]              # copy-2 future == original
            tgt = X[tau : tau + F, CNR_IDX]
        else:
            t_hi = max(t_min, T - F)
            t = int(rng.integers(t_min, t_hi + 1))
            ctx = X[0:t]
            fut_flu = X[t : t + F, FLU_IDX]
            tgt = X[t : t + F, CNR_IDX]

        ctx_std = ((ctx - self.mean) / self.std).astype(np.float32)
        fut_flu_std = ((fut_flu - self.mean[FLU_IDX]) / self.std[FLU_IDX]).astype(np.float32)
        # Target is standardized cnr too (consistent with the standardized
        # encoder input and the decoder's autoregressive feedback). Denormalize
        # with the cnr stats to recover absolute CNR at the output boundary.
        tgt_std = ((tgt - self.mean[CNR_IDX]) / self.std[CNR_IDX]).astype(np.float32)
        return {
            "ctx": torch.from_numpy(ctx_std),                # (L, C)
            "fut_flu": torch.from_numpy(fut_flu_std)[:, None],  # (F, 1)
            "tgt": torch.from_numpy(tgt_std),                # (F,) standardized cnr
            "len": int(ctx_std.shape[0]),
        }


def collate_history(batch: list[dict]) -> dict:
    """Right-pad variable-length contexts; return lengths for pack_padded."""
    lens = [b["len"] for b in batch]
    Lmax = max(lens)
    C = batch[0]["ctx"].shape[1]
    ctx = torch.zeros(len(batch), Lmax, C, dtype=torch.float32)
    for i, b in enumerate(batch):
        ctx[i, : b["len"]] = b["ctx"]
    return {
        "ctx": ctx,                                          # (B, Lmax, C)
        "lengths": torch.tensor(lens, dtype=torch.long),     # (B,)
        "fut_flu": torch.stack([b["fut_flu"] for b in batch]),  # (B, F, 1)
        "tgt": torch.stack([b["tgt"] for b in batch]),       # (B, F)
    }


def make_history_collate(f_min: int, f_max: int, seed: int = 0):
    """Collate that samples one prediction horizon ``F ~ U[f_min, f_max]`` per
    batch and truncates the (F_max-length) future/target to it.

    Items are always built at ``f_max`` (so the context/concat cap reserves the
    longest horizon and stays leak-safe); per-batch truncation gives multi-length
    training with no padding/masking. ``f_min >= f_max`` → fixed horizon.
    """
    rng = np.random.default_rng(seed)

    def _collate(batch):
        out = collate_history(batch)
        if f_min < f_max:
            f = int(rng.integers(f_min, f_max + 1))
            out["fut_flu"] = out["fut_flu"][:, :f]
            out["tgt"] = out["tgt"][:, :f]
        return out

    return _collate


if __name__ == "__main__":
    # Regenerate the frozen normalization stats from the canonical training bundle
    # over the seed-0 train split — the exact split the training notebook uses. Run
    # this whenever HISTORY_FEATURES or the training bundle changes; the notebook
    # and serving both load the saved file via NormStats.load(), so a stale file
    # silently mismatches the channel count. One file per cnr_mode: the cnr channel
    # mean/std differ between baseline-normalized and raw CNR.
    from optoerk.data.history_data import CNR_MODES, load_history_tracks

    for _mode in CNR_MODES:
        _cnr, _feats, _cond, _ = load_history_tracks(cnr_mode=_mode)
        _split = make_split(_cond, seed=0)
        _stats = compute_norm_stats(_cnr, _feats, _split["train"])
        _stats.save(cnr_mode=_mode)
        print(
            f"[{_mode}] saved {len(_stats.channels)}-channel norm stats "
            f"-> {stats_path_for_mode(_mode)}\n"
            f"  channels: {_stats.channels}\n"
            f"  mean:     {[round(m, 4) for m in _stats.mean]}\n"
            f"  std:      {[round(s, 4) for s in _stats.std]}"
        )
