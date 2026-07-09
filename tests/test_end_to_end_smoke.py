"""End-to-end smoke test on the canonical bundle parquets.

Skipped when the real dataset files are absent (e.g. CI without the mount).
"""
from pathlib import Path

import pytest

from optoerk.data.seq2seq_data import Seq2SeqDataset, load

_ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(
    not (_ROOT / "dataset.parquet.v0").exists(),
    reason="dataset.parquet.v0 not present",
)


def test_real_windowed_and_track_contracts():
    cnr_w, stim_w, cond_w = load("real", window_size=20, stride=5)
    assert cnr_w.ndim == 2 and stim_w.shape[1] == 9
    assert cnr_w.shape[0] == stim_w.shape[0] == len(cond_w)

    cnr_t, stim_t, cond_t = load("real")
    assert len(cnr_t) == len(cond_t)
    assert stim_t[0].shape[0] == 9  # channel-first, 9 stim features


def test_dataset_shapes_from_tracks():
    cnr_t, stim_t, cond_t = load("real")
    ds = Seq2SeqDataset(cnr_t, stim_t, cond_t, 10, 5, stride=5)
    assert len(ds) > 0
    enc_in, dec_stim, dec_target = ds[0]
    assert tuple(enc_in.shape) == (10, 10)   # (H, 1 + n_stim)
    assert tuple(dec_stim.shape) == (5, 9)   # (F, n_stim)
    assert tuple(dec_target.shape) == (5,)   # (F,)
