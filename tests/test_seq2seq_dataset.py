"""Consolidated Seq2SeqDataset == the old inline impl (2D and object arrays)."""
import numpy as np

from optoerk.data.seq2seq_data import Seq2SeqDataset


class _OldSeq2SeqDataset:
    """Frozen copy of the pre-refactor 2D-only inline dataset (regression guard)."""

    def __init__(self, cnr, stim, conditions, history_len, future_len, stride=5):
        self.samples = []
        self.sample_conditions = []
        total = history_len + future_len
        for i in range(len(cnr)):
            t = 0
            while t + total <= cnr.shape[1]:
                enc_cnr = cnr[i, t : t + history_len]
                enc_stim = stim[i, :, t : t + history_len]
                dec_stim = stim[i, :, t + history_len : t + total]
                full_window = cnr[i, t : t + total]
                dec_target = np.diff(full_window)[history_len - 1 : history_len - 1 + future_len]
                enc_in = np.concatenate([enc_cnr[:, None], enc_stim.T], axis=-1)
                self.samples.append((enc_in, dec_stim.T, dec_target))
                self.sample_conditions.append(str(conditions[i]))
                t += stride


def _random_2d(n=5, T=30, n_stim=9, seed=0):
    rng = np.random.default_rng(seed)
    cnr = rng.standard_normal((n, T)).astype(np.float32)
    stim = rng.standard_normal((n, n_stim, T)).astype(np.float32)
    conditions = np.array([f"cond_{i % 2}" for i in range(n)])
    return cnr, stim, conditions


def _to_object(cnr, stim):
    co = np.empty(len(cnr), dtype=object)
    so = np.empty(len(stim), dtype=object)
    for i in range(len(cnr)):
        co[i] = cnr[i]
        so[i] = stim[i]
    return co, so


def _assert_same_samples(a, b):
    assert len(a.samples) == len(b.samples)
    assert a.sample_conditions == b.sample_conditions
    for (e1, d1, t1), (e2, d2, t2) in zip(a.samples, b.samples):
        assert np.array_equal(e1, e2)
        assert np.array_equal(d1, d2)
        assert np.array_equal(t1, t2)


def test_parity_with_old_impl_2d():
    cnr, stim, cond = _random_2d()
    old = _OldSeq2SeqDataset(cnr, stim, cond, 10, 5, stride=5)
    new = Seq2SeqDataset(cnr, stim, cond, 10, 5, stride=5)
    _assert_same_samples(old, new)
    assert len(new) > 0


def test_2d_and_object_arrays_equivalent():
    cnr, stim, cond = _random_2d()
    new_2d = Seq2SeqDataset(cnr, stim, cond, 10, 5, stride=5)
    co, so = _to_object(cnr, stim)
    new_obj = Seq2SeqDataset(co, so, cond, 10, 5, stride=5)
    _assert_same_samples(new_2d, new_obj)


def test_object_arrays_variable_length():
    # variable-T tracks: only cells long enough yield windows
    rng = np.random.default_rng(1)
    lengths = [40, 16, 25]
    cnr = np.empty(3, dtype=object)
    stim = np.empty(3, dtype=object)
    for i, L in enumerate(lengths):
        cnr[i] = rng.standard_normal(L).astype(np.float32)
        stim[i] = rng.standard_normal((9, L)).astype(np.float32)
    ds = Seq2SeqDataset(cnr, stim, None, 10, 5, stride=5)
    # sample_conditions are None when conditions is None
    assert set(ds.sample_conditions) == {None}
    assert len(ds) > 0


def test_delta_target_and_channel_order():
    # ramp CNR -> constant deltas of 1.0; enc_in channel 0 == CNR window
    T = 20
    cnr = np.arange(T, dtype=np.float32)[None, :]          # (1, T) ramp
    stim = np.zeros((1, 9, T), dtype=np.float32)
    stim[0, 0, :] = np.arange(T)                            # u_t channel distinct
    ds = Seq2SeqDataset(cnr, stim, ["c"], history_len=8, future_len=4, stride=100)
    enc_in, dec_stim, dec_target = ds.samples[0]
    assert np.allclose(dec_target, 1.0)                     # ramp deltas
    assert np.array_equal(enc_in[:, 0], cnr[0, :8])         # ch0 = CNR
    assert np.array_equal(enc_in[:, 1:], stim[0, :, :8].T)  # ch1.. = STIM_COLS
    assert dec_stim.shape == (4, 9)
