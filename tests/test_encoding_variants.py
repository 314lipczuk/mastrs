"""The four encoding variants: do they build, differ, and stay one-factor?

The comparison is only meaningful if each run differs from the baseline in
exactly one thing and everything else is identical. These tests pin that, plus
the two places the wiring can silently go wrong: the interaction channel must
reach the DECODER (encoder-only would be invisible across the rollout), and the
channel order must travel on the bundle so serving assembles inputs the same way.
"""
import pytest
import torch

from optoerk.data.history_data import (
    DERIVED,
    FEATURE_SETS,
    FUTURE_CHANNELS,
    U_X_EXPR,
    resolve_feature_set,
)
from optoerk.data.history_dataset import channels_for
from optoerk.models.seq2scal_history import HistoryConfig, Seq2ScalarHistory


def _cfg(feature_set="base", film_cond="fluence", **kw):
    feats, fut = resolve_feature_set(feature_set)
    ch = channels_for(feats)
    return HistoryConfig(
        input_dim=len(ch), stim_dim=len(fut), future_channels=fut,
        hidden_dim=16, num_layers=2, future_len=6, film="output",
        film_cond=film_cond, norm_channels=ch,
        norm_mean=[0.0] * len(ch), norm_std=[1.0] * len(ch), **kw,
    )


def test_the_four_variants_are_one_factor_apart():
    """a vs b vs c isolates the encoding; a vs d isolates area. Any other
    difference between them would make the comparison uninterpretable."""
    a = _cfg("base", "fluence")
    b = _cfg("interaction", "fluence")
    c = _cfg("base", "expr")
    d = _cfg("area", "fluence")

    assert a.norm_channels == c.norm_channels, "c must change ONLY the FiLM input"
    assert a.film_cond == "fluence" and c.film_cond == "expr"
    assert b.norm_channels == a.norm_channels + [U_X_EXPR]
    assert d.norm_channels == a.norm_channels + ["nuc_area"]
    assert b.film_cond == d.film_cond == "fluence"
    # everything not under test is shared
    for other in (b, c, d):
        assert (other.hidden_dim, other.num_layers, other.future_len, other.film) == \
               (a.hidden_dim, a.num_layers, a.future_len, a.film)


def test_the_interaction_reaches_the_decoder():
    """The decoder's only future input is fluence, so an encoder-only interaction
    channel would be invisible across the whole rollout — the regime that matters.
    It rides along as a second stim channel because it is a deterministic function
    of the commanded dose and a static per-cell value."""
    assert FUTURE_CHANNELS["interaction"] == ["u_t", U_X_EXPR]
    assert FUTURE_CHANNELS["base"] == ["u_t"]
    assert FUTURE_CHANNELS["area"] == ["u_t"], "area is not knowable ahead of time"
    b = _cfg("interaction")
    assert b.stim_dim == 2


def test_the_interaction_is_the_product_of_its_inputs():
    assert DERIVED[U_X_EXPR] == ("u_t", "optortk_expr")


@pytest.mark.parametrize("feature_set,film_cond", [
    ("base", "fluence"), ("interaction", "fluence"),
    ("base", "expr"), ("area", "fluence"),
])
def test_each_variant_forwards(feature_set, film_cond):
    cfg = _cfg(feature_set, film_cond)
    m = Seq2ScalarHistory(cfg).eval()
    B, L, F = 3, 12, cfg.future_len
    pi, mu, sigma = m(torch.randn(B, L, cfg.input_dim),
                      torch.full((B,), L), torch.randn(B, F, cfg.stim_dim))
    assert pi.shape == (B, F, cfg.n_gaussians)
    assert torch.isfinite(mu).all() and (sigma > 0).all()


def test_film_on_expr_actually_uses_the_expression_channel():
    """Not just that it runs — that changing expression changes the output. If the
    channel index were wrong the model would happily modulate on crowding."""
    cfg = _cfg("base", "expr")
    m = Seq2ScalarHistory(cfg).eval()
    i = cfg.norm_channels.index("optortk_expr")
    B, L, F = 4, 12, cfg.future_len
    ctx = torch.randn(B, L, cfg.input_dim)
    lens, fut = torch.full((B,), L), torch.randn(B, F, cfg.stim_dim)
    with torch.no_grad():
        lo = m(ctx.clone().index_fill_(2, torch.tensor([i]), -1.7), lens, fut)[1]
        hi = m(ctx.clone().index_fill_(2, torch.tensor([i]), +1.7), lens, fut)[1]
    assert not torch.allclose(lo, hi), "FiLM-on-expr ignored the expression channel"

    # ...while the fluence-conditioned baseline reads the same channel only through
    # the encoder, so it must not blow up either
    base = Seq2ScalarHistory(_cfg("base", "fluence")).eval()
    with torch.no_grad():
        assert torch.isfinite(base(ctx, lens, fut)[1]).all()


def test_stim_dim_must_match_future_channels():
    """The decoder input width IS the number of known-future channels; a mismatch
    is a silent shape bug at rollout time."""
    with pytest.raises(ValueError, match="must equal len\\(future_channels\\)"):
        HistoryConfig(input_dim=6, stim_dim=1, future_channels=["u_t", U_X_EXPR],
                      norm_channels=channels_for(FEATURE_SETS["interaction"]))
    with pytest.raises(ValueError, match="not among norm_channels"):
        HistoryConfig(input_dim=5, stim_dim=2, future_channels=["u_t", "nope"],
                      norm_channels=channels_for(FEATURE_SETS["base"]))


def test_unknown_feature_set_is_rejected():
    with pytest.raises(ValueError, match="unknown feature_set"):
        resolve_feature_set("does_not_exist")


def test_every_feature_set_has_future_channels():
    """A set without an entry would silently fall back to fluence-only."""
    assert set(FEATURE_SETS) == set(FUTURE_CHANNELS)
    for name, fut in FUTURE_CHANNELS.items():
        assert set(fut) <= set(FEATURE_SETS[name]), name


def test_channel_order_is_cnr_first():
    """cnr is the target and is read at a fixed index by the decoder feedback."""
    for name in FEATURE_SETS:
        feats, _ = resolve_feature_set(name)
        assert channels_for(feats)[0] == "cnr"


def test_area_lean_is_area_minus_fov_density():
    """The candidate is one channel away from `area`, and that channel is the one
    the comparison showed the model does not use (perm importance 0.0046, the
    lowest of any) and structurally cannot use per-cell — fov_density is identical
    for every cell in a frame."""
    area, _ = resolve_feature_set("area")
    lean, lean_fut = resolve_feature_set("area_lean")
    assert set(area) - set(lean) == {"fov_density"}
    assert set(lean) - set(area) == set()
    assert lean_fut == ["u_t"], "no new known-future input"
    # ...and it keeps the two channels that ARE used
    assert "optortk_expr" in lean and "nuc_area" in lean


def test_area_lean_is_encoding_a():
    """Encoding (a) = expression as a plain encoder channel: no interaction
    channel, FiLM on fluence. Same as the winning d_area run."""
    lean, _ = resolve_feature_set("area_lean")
    assert U_X_EXPR not in lean
    cfg = _cfg("area_lean", "fluence")
    assert cfg.stim_dim == 1 and cfg.film_cond == "fluence"
    assert Seq2ScalarHistory(cfg).expr_idx is None   # FiLM reads fluence, not expr


# ---------------------------------------------------------------------------
# eval/video path: channels must be matched BY NAME, not by row position
# ---------------------------------------------------------------------------


def test_predict_history_cell_aligns_channels_by_name():
    """`base` and `area_lean` have the SAME channel count (5) but different
    channels, so a width check cannot separate them — only names can.

    This used to unpack `cell.stim` by fixed row index, hardcoded to the historical
    four features. Under `area_lean`, which drops `fov_density`, that fed the model
    crowding as density, expression as crowding and cell area as expression, with
    nothing anywhere to say so. Each feature row here carries a unique sentinel and
    the assertion is that every channel receives the row bearing its own name.
    """
    import types

    import numpy as np

    from optoerk.eval.history_predict import predict_history_cell

    assert len(channels_for(resolve_feature_set("base")[0])) == \
           len(channels_for(resolve_feature_set("area_lean")[0])), \
        "the premise: equal width, so only names can disambiguate"

    for fs in ("base", "area_lean", "interaction"):
        feats, fut = resolve_feature_set(fs)
        ch = channels_for(feats)
        cfg = HistoryConfig(
            input_dim=len(ch), stim_dim=len(fut), future_channels=fut,
            hidden_dim=8, num_layers=1, future_len=4, norm_channels=ch,
            norm_mean=[0.0] * len(ch), norm_std=[1.0] * len(ch),
        )
        m = Seq2ScalarHistory(cfg).eval()
        seen = {}
        _orig = m.forward

        def _spy(ctx, lens, futr, *a, _o=_orig, _s=seen, **k):
            _s["ctx"] = ctx.detach().clone()
            return _o(ctx, lens, futr, *a, **k)

        m.forward = _spy
        T = 30
        cell = types.SimpleNamespace(
            cnr=np.full(T, 7.0, np.float32),
            stim=np.stack([np.full(T, 100.0 + i, np.float32)
                           for i in range(len(feats))]),
        )
        predict_history_cell(m, cell, 15, None, None, "cpu")
        got = seen["ctx"][0, -1].numpy()
        want = np.array([7.0] + [100.0 + i for i in range(len(feats))], np.float32)
        for name, g, w in zip(ch, got, want):
            assert abs(float(g) - float(w)) < 1e-5, f"{fs}: channel {name} misaligned"


def test_predict_history_cell_rejects_a_width_mismatch():
    """The backstop for the gross case, where the counts differ at all."""
    import types

    import numpy as np
    import pytest as _pytest

    from optoerk.eval.history_predict import predict_history_cell

    feats, fut = resolve_feature_set("interaction")     # 5 features
    ch = channels_for(feats)
    cfg = HistoryConfig(input_dim=len(ch), stim_dim=len(fut), future_channels=fut,
                        hidden_dim=8, num_layers=1, future_len=4, norm_channels=ch,
                        norm_mean=[0.0] * len(ch), norm_std=[1.0] * len(ch))
    m = Seq2ScalarHistory(cfg).eval()
    cell = types.SimpleNamespace(cnr=np.ones(30, np.float32),
                                 stim=np.ones((4, 30), np.float32))   # wrong width
    with _pytest.raises(ValueError, match="feature rows but the model expects"):
        predict_history_cell(m, cell, 15, None, None, "cpu")


def test_predict_many_builds_the_decoder_input_from_future_channels():
    """A stim_dim=2 model needs both known-future channels; hardcoding u_t would
    hand it a half-width tensor and fail at the decoder."""
    import numpy as np

    from optoerk.eval.history_predict import predict_many

    feats, fut = resolve_feature_set("interaction")
    ch = channels_for(feats)
    assert len(fut) == 2
    cfg = HistoryConfig(input_dim=len(ch), stim_dim=len(fut), future_channels=fut,
                        hidden_dim=8, num_layers=1, future_len=4, norm_channels=ch,
                        norm_mean=[0.0] * len(ch), norm_std=[1.0] * len(ch))
    m = Seq2ScalarHistory(cfg).eval()
    T = 30
    mu, sd = predict_many(
        m, np.full(T, 1.0, np.float32), np.full(T, 2.0, np.float32), [15],
        channels={c: np.full(T, 3.0, np.float32) for c in feats},
    )
    assert mu.shape == (1, 4) and np.isfinite(mu).all()
