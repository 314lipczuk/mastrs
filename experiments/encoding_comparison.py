import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import torch
    from scipy import stats

    from optoerk.core.utils import results_write_path

    # Categorical hues, fixed order, never cycled. Validated for CVD separation.
    HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    INK, MUT, SURF = "#1a1a19", "#6b6a63", "#fcfcfb"
    return (
        HUES,
        INK,
        MUT,
        Path,
        SURF,
        mo,
        np,
        pl,
        plt,
        results_write_path,
        stats,
        torch,
    )


@app.cell
def _(mo):
    mo.md("""
    # optoRTK expression: which encoding, and does area earn its place?

    Four runs, each differing from the baseline in exactly one thing. All trained on
    `dataset_all_mcitrine.parquet` — the **real** mCitrine measurement, not the C0
    surrogate every earlier checkpoint used.

    | run | what it changes |
    |---|---|
    | `a_channel` | baseline: expression as a constant encoder channel |
    | `b_interaction` | adds `u_t × expr`, also fed to the decoder as a known future input |
    | `c_film_expr` | FiLM conditioned on expression instead of fluence |
    | `d_area` | baseline + `nuc_area` |

    `a` vs `b` vs `c` isolates the **encoding**. `a` vs `d` isolates whether
    **area** is worth a channel.

    ## What decides it

    Held-out NLL alone does not. A covariate can shave the average while being used
    for the wrong thing, so three further readouts carry equal weight:

    - **Gain test** — Spearman(expression, the model's believed dose effect).
      optoRTK expression physically *is* a gain on the dose–response, so a model
      that learned it predicts a bigger light effect for higher expressers. This is
      the question the encoding variants exist to answer.
    - **Permutation importance** — shuffle a channel across cells and watch NLL.
      A channel whose shuffle is free is a channel the model ignores. This is what
      decides `nuc_area`.
    - **Decile error** — helping on average while leaving the high expressers just
      as wrong is not success.

    Every run scores the same TEST samples in the same order, so the differences
    below are **paired**.
    """)
    return


@app.cell
def _(mo, results_write_path):
    results_input = mo.ui.text(
        value=str(results_write_path()),
        label="Results directory (searched recursively for run bundles)",
        full_width=True,
    )
    results_input
    return (results_input,)


@app.cell
def _(Path, mo, results_input, torch):
    # Runs are matched by the `enc_` name prefix the launcher gives them. Each
    # bundle carries its own encoding metrics, so nothing needs recomputing here —
    # and nothing can drift between what was trained and what is compared.
    RUNS = {
        "a_channel": "expr as a channel",
        "b_interaction": "+ u_t x expr (decoder too)",
        "c_film_expr": "FiLM on expr",
        "d_area": "+ nuc_area",
        "e_area_lean": "+ nuc_area, - fov_density",
    }
    # Which run everything is paired against. `a_channel` is right for the
    # encoding comparison, but `e_area_lean` differs from `d_area` by one channel
    # and from `a_channel` by TWO — paired against the baseline it would confound
    # the area gain with the density drop and neither could be attributed.
    BASELINE = "d_area" if "e_area_lean" in RUNS else "a_channel"
    _root = Path(results_input.value.strip())

    def _load(tag):
        # Metrics live in the bundle, written by the training notebook's
        # `save_bundle(metrics=...)`. Read from there rather than recomputing, so
        # what is compared is exactly what was trained. Newest run wins if a tag
        # was launched more than once.
        hits = sorted(_root.glob(f"*enc_{tag}*/bundle.pt"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        for h in hits:
            try:
                b = torch.load(h, map_location="cpu", weights_only=False)
            except Exception:
                continue
            enc = (b.get("metrics") or {}).get("encoding")
            if enc:
                return enc, h
        return None, None

    found, missing = {}, []
    for _tag in RUNS:
        _enc, _src = _load(_tag)
        if _enc is None:
            missing.append(_tag)
        else:
            found[_tag] = _enc

    mo.stop(
        not found,
        mo.md(
            f"**No run bundles found under `{_root}`.** Launch the four jobs in "
            f"`launcher.py` (the `enc_*` block) first; each writes its encoding "
            f"metrics into its own bundle."
        ),
    )
    mo.md(
        f"Loaded **{len(found)}** of {len(RUNS)} runs"
        + (f" — missing `{'`, `'.join(missing)}`" if missing else "")
    )
    return BASELINE, RUNS, found


@app.cell
def _(RUNS, found, mo, pl):
    # The headline table. Sorted by NLL, but read the whole row: a run that wins on
    # NLL and shows gain_rho ~ 0 has bought accuracy without learning the thing the
    # covariate is for.
    _rows = []
    for _tag, _label in RUNS.items():
        _m = found.get(_tag)
        if _m is None:
            continue
        _rows.append({
            "run": _tag,
            "change": _label,
            "NLL": round(_m["test_nll"], 4),
            "MAE": round(_m["test_mae"], 4),
            "RMSE": round(_m["test_rmse"], 4),
            "cov68": round(_m["cov68"], 3),
            "cov95": round(_m["cov95"], 3),
            "gain_rho": round(_m["gain_spearman"], 3),
            "decile_spread": round(_m["decile_mae_spread"], 4),
            "n_test": _m["n_test_samples"],
        })
    summary = pl.DataFrame(_rows).sort("NLL")
    mo.output.append(mo.md("### Headline — lower NLL/MAE better, higher `gain_rho` better"))
    mo.output.append(summary)
    return


@app.cell
def _(BASELINE, RUNS, found, mo, np, pl, stats):
    # PAIRED comparison against the baseline. Same test samples in the same order,
    # so the per-sample difference is meaningful and its spread gives a real
    # uncertainty — a difference of two independent means would not.
    _base = found.get(BASELINE)
    mo.stop(_base is None,
            mo.md(f"_No `{BASELINE}` baseline loaded — no paired test._"))

    _b_nll = np.asarray(_base["_per_sample_nll"])
    _b_err = np.asarray(_base["_per_sample_abs_err"])
    _rows = []
    for _tag, _label in RUNS.items():
        if _tag == BASELINE or _tag not in found:
            continue
        _m = found[_tag]
        _n = np.asarray(_m["_per_sample_nll"])
        _e = np.asarray(_m["_per_sample_abs_err"])
        if len(_n) != len(_b_nll):
            _rows.append({"run": _tag, "note": "different n — not paired"})
            continue
        _dn, _de = _n - _b_nll, _e - _b_err
        # TWO different uncertainties, and reading the wrong one flips the answer.
        # `se` is the standard error OF THE MEAN difference — is the average shift
        # real. `spread` is the per-sample scatter — does the variant win on most
        # individual samples or just on average. The scatter here is ~0.8 while the
        # mean shifts are ~0.01, so quoting only the spread reads as "nothing is
        # significant" when the paired means are in fact 3-5 sigma apart.
        _se = float(_dn.std(ddof=1) / np.sqrt(len(_dn)))
        _t = float(_dn.mean() / _se) if _se else float("nan")
        _tn = stats.wilcoxon(_n, _b_nll)
        _rows.append({
            "run": _tag,
            "change": _label,
            "d_NLL": round(float(_dn.mean()), 5),
            "se": round(_se, 5),
            "t": round(_t, 2),
            "p": f"{2 * stats.norm.sf(abs(_t)):.1e}",
            "wilcoxon_p": f"{_tn.pvalue:.1e}",
            "spread_p2.5_97.5": f"[{np.percentile(_dn, 2.5):+.3f}, {np.percentile(_dn, 97.5):+.3f}]",
            "d_MAE": round(float(_de.mean()), 5),
            "verdict": ("better" if _t < -2 else "worse" if _t > 2 else "no difference"),
        })
    paired = pl.DataFrame(_rows)
    mo.output.append(mo.md(
        f"### Paired vs `{BASELINE}` — negative `d_NLL` means the variant is better\n\n"
        "`t` and `p` test whether the **mean** paired difference is real. "
        "`spread_p2.5_97.5` is the scatter of the per-sample differences and is a "
        "different question — how *consistently* a variant wins. The scatter is "
        "~80x the mean shift here, so a variant can be reliably better on average "
        "while losing on plenty of individual samples. Read both; neither alone "
        "settles it."
    ))
    mo.output.append(paired)
    return


@app.cell
def _(HUES, INK, MUT, RUNS, SURF, found, mo, np, plt):
    # Permutation importance per run. This is the panel that answers `nuc_area`:
    # a bar at ~0 is a channel the model does not use.
    _tags = [t for t in RUNS if t in found]
    _all_ch = []
    for _t in _tags:
        for _c in found[_t]["perm_delta_nll"]:
            if _c not in _all_ch:
                _all_ch.append(_c)

    _fig, _ax = plt.subplots(figsize=(9.5, 4.2), dpi=170)
    _fig.patch.set_facecolor(SURF)
    _ax.set_facecolor(SURF)
    for _s in ("top", "right"):
        _ax.spines[_s].set_visible(False)
    _w = 0.8 / max(len(_tags), 1)
    _x = np.arange(len(_all_ch))
    for _i, _t in enumerate(_tags):
        _v = [found[_t]["perm_delta_nll"].get(_c, np.nan) for _c in _all_ch]
        _ax.bar(_x + _i * _w, _v, _w, color=HUES[_i % len(HUES)], label=_t)
    _ax.axhline(0, color=MUT, lw=1)
    _ax.set_xticks(_x + 0.4 - _w / 2)
    _ax.set_xticklabels(_all_ch, rotation=20, ha="right", color=INK, fontsize=8.5)
    _ax.set_ylabel("Δ NLL when shuffled", color=MUT, fontsize=9)
    _ax.tick_params(colors=MUT, labelsize=8)
    _ax.grid(axis="y", alpha=.18, lw=.6)
    _ax.legend(frameon=False, fontsize=8, labelcolor=MUT, ncol=len(_tags))
    _ax.set_title("Permutation importance — a bar at zero is a channel the model ignores",
                  color=INK, fontsize=10, loc="left")
    _fig.tight_layout()
    mo.output.append(_ax.get_figure())
    return


@app.cell
def _(HUES, INK, MUT, RUNS, SURF, found, mo, np, plt):
    _tags = [t for t in RUNS if t in found]
    _fig, _ax = plt.subplots(1, 3, figsize=(14, 3.8), dpi=170)
    _fig.patch.set_facecolor(SURF)
    for _a in _ax:
        _a.set_facecolor(SURF)
        for _s in ("top", "right"):
            _a.spines[_s].set_visible(False)
        _a.tick_params(colors=MUT, labelsize=8)
        _a.grid(alpha=.18, lw=.6)

    for _i, _t in enumerate(_tags):
        _m = found[_t]
        _c = HUES[_i % len(HUES)]
        _ax[0].plot(np.arange(1, len(_m["mae_per_step"]) + 1), _m["mae_per_step"],
                    color=_c, lw=2, label=_t)
        _ax[1].plot(np.arange(1, 11), _m["decile_mae"], color=_c, lw=2, marker="o", ms=4)
        _ax[2].bar(_i, _m["gain_spearman"], color=_c, width=.65)

    _ax[0].set_xlabel("forecast step", color=INK, fontsize=9)
    _ax[0].set_ylabel("MAE (CNR)", color=MUT, fontsize=9)
    _ax[0].set_title("Error vs horizon", color=INK, fontsize=10, loc="left")
    _ax[0].legend(frameon=False, fontsize=8, labelcolor=MUT)

    _ax[1].set_xlabel("expression decile (low → high)", color=INK, fontsize=9)
    _ax[1].set_ylabel("MAE (CNR)", color=MUT, fontsize=9)
    _ax[1].set_title("Error across the expression range", color=INK, fontsize=10, loc="left")

    _ax[2].axhline(0, color=MUT, lw=1)
    _ax[2].set_xticks(range(len(_tags)))
    _ax[2].set_xticklabels(_tags, rotation=20, ha="right", fontsize=8)
    _ax[2].set_ylabel("Spearman(expr, dose effect)", color=MUT, fontsize=9)
    _ax[2].set_title("Gain test — did it learn expression AS a gain?",
                     color=INK, fontsize=10, loc="left")
    _fig.tight_layout()
    mo.output.append(_fig)
    return


@app.cell
def _(found, mo):
    # A verdict has to be stated, not left for the reader to assemble — but naming
    # a "winner" on an unchecked gap is worse than naming none. Every claim below
    # is gated on the spread across runs actually exceeding what the runs disagree
    # by; where it does not, the honest answer is "no difference" and it says so.
    _nll = {t: m["test_nll"] for t, m in found.items()}
    _gain = {t: m["gain_spearman"] for t, m in found.items()}
    _best_nll = min(_nll, key=_nll.get) if _nll else None
    _gain_spread = (max(_gain.values()) - min(_gain.values())) if _gain else 0.0
    # The gain test only names a winner if the runs are further apart than a
    # Spearman on this many samples can wobble (~1/sqrt(n), n = test samples).
    _n = next(iter(found.values()))["n_test_samples"] if found else 1
    _gain_noise = 1.0 / max(_n, 1) ** 0.5
    _gain_line = (
        "**Gain test: a dead heat.** rho = "
        + ", ".join(f"`{t}` {v:.3f}" for t, v in _gain.items())
        + f" — spread {_gain_spread:.4f} against a sampling wobble of ~{_gain_noise:.4f}. "
        "Every encoding extracts the same gain relationship, so the encoding is "
        "**not** what limits it."
        if _gain_spread < 3 * _gain_noise else
        f"**Strongest gain relationship:** `{max(_gain, key=_gain.get)}` "
        f"(spread {_gain_spread:.4f} clears the ~{_gain_noise:.4f} noise floor)."
    )

    _area_imp = (found.get("d_area") or {}).get("perm_delta_nll", {}).get("nuc_area")
    # Judge area against the channels known to be near-useless, not against zero:
    # crowding is the natural "unused channel" yardstick this dataset provides.
    _crowd = [v for m in found.values()
              for k, v in m["perm_delta_nll"].items()
              if k in ("fov_density", "n_cells_200px")]
    _crowd_ref = max(_crowd) if _crowd else 0.0
    _area_line = (
        f"`nuc_area` permutation importance **{_area_imp:+.4f}** NLL, against "
        f"**{_crowd_ref:+.4f}** for the most-used crowding channel — the yardstick "
        f"for a channel the model does not need. "
        + ("Area clears it, so it carries real information."
           if _area_imp and _area_imp > 2 * _crowd_ref else
           "Area does not clear it; the channel is not earning its place.")
        if _area_imp is not None else
        "`d_area` did not load, so area is undecided."
    )

    _interaction_caveat = (
        "\n    **One number not to read at face value.** In `b_interaction`, "
        "permuting `optortk_expr` also permutes `u_t_x_expr` (they are linked by "
        "construction), and that channel is a **decoder input** — so the shuffle "
        "corrupts a dose-carrying future signal, not just expression. Its "
        "expression importance is therefore not comparable to the other runs'. "
        "Decompose it by permuting each channel alone before concluding anything "
        "from it.\n"
        if "b_interaction" in found else ""
    )

    mo.md(
        f"""
    ## Verdict

    - **Lowest NLL:** `{_best_nll}` — but read the paired table above, not this
      line: only a difference with |t| > 2 there is real.
    - {_gain_line}
    - {_area_line}
    {_interaction_caveat}
    **How to read a disagreement.** If one run wins on NLL but another wins the
    gain test, prefer the gain test for the serving model. Serving does not average
    over the population — it steers individual cells, and the per-cell gain is
    exactly the quantity it needs to get right. A lower average NLL bought without
    a gain relationship means the covariate was absorbed as something else.

    **The decile curve is the result to take seriously.** If error still climbs
    from the low to the high expression deciles in every variant, then none of the
    encodings fixed what the covariate exists to fix, and that is a more useful
    finding than whichever run edged the NLL.

    **Before adopting a winner:** confirm its `cov68`/`cov95` are near the nominal
    68/95. The controller's band kernel scores plans under the full predictive
    mixture, so a mis-calibrated model degrades control in a way that no point
    error reveals.
    """
    )
    return


if __name__ == "__main__":
    app.run()
