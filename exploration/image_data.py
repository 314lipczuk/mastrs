import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from hastyplot import qplot
    import pandas as pd
    import numpy as np
    from PIL import Image
    import subprocess
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from pathlib import Path
    return Image, Path, cv2, mo, mpatches, np, pd, plt, qplot, subprocess


@app.cell
def _(mo):
    mo.md(r"""
    # Purpouse
    This notebook serves as an:
    - exploration of the image data that I have available for my experiments
    - maybe a workspace to test different pre-trained embeddings of this data into my model.
    """)
    return


@app.cell
def _(subprocess):
    subprocess.run(['nvidia-smi']).stdout
    return


@app.cell
def _():
    exp_paths = [
        "/Volumes/imaging.data/PertzLab/optoRTK_CedricZ/experimental_data/2025-11-03_3-2-1minIntervals/",  # 660 cells, 180 min
        "/Volumes/imaging.data/PertzLab/optoRTK_CedricZ/experimental_data/2025-10-12_DoseResponse",        # 800 cells, 40 min
        "/Volumes/imaging.data/PertzLab/optoRTK_CedricZ/experimental_data/2025-11-02_Sustained_1min",      # 1350 cells, 120 min
        "/Volumes/imaging.data/PertzLab/optoRTK_CedricZ/experimental_data/2025-09-04_RampReverse",
    ]
    return (exp_paths,)


@app.cell
def _(Path, exp_paths):
    # inspect raw/ dir of each experiment to learn channel/file naming
    for _ep in exp_paths:
        _raw = Path(_ep.replace("/Volumes/", "/mnt/")) / "raw"
        if not _raw.exists():
            print(f"MISSING: {_raw}")
            continue
        _files = sorted(_raw.iterdir())
        print(f"\n=== {Path(_ep).name}")
        print(f"  raw/: {len(_files)} files")
        for _f in _files[:6]:
            print(f"    {_f.name}")
    return


@app.cell
def _(Image, Path, exp_paths, np):
    # load 1st tiff from each exp; inspect shape, dtype, n_pages
    def _inspect(path):
        with Image.open(path) as _img:
            n_pages = getattr(_img, "n_frames", 1)
            pages = []
            for _i in range(n_pages):
                _img.seek(_i)
                pages.append(np.array(_img))
        return n_pages, [(p.shape, str(p.dtype), float(p.min()), float(p.max())) for p in pages]

    for _ep in exp_paths:
        _raw = Path(_ep.replace("/Volumes/", "/mnt/")) / "raw"
        _f = _raw / "000_00000.tiff"
        if not _f.exists():
            print(f"MISSING: {_f}")
            continue
        _n, _info = _inspect(_f)
        print(f"\n=== {Path(_ep).name}")
        print(f"  file: {_f.name}  pages={_n}")
        for _i, (_s, _d, _lo, _hi) in enumerate(_info):
            print(f"    page {_i}: shape={_s} dtype={_d} range=[{_lo:.0f}, {_hi:.0f}]")
    return


@app.cell
def _(pd):
    _PATH = '/mnt/imaging.data/PertzLab/optoRTK_CedricZ/experimental_data/2025-11-03_3-2-1minIntervals/data_filtered.parquet'
    df = pd.read_parquet(_PATH)
    df
    return (df,)


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(Image):
    im = Image.open(
        '/mnt/imaging.data/PertzLab/optoRTK_CedricZ/experimental_data/2025-11-03_3-2-1minIntervals/raw/021_00111.tiff'
    )
    return (im,)


@app.cell
def _(im):
    im
    return


@app.cell
def _(im, np):
    im_ar = np.array(im)
    im_ar.dtype, im_ar.min(), im_ar.max(), np.percentile(im_ar, [1, 50, 99, 99.9])
    return (im_ar,)


@app.cell
def _(Image, im_ar, np):
    p_low, p_high = np.percentile(im_ar, (1, 99.5))
    stretched = np.clip(im_ar, p_low, p_high)
    stretched = ((stretched - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    Image.fromarray(stretched)
    return (stretched,)


@app.cell
def _(Image, np, stretched):
    gamma = 0.7   # < 1 brightens midtones; try 0.4–0.7
    corrected = np.power(stretched / 255.0, gamma)
    corrected = (corrected * 255).astype(np.uint8)
    Image.fromarray(corrected)
    return


@app.cell
def _(Image, cv2, stretched):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe.apply(stretched)   # input must be uint8
    Image.fromarray(result)
    return (result,)


@app.cell
def _(im_ar, plt, result, stretched):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(im_ar, cmap='gray');     axes[0].set_title('Raw')
    axes[1].imshow(stretched, cmap='gray'); axes[1].set_title('Percentile stretch')
    axes[2].imshow(result, cmap='gray');    axes[2].set_title('+ CLAHE')
    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Bit hard to tell between simple percentile stretch and CLAHE. Second one seems a bit more grainy in the background, but the features seem to be preserved on the same level;
    """)
    return


@app.cell
def _(Image, Path, np):
    HALF = 32
    RAW_DIR = Path("/mnt/imaging.data/PertzLab/optoRTK_CedricZ/experimental_data/2025-11-03_3-2-1minIntervals/raw")


    def normalize(arr: np.ndarray, p_low=1, p_high=99) -> np.ndarray:
        lo, hi = np.percentile(arr, (p_low, p_high))
        return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1)


    def load_frame(fov: int, frame: int, channel: int = 1) -> np.ndarray:
        """Load one channel page from a multi-page tiff and percentile-normalize."""
        path = RAW_DIR / f"{fov:03d}_{frame:05d}.tiff"
        with Image.open(path) as img:
            img.seek(channel)
            arr = np.array(img)
        return normalize(arr)


    def crop(arr, x, y, half=HALF):
        # x → row, y → col (matches centroid convention used elsewhere)
        row, col = int(round(x)), int(round(y))
        r0, r1 = row - half, row + half
        c0, c1 = col - half, col + half
        if r0 < 0 or c0 < 0 or r1 > arr.shape[0] or c1 > arr.shape[1]:
            return None
        return arr[r0:r1, c0:c1]
    return HALF, crop, load_frame


@app.cell
def _(Path, exp_paths, pd):
    EXP_DIR_MAP = {}
    for _ep_path in exp_paths:
        _exp = Path(_ep_path.replace("/Volumes/", "/mnt/"))
        _pq = _exp / "data_filtered.parquet"
        if not _pq.exists():
            continue
        _patterns = pd.read_parquet(_pq, columns=["ramp_pattern_name"])["ramp_pattern_name"].unique()
        for _p in _patterns:
            EXP_DIR_MAP[_p] = _exp
    EXP_DIR_MAP
    return (EXP_DIR_MAP,)


@app.cell
def _(df, mo):
    _fovs = sorted(df["fov"].unique().tolist())
    _frames = sorted(df["frame"].unique().tolist())
    _default_fov = "21" if 21 in _fovs else str(_fovs[0])
    _default_frame = 111 if 111 in _frames else int(min(_frames))
    fov_pick = mo.ui.dropdown(
        options=[str(v) for v in _fovs], value=_default_fov, label="FOV"
    )
    frame_pick = mo.ui.slider(
        start=int(min(_frames)), stop=int(max(_frames)),
        value=_default_frame, step=1, label="Frame", show_value=True,
    )
    channel_pick = mo.ui.radio(
        options=["C0 (nuc)", "C1 (ERK)"], value="C1 (ERK)", label="Channel",
    )
    mo.hstack([fov_pick, frame_pick, channel_pick])
    return channel_pick, fov_pick, frame_pick


@app.cell
def _(HALF, channel_pick, crop, df, fov_pick, frame_pick, load_frame, mo, mpatches, np, plt):
    _fov = int(fov_pick.value)
    _frame = int(frame_pick.value)
    _ch = 0 if channel_pick.value.startswith("C0") else 1

    arr = load_frame(_fov, _frame, channel=_ch)
    subset = df[(df["fov"] == _fov) & (df["frame"] == _frame)]

    records = []
    for row in subset.itertuples():
        patch = crop(arr, row.x, row.y)
        if patch is not None:
            records.append({"uid": row.uid, "x": row.x, "y": row.y, "patch": patch})

    print(f"{len(records)} / {len(subset)} cells cropped")

    # full image with crop boxes overlaid
    _fig1, _ax1 = plt.subplots()
    for r in records:
        cx, cy = int(round(r["x"])), int(round(r["y"]))
        _ax1.add_patch(mpatches.Rectangle(
            (cy - HALF, cx - HALF), 2 * HALF, 2 * HALF,
            linewidth=0.8, edgecolor="lime", facecolor="none",
        ))
    _ax1.imshow(arr, cmap="gray")
    _ax1.set_title(f"FOV {_fov}  frame {_frame}  ch{_ch}  —  {len(records)} cells")
    _ax1.axis("off")
    _fig1.tight_layout()
    _full = mo.ui.matplotlib(_ax1)

    # patch grid
    if len(records) == 0:
        _grid = mo.md("No cells in view.")
    else:
        _ncols = min(len(records), 10)
        _nrows = (len(records) + _ncols - 1) // _ncols
        _fig2, _axes = plt.subplots(_nrows, _ncols, figsize=(_ncols * 1.2, _nrows * 1.2))
        _axes_flat = np.array(_axes).flatten()
        for _ax, _r in zip(_axes_flat, records):
            _ax.imshow(_r["patch"], cmap="gray")
            _ax.set_title(_r["uid"][:8], fontsize=6)
            _ax.axis("off")
        for _ax in _axes_flat[len(records):]:
            _ax.set_visible(False)
        _fig2.suptitle(f"Patches — FOV {_fov} frame {_frame} ch{_ch}", y=1.01)
        _fig2.tight_layout()
        _grid = mo.ui.matplotlib(_axes_flat[0])

    mo.vstack([_full, _grid])
    return arr, subset


@app.cell
def _(arr, subset):
    print(f"Image shape (H, W): {arr.shape}")
    print(f"x range: {subset['x'].min():.1f} → {subset['x'].max():.1f}")
    print(f"y range: {subset['y'].min():.1f} → {subset['y'].max():.1f}")
    return


@app.cell
def _(arr, plt, subset):
    _fig, _ax = plt.subplots()
    _ax.imshow(arr, cmap='gray', vmin=0)
    _ax.scatter(subset['y'], subset['x'], s=10, c='lime', linewidths=0)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ok cool cropping is sorta working.

    Question: Which arch makes the most sense here?
    - dino emb
    - simple VAE
    - VAE-GAN?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## HDF5 patch store

    Single sbatch job (`experiments/extract_patches.py` + `notebooks/experiment/extract_patches.py`)
    extracts every (uid, frame) crop from all 4 experiments into one HDF5:

    - `patches`: (N, C, H, W) float16
    - `meta/uid, frame, fov, particle, x, y, ramp_pattern_name`: (N,)

    Look up a patch from any `dataset.parquet` row via `(uid, frame)` → `patch_idx`
    using `attach_patch_idx(df, h5)` from the module.
    """)
    return


if __name__ == "__main__":
    app.run()
