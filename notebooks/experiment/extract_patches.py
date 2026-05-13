"""Pre-extract per-cell image patches into a single HDF5 file.

Layout produced by ``extract_patches``::

    patches.h5
      patches             (N, C, H, W) float16   normalized cell crops
      meta/uid            (N,) S
      meta/frame          (N,) int32
      meta/fov            (N,) int32
      meta/particle       (N,) int32
      meta/x              (N,) float32           original centroid (row-axis)
      meta/y              (N,) float32           original centroid (col-axis)
      meta/ramp_pattern_name (N,) S
      attrs: half, channels, p_low, p_high

Patches are addressed by integer index into the first axis of ``patches``.
The (uid, frame) pair uniquely identifies a row in the source dataframe and
maps to exactly one patch — use ``attach_patch_idx`` to add that index column.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np
import pandas as pd
from PIL import Image


def _normalize(arr: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    lo, hi = np.percentile(arr, (p_low, p_high))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1)


def _crop(arr: np.ndarray, x: float, y: float, half: int):
    # NB: image_data.py treats x as row, y as col — keep the same convention here
    row, col = int(round(x)), int(round(y))
    r0, r1, c0, c1 = row - half, row + half, col - half, col + half
    if r0 < 0 or c0 < 0 or r1 > arr.shape[0] or c1 > arr.shape[1]:
        return None
    return arr[r0:r1, c0:c1]


def extract_patches(
    df: pd.DataFrame,
    exp_dir_map: Mapping[str, str | Path],
    out_h5: str | Path,
    half: int = 32,
    channels: Iterable[int] = (0, 1),
    p_low: float = 1.0,
    p_high: float = 99.0,
    compression: str | None = None,
    progress: bool = True,
) -> Path:
    """Crop and store per-(uid, frame) image patches into a single HDF5 file.

    Parameters
    ----------
    df
        Long-format dataframe with at least ``uid, frame, fov, x, y,
        ramp_pattern_name, particle``.
    exp_dir_map
        Maps each ``ramp_pattern_name`` value to its experiment directory (the
        one that contains ``raw/{fov:03d}_{frame:05d}.tiff``).
    out_h5
        Output path. Parent dirs are created.
    half
        Half-edge of crop in pixels; output side is ``2 * half``.
    channels
        Tiff page indices to extract. The HDF5 ``patches`` axis 1 is in this
        order.
    p_low, p_high
        Percentile range for per-image, per-channel min/max normalization.
    """
    out = Path(out_h5)
    out.parent.mkdir(parents=True, exist_ok=True)

    side = 2 * half
    chans = tuple(channels)
    n_ch = len(chans)

    cols = ["uid", "frame", "fov", "x", "y", "ramp_pattern_name", "particle"]
    rows = df[cols].copy()
    rows["frame"] = rows["frame"].astype(np.int32)
    rows["fov"] = rows["fov"].astype(np.int32)
    rows["particle"] = rows["particle"].astype(np.int32)
    # one patch per (uid, frame) — drop accidental duplicates
    rows = rows.drop_duplicates(subset=["uid", "frame"])

    # group by (ramp, fov, frame): one tiff per group
    groups = rows.groupby(["ramp_pattern_name", "fov", "frame"], sort=False)
    n_groups = len(groups)

    # buffers
    meta_buf = {k: [] for k in ["uid", "frame", "fov", "particle", "x", "y", "ramp_pattern_name"]}

    # upper bound on patches = number of unique (uid, frame) rows; we trim later
    n_max = len(rows)

    with h5py.File(out, "w") as f:
        ds_kwargs = dict(
            shape=(n_max, n_ch, side, side),
            maxshape=(None, n_ch, side, side),
            dtype="float16",
            chunks=(64, n_ch, side, side),
        )
        if compression:
            ds_kwargs["compression"] = compression
            if compression == "gzip":
                ds_kwargs["compression_opts"] = 4
        ds = f.create_dataset("patches", **ds_kwargs)
        f.attrs["half"] = half
        f.attrs["channels"] = list(chans)
        f.attrs["p_low"] = p_low
        f.attrs["p_high"] = p_high

        n_written = 0
        n_skipped_missing = 0
        n_skipped_border = 0

        for gi, ((ramp, fov, frame), grp) in enumerate(groups):
            exp_dir = exp_dir_map.get(ramp)
            if exp_dir is None:
                n_skipped_missing += len(grp)
                continue
            tiff = Path(exp_dir) / "raw" / f"{int(fov):03d}_{int(frame):05d}.tiff"
            if not tiff.exists():
                n_skipped_missing += len(grp)
                continue

            # load only the requested channels once per tiff
            arrs = []
            with Image.open(tiff) as im:
                for ch in chans:
                    im.seek(ch)
                    arrs.append(_normalize(np.array(im), p_low, p_high))

            # crop every cell in this frame
            patches = []
            keep_rows = []
            for r in grp.itertuples():
                stack = []
                ok = True
                for a in arrs:
                    p = _crop(a, r.x, r.y, half)
                    if p is None:
                        ok = False
                        break
                    stack.append(p)
                if not ok:
                    n_skipped_border += 1
                    continue
                patches.append(np.stack(stack, axis=0).astype(np.float16))
                keep_rows.append(r)

            if not patches:
                continue

            batch = np.stack(patches, axis=0)
            new_n = n_written + batch.shape[0]
            ds[n_written:new_n] = batch

            for r in keep_rows:
                meta_buf["uid"].append(r.uid)
                meta_buf["frame"].append(int(r.frame))
                meta_buf["fov"].append(int(r.fov))
                meta_buf["particle"].append(int(r.particle))
                meta_buf["x"].append(float(r.x))
                meta_buf["y"].append(float(r.y))
                meta_buf["ramp_pattern_name"].append(r.ramp_pattern_name)

            n_written = new_n

            if progress and (gi % 50 == 0 or gi == n_groups - 1):
                print(
                    f"[{gi + 1}/{n_groups}] written={n_written} "
                    f"missing={n_skipped_missing} border={n_skipped_border}",
                    flush=True,
                )

        # trim to actual count
        if n_written < n_max:
            ds.resize((n_written, n_ch, side, side))

        # write meta as parallel 1D datasets
        f.create_dataset("meta/uid", data=np.array(meta_buf["uid"], dtype="S64"))
        f.create_dataset("meta/frame", data=np.array(meta_buf["frame"], dtype=np.int32))
        f.create_dataset("meta/fov", data=np.array(meta_buf["fov"], dtype=np.int32))
        f.create_dataset("meta/particle", data=np.array(meta_buf["particle"], dtype=np.int32))
        f.create_dataset("meta/x", data=np.array(meta_buf["x"], dtype=np.float32))
        f.create_dataset("meta/y", data=np.array(meta_buf["y"], dtype=np.float32))
        f.create_dataset(
            "meta/ramp_pattern_name",
            data=np.array(meta_buf["ramp_pattern_name"], dtype="S64"),
        )

    return out


def load_meta(h5_path: str | Path) -> pd.DataFrame:
    """Read the per-patch metadata table from an extraction HDF5."""
    with h5py.File(h5_path, "r") as f:
        return pd.DataFrame(
            {
                "uid": [s.decode() for s in f["meta/uid"][:]],
                "frame": f["meta/frame"][:],
                "fov": f["meta/fov"][:],
                "particle": f["meta/particle"][:],
                "x": f["meta/x"][:],
                "y": f["meta/y"][:],
                "ramp_pattern_name": [
                    s.decode() for s in f["meta/ramp_pattern_name"][:]
                ],
                "patch_idx": np.arange(f["meta/uid"].shape[0], dtype=np.int64),
            }
        )


def attach_patch_idx(df: pd.DataFrame, h5_path: str | Path) -> pd.DataFrame:
    """Join ``patch_idx`` onto ``df`` by ``(uid, frame)``. NaN where no patch."""
    meta = load_meta(h5_path)[["uid", "frame", "patch_idx"]]
    return df.merge(meta, on=["uid", "frame"], how="left")


class PatchStore:
    """Random-access reader over the ``patches`` dataset.

    Use as a context manager or call ``close()`` when done.
    """

    def __init__(self, h5_path: str | Path):
        self.path = Path(h5_path)
        self._f = h5py.File(self.path, "r")
        self.patches = self._f["patches"]
        self.half = int(self._f.attrs["half"])
        self.channels = tuple(int(c) for c in self._f.attrs["channels"])

    def __len__(self) -> int:
        return self.patches.shape[0]

    def __getitem__(self, idx):
        return self.patches[idx]

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
