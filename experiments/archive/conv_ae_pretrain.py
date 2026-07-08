"""Convolutional autoencoder pretraining on per-cell image patches.

Goal: produce a pretrained backbone whose first 4 conv blocks match the
``ImageEncoder`` in ``experiments/lstm_seq2scal_mdn_image.py``. Those blocks
can then be loaded into the LSTM model to give the image branch a head start.

Training is unsupervised reconstruction (MSE) on patches pulled from the
extraction HDF5 produced by ``experiments/extract_patches.py``.

Outputs (under ``--results-dir``):
    encoder_backbone.pt   state_dict of the 4 shared conv blocks
                          (key shape matches ImageEncoder.conv[:8])
    encoder_full.pt       state_dict of backbone + extra conv blocks + projection
    bundle.pt             full AE state for resume / inspection
    notebook.html         this notebook rendered headless
"""

import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")

with app.setup:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import os
    import random
    import tempfile

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    from utils import get_device, results_write_path


    def make_backbone() -> nn.Sequential:
        """4 conv blocks identical to lstm_seq2scal_mdn_image.ImageEncoder.conv[:8].

        Stride-2 each layer → 80×80 input → 5×5 feature map (64 channels).
        """
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )


    class ConvAE(nn.Module):
        """Encoder = shared backbone + ``n_extra_conv`` stride-1 conv blocks +
        AdaptiveAvgPool + Linear → embed_dim. Decoder mirrors back to (1, H, W)
        with ConvTranspose2d, ending in a sigmoid for [0, 1] outputs.

        The backbone state_dict matches ``ImageEncoder.conv`` keys 0..7 exactly,
        so it transfers cleanly to the LSTM model.
        """

        def __init__(self, embed_dim: int = 16, n_extra_conv: int = 2,
                     bottleneck_spatial: int = 5):
            super().__init__()
            self.embed_dim = embed_dim
            self.n_extra_conv = n_extra_conv
            self.bottleneck_spatial = bottleneck_spatial

            self.backbone = make_backbone()
            extra = []
            for _ in range(n_extra_conv):
                extra += [
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                ]
            self.extra = nn.Sequential(*extra)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.proj = nn.Linear(64, embed_dim)

            # decoder: lift back to 64 × bottleneck_spatial × bottleneck_spatial
            self.unproj = nn.Linear(embed_dim, 64 * bottleneck_spatial * bottleneck_spatial)
            extra_dec = []
            for _ in range(n_extra_conv):
                extra_dec += [
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                ]
            self.extra_dec = nn.Sequential(*extra_dec)
            # mirror the 4 stride-2 conv blocks with ConvTranspose2d
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid(),
            )

        def encode(self, x):
            h = self.backbone(x)
            h = self.extra(h)
            z_pre = self.pool(h).flatten(1)
            return self.proj(z_pre)

        def decode(self, z):
            B = z.shape[0]
            S = self.bottleneck_spatial
            h = self.unproj(z).view(B, 64, S, S)
            h = self.extra_dec(h)
            return self.upsample(h)

        def forward(self, x):
            return self.decode(self.encode(x))


    class PatchDataset(Dataset):
        """Reads a single channel out of the patch HDF5.

        For training-time IO simplicity, we open a fresh h5 handle inside the
        worker and slice on demand. With num_workers=0 (default below) this is
        single-process and fine on macOS / SLURM.
        """

        def __init__(self, h5_path: str | Path, indices, image_channel: int = 1):
            import h5py
            self.h5_path = str(h5_path)
            self.indices = np.asarray(indices, dtype=np.int64)
            self.image_channel = int(image_channel)
            self._f = h5py.File(self.h5_path, "r")
            self.patches = self._f["patches"]

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, i):
            idx = int(self.indices[i])
            patch = self.patches[idx, self.image_channel].astype(np.float32)
            return torch.from_numpy(patch).unsqueeze(0)  # (1, H, W)

        def close(self):
            self._f.close()


@app.cell
def _():
    import marimo as mo
    from utils import parse_bool
    args = mo.cli_args()
    EXPERIMENT_NAME = args.get("name", "conv_ae_pretrain")
    PATCHES_H5 = args.get("patches-h5", None)
    EMBED_DIM = int(args.get("embed_dim", 16))
    N_EXTRA_CONV = int(args.get("n_extra_conv", 2))
    IMAGE_CHANNEL = int(args.get("image_channel", 1))
    EPOCHS = int(args.get("epochs", 30))
    BATCH_SIZE = int(args.get("batch_size", 256))
    LR = float(args.get("lr", 1e-3))
    VAL_FRAC = float(args.get("val_frac", 0.1))
    SEED = int(args.get("seed", 42))
    DRY_RUN = parse_bool(args.get("dry_run", False))
    RESULTS_DIR = args.get("results-dir", f"{results_write_path()}/{EXPERIMENT_NAME}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if PATCHES_H5 is None:
        raise ValueError("--patches-h5 is required (path to the extraction HDF5)")

    mo.md(f"""
    # ConvAE pretrain — `{EXPERIMENT_NAME}`

    | param | value |
    |-------|-------|
    | patches-h5 | `{PATCHES_H5}` |
    | embed_dim | {EMBED_DIM} |
    | n_extra_conv | {N_EXTRA_CONV} |
    | image_channel | {IMAGE_CHANNEL} |
    | epochs | {EPOCHS} |
    | batch_size | {BATCH_SIZE} |
    | lr | {LR} |
    | val_frac | {VAL_FRAC} |
    | dry_run | {DRY_RUN} |
    | results | `{RESULTS_DIR}` |
    """)
    return (
        BATCH_SIZE,
        DRY_RUN,
        EMBED_DIM,
        EPOCHS,
        IMAGE_CHANNEL,
        LR,
        N_EXTRA_CONV,
        PATCHES_H5,
        RESULTS_DIR,
        SEED,
        VAL_FRAC,
        mo,
    )


@app.cell
def _(IMAGE_CHANNEL, PATCHES_H5, SEED, VAL_FRAC, mo):
    import h5py as _h5py
    with _h5py.File(PATCHES_H5, "r") as _f:
        n_patches = int(_f["patches"].shape[0])
        patch_shape = tuple(_f["patches"].shape[1:])
        h5_half = int(_f.attrs["half"])
        h5_channels = list(_f.attrs["channels"])

    rng = np.random.default_rng(SEED)
    _all_idx = rng.permutation(n_patches)
    _n_val = int(round(n_patches * VAL_FRAC))
    val_indices = _all_idx[:_n_val]
    train_indices = _all_idx[_n_val:]

    mo.md(f"""
    **HDF5:** {n_patches:,} patches · shape per patch {patch_shape} (C, H, W) ·
    half={h5_half} · channels={h5_channels}

    Using channel **{IMAGE_CHANNEL}** for AE input.
    Train / val: {len(train_indices):,} / {len(val_indices):,}.
    """)
    return patch_shape, train_indices, val_indices


@app.cell
def _(
    BATCH_SIZE,
    DRY_RUN,
    EMBED_DIM,
    EPOCHS,
    IMAGE_CHANNEL,
    LR,
    N_EXTRA_CONV,
    PATCHES_H5,
    RESULTS_DIR,
    SEED,
    mo,
    patch_shape,
    train_indices,
    val_indices,
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if DRY_RUN:
        train_indices_used = train_indices[: min(2000, len(train_indices))]
        val_indices_used = val_indices[: min(500, len(val_indices))]
        epochs_used = min(EPOCHS, 2)
    else:
        train_indices_used = train_indices
        val_indices_used = val_indices
        epochs_used = EPOCHS

    train_ds = PatchDataset(PATCHES_H5, train_indices_used, image_channel=IMAGE_CHANNEL)
    val_ds = PatchDataset(PATCHES_H5, val_indices_used, image_channel=IMAGE_CHANNEL)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # bottleneck spatial size: stride-2 four times, so input // 16
    _bottleneck = max(1, patch_shape[1] // 16)
    device = get_device()
    print(f"Device: {device}  bottleneck_spatial: {_bottleneck}")

    model = ConvAE(
        embed_dim=EMBED_DIM,
        n_extra_conv=N_EXTRA_CONV,
        bottleneck_spatial=_bottleneck,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ConvAE params: {n_params:,}")

    history = {"train_loss": [], "val_loss": []}
    ckpt_fd, ckpt = tempfile.mkstemp(suffix=".pt")
    os.close(ckpt_fd)
    torch.save(model.state_dict(), ckpt)
    best_val = float("inf")

    for ep in range(epochs_used):
        model.train()
        _tr_losses = []
        for _x in train_loader:
            _x = _x.to(device)
            _xh = model(_x)
            _loss = loss_fn(_xh, _x)
            opt.zero_grad()
            _loss.backward()
            opt.step()
            _tr_losses.append(float(_loss.item()))
        train_loss = float(np.mean(_tr_losses))

        model.eval()
        _va_losses = []
        with torch.no_grad():
            for _x in val_loader:
                _x = _x.to(device)
                _xh = model(_x)
                _va_losses.append(float(loss_fn(_xh, _x).item()))
        val_loss = float(np.mean(_va_losses)) if _va_losses else float("nan")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {ep:3d} | T:{train_loss:.6f}  V:{val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt)

    # restore best
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    os.remove(ckpt)

    # save artifacts
    _backbone_path = Path(RESULTS_DIR) / "encoder_backbone.pt"
    _full_path = Path(RESULTS_DIR) / "encoder_full.pt"
    _bundle_path = Path(RESULTS_DIR) / "bundle.pt"
    torch.save(model.backbone.state_dict(), _backbone_path)
    torch.save(
        {
            "backbone": model.backbone.state_dict(),
            "extra": model.extra.state_dict(),
            "pool": {},  # AdaptiveAvgPool has no params
            "proj": model.proj.state_dict(),
        },
        _full_path,
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "embed_dim": EMBED_DIM,
                "n_extra_conv": N_EXTRA_CONV,
                "bottleneck_spatial": _bottleneck,
                "image_channel": IMAGE_CHANNEL,
                "patch_shape": list(patch_shape),
            },
            "history": history,
            "best_val_loss": best_val,
        },
        _bundle_path,
    )

    train_ds.close()
    val_ds.close()

    mo.md(f"""
    **Done.** best val MSE = {best_val:.6f} after {len(history['train_loss'])} epochs.

    Saved:
    - `{_backbone_path}` — load with `image_encoder.conv[:8].load_state_dict(...)`
    - `{_full_path}` — backbone + extra + proj (full encoder)
    - `{_bundle_path}` — full AE bundle
    """)
    return device, model


@app.cell
def _(IMAGE_CHANNEL, PATCHES_H5, RESULTS_DIR, device, mo, model, val_indices):
    """Show a handful of reconstructions for sanity check."""
    import matplotlib.pyplot as _plt
    import h5py as _h5py_recon

    n_show = 8
    # h5py fancy-indexing wants strictly increasing indices
    _idx = np.sort(val_indices[:n_show])
    with _h5py_recon.File(PATCHES_H5, "r") as _f:
        _patches = _f["patches"][_idx, IMAGE_CHANNEL].astype(np.float32)

    _x = torch.from_numpy(_patches).unsqueeze(1).to(device)
    model.eval()
    with torch.no_grad():
        _xh = model(_x).cpu().numpy()[:, 0]

    _fig, _axes = _plt.subplots(2, n_show, figsize=(n_show * 1.3, 2.6))
    for _i in range(n_show):
        _axes[0, _i].imshow(_patches[_i], cmap="gray")
        _axes[0, _i].axis("off")
        _axes[1, _i].imshow(_xh[_i], cmap="gray")
        _axes[1, _i].axis("off")
    _axes[0, 0].set_title("input", loc="left", fontsize=8)
    _axes[1, 0].set_title("recon", loc="left", fontsize=8)
    _fig.tight_layout()

    _fig_path = Path(RESULTS_DIR) / "reconstructions.png"
    _fig.savefig(_fig_path, dpi=120, bbox_inches="tight")
    mo.vstack([mo.md(f"Saved `{_fig_path}`."), mo.ui.matplotlib(_axes[0, 0])])
    return


if __name__ == "__main__":
    app.run()
