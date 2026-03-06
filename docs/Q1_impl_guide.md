# Q1 Implementation Guide: Conditional β-VAE for ERK Trajectory Encoding

## Overview

This guide walks through building a conditional β-VAE that encodes single-cell ERK trajectories into an interpretable latent space, conditioned on the light stimulus. The goal: the latent space captures **only cell-intrinsic heterogeneity**, because the stimulus is provided to the decoder through a bypass pathway and doesn't need to pass through the bottleneck.

```
                        ┌─────────────┐
  ERK trace ──────┐     │             │     ┌──── z ────┐
                  ├───► │   Encoder   ├──►  │  μ, logσ² │
  Stimulus feat. ─┘     │  (1D conv)  │     └─────┬─────┘
                        └─────────────┘           │
                                            reparametrize
                                                  │
                        ┌─────────────┐           │
  Reconstructed ◄────── │   Decoder   │ ◄── z ───┘
  ERK trace             │ (1D deconv) │ ◄── Stimulus feat. (BYPASS)
                        └─────────────┘
```

The encoder sees both ERK and stimulus (so it can factor out the expected response). The decoder receives z **and** the stimulus through a separate path. Because the decoder already knows what stimulus was given, z only needs to encode the cell-specific deviation from the expected response.

---

## 1. Data Preprocessing

### 1.1 Loading and structuring

Your parquet files contain per-cell, per-timepoint rows with ERK fluorescence and light stimulation columns. Reshape into two parallel matrices:

```python
import numpy as np
import pandas as pd

def load_experiment(parquet_path):
    """
    Returns:
        erk: np.array of shape (n_cells, n_timepoints)
        stim: np.array of shape (n_cells, n_timepoints)
        metadata: DataFrame with cell_id, condition, replicate, etc.
    """
    df = pd.read_parquet(parquet_path)

    # Pivot so each row is a cell, each column is a timepoint
    erk = df.pivot(index="cell_id", columns="timepoint", values="erk_fluorescence").values
    stim = df.pivot(index="cell_id", columns="timepoint", values="light_stimulus").values

    metadata = df.groupby("cell_id").first()[["condition", "replicate"]].reset_index()

    return erk.astype(np.float32), stim.astype(np.float32), metadata
```

Adapt column names to match your actual parquet schema.

### 1.2 Normalisation

This is a critical design decision. Options:

**Per-cell z-score** normalises each cell to zero mean and unit variance. This removes amplitude information entirely and forces the model to focus on temporal shape. Use this if you believe dynamics (oscillation pattern, adaptation curve) matter more than magnitude.

**Global z-score** computes mean and std across all cells and timepoints. Preserves relative amplitude differences between cells. Use this if amplitude variation is part of the heterogeneity you want to capture.

**Recommendation:** Start with global z-score. Amplitude is likely biologically meaningful (different expression levels of the reporter, different receptor densities). You can always ablate with per-cell normalisation later to see what changes in the latent space.

```python
def normalise_global(erk, fit_data=None):
    """Global z-score. Optionally fit on training set only."""
    if fit_data is None:
        fit_data = erk
    mu = fit_data.mean()
    sigma = fit_data.std()
    return (erk - mu) / (sigma + 1e-8), mu, sigma
```

### 1.3 Stimulus feature engineering

The raw stimulus is a sparse binary signal (mostly zeros, occasional short pulses). This is unfriendly to convolutions. Transform it into a richer per-timestep representation:

```python
def engineer_stimulus_features(stim_raw, dt=1.0):
    """
    Transform sparse binary stimulus into informative per-timestep features.

    Input: stim_raw, shape (n_cells, T), binary pulse signal
    Output: stim_feat, shape (n_cells, T, n_features)

    Features (per timestep):
      0: raw stimulus value (0 or 1)
      1: time since last pulse onset (minutes), clipped
      2: time since last pulse offset (minutes), clipped
      3: cumulative dose so far (integral of stimulus up to t)
      4: local pulse frequency (pulses in trailing window)
    """
    n_cells, T = stim_raw.shape
    n_features = 5
    stim_feat = np.zeros((n_cells, T, n_features), dtype=np.float32)

    for i in range(n_cells):
        s = stim_raw[i]
        stim_feat[i, :, 0] = s  # raw

        # Time since last onset / offset
        last_onset = -100.0
        last_offset = -100.0
        for t in range(T):
            if t > 0 and s[t] == 1 and s[t-1] == 0:
                last_onset = t * dt
            if t > 0 and s[t] == 0 and s[t-1] == 1:
                last_offset = t * dt
            stim_feat[i, t, 1] = np.clip((t * dt - last_onset), 0, 60) / 60.0
            stim_feat[i, t, 2] = np.clip((t * dt - last_offset), 0, 60) / 60.0

        # Cumulative dose
        stim_feat[i, :, 3] = np.cumsum(s) * dt
        stim_feat[i, :, 3] /= (stim_feat[i, :, 3].max() + 1e-8)  # normalise to [0, 1]

        # Local frequency: count onsets in trailing 10-min window
        window = int(10.0 / dt)
        onsets = np.diff(s, prepend=0) == 1
        for t in range(T):
            start = max(0, t - window)
            stim_feat[i, t, 4] = onsets[start:t+1].sum()
        stim_feat[i, :, 4] /= (stim_feat[i, :, 4].max() + 1e-8)

    return stim_feat
```

This is deliberately simple and interpretable. If some features turn out to be uninformative, you can drop them. If you later want a learned stimulus encoder, you can replace this with a small 1D conv block and compare.

### 1.4 Handling variable-length trajectories

If experiments have different durations (2–8h at 1 frame/min → 120–480 timepoints), you have two options:

**Option A: Pad and mask.** Pad shorter trajectories to the max length, create a boolean mask, and ignore padded positions in the loss. More principled but adds implementation complexity.

**Option B: Crop to a fixed window.** Choose a duration that covers most experiments (e.g., 120 timepoints = 2h) and crop or truncate. Simpler, and fine for a first pass. You lose late dynamics in long experiments.

Start with Option B at a window that captures the interesting dynamics. Move to Option A if you need the full trajectory.

### 1.5 Dataset and DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader

class ERKDataset(Dataset):
    def __init__(self, erk, stim_features):
        """
        erk: (n_cells, T) normalised ERK traces
        stim_features: (n_cells, T, n_stim_feat) engineered stimulus features
        """
        # Encoder input: ERK as channel 0, stim features as channels 1..n
        # Shape: (n_cells, 1 + n_stim_feat, T)
        erk_expanded = erk[:, np.newaxis, :]  # (N, 1, T)
        stim_transposed = stim_features.transpose(0, 2, 1)  # (N, n_stim_feat, T)
        self.encoder_input = torch.tensor(
            np.concatenate([erk_expanded, stim_transposed], axis=1)
        )
        # Decoder conditioning: just the stim features (bypass)
        self.stim_cond = torch.tensor(stim_transposed.copy())
        # Reconstruction target: ERK only
        self.target = torch.tensor(erk_expanded.copy())

    def __len__(self):
        return self.encoder_input.shape[0]

    def __getitem__(self, idx):
        return self.encoder_input[idx], self.stim_cond[idx], self.target[idx]
```

Use an 80/20 train/validation split, **stratified by experimental condition** so each split contains cells from all stimulation protocols.

---

## 2. Model Architecture

### 2.1 Encoder

Stack of 1D dilated causal convolutions that progressively downsample the temporal dimension, ending in two linear heads for μ and log σ².

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """1D convolution with left-padding for causal (or non-causal) operation."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, causal=True):
        super().__init__()
        self.causal = causal
        self.pad = (kernel_size - 1) * dilation if causal else ((kernel_size - 1) * dilation) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        if self.causal:
            x = F.pad(x, (self.pad, 0))
        else:
            x = F.pad(x, (self.pad, self.pad))
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, latent_dim=3):
        """
        in_channels: 1 (ERK) + n_stim_features
        hidden_dim: width of conv layers
        latent_dim: dimensionality of z
        """
        super().__init__()

        # Non-causal is fine for the encoder (it sees the full trajectory).
        # Causal only matters for Q2's prediction task.
        self.net = nn.Sequential(
            CausalConv1d(in_channels, hidden_dim, kernel_size=7, causal=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),

            CausalConv1d(hidden_dim, hidden_dim, kernel_size=5, dilation=2, causal=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),

            CausalConv1d(hidden_dim, hidden_dim, kernel_size=5, dilation=4, causal=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),

            CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=8, causal=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Global average pooling collapses time → single vector per cell
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        """x: (batch, in_channels, T) → mu, logvar: (batch, latent_dim)"""
        h = self.net(x)                  # (batch, hidden_dim, T)
        h = h.mean(dim=-1)               # (batch, hidden_dim)  global average pool
        return self.fc_mu(h), self.fc_logvar(h)
```

**Design notes:**
- Dilated convolutions (2, 4, 8) give a large receptive field without many layers — important with ~1600 cells. With kernel sizes of 7, 5, 5, 3 and dilations of 1, 2, 4, 8, the total receptive field is approximately 60 timepoints.
- Global average pooling over time is simple and effective. It forces the network to produce features that are useful across the whole trajectory, not just at specific timepoints.
- GELU activations over ReLU — smoother gradients, standard in modern architectures.
- Non-causal convolutions are appropriate here because the encoder sees the full trajectory. Switch to causal only for Q2.

### 2.2 Decoder

The decoder must reconstruct the ERK trajectory from z **and** the stimulus. The stimulus enters through a bypass — it is never compressed through the bottleneck.

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim=3, stim_channels=5, hidden_dim=64, out_length=120):
        """
        latent_dim: dimensionality of z
        stim_channels: number of engineered stimulus features
        hidden_dim: width of conv layers
        out_length: number of timepoints to reconstruct
        """
        super().__init__()
        self.out_length = out_length

        # Project z to a time-distributed representation
        self.z_proj = nn.Linear(latent_dim, hidden_dim)

        # Process stimulus bypass
        self.stim_net = nn.Sequential(
            CausalConv1d(stim_channels, hidden_dim // 2, kernel_size=5, causal=False),
            nn.GELU(),
            CausalConv1d(hidden_dim // 2, hidden_dim, kernel_size=5, causal=False),
            nn.GELU(),
        )

        # Merge z (broadcast over time) + processed stimulus, then decode
        self.decode_net = nn.Sequential(
            CausalConv1d(hidden_dim * 2, hidden_dim, kernel_size=5, dilation=1, causal=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),

            CausalConv1d(hidden_dim, hidden_dim, kernel_size=5, dilation=2, causal=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),

            CausalConv1d(hidden_dim, hidden_dim // 2, kernel_size=5, dilation=4, causal=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),

            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),  # project to 1 channel (ERK)
        )

    def forward(self, z, stim_cond):
        """
        z: (batch, latent_dim)
        stim_cond: (batch, stim_channels, T)  ← bypass input
        Returns: (batch, 1, T) reconstructed ERK
        """
        # Broadcast z across time
        z_time = self.z_proj(z).unsqueeze(-1).expand(-1, -1, self.out_length)  # (B, H, T)

        # Process stimulus
        s = self.stim_net(stim_cond)  # (B, H, T)

        # Concatenate and decode
        combined = torch.cat([z_time, s], dim=1)  # (B, 2*H, T)
        return self.decode_net(combined)           # (B, 1, T)
```

**Design notes:**
- z is projected to `hidden_dim` and broadcast across all timepoints. This means z provides a constant "cell identity" signal that modulates how the stimulus is decoded. This is the core conditional mechanism.
- The stimulus gets its own small processing network before being merged with z. This lets the model learn nonlinear transformations of the stimulus features.
- The merge is a simple concatenation followed by conv layers. Alternatives like FiLM conditioning (where z modulates the stimulus path via learned scale/shift) are more expressive but harder to interpret. Start simple.

### 2.3 Full CVAE

```python
class ConditionalBetaVAE(nn.Module):
    def __init__(self, in_channels=6, stim_channels=5, hidden_dim=64,
                 latent_dim=3, seq_length=120, beta=1.0):
        super().__init__()
        self.beta = beta
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, stim_channels, hidden_dim, seq_length)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, encoder_input, stim_cond):
        """
        encoder_input: (B, 1+stim_channels, T)  — ERK + stim features
        stim_cond: (B, stim_channels, T)         — stim features only (bypass)
        Returns: recon (B, 1, T), mu, logvar
        """
        mu, logvar = self.encoder(encoder_input)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, stim_cond)
        return recon, mu, logvar

    def loss(self, recon, target, mu, logvar):
        """
        recon, target: (B, 1, T)
        Returns: total_loss, recon_loss, kl_loss (all scalar)
        """
        recon_loss = F.mse_loss(recon, target, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + self.beta * kl_loss
        return total, recon_loss, kl_loss
```

---

## 3. Training

### 3.1 Hyperparameters to start with

| Parameter | Starting value | Rationale |
|---|---|---|
| `latent_dim` | 3 | Enough for interpretable structure, small enough to visualise directly |
| `hidden_dim` | 64 | Moderate capacity for ~1600 cells |
| `beta` | 0.1 → 1.0 (annealed) | See §3.2 |
| `learning_rate` | 1e-3 | Adam with weight decay 1e-5 |
| `batch_size` | 64 | Small dataset, small batches fine |
| `epochs` | 300–500 | Monitor validation loss for early stopping |

### 3.2 β annealing

Don't start with β = 1.0. The KL term will dominate early in training and push the posterior to match the prior before the decoder has learned anything useful, collapsing the latent space (posterior collapse). Instead:

```python
def beta_schedule(epoch, warmup=50, target_beta=1.0):
    """Linear warmup from 0 to target_beta over warmup epochs."""
    return min(target_beta, target_beta * epoch / warmup)
```

Start with β = 0, linearly ramp to your target over ~50 epochs. This lets the autoencoder first learn a useful reconstruction, then gradually regularise the latent space.

**Choosing the final β:** This is the key tuning knob. Lower β → better reconstruction, less structured latent space. Higher β → more disentangled latent dimensions, worse reconstruction. For your purpose (interpretable latent space), try final β values of 0.1, 0.5, 1.0, 2.0, and 4.0. Evaluate each with the metrics in §4.

### 3.3 Training loop

```python
def train(model, train_loader, val_loader, epochs=300, lr=1e-3, warmup=50,
          target_beta=1.0, patience=30):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    best_val = float("inf")
    wait = 0

    for epoch in range(epochs):
        model.beta = beta_schedule(epoch, warmup, target_beta)
        model.train()
        train_losses = []

        for enc_in, stim_cond, target in train_loader:
            recon, mu, logvar = model(enc_in, stim_cond)
            loss, recon_l, kl_l = model.loss(recon, target, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append((loss.item(), recon_l.item(), kl_l.item()))

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for enc_in, stim_cond, target in val_loader:
                recon, mu, logvar = model(enc_in, stim_cond)
                loss, recon_l, kl_l = model.loss(recon, target, mu, logvar)
                val_losses.append((loss.item(), recon_l.item(), kl_l.item()))

        val_mean = np.mean([v[0] for v in val_losses])
        scheduler.step(val_mean)

        # Early stopping
        if val_mean < best_val:
            best_val = val_mean
            wait = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 20 == 0:
            t = np.mean(train_losses, axis=0)
            v = np.mean(val_losses, axis=0)
            print(f"Epoch {epoch:3d} | β={model.beta:.3f} | "
                  f"Train: total={t[0]:.4f} recon={t[1]:.4f} kl={t[2]:.4f} | "
                  f"Val: total={v[0]:.4f} recon={v[1]:.4f} kl={v[2]:.4f}")
```

### 3.4 Things that can go wrong

**Posterior collapse** — KL loss drops to near zero, all cells map to the same point. Diagnosis: plot KL loss over training; if it flatlines near zero, β warmup is too fast or target β is too high. Fix: slower warmup, lower target β, or use KL-free bits (minimum KL per dimension).

**Reconstruction is good but latent space is random** — the model is memorising through the decoder's stimulus path and ignoring z. Diagnosis: sample random z values with the same stimulus — if reconstructions don't change, z is unused. Fix: increase latent dim, decrease decoder capacity, or add dropout to the stimulus bypass.

**Overfitting** — training loss keeps dropping but validation plateaus. With ~1600 cells this is a real risk. Fix: reduce `hidden_dim` (try 32), add dropout (0.1–0.2 after GELU activations), or use stronger weight decay.

---

## 4. Evaluation and Interpretation

### 4.1 Reconstruction quality

Plot overlaid original vs. reconstructed trajectories for random cells from each condition. Visually check that the model captures the main dynamics (peaks, oscillations, adaptation) and not just the mean trajectory.

```python
def plot_reconstructions(model, dataset, n=8):
    model.eval()
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    indices = np.random.choice(len(dataset), n, replace=False)

    for ax, idx in zip(axes.flat, indices):
        enc_in, stim_cond, target = dataset[idx]
        with torch.no_grad():
            recon, _, _ = model(enc_in.unsqueeze(0), stim_cond.unsqueeze(0))

        ax.plot(target[0].numpy(), "k", alpha=0.7, label="original")
        ax.plot(recon[0, 0].numpy(), "r", alpha=0.7, label="reconstructed")
        ax.set_xlabel("time (min)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("reconstructions.png", dpi=150)
```

### 4.2 Latent space visualisation

With `latent_dim=3`, you can directly scatter-plot the latent space colour-coded by experimental condition, then by other metadata (replicate, cell density, etc.).

```python
def encode_all(model, dataset):
    """Get latent coordinates for all cells."""
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    mus = []
    for enc_in, stim_cond, _ in loader:
        with torch.no_grad():
            mu, _ = model.encoder(enc_in)
        mus.append(mu.numpy())
    return np.concatenate(mus, axis=0)  # (n_cells, latent_dim)
```

**What to look for:**
- Cells from different stimulation conditions should separate (the conditioning works)
- Within a condition, look for subclusters or continuous gradients (the heterogeneity you're after)
- If everything is a single blob, the model may not be learning meaningful variation — try lower β or more latent dims

### 4.3 Latent traversals (most important for interpretability)

Fix a reference cell, then move along one latent dimension at a time and decode. This reveals what each dimension controls.

```python
def latent_traversal(model, z_ref, stim_cond, dim, n_steps=7, range_val=2.0):
    """Traverse one latent dimension and decode."""
    model.eval()
    zs = z_ref.unsqueeze(0).repeat(n_steps, 1)
    values = torch.linspace(-range_val, range_val, n_steps)
    zs[:, dim] = values

    stim_repeated = stim_cond.unsqueeze(0).repeat(n_steps, 1, 1)
    with torch.no_grad():
        recons = model.decoder(zs, stim_repeated)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(n_steps):
        alpha = 0.3 + 0.7 * (i / (n_steps - 1))
        ax.plot(recons[i, 0].numpy(), alpha=alpha,
                label=f"z[{dim}]={values[i]:.1f}")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("ERK (normalised)")
    ax.set_title(f"Traversal along latent dimension {dim}")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"traversal_dim{dim}.png", dpi=150)
```

If dim 0 controls amplitude, you'll see the curves scale vertically. If dim 1 controls oscillation frequency, you'll see the number of peaks change. If a dimension doesn't produce interpretable variation, it might be encoding noise or the model needs re-tuning.

### 4.4 Quantitative metrics

**Mutual information with known parameters.** For each latent dimension, compute mutual information (or simple Spearman correlation) with known experimental variables: stimulus condition, pulse duration, pulse frequency, replicate, mean ERK level. High MI between a latent dimension and a known variable confirms the model is encoding something real.

**KL per dimension.** Compute the KL divergence for each latent dimension separately. Dimensions with KL ≈ 0 are "dead" (collapsed to prior, not encoding information). Dimensions with high KL are carrying the most signal.

```python
def kl_per_dimension(model, dataset):
    """Average KL divergence per latent dimension across all cells."""
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    all_mu, all_logvar = [], []
    for enc_in, stim_cond, _ in loader:
        with torch.no_grad():
            mu, logvar = model.encoder(enc_in)
        all_mu.append(mu.numpy())
        all_logvar.append(logvar.numpy())
    mu = np.concatenate(all_mu)
    logvar = np.concatenate(all_logvar)
    # KL per dim: 0.5 * (mu^2 + exp(logvar) - logvar - 1), averaged over cells
    kl = 0.5 * (mu**2 + np.exp(logvar) - logvar - 1)
    return kl.mean(axis=0)  # (latent_dim,)
```

---

## 5. Iteration Checklist

Once the basic pipeline runs, work through these systematically:

**β sweep.** Train models with final β ∈ {0.01, 0.1, 0.5, 1.0, 4.0}. For each, record reconstruction MSE, total KL, KL per dimension, and qualitatively assess traversals. You're looking for the sweet spot where traversals are interpretable but reconstruction is still faithful.

**Latent dim sweep.** Try `latent_dim` ∈ {2, 3, 4, 6}. With 2, you can plot the full latent space directly. With 3–4, you get more expressiveness. Above 4, risk of dead dimensions and overfitting grows with your data scale.

**Stimulus bypass ablation.** Train a version where the decoder does NOT receive the stimulus (standard VAE, no conditioning). Compare latent spaces. If the conditional version produces cleaner within-condition structure, the bypass is working as intended. If they look the same, the stimulus signal may be weak or the model is not using it.

**Normalisation ablation.** Compare global vs. per-cell normalisation. If per-cell normalisation kills a latent dimension that was present with global normalisation, that dimension was encoding amplitude.

**Decoder capacity check.** If you suspect the decoder is ignoring z and relying entirely on the stimulus bypass, reduce the stimulus processing network (fewer layers, smaller hidden dim) or add dropout to the bypass path. The decoder should *need* z to produce good reconstructions.

---

## 6. From Here to Q2 and Q3

Once you have a working CVAE with interpretable latent space:

**→ Q2:** Replace the non-causal encoder with a causal version (only left-padding in convolutions). Add a trajectory prediction head: given z encoded from timepoints 0..t, predict ERK at timepoints t+1..T. Sweep t and plot prediction error as a function of observation window.

**→ Q3:** Freeze the trained encoder. Separately fit ODE models to each cell's trajectory. Regress ODE parameters against latent coordinates. Correlation reveals which mechanistic parameters the network has implicitly learned; lack of correlation identifies unexplained variation.