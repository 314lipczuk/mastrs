# Evaluation Module Spec: `eval_cvae.py`

## Purpose

A single importable Python module that evaluates trained CVAE models across all
metrics discussed. Designed to be called from Jupyter notebooks, producing
publication-ready figures and a summary report for each experiment.

Must handle architectural variation between experiments (different latent dims,
different loss functions, different encoder/decoder configs) without code changes.

---

## Design Principles

**Model-agnostic interface.** The module never instantiates models itself. It
receives a trained model object, a dataset, and metadata. It interrogates the
model to discover properties (latent dim, etc.) rather than requiring them as
arguments. This means any architecture that exposes `.encoder(input) → (mu, logvar)`
and `.decoder(z, stim) → reconstruction` works without modification.

**Experiment as a unit.** All evaluation for one experiment is bundled into a
single `ExperimentResult` object that holds every metric, every figure, and all
raw data. Results can be compared across experiments via a separate comparison
function.

**Figures returned, not shown.** Every plotting function returns a matplotlib
`Figure` object. The notebook decides whether to `display()`, `savefig()`, or
ignore. No `plt.show()` calls inside the module.

**No global state.** No module-level variables, no device assumptions. Device is
inferred from model parameters. Everything is passed explicitly.

---

## Public API

### Core evaluation

```python
def evaluate(
    model: nn.Module,
    dataset: Dataset,
    metadata: pd.DataFrame,
    condition_col: str = "condition",
    dt: float = 1.0,
    n_recon_examples: int = 12,
    n_traversal_steps: int = 7,
    traversal_range: float = 2.0,
    kl_active_threshold: float = 0.05,
) -> ExperimentResult:
```

This is the main entry point. Runs all metrics, returns everything in one object.
The notebook usage pattern:

```python
from eval_cvae import evaluate, compare_experiments

result = evaluate(model, dataset, metadata)

# Look at individual outputs
result.summary()                     # prints text summary
result.figures["reconstructions"]    # access specific figure
result.metrics["mse_per_cell"]       # access raw metric arrays
result.save("experiments/exp01/")    # dump everything to disk
```

### Experiment comparison

```python
def compare_experiments(
    results: dict[str, ExperimentResult],
    metrics: list[str] | None = None,
) -> Figure:
```

Takes a dict mapping experiment names to `ExperimentResult` objects. Produces
a comparison dashboard (bar charts, overlaid distributions, etc).

```python
# In notebook:
results = {
    "baseline_L4": result_baseline,
    "spectral_L4": result_spectral,
    "baseline_L10": result_l10,
    "low_beta_L4": result_lowbeta,
}
fig = compare_experiments(results)
```

---

## ExperimentResult dataclass

```python
@dataclass
class ExperimentResult:
    # --- Identity ---
    name: str                          # experiment label (user-provided or auto)
    timestamp: str                     # when evaluation was run
    model_config: dict                 # auto-discovered: latent_dim, hidden_dim, beta, etc.

    # --- Raw metric arrays ---
    metrics: dict[str, np.ndarray]
    # Keys:
    #   "mse_per_cell"          (n_cells,)        per-cell reconstruction MSE
    #   "z_mu"                  (n_cells, L)      latent means for all cells
    #   "z_logvar"              (n_cells, L)      latent log-variances
    #   "kl_per_dim"            (L,)              avg KL per latent dimension
    #   "kl_per_cell"           (n_cells, L)      KL per cell per dimension
    #   "active_dims"           (L,)              boolean mask of active dims
    #   "n_active_dims"         int
    #   "orig_psd"              (n_cells, F)      power spectra of originals
    #   "recon_psd"             (n_cells, F)      power spectra of reconstructions
    #   "psd_freqs"             (F,)              frequency axis
    #   "condition_clf_acc"     float             stimulus invariance accuracy
    #   "condition_clf_chance"  float             chance level

    # --- Figures ---
    figures: dict[str, Figure]
    # Keys:
    #   "reconstructions"           grid of original vs reconstructed traces
    #   "mse_distribution"          histogram of per-cell MSE
    #   "mse_by_condition"          box plot of MSE grouped by condition
    #   "power_spectra"             mean PSD comparison + ratio plot
    #   "kl_per_dim"                bar chart of KL per dimension
    #   "latent_space"              pairwise scatter of active dims, coloured by condition
    #   "latent_traversals"         one figure per active dimension
    #   "stimulus_invariance"       latent scatter + classifier accuracy annotation
    #   "within_condition"          per-condition latent distributions
    #   "encoder_uncertainty"       distribution of sigma per cell

    # --- Methods ---
    def summary(self) -> None:
        """Print a concise text summary of all metrics."""

    def save(self, directory: str) -> None:
        """Save all figures as PNG, metrics as npz, summary as txt."""

    def _repr_html_(self) -> str:
        """Jupyter-friendly display: shows summary + key figures inline."""
```

---

## Internal Functions (private)

Each metric is computed by a separate internal function. This keeps the code
modular and testable. All follow the same pattern: receive model + data,
return raw metric arrays and a figure.

### `_discover_model_config(model) -> dict`

Interrogates the model to extract architectural details without the caller
specifying them. Implementation:

```python
def _discover_model_config(model):
    config = {}

    # Discover latent dim from encoder output
    # Feed a dummy input through the encoder, check mu shape
    # This works regardless of encoder architecture
    dummy_input = torch.zeros(1, *dataset_input_shape).to(device)
    # ... but we don't have the dataset here.
    #
    # Better: inspect parameter shapes directly.
    # fc_mu.weight has shape (latent_dim, something)
    for name, param in model.named_parameters():
        if "fc_mu.weight" in name:
            config["latent_dim"] = param.shape[0]
        if "fc_logvar.weight" in name:
            config["latent_dim_check"] = param.shape[0]

    # Discover beta
    if hasattr(model, "beta"):
        config["beta"] = model.beta

    # Discover total params
    config["total_params"] = sum(p.numel() for p in model.parameters())
    config["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return config
```

Falls back gracefully if attribute names differ. The dummy-forward approach is
the most robust fallback: create a zero tensor matching the dataset's input
shape, run it through the encoder, and check output dimensions. This requires
passing one sample from the dataset, which `evaluate()` does.

### `_encode_all(model, dataset) -> (z_mu, z_logvar)`

Encode all cells, return mu and logvar arrays. Shape: `(n_cells, latent_dim)`.
Handles batching internally. Infers device from model.

### `_reconstruct_all(model, dataset) -> (recons, targets)`

Reconstruct all cells, return both. Shape: `(n_cells, 1, T)`.
Used by multiple downstream metrics.

### `_plot_reconstructions(recons, targets, metadata, condition_col, n_examples) -> Figure`

Grid of subplots, one per cell. Each shows original (black) and reconstruction
(red) overlaid. Cells are sampled to cover all conditions roughly equally.
Subplot titles show condition label and per-cell MSE.

Layout: automatic grid sizing based on `n_examples`. Default 3×4 = 12 cells.

### `_compute_mse_per_cell(recons, targets) -> np.ndarray`

Simple: MSE computed per cell across timepoints. Returns `(n_cells,)`.

### `_plot_mse_distribution(mse_per_cell, metadata, condition_col) -> Figure`

Two panels side by side:
- Left: histogram of MSE across all cells, with mean and median annotated
- Right: box plot of MSE grouped by condition

### `_compute_frequency_analysis(recons, targets, dt) -> (orig_psd, recon_psd, freqs)`

Compute power spectral density via FFT for each cell's original and
reconstructed trace. Returns arrays for downstream plotting.

### `_plot_power_spectra(orig_psd, recon_psd, freqs) -> Figure`

Two panels:
- Left: mean PSD of originals vs reconstructions on log-y scale
- Right: per-frequency power ratio (reconstruction / original)
  Horizontal line at 1.0. Shaded region showing ±1 std across cells.

### `_compute_kl_per_dimension(z_mu, z_logvar) -> (kl_per_dim, kl_per_cell)`

KL divergence per dimension averaged over cells, and the full per-cell array
for downstream use. `kl_per_dim` has shape `(latent_dim,)`.

### `_plot_kl_per_dimension(kl_per_dim, threshold) -> Figure`

Bar chart. Active dimensions in colour, dead dimensions in grey.
Threshold line. Title shows count of active/total.

### `_compute_latent_traversals(model, dataset, z_mu, active_dims, n_steps, range_val) -> dict`

For each active latent dimension:
- Pick a reference cell near the centroid of the latent space
- Sweep that dimension from -range_val to +range_val in n_steps
- Decode each point using the reference cell's stimulus
- Store the decoded trajectories

Returns dict mapping dimension index to array of decoded trajectories.

### `_plot_latent_traversals(traversals, active_dims) -> Figure`

One subplot per active dimension. Each subplot shows the family of decoded
trajectories coloured by the swept latent value (using a sequential colourmap).
Consistent y-axis across subplots for comparability.

### `_compute_stimulus_invariance(z_mu, metadata, condition_col) -> (accuracy, chance, scores)`

Train logistic regression classifier to predict condition from z.
5-fold cross-validation. Returns mean accuracy, chance level, and per-fold scores.

### `_plot_stimulus_invariance(z_mu, metadata, condition_col, accuracy, chance, active_dims) -> Figure`

Pairwise scatter plots of active latent dimensions, coloured by condition.
Accuracy and chance level annotated as text on the figure.
Maximum 3 pairwise plots (for the first 3 active dimensions).

### `_plot_within_condition(z_mu, metadata, condition_col, active_dims) -> Figure`

One row of subplots per condition. Each row shows the marginal distribution
(histogram or KDE) of each active latent dimension for cells in that condition.

This reveals whether within-condition distributions are unimodal (one phenotype),
bimodal (two subpopulations), or broad (continuous gradient).

### `_plot_encoder_uncertainty(z_logvar, metadata, condition_col) -> Figure`

Two panels:
- Left: distribution of mean sigma (averaged across latent dims) per cell
- Right: box plot of mean sigma grouped by condition

Cells with high uncertainty are candidates for stochastic/transitional behaviour.

---

## Handling Architectural Variation

The module must work with models that differ in:
- Latent dimensionality (2, 3, 4, 10)
- Loss function (MSE-only vs MSE + spectral)
- Encoder architecture (global pool vs multi-scale pool vs attention pool)
- Stimulus encoding (hand-crafted vs learned vs hybrid)
- Beta value

The strategy:

**Duck typing for model interface.** The module assumes:
- `model.encoder(x) -> (mu, logvar)` where x has the same channel layout as
  the dataset's encoder_input
- `model.decoder(z, stim) -> reconstruction`
- `model.beta` exists (optional, just for reporting)

It does NOT assume specific layer names, hidden dimensions, or internal
architecture. If a model uses a different attribute name for beta, or doesn't
have one, the config discovery step skips it gracefully.

**Dataset provides the contract.** The module assumes `dataset[i]` returns
`(encoder_input, stim_cond, target)` as tensors. This is already the contract
from the implementation guide's ERKDataset. If someone builds a different
dataset class, it must follow this same 3-tuple convention.

**Automatic figure layout.** Plotting functions adapt to the number of active
latent dimensions. With 2 active dims you get 1 pairwise scatter. With 4 you
get 6 (or capped at 3 most informative pairs by KL ranking). Grid sizes for
reconstruction plots adapt to `n_examples`.

---

## Comparison Dashboard: `compare_experiments`

Takes a dict of `ExperimentResult` objects. Produces a multi-panel figure:

**Panel 1: Reconstruction MSE.** Violin or box plot of per-cell MSE for each
experiment, side by side. Shows whether spectral loss or lower beta actually
improved reconstruction.

**Panel 2: Active latent dimensions.** Grouped bar chart showing KL per
dimension for each experiment. Quickly reveals whether adding dimensions
helps or just adds dead ones.

**Panel 3: Stimulus invariance.** Bar chart of condition classification accuracy
per experiment, with chance level marked. Shows which architectures best
separate stimulus from intrinsic variation.

**Panel 4: Spectral fidelity.** Mean power ratio curves overlaid for all
experiments. Shows which experiments best preserve oscillatory content.

Layout: 2×2 grid. Colour-coded by experiment name with shared legend.

```python
def compare_experiments(
    results: dict[str, ExperimentResult],
    metrics: list[str] | None = None,
) -> Figure:
    """
    results: {"experiment_name": ExperimentResult, ...}
    metrics: optional subset of ["mse", "kl", "invariance", "spectral"]
             defaults to all four
    """
```

---

## File Structure

```
eval_cvae.py
├── ExperimentResult (dataclass)
├── evaluate()                     # main entry point
├── compare_experiments()          # cross-experiment comparison
├── _discover_model_config()
├── _encode_all()
├── _reconstruct_all()
├── _plot_reconstructions()
├── _compute_mse_per_cell()
├── _plot_mse_distribution()
├── _compute_frequency_analysis()
├── _plot_power_spectra()
├── _compute_kl_per_dimension()
├── _plot_kl_per_dimension()
├── _compute_latent_traversals()
├── _plot_latent_traversals()
├── _compute_stimulus_invariance()
├── _plot_stimulus_invariance()
├── _plot_within_condition()
├── _plot_encoder_uncertainty()
└── _plot_comparison_dashboard()   # used by compare_experiments
```

Single file. No submodules. All private functions prefixed with underscore.

---

## Dependencies

```
torch
numpy
pandas
matplotlib
scipy           (for spearmanr in correlation analysis, if added later)
scikit-learn    (for LogisticRegression in stimulus invariance test)
```

---

## Notebook Usage Pattern

```python
import torch
from eval_cvae import evaluate, compare_experiments

# --- Evaluate a single experiment ---
model_baseline = ConditionalBetaVAE(latent_dim=4, ...)
model_baseline.load_state_dict(torch.load("checkpoints/baseline_L4.pt"))

result_baseline = evaluate(
    model=model_baseline,
    dataset=val_dataset,
    metadata=metadata_df,
    condition_col="stim_condition",
)

result_baseline.summary()
result_baseline.figures["reconstructions"]
result_baseline.figures["latent_traversals"]
result_baseline.save("results/baseline_L4/")

# --- Evaluate another experiment ---
model_spectral = ConditionalBetaVAE(latent_dim=4, ...)
model_spectral.load_state_dict(torch.load("checkpoints/spectral_L4.pt"))

result_spectral = evaluate(
    model=model_spectral,
    dataset=val_dataset,
    metadata=metadata_df,
)

# --- Compare ---
fig = compare_experiments({
    "Baseline L=4":       result_baseline,
    "Spectral loss L=4":  result_spectral,
})
fig.savefig("comparison.png", dpi=200)
```

---

## On the Windowing Question

The user raised a concern: if non-overlapping windows are extracted from
trajectories, won't the model cluster by window position (early vs late
in the trajectory) rather than by biological phenotype?

This is a real risk. If a trajectory is 480 timepoints and windows are 120,
you get 4 windows per cell. Window 1 always captures the initial response,
window 4 always captures late adaptation. The model will trivially learn
to distinguish "early" from "late" and this will dominate the latent space.

**Recommended approach: one window per cell, randomly positioned.**
For each cell, sample one random start index per epoch (data augmentation).
This means:
- No duplicate data within a single epoch (each cell contributes one window)
- Across epochs, the model sees different temporal slices of each cell
- No systematic bias toward early or late windows

Implementation belongs in the Dataset class, not in eval_cvae.py.
But the evaluation module should document this assumption: it expects
one sample per cell in the dataset, and metadata should have one row per cell.

If the user later wants overlapping windows for training (to increase
effective dataset size), the evaluation should still be run on one canonical
window per cell (e.g., always starting at t=0, or at stimulation onset) to
ensure metrics are not inflated by pseudo-replication.

The eval module handles this by checking that `len(dataset) == len(metadata)`
and warning if they don't match:

```python
if len(dataset) != len(metadata):
    warnings.warn(
        f"Dataset has {len(dataset)} samples but metadata has "
        f"{len(metadata)} rows. If using windowed data, ensure "
        f"evaluation uses one canonical window per cell."
    )
```
