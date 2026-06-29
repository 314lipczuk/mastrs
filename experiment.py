"""
Standardized experiment result serialization and deserialization.

Save a complete training run (model weights, configs, metrics, figures)
to a directory and reload it later — even if the model class has changed.

Usage:
    from experiment import save_experiment, load_experiment

    # After training + evaluation:
    save_experiment(
        "results/my_exp",
        model=model,
        model_config={"input_dim": 5, "hidden_dims": (128, 128), "latent_dim": 3},
        training_config={"lr": 1e-3, "epochs": 300, "batch_size": 256},
        training_results={"train_loss": [...], "val_loss": [...], "best_epoch": 42},
        metrics=result.metrics,
        figures=result.figures,
    )

    # Loading:
    bundle = load_experiment("results/my_exp")
    bundle.summary()
    bundle.display_figures()
    model = bundle.reconstruct_model()
"""

from __future__ import annotations

import importlib
import io
import json
import contextlib
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_type_string(model: nn.Module) -> str:
    """Derive fully qualified class path from a model instance."""
    cls = type(model)
    return f"{cls.__module__}.{cls.__qualname__}"


def _resolve_model_class(model_type: str) -> type:
    """Import and return the model class from its fully qualified path."""
    module_path, class_name = model_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _infer_config_from_state_dict(state_dict: dict) -> dict:
    """Best-effort inference of model config from state_dict weight shapes.

    Detects MLP autoencoders (encoder.net.N.weight) vs Conv1d conditional
    models (encoder.net.N.conv.weight + decoder.stim_net) and extracts
    latent_dim, hidden_dims/hidden_dim, input_dim, in_channels, stim_channels.
    """
    keys = set(state_dict.keys())
    config: dict = {}

    is_conv = any("conv.weight" in k for k in keys)

    if is_conv:
        # Conv1d architecture (ConditionalBetaVAE / ConditionalAE / BetaVAE)
        # hidden_dim from first encoder conv
        first_conv = state_dict.get("encoder.net.0.conv.weight")
        if first_conv is not None:
            config["hidden_dim"] = first_conv.shape[0]
            config["in_channels"] = first_conv.shape[1]

        # latent_dim from fc_mu (VAE) or fc (AE)
        if "encoder.fc_mu.weight" in keys:
            config["latent_dim"] = state_dict["encoder.fc_mu.weight"].shape[0]
        elif "encoder.fc.weight" in keys:
            config["latent_dim"] = state_dict["encoder.fc.weight"].shape[0]

        # stim_channels from decoder.stim_net (conditional models)
        stim_conv = state_dict.get("decoder.stim_net.0.conv.weight")
        if stim_conv is not None:
            config["stim_channels"] = stim_conv.shape[1]
    else:
        # MLP architecture (AutoEncoder / VAE)
        # Extract layer sizes: encoder.net.0.weight, encoder.net.2.weight, ...
        enc_weights = sorted(
            [(k, v) for k, v in state_dict.items()
             if k.startswith("encoder.net.") and k.endswith(".weight")],
            key=lambda x: x[0],
        )
        if enc_weights:
            config["input_dim"] = enc_weights[0][1].shape[1]
            config["latent_dim"] = enc_weights[-1][1].shape[0]
            hidden_dims = tuple(w.shape[0] for _, w in enc_weights[:-1])
            if hidden_dims:
                config["hidden_dims"] = hidden_dims

        # Detect VAE: has fc_mu / fc_logvar
        if "encoder.fc_mu.weight" in keys:
            config["latent_dim"] = state_dict["encoder.fc_mu.weight"].shape[0]

    return config


def _infer_model_type_from_state_dict(state_dict: dict) -> str:
    """Best-effort inference of fully qualified model class from state_dict keys."""
    keys = set(state_dict.keys())
    is_conv = any("conv.weight" in k for k in keys)
    has_fc_mu = "encoder.fc_mu.weight" in keys
    has_stim_net = any(k.startswith("decoder.stim_net.") for k in keys)

    if is_conv:
        if has_stim_net and has_fc_mu:
            return "model.dl.cvae.ConditionalBetaVAE"
        elif has_stim_net:
            return "model.dl.cae.ConditionalAE"
        elif has_fc_mu:
            return "model.dl.vae.BetaVAE"
    else:
        if has_fc_mu:
            return "model.dl.vae.VAE"
        else:
            return "model.dl.ae.AutoEncoder"
    return ""


def _prepare_config_for_save(model_config: dict) -> dict:
    """Convert non-serializable config values (e.g. activation classes) to strings."""
    out = {}
    for k, v in model_config.items():
        if isinstance(v, type) and issubclass(v, nn.Module):
            out[k] = v.__name__
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _prepare_config_for_load(model_config: dict) -> dict:
    """Convert string activation names back to classes."""
    out = dict(model_config)
    if "activation" in out and isinstance(out["activation"], str):
        out["activation"] = getattr(nn, out["activation"])
    # Restore tuples for hidden_dims
    if "hidden_dims" in out and isinstance(out["hidden_dims"], list):
        out["hidden_dims"] = tuple(out["hidden_dims"])
    return out


# ---------------------------------------------------------------------------
# Training statistics
# ---------------------------------------------------------------------------

def _get_device_name() -> str:
    """Return the name of the active compute device."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return f"Apple MPS ({platform.processor()})"
    return "CPU"


def compute_training_stats(
    train_elapsed_s: float,
    history: dict,
    n_train_samples: int,
    n_val_samples: int = 0,
    model: nn.Module | None = None,
    device_name: str | None = None,
) -> dict:
    """Compute standardized training statistics from a training run.

    Args:
        train_elapsed_s: total wall-clock training time in seconds.
        history: dict with at least 'train_loss' and optionally 'val_loss' as lists.
        n_train_samples: number of samples in training set.
        n_val_samples: number of samples in validation set.
        model: trained model (used to count parameters).
        device_name: override device name string. Auto-detected if None.

    Returns:
        Dict of training statistics ready to be stored in training_results.
    """
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    epochs = len(train_loss)

    stats: dict = {
        "train_elapsed_s": round(train_elapsed_s, 2),
        "epochs_completed": epochs,
        "time_per_epoch_s": round(train_elapsed_s / max(epochs, 1), 3),
        "n_train_samples": n_train_samples,
        "n_val_samples": n_val_samples,
        "total_samples_seen": n_train_samples * epochs,
        "samples_per_second": round(n_train_samples * epochs / max(train_elapsed_s, 1e-6), 1),
        "device": device_name or _get_device_name(),
    }

    if model is not None:
        stats["n_model_parameters"] = sum(p.numel() for p in model.parameters())

    if train_loss:
        stats["final_train_loss"] = round(train_loss[-1], 6)
    if val_loss:
        stats["final_val_loss"] = round(val_loss[-1], 6)
        stats["best_val_loss"] = round(min(val_loss), 6)
        stats["best_epoch"] = int(np.argmin(val_loss))

        # Convergence epoch: first epoch reaching within 5% of best improvement
        first_val = val_loss[0]
        best_val = min(val_loss)
        improvement = first_val - best_val
        if improvement > 0:
            threshold = first_val - 0.95 * improvement
            convergence = next(
                (i for i, v in enumerate(val_loss) if v <= threshold), epochs - 1
            )
            stats["convergence_epoch_95pct"] = convergence

    return stats


# ---------------------------------------------------------------------------
# ExperimentBundle
# ---------------------------------------------------------------------------

@dataclass
class ExperimentBundle:
    name: str
    timestamp: str
    model_type: str
    model_config: dict
    model_state_dict: dict | None
    training_config: dict
    training_results: dict
    metrics: dict
    figures: dict[str, Figure | np.ndarray]
    normalization: dict | None = None
    save_dir: str | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> None:
        print(f"=== {self.name} ({self.timestamp}) ===")
        print(f"  Model type     : {self.model_type}")
        print(f"  Model config   : {self.model_config}")
        print()
        if self.training_config:
            print("  Training config:")
            for k, v in self.training_config.items():
                print(f"    {k}: {v}")
            print()
        # Display training stats (from compute_training_stats) prominently
        stats = self.training_results.get("stats", {})
        if stats:
            print("  Training stats:")
            elapsed = stats.get("train_elapsed_s", 0)
            if elapsed >= 3600:
                time_str = f"{elapsed / 3600:.1f}h"
            elif elapsed >= 60:
                time_str = f"{elapsed / 60:.1f}m"
            else:
                time_str = f"{elapsed:.1f}s"
            print(f"    Wall time        : {time_str}")
            print(f"    Epochs           : {stats.get('epochs_completed', '?')}")
            print(f"    Time/epoch       : {stats.get('time_per_epoch_s', '?')}s")
            if "n_model_parameters" in stats:
                print(f"    Parameters       : {stats['n_model_parameters']:,}")
            print(f"    Train samples    : {stats.get('n_train_samples', '?'):,}")
            print(f"    Val samples      : {stats.get('n_val_samples', '?'):,}")
            print(f"    Total seen       : {stats.get('total_samples_seen', '?'):,}")
            print(f"    Throughput       : {stats.get('samples_per_second', '?')} samples/s")
            print(f"    Device           : {stats.get('device', '?')}")
            if "best_val_loss" in stats:
                print(f"    Best val loss    : {stats['best_val_loss']:.6f} (epoch {stats.get('best_epoch', '?')})")
            if "final_train_loss" in stats:
                print(f"    Final train loss : {stats['final_train_loss']:.6f}")
            if "final_val_loss" in stats:
                print(f"    Final val loss   : {stats['final_val_loss']:.6f}")
            if "convergence_epoch_95pct" in stats:
                print(f"    95% convergence  : epoch {stats['convergence_epoch_95pct']}")
            print()
        if self.training_results:
            other_results = {k: v for k, v in self.training_results.items() if k != "stats"}
            if other_results:
                print("  Training results:")
                for k, v in other_results.items():
                    if isinstance(v, (list, np.ndarray)) and len(v) > 5:
                        print(f"    {k}: [{len(v)} values]")
                    else:
                        print(f"    {k}: {v}")
                print()
        # Print scalar metrics
        scalar_metrics = {
            k: v for k, v in self.metrics.items()
            if isinstance(v, (int, float, str))
        }
        if scalar_metrics:
            print("  Metrics:")
            for k, v in scalar_metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.6f}")
                else:
                    print(f"    {k}: {v}")
            print()
        print(f"  Figures        : {list(self.figures.keys())}")
        if self.warnings:
            print()
            print("  WARNINGS:")
            for w in self.warnings:
                print(f"    ⚠ {w}")
        print()

    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.save_dir = str(d)

        # Save figures as PNGs
        fig_dir = d / "figures"
        fig_dir.mkdir(exist_ok=True)
        for name, fig in self.figures.items():
            if isinstance(fig, Figure):
                fig.savefig(fig_dir / f"{name}.png", dpi=200, bbox_inches="tight")
            elif isinstance(fig, np.ndarray):
                plt.imsave(fig_dir / f"{name}.png", fig)

        # Build bundle dict (everything except figures)
        bundle_data = {
            "name": self.name,
            "timestamp": self.timestamp,
            "model_type": self.model_type,
            "model_config": _prepare_config_for_save(self.model_config),
            "model_state_dict": self.model_state_dict,
            "training_config": self.training_config,
            "training_results": self.training_results,
            "metrics": self.metrics,
            "normalization": self.normalization,
        }
        torch.save(bundle_data, d / "bundle.pt")

        # Human-readable summary
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.summary()
        (d / "summary.txt").write_text(buf.getvalue())

    @classmethod
    def load(cls, directory: str) -> ExperimentBundle:
        d = Path(directory)
        warnings: list[str] = []

        bundle_path = d / "bundle.pt"
        if bundle_path.exists():
            data = torch.load(bundle_path, map_location="cpu", weights_only=False)
        else:
            # Legacy format: model_*.pt + metrics.npz + scalars.json
            data = cls._load_legacy(d, warnings)

        # Load figures from PNGs (best-effort)
        figures: dict[str, np.ndarray] = {}
        fig_dir = d / "figures"
        if fig_dir.is_dir():
            for png in sorted(fig_dir.glob("*.png")):
                try:
                    figures[png.stem] = mpimg.imread(str(png))
                except Exception as e:
                    warnings.append(f"Could not load figure '{png.name}': {e}")

        # Infer model_type from state_dict if missing
        model_type = data.get("model_type", "")
        model_state_dict = data.get("model_state_dict")
        if not model_type and model_state_dict:
            model_type = _infer_model_type_from_state_dict(model_state_dict)
            if model_type:
                warnings.append(f"Inferred model_type from state_dict: {model_type}")

        if model_type:
            try:
                _resolve_model_class(model_type)
            except Exception as e:
                warnings.append(
                    f"Model class '{model_type}' could not be imported: {e}. "
                    f"Figures and metrics are still available, but "
                    f"reconstruct_model() will fail."
                )

        # Validate state dict is loadable (basic shape check deferred to reconstruct_model)
        if model_state_dict is None:
            warnings.append("No model_state_dict found in bundle. Model reconstruction not possible.")

        # Impute missing model_config from state_dict shapes
        model_config = data.get("model_config", {})
        if model_state_dict and not model_config.get("latent_dim"):
            inferred = _infer_config_from_state_dict(model_state_dict)
            for k, v in inferred.items():
                if k not in model_config:
                    model_config[k] = v
            if inferred:
                warnings.append(
                    f"Inferred model_config from state_dict: {inferred}"
                )

        return cls(
            name=data.get("name", d.name),
            timestamp=data.get("timestamp", "unknown"),
            model_type=model_type,
            model_config=model_config,
            model_state_dict=model_state_dict,
            training_config=data.get("training_config", {}),
            training_results=data.get("training_results", {}),
            metrics=data.get("metrics", {}),
            figures=figures,
            normalization=data.get("normalization"),
            warnings=warnings,
        )

    @staticmethod
    def _load_legacy(d: Path, warnings: list[str]) -> dict:
        """Load from the legacy format: model_*.pt + metrics.npz + scalars.json."""
        import json
        # Find the .pt checkpoint (use the latest one)
        pt_files = sorted(d.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(
                f"No .pt files found in {d}. Cannot load experiment."
            )
        checkpoint = torch.load(pt_files[-1], map_location="cpu", weights_only=False)

        # Build model_config from the legacy 'config' key
        config = checkpoint.get("config", {})
        training_kwargs = checkpoint.get("training_kwargs", {})

        # Merge metrics from scalars.json and metrics.npz
        metrics: dict = {}
        scalars_path = d / "scalars.json"
        if scalars_path.exists():
            with open(scalars_path) as f:
                metrics.update(json.load(f))
        npz_path = d / "metrics.npz"
        if npz_path.exists():
            npz = np.load(npz_path, allow_pickle=True)
            metrics.update({k: npz[k] for k in npz.files})

        # Build training_results from history
        training_results: dict = {}
        if "history" in checkpoint:
            training_results["history"] = checkpoint["history"]
        if "train_elapsed_s" in checkpoint:
            training_results["train_elapsed_s"] = checkpoint["train_elapsed_s"]

        # Build normalization
        normalization = None
        if "erk_mu" in checkpoint or "erk_sigma" in checkpoint:
            normalization = {}
            if "erk_mu" in checkpoint:
                normalization["erk_mu"] = checkpoint["erk_mu"]
            if "erk_sigma" in checkpoint:
                normalization["erk_sigma"] = checkpoint["erk_sigma"]

        return {
            "name": d.name,
            "timestamp": checkpoint.get("train_start", "unknown"),
            "model_type": "",
            "model_config": config,
            "model_state_dict": checkpoint.get("model_state_dict"),
            "training_config": training_kwargs,
            "training_results": training_results,
            "metrics": metrics,
            "normalization": normalization,
        }

    def reconstruct_model(self) -> nn.Module:
        """Instantiate the model class, load weights, return in eval mode.

        Raises RuntimeError if the model class can't be imported or weights
        don't match the architecture.
        """
        if self.model_state_dict is None:
            raise RuntimeError(
                "Cannot reconstruct model: no model_state_dict in this bundle."
            )

        try:
            model_cls = _resolve_model_class(self.model_type)
        except Exception as e:
            raise RuntimeError(
                f"Cannot import model class '{self.model_type}': {e}. "
                f"The model code may have been moved or renamed."
            ) from e

        config = _prepare_config_for_load(self.model_config)

        try:
            model = model_cls(**config)
        except TypeError as e:
            raise RuntimeError(
                f"Cannot instantiate {self.model_type} with config {self.model_config}: {e}. "
                f"The model constructor signature may have changed."
            ) from e

        try:
            model.load_state_dict(self.model_state_dict)
        except RuntimeError as e:
            raise RuntimeError(
                f"Cannot load state dict into {self.model_type}: {e}. "
                f"The model architecture may have changed since this bundle was saved."
            ) from e

        model.eval()
        return model

    def display_figures(self) -> None:
        """Display all figures in a Jupyter notebook."""
        try:
            from IPython.display import display as ipy_display
        except ImportError:
            print("IPython not available. Access figures via bundle.figures dict.")
            return

        for name, img in self.figures.items():
            print(f"--- {name} ---")
            if isinstance(img, Figure):
                ipy_display(img)
            elif isinstance(img, np.ndarray):
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(img)
                ax.set_axis_off()
                ax.set_title(name)
                fig.tight_layout()
                ipy_display(fig)
                plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def _make_experiment_directory(directory: str) -> str:
    """Append a timestamp and SLURM job ID suffix to the directory name.

    Turns e.g. "results/my_exp" into "results/my_exp_20260401_143012_j12345"
    (or "_local" when not running under SLURM).
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    slurm_id = os.environ.get("SLURM_JOB_ID", "local")
    return f"{directory}_{ts}_j{slurm_id}"


class ExperimentTracker:
    """Manages an experiment directory with periodic checkpointing.

    Creates a unique timestamped directory on ``register_start()``.
    Call ``checkpoint()`` during training (e.g. every epoch) — it will
    save an intermediate bundle to a ``checkpoints/`` subdirectory if
    enough wall-clock time has elapsed since the last save.
    Call ``save_final()`` at the end to write the completed bundle to
    the experiment root.

    Usage::

        tracker = ExperimentTracker(
            directory="results/my_exp",
            name="my_exp",
            model_config={...},
            training_config=config,
            checkpoint_interval_s=3600,
        )
        tracker.register_start()

        for epoch in range(epochs):
            train(...)
            tracker.checkpoint(model, {"history": hist})

        tracker.save_final(model, training_results, metrics, figures)
    """

    def __init__(
        self,
        directory: str,
        name: str,
        model_config: dict,
        training_config: dict,
        checkpoint_interval_s: float = 3600,
        normalization: dict | None = None,
    ):
        self.base_directory = directory
        self.directory: str | None = None
        self.name = name
        self.model_config = model_config
        self.training_config = training_config
        self.checkpoint_interval_s = checkpoint_interval_s
        self.normalization = normalization
        self._start_time: float | None = None
        self._last_checkpoint_time: float | None = None
        self._checkpoint_count = 0
        self._parent: ExperimentTracker | None = None
        self._variant: str | None = None
        self._subexperiments: dict[str, ExperimentTracker] = {}

    def register_start(self) -> str:
        """Create the experiment directory and record the start time.

        For sub-experiments (created via ``make_subexperiment``), the directory
        is a subdirectory of the parent — no timestamp suffix is appended.

        Returns the resolved directory path.
        """
        if self._parent is not None:
            # Sub-experiment: directory was pre-assigned by make_subexperiment
            assert self.directory is not None
            Path(self.directory).mkdir(parents=True, exist_ok=True)
        else:
            if Path(self.base_directory).is_dir():
                self.directory = self.base_directory
            else:
                self.directory = _make_experiment_directory(self.base_directory)
            Path(self.directory).mkdir(parents=True, exist_ok=True)

        self._start_time = time.time()
        self._last_checkpoint_time = self._start_time

        (Path(self.directory) / "started.txt").write_text(
            f"name: {self.name}\n"
            f"started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"slurm_job_id: {os.environ.get('SLURM_JOB_ID', 'local')}\n"
            f"pid: {os.getpid()}\n"
        )

        # Write initial manifest for parent experiments with sub-experiments
        if self._parent is None and self._subexperiments:
            self._write_manifest()

        print(f"[experiment] Started {self.name} → {self.directory}")
        return self.directory

    def checkpoint(
        self,
        model: nn.Module,
        training_results: dict | None = None,
        metrics: dict | None = None,
        force: bool = False,
    ) -> bool:
        """Save a checkpoint if enough time has passed (or force=True).

        Overwrites the previous checkpoint each time (keeps only the latest).
        Returns True if a checkpoint was written.
        """
        assert self.directory is not None, "Call register_start() first"

        now = time.time()
        if not force and (now - self._last_checkpoint_time) < self.checkpoint_interval_s:
            return False

        ckpt_dir = str(Path(self.directory) / "checkpoints")
        elapsed = now - self._start_time

        bundle = ExperimentBundle(
            name=f"{self.name}_ckpt",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_type=_model_type_string(model),
            model_config=self.model_config,
            model_state_dict=model.state_dict(),
            training_config=self.training_config,
            training_results={**(training_results or {}), "train_elapsed_s": elapsed},
            metrics=metrics or {},
            figures={},
            normalization=self.normalization,
        )
        bundle.save(ckpt_dir)

        self._last_checkpoint_time = now
        self._checkpoint_count += 1
        print(f"[ckpt] #{self._checkpoint_count} at {elapsed:.0f}s → {ckpt_dir}")
        return True

    def save_final(
        self,
        model: nn.Module,
        training_results: dict,
        metrics: dict,
        figures: dict[str, Figure],
    ) -> ExperimentBundle:
        """Write the final experiment bundle to the experiment root."""
        assert self.directory is not None, "Call register_start() first"

        bundle = ExperimentBundle(
            name=self.name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_type=_model_type_string(model),
            model_config=self.model_config,
            model_state_dict=model.state_dict(),
            training_config=self.training_config,
            training_results=training_results,
            metrics=metrics,
            figures=figures,
            normalization=self.normalization,
        )
        bundle.save(self.directory)

        # Update parent manifest if this is a sub-experiment
        if self._parent is not None:
            self._parent._write_manifest()

        print(f"[experiment] Final save → {self.directory}")
        return bundle

    def make_subexperiment(
        self,
        variant: str,
        model_config: dict,
        training_config: dict | None = None,
        checkpoint_interval_s: float | None = None,
        normalization: dict | None = None,
    ) -> "ExperimentTracker":
        """Create a sub-experiment that saves into a subdirectory.

        The child tracker works like a normal ``ExperimentTracker`` — it has
        its own ``register_start()``, ``checkpoint()``, and ``save_final()``.
        Its directory is ``self.directory / variant`` (no timestamp suffix).

        Args:
            variant: subdirectory name (e.g. "ar", "baseline").
            model_config: model constructor kwargs for this variant.
            training_config: override parent's training config (default: inherit).
            checkpoint_interval_s: override parent's interval (default: inherit).
            normalization: override parent's normalization (default: inherit).

        Returns:
            A child ExperimentTracker.
        """
        assert self.directory is not None, "Call register_start() on the parent first"

        child = ExperimentTracker(
            directory=str(Path(self.directory) / variant),
            name=f"{self.name}_{variant}",
            model_config=model_config,
            training_config=training_config if training_config is not None else self.training_config,
            checkpoint_interval_s=checkpoint_interval_s if checkpoint_interval_s is not None else self.checkpoint_interval_s,
            normalization=normalization if normalization is not None else self.normalization,
        )
        child._parent = self
        child._variant = variant
        # Pre-assign directory so register_start() doesn't append a timestamp
        child.directory = str(Path(self.directory) / variant)
        self._subexperiments[variant] = child
        return child

    def _write_manifest(self) -> None:
        """Write experiment.json to the experiment directory."""
        assert self.directory is not None

        subexps = {}
        for variant, child in self._subexperiments.items():
            child_dir = Path(child.directory) if child.directory else None
            has_bundle = child_dir is not None and (child_dir / "bundle.pt").exists()
            subexps[variant] = {
                "name": child.name,
                "model_config": _prepare_config_for_save(child.model_config),
                "training_config": child.training_config,
                "status": "completed" if has_bundle else "started",
            }

        manifest = {
            "name": self.name,
            "variants": list(self._subexperiments.keys()),
            "shared_config": self.training_config,
            "created": datetime.now().isoformat(),
            "subexperiments": subexps,
        }

        (Path(self.directory) / "experiment.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )


def save_experiment(
    directory: str,
    model: nn.Module,
    model_config: dict,
    training_config: dict,
    training_results: dict,
    metrics: dict,
    figures: dict[str, Figure],
    name: str = "experiment",
    normalization: dict | None = None,
) -> ExperimentBundle:
    """Save a complete experiment to a directory.

    A timestamp and SLURM job ID are appended to the directory name
    so that concurrent or repeated runs never overwrite each other.

    Args:
        directory: base path to save to (e.g. "results/my_exp")
        model: trained PyTorch model
        model_config: constructor kwargs that reproduce the model
        training_config: hyperparameters (lr, epochs, batch_size, etc.)
        training_results: training outcomes (loss curves, elapsed time, etc.)
        metrics: evaluation metrics dict (from eval_ae or eval_cvae)
        figures: evaluation figures dict (from eval_ae or eval_cvae)
        name: experiment name
        normalization: optional data normalization params (e.g. erk_mu, erk_sigma)

    Returns:
        The saved ExperimentBundle.
    """
    directory = _make_experiment_directory(directory)

    bundle = ExperimentBundle(
        name=name,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_type=_model_type_string(model),
        model_config=model_config,
        model_state_dict=model.state_dict(),
        training_config=training_config,
        training_results=training_results,
        metrics=metrics,
        figures=figures,
        normalization=normalization,
    )
    bundle.save(directory)
    print(f"[save_experiment] Saved to {directory}")
    return bundle


def is_experiment_dir(directory: Path) -> bool:
    """Check whether a directory contains a loadable experiment.

    Recognizes:
    - Standard bundles (``bundle.pt``)
    - In-progress experiments (``checkpoints/bundle.pt``)
    - Grouped experiments (``experiment.json`` with variant subdirs)
    - Legacy format (any ``.pt`` file at the top level)
    """
    if (directory / "slurm.log").exists():
        return True
    if (directory / "notebook.html").exists():
        return True
    if not directory.is_dir():
        return False
    if (directory / "bundle.pt").exists():
        return True
    if (directory / "checkpoints" / "bundle.pt").exists():
        return True
    if (directory / "experiment.json").exists():
        return True
    if any(directory.glob("*.pt")):
        return True
    return False


def discover_experiments(results_path: str | Path) -> list[str]:
    """Return directory names under *results_path* that contain loadable experiments.

    Results are sorted by directory creation time, newest first.
    """
    results_path = Path(results_path)
    if not results_path.is_dir():
        return []
    experiment_subdirs = [d for d in results_path.iterdir() if is_experiment_dir(d)]
    experiment_subdirs.sort(key=lambda d: d.stat().st_birthtime, reverse=True)
    return [d.name for d in experiment_subdirs]


def load_experiment(directory: str) -> ExperimentBundle:
    """Load an experiment from a directory.

    Returns an ExperimentBundle with all available data. Check
    bundle.warnings for any compatibility issues.
    """
    bundle = ExperimentBundle.load(directory)
    if bundle.warnings:
        print(f"Loaded with {len(bundle.warnings)} warning(s):")
        for w in bundle.warnings:
            print(f"  ⚠ {w}")
    return bundle


def load_experiment_group(directory: str) -> dict[str, ExperimentBundle]:
    """Load all sub-experiments from a grouped experiment directory.

    Reads ``experiment.json`` to discover variants, then loads each
    variant's ``ExperimentBundle`` from its subdirectory.

    Returns:
        Dict mapping variant name to ExperimentBundle.

    Raises:
        FileNotFoundError: if ``experiment.json`` is missing.
    """
    d = Path(directory)
    manifest_path = d / "experiment.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No experiment.json in {d}. Use load_experiment() for single-bundle experiments."
        )

    manifest = json.loads(manifest_path.read_text())
    bundles: dict[str, ExperimentBundle] = {}

    for variant in manifest.get("variants", []):
        variant_dir = d / variant
        if variant_dir.is_dir() and (variant_dir / "bundle.pt").exists():
            bundles[variant] = ExperimentBundle.load(str(variant_dir))
        else:
            print(f"  ⚠ Variant '{variant}' listed in manifest but not found on disk")

    if bundles:
        print(f"Loaded {len(bundles)} variant(s) from {d.name}: {list(bundles.keys())}")
    return bundles
