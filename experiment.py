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
import contextlib
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
        if self.training_results:
            print("  Training results:")
            for k, v in self.training_results.items():
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

    Args:
        directory: path to save to (e.g. "results/my_exp")
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
    return bundle


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
