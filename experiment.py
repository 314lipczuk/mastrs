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

        # Load bundle.pt
        bundle_path = d / "bundle.pt"
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"No bundle.pt found in {directory}. "
                f"This directory may use an older save format."
            )

        data = torch.load(bundle_path, map_location="cpu", weights_only=False)

        # Load figures from PNGs (best-effort)
        figures: dict[str, np.ndarray] = {}
        fig_dir = d / "figures"
        if fig_dir.is_dir():
            for png in sorted(fig_dir.glob("*.png")):
                try:
                    figures[png.stem] = mpimg.imread(str(png))
                except Exception as e:
                    warnings.append(f"Could not load figure '{png.name}': {e}")

        # Validate model class is importable
        model_type = data.get("model_type", "")
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
        model_state_dict = data.get("model_state_dict")
        if model_state_dict is None:
            warnings.append("No model_state_dict found in bundle. Model reconstruction not possible.")

        return cls(
            name=data.get("name", d.name),
            timestamp=data.get("timestamp", "unknown"),
            model_type=model_type,
            model_config=data.get("model_config", {}),
            model_state_dict=model_state_dict,
            training_config=data.get("training_config", {}),
            training_results=data.get("training_results", {}),
            metrics=data.get("metrics", {}),
            figures=figures,
            normalization=data.get("normalization"),
            warnings=warnings,
        )

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
