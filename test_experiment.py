"""Tests for experiment save/load round-trip and legacy format loading."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from experiment import ExperimentBundle, save_experiment, load_experiment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Minimal model for testing."""
    def __init__(self, input_dim: int = 4, hidden_dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x)


def _make_figure(title: str = "test") -> Figure:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title(title)
    return fig


@pytest.fixture
def bundle_dir(tmp_path):
    """Save a bundle via save_experiment and return the directory path."""
    model = _TinyModel(input_dim=4, hidden_dim=8)
    model_config = {"input_dim": 4, "hidden_dim": 8}
    training_config = {"lr": 1e-3, "epochs": 100, "batch_size": 32}
    training_results = {"train_loss": [0.5, 0.3, 0.1], "best_epoch": 2}
    metrics = {"mse_overall": 0.05, "per_sample": np.array([0.04, 0.06])}
    figures = {"loss_curve": _make_figure("Loss"), "scatter": _make_figure("Scatter")}

    d = str(tmp_path / "test_exp")
    save_experiment(
        directory=d,
        model=model,
        model_config=model_config,
        training_config=training_config,
        training_results=training_results,
        metrics=metrics,
        figures=figures,
        name="test_exp",
        normalization={"erk_mu": 1.5, "erk_sigma": 0.8},
    )
    plt.close("all")
    return d


@pytest.fixture
def legacy_dir(tmp_path):
    """Create a legacy-format experiment directory (model_*.pt + metrics.npz + scalars.json)."""
    d = tmp_path / "legacy_exp"
    d.mkdir()

    # model checkpoint (legacy format)
    torch.save({
        "model_state_dict": {"fc.weight": torch.randn(8, 4), "fc.bias": torch.randn(8)},
        "history": {"train": [0.5, 0.3], "val": [0.6, 0.4]},
        "config": {"hidden_dim": 8, "latent_dim": 3, "seq_length": 40},
        "training_kwargs": {"lr": 0.001, "epochs": 200, "batch_size": 64},
        "train_start": "2026-03-10T10:00:00",
        "train_elapsed_s": 12.5,
        "erk_mu": 1.6,
        "erk_sigma": 0.75,
        "device": "cpu",
    }, d / "model_20260310_100000.pt")

    # metrics
    np.savez(d / "metrics.npz", mse_per_cell=np.array([0.1, 0.2, 0.15]))
    with open(d / "scalars.json", "w") as f:
        json.dump({"n_active_dims": 3.0, "condition_clf_acc": 0.95}, f)

    # figures
    fig_dir = d / "figures"
    fig_dir.mkdir()
    fig = _make_figure("test_legacy")
    fig.savefig(fig_dir / "recon.png", dpi=72)
    plt.close(fig)

    return str(d)


# ---------------------------------------------------------------------------
# Tests: bundle.pt format (round-trip)
# ---------------------------------------------------------------------------

class TestBundleRoundTrip:
    def test_bundle_pt_exists(self, bundle_dir):
        assert (Path(bundle_dir) / "bundle.pt").exists()

    def test_figures_saved_as_png(self, bundle_dir):
        fig_dir = Path(bundle_dir) / "figures"
        pngs = sorted(f.stem for f in fig_dir.glob("*.png"))
        assert "loss_curve" in pngs
        assert "scatter" in pngs

    def test_summary_txt_exists(self, bundle_dir):
        assert (Path(bundle_dir) / "summary.txt").exists()

    def test_load_name(self, bundle_dir):
        bundle = load_experiment(bundle_dir)
        assert bundle.name == "test_exp"

    def test_load_timestamp(self, bundle_dir):
        bundle = load_experiment(bundle_dir)
        assert bundle.timestamp != "unknown"

    def test_load_model_config(self, bundle_dir):
        bundle = load_experiment(bundle_dir)
        assert bundle.model_config["input_dim"] == 4
        assert bundle.model_config["hidden_dim"] == 8

    def test_load_training_config(self, bundle_dir):
        bundle = load_experiment(bundle_dir)
        assert bundle.training_config["lr"] == 1e-3
        assert bundle.training_config["epochs"] == 100
        assert bundle.training_config["batch_size"] == 32

    def test_load_training_results(self, bundle_dir):
        bundle = load_experiment(bundle_dir)
        assert bundle.training_results["best_epoch"] == 2
        assert len(bundle.training_results["train_loss"]) == 3

    def test_load_metrics(self, bundle_dir):
        bundle = load_experiment(bundle_dir)
        assert bundle.metrics["mse_overall"] == pytest.approx(0.05)
        assert isinstance(bundle.metrics["per_sample"], np.ndarray)
        assert len(bundle.metrics["per_sample"]) == 2

    def test_load_normalization(self, bundle_dir):
        bundle = load_experiment(bundle_dir)
        assert bundle.normalization["erk_mu"] == pytest.approx(1.5)
        assert bundle.normalization["erk_sigma"] == pytest.approx(0.8)

    def test_load_figures(self, bundle_dir):
        bundle = load_experiment(bundle_dir)
        assert "loss_curve" in bundle.figures
        assert "scatter" in bundle.figures
        for fig in bundle.figures.values():
            assert isinstance(fig, np.ndarray)
            assert fig.ndim == 3  # H x W x channels

    def test_load_model_state_dict(self, bundle_dir):
        bundle = load_experiment(bundle_dir)
        assert "fc.weight" in bundle.model_state_dict
        assert "fc.bias" in bundle.model_state_dict

    def test_load_no_warnings(self, bundle_dir):
        bundle = load_experiment(bundle_dir)
        assert bundle.warnings == []

    def test_reconstruct_model(self, bundle_dir):
        bundle = load_experiment(bundle_dir)
        model = bundle.reconstruct_model()
        assert isinstance(model, _TinyModel)
        assert not model.training  # eval mode
        # Verify weights match
        assert torch.equal(model.fc.weight, bundle.model_state_dict["fc.weight"])


# ---------------------------------------------------------------------------
# Tests: legacy format
# ---------------------------------------------------------------------------

class TestLegacyFormat:
    def test_load_name(self, legacy_dir):
        bundle = load_experiment(legacy_dir)
        assert bundle.name == "legacy_exp"

    def test_load_timestamp(self, legacy_dir):
        bundle = load_experiment(legacy_dir)
        assert bundle.timestamp == "2026-03-10T10:00:00"

    def test_load_model_config(self, legacy_dir):
        bundle = load_experiment(legacy_dir)
        assert bundle.model_config["hidden_dim"] == 8
        assert bundle.model_config["latent_dim"] == 3

    def test_load_training_config(self, legacy_dir):
        bundle = load_experiment(legacy_dir)
        assert bundle.training_config["lr"] == 0.001
        assert bundle.training_config["epochs"] == 200

    def test_load_training_results(self, legacy_dir):
        bundle = load_experiment(legacy_dir)
        assert "history" in bundle.training_results
        assert bundle.training_results["train_elapsed_s"] == pytest.approx(12.5)

    def test_load_scalar_metrics(self, legacy_dir):
        bundle = load_experiment(legacy_dir)
        assert bundle.metrics["n_active_dims"] == pytest.approx(3.0)
        assert bundle.metrics["condition_clf_acc"] == pytest.approx(0.95)

    def test_load_array_metrics(self, legacy_dir):
        bundle = load_experiment(legacy_dir)
        assert isinstance(bundle.metrics["mse_per_cell"], np.ndarray)
        np.testing.assert_allclose(bundle.metrics["mse_per_cell"], [0.1, 0.2, 0.15])

    def test_load_normalization(self, legacy_dir):
        bundle = load_experiment(legacy_dir)
        assert bundle.normalization["erk_mu"] == pytest.approx(1.6)
        assert bundle.normalization["erk_sigma"] == pytest.approx(0.75)

    def test_load_figures(self, legacy_dir):
        bundle = load_experiment(legacy_dir)
        assert "recon" in bundle.figures

    def test_load_model_state_dict(self, legacy_dir):
        bundle = load_experiment(legacy_dir)
        assert "fc.weight" in bundle.model_state_dict
        assert bundle.model_state_dict["fc.weight"].shape == (8, 4)

    def test_no_bundle_pt_needed(self, legacy_dir):
        assert not (Path(legacy_dir) / "bundle.pt").exists()


# ---------------------------------------------------------------------------
# Tests: real experiment directories (smoke tests)
# ---------------------------------------------------------------------------

_RESULTS_DIR = Path(__file__).parent / "results"

@pytest.mark.skipif(not _RESULTS_DIR.is_dir(), reason="No results/ directory")
class TestRealExperiments:
    @pytest.fixture(params=[
        d.name for d in sorted(_RESULTS_DIR.iterdir())
        if d.is_dir() and any(d.glob("*.pt"))
    ])
    def exp_dir(self, request):
        return str(_RESULTS_DIR / request.param)

    def test_loads_without_error(self, exp_dir):
        bundle = load_experiment(exp_dir)
        assert bundle.name

    def test_has_training_config(self, exp_dir):
        bundle = load_experiment(exp_dir)
        assert isinstance(bundle.training_config, dict)
        assert len(bundle.training_config) > 0, f"training_config is empty for {exp_dir}"

    def test_has_metrics(self, exp_dir):
        bundle = load_experiment(exp_dir)
        assert isinstance(bundle.metrics, dict)
        assert len(bundle.metrics) > 0, f"metrics is empty for {exp_dir}"

    def test_has_model_state_dict(self, exp_dir):
        bundle = load_experiment(exp_dir)
        assert bundle.model_state_dict is not None
