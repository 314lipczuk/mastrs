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

from experiment import ExperimentBundle, ExperimentTracker, save_experiment, load_experiment


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
    bundle = save_experiment(
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
    # save_experiment appends a timestamp+job suffix to the directory
    return bundle.save_dir


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
# Tests: ExperimentTracker
# ---------------------------------------------------------------------------

class TestExperimentTracker:
    def _make_tracker(self, tmp_path, interval_s=3600):
        return ExperimentTracker(
            directory=str(tmp_path / "tracked_exp"),
            name="tracked_exp",
            model_config={"input_dim": 4, "hidden_dim": 8},
            training_config={"lr": 1e-3, "epochs": 10},
            checkpoint_interval_s=interval_s,
        )

    def test_register_start_creates_directory(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        result_dir = tracker.register_start()
        assert Path(result_dir).is_dir()
        assert "tracked_exp" in result_dir

    def test_register_start_creates_started_txt(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        result_dir = tracker.register_start()
        started = Path(result_dir) / "started.txt"
        assert started.exists()
        content = started.read_text()
        assert "name: tracked_exp" in content
        assert "pid:" in content

    def test_directory_has_timestamp_suffix(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        result_dir = tracker.register_start()
        # Should have _YYYYMMDD_HHMMSS_jlocal suffix
        assert "_jlocal" in result_dir or "_j" in result_dir

    def test_checkpoint_before_register_raises(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        model = _TinyModel()
        with pytest.raises(AssertionError, match="register_start"):
            tracker.checkpoint(model)

    def test_checkpoint_skipped_when_interval_not_elapsed(self, tmp_path):
        tracker = self._make_tracker(tmp_path, interval_s=3600)
        tracker.register_start()
        model = _TinyModel()
        saved = tracker.checkpoint(model, training_results={"loss": 0.5})
        assert saved is False
        assert tracker._checkpoint_count == 0

    def test_checkpoint_force_saves_immediately(self, tmp_path):
        tracker = self._make_tracker(tmp_path, interval_s=3600)
        tracker.register_start()
        model = _TinyModel()
        saved = tracker.checkpoint(model, training_results={"loss": 0.5}, force=True)
        assert saved is True
        assert tracker._checkpoint_count == 1
        ckpt_dir = Path(tracker.directory) / "checkpoints"
        assert ckpt_dir.is_dir()
        assert (ckpt_dir / "bundle.pt").exists()

    def test_checkpoint_saves_when_interval_elapsed(self, tmp_path):
        tracker = self._make_tracker(tmp_path, interval_s=0)
        tracker.register_start()
        model = _TinyModel()
        saved = tracker.checkpoint(model, training_results={"loss": 0.5})
        assert saved is True
        assert tracker._checkpoint_count == 1

    def test_checkpoint_overwrites_previous(self, tmp_path):
        tracker = self._make_tracker(tmp_path, interval_s=0)
        tracker.register_start()
        model = _TinyModel()

        tracker.checkpoint(model, training_results={"epoch": 1}, force=True)
        tracker.checkpoint(model, training_results={"epoch": 5}, force=True)
        assert tracker._checkpoint_count == 2

        # Load the checkpoint — should have the latest training_results
        ckpt_dir = Path(tracker.directory) / "checkpoints"
        bundle = load_experiment(str(ckpt_dir))
        assert bundle.training_results["epoch"] == 5

    def test_checkpoint_contains_elapsed_time(self, tmp_path):
        tracker = self._make_tracker(tmp_path, interval_s=0)
        tracker.register_start()
        model = _TinyModel()
        tracker.checkpoint(model, training_results={"loss": 0.1}, force=True)

        ckpt_dir = Path(tracker.directory) / "checkpoints"
        bundle = load_experiment(str(ckpt_dir))
        assert "train_elapsed_s" in bundle.training_results
        assert bundle.training_results["train_elapsed_s"] >= 0

    def test_checkpoint_preserves_model_weights(self, tmp_path):
        tracker = self._make_tracker(tmp_path, interval_s=0)
        tracker.register_start()
        model = _TinyModel()

        # Record weights before checkpoint
        weights_before = model.fc.weight.clone()
        tracker.checkpoint(model, force=True)
        # Weights should be unchanged after checkpoint
        assert torch.equal(model.fc.weight, weights_before)

    def test_save_final_writes_to_root(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.register_start()
        model = _TinyModel()
        fig = _make_figure("final")

        bundle = tracker.save_final(
            model=model,
            training_results={"loss": 0.01},
            metrics={"mse": 0.05},
            figures={"loss_curve": fig},
        )
        plt.close(fig)

        assert (Path(tracker.directory) / "bundle.pt").exists()
        assert (Path(tracker.directory) / "figures" / "loss_curve.png").exists()
        assert bundle.name == "tracked_exp"
        assert bundle.metrics["mse"] == pytest.approx(0.05)

    def test_save_final_before_register_raises(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        model = _TinyModel()
        with pytest.raises(AssertionError, match="register_start"):
            tracker.save_final(model, {}, {}, {})

    def test_checkpoint_and_final_coexist(self, tmp_path):
        tracker = self._make_tracker(tmp_path, interval_s=0)
        tracker.register_start()
        model = _TinyModel()

        tracker.checkpoint(model, training_results={"epoch": 3}, force=True)

        fig = _make_figure("final")
        tracker.save_final(model, {"epoch": 10}, {"mse": 0.01}, {"fig": fig})
        plt.close(fig)

        # Both should exist
        root = Path(tracker.directory)
        assert (root / "bundle.pt").exists()
        assert (root / "checkpoints" / "bundle.pt").exists()

        # Final bundle has final data
        final = load_experiment(str(root))
        assert final.training_results["epoch"] == 10

        # Checkpoint has intermediate data
        ckpt = load_experiment(str(root / "checkpoints"))
        assert ckpt.training_results["epoch"] == 3

    def test_two_trackers_get_different_directories(self, tmp_path):
        t1 = self._make_tracker(tmp_path)
        d1 = t1.register_start()
        # Small delay not needed — timestamp includes seconds, but
        # _make_experiment_directory uses the same base so they should differ
        # if called at different times. Force a different name to be safe.
        import time as _time
        _time.sleep(1)
        t2 = self._make_tracker(tmp_path)
        d2 = t2.register_start()
        assert d1 != d2


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
