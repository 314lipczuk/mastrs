"""Model runtime: load the checkpoint, advance the online encoder, and invert
the CNR predictor into a per-cell exposure via a short-horizon search.

Two interchangeable engines behind a common ``decide(frames) -> [ms]`` seam:

  * :class:`RealModelEngine` — wraps a trained ``Seq2ScalarHistory``. Per call it
    (1) advances each cell's encoder LSTM by exactly one step with the carried
    ``(h, c)`` (equivalent to full-history encoding), then (2) runs the control
    law: for a grid of candidate exposures it rolls the decoder forward
    ``control_horizon`` steps and picks the exposure whose predicted CNR best
    tracks ``target_cnr``.

  * :class:`StubEngine` — no neural net; a deterministic proportional policy
    (``exposure = gain * max(0, target - cnr)``) so faro can integrate before a
    real checkpoint exists.

Use :func:`load_engine` to get whichever is appropriate for the config. All
engine methods assume the caller holds the service lock (torch models and the
state tensors are not thread-safe).
"""
from __future__ import annotations
from pprint import pprint

import time
from dataclasses import dataclass

import numpy as np
import torch

from optoerk.serving.calibration import FluenceCalibration
from optoerk.serving.config import ServerConfig
from optoerk.serving.state import CellState

CNR = 0  # channel index of cnr (first in norm_channels: [cnr, u_t, ...])
FLU = 1  # channel index of u_t (fluence; second)


@dataclass
class CellFrame:
    """One cell's inputs for the current frame (baseline-normalized cnr)."""
    state: CellState
    cnr_norm: float
    fov_density: float
    n_cells_200px: float


# ---------------------------------------------------------------------------
# model / norm-stats loading
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_norm_stats(model) -> tuple[np.ndarray, np.ndarray]:
    """Frozen standardization stats, preferring those carried on the config."""
    cfg = model.cfg
    mean = np.asarray(getattr(cfg, "norm_mean", []) or [], np.float32)
    std = np.asarray(getattr(cfg, "norm_std", []) or [], np.float32)
    if mean.size and mean.size == std.size:
        return mean, std
    # Fall back to the checked-in train-population stats file.
    from optoerk.data.history_dataset import NormStats

    stats = NormStats.load()
    return np.asarray(stats.mean, np.float32), np.asarray(stats.std, np.float32)


def load_engine(cfg: ServerConfig):
    """Return a (engine, info) pair. Falls back to the stub on any load failure."""
    calib = FluenceCalibration(cfg.instrument, cfg.stim_power_pct)
    if cfg.checkpoint_dir:
        try:
            from optoerk.core.experiment import load_experiment

            bundle = load_experiment(cfg.checkpoint_dir)
            model = bundle.reconstruct_model()
            device = _resolve_device(cfg.device)
            model.to(device).eval()
            mean, std = _load_norm_stats(model)
            engine = RealModelEngine(model, mean, std, calib, cfg, device)
            info = {
                "model_loaded": True,
                "model_type": bundle.model_type,
                "checkpoint_dir": cfg.checkpoint_dir,
                "device": str(device),
                "future_len": int(model.cfg.future_len),
                "norm_channels": list(getattr(model.cfg, "norm_channels", []))
                or ["cnr", "u_t", "fov_density", "n_cells_200px"],
            }
            if cfg.warmup and device.type == "cuda":
                warmup_engine(engine)
            return engine, info
        except Exception as e:  # noqa: BLE001 - degrade gracefully to the stub
            print(f"[serving] checkpoint load failed ({e!r}); using STUB policy")
    engine = StubEngine(calib, cfg)
    return engine, {"model_loaded": False, "policy": "stub", "checkpoint_dir": cfg.checkpoint_dir}


def warmup_engine(engine, batch_sizes: tuple[int, ...] = (1, 8, 32, 64)) -> None:
    """Prime the GPU before the first real frame by running throwaway ``decide``
    calls across a few batch sizes. Triggers CUDA context creation, cuDNN
    autotune and allocator-pool growth up front — the cold-start work that
    otherwise lands on the first predictions. Best-effort; never raises."""
    try:
        t0 = time.perf_counter()
        for n in batch_sizes:
            frames = [
                CellFrame(state=CellState(), cnr_norm=1.0, fov_density=float(n),
                          n_cells_200px=1.0)
                for _ in range(n)
            ]
            engine.decide(frames)
        dev = getattr(engine, "device", None)
        if dev is not None and dev.type == "cuda":
            torch.cuda.synchronize(dev)
        print(f"[serving] warmup: {len(batch_sizes)} batches in {time.perf_counter() - t0:.2f}s")
    except Exception as e:  # noqa: BLE001 - warmup is optional, never fatal
        print(f"[serving] warmup skipped: {e!r}")


# ---------------------------------------------------------------------------
# real model engine
# ---------------------------------------------------------------------------

class RealModelEngine:
    def __init__(self, model, mean, std, calib: FluenceCalibration, cfg: ServerConfig, device):
        self.model = model
        self.calib = calib
        self.cfg = cfg
        self.device = device
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device)  # (C,)
        self.std = torch.tensor(std, dtype=torch.float32, device=device)
        self.mean_np = np.asarray(mean, np.float32)
        # Model's channel order; unsupplied channels default to the pop. mean.
        self.channels = list(getattr(model.cfg, "norm_channels", []) or
                             ["cnr", "u_t", "fov_density", "n_cells_200px"])
        # Optional server-side override of the optoRTK-expression channel: feed a
        # fixed raw value for every cell (ignoring the payload). Defaults to the
        # channel's population mean (history_norm_stats.json), i.e. neutral.
        self._optortk_channel = "optortk_expr"
        self._optortk_override: float | None = None
        if cfg.override_optortk_expr and self._optortk_channel in self.channels:
            idx = self.channels.index(self._optortk_channel)
            val = cfg.optortk_expr_value
            self._optortk_override = float(self.mean_np[idx] if val is None else val)
        # Value actually fed on the optoRTK channel (override, else the channel
        # population mean); exposed for the prediction log. None if unused.
        if self._optortk_channel in self.channels:
            _idx = self.channels.index(self._optortk_channel)
            self.optortk_fed: float | None = (
                self._optortk_override
                if self._optortk_override is not None
                else float(self.mean_np[_idx])
            )
        else:
            self.optortk_fed = None
        self.num_layers = model.cfg.num_layers
        self.hidden = model.cfg.hidden_dim
        self.horizon = min(cfg.control_horizon, int(model.cfg.future_len))
        # Candidate exposure grid (ms) -> fluence (mJ/cm2) -> standardized u_t.
        ms_grid = np.linspace(cfg.min_exposure_ms, cfg.max_exposure_ms, cfg.n_candidates)
        self.cand_ms = torch.tensor(ms_grid, dtype=torch.float32, device=device)  # (M,)
        cand_flu = self.calib.ms_to_fluence(ms_grid)  # (M,) mJ/cm2
        cand_flu_std = (cand_flu - float(mean[FLU])) / float(std[FLU])
        self.cand_flu_std = torch.tensor(cand_flu_std, dtype=torch.float32, device=device)
        self.target_std = (cfg.target_cnr - float(mean[CNR])) / float(std[CNR])

    @torch.no_grad()
    def decide(self, frames: list[CellFrame]) -> list[float]:
        #pprint(self)
        #pprint(self.device)
        #pprint(self.model)
        if not frames:
            return []
        N = len(frames)
        M = self.cand_ms.shape[0]
        model = self.model
        L, H = self.num_layers, self.hidden

        # --- raw per-frame channels, assembled by NAME in the model's channel
        # order (robust to CHANNELS growing/reordering). Channels the live payload
        # doesn't carry — currently optortk_expr — default to the population mean
        # (→ 0 after standardizing, i.e. neutral median expression).
        # TODO: feed the real session-relative optoRTK expression rank once the
        # server accumulates per-cell baseline C0 (see NIESEN_TOCHECK.md).
        _supplied = {
            "cnr": lambda f: f.cnr_norm,
            "u_t": lambda f: f.state.last_fluence,
            "fov_density": lambda f: f.fov_density,
            "n_cells_200px": lambda f: f.n_cells_200px,
        }
        # Server-side hardcode of the optoRTK-expression channel, if enabled.
        if self._optortk_override is not None:
            _supplied[self._optortk_channel] = lambda f, v=self._optortk_override: v
        raw = torch.tensor(
            [[_supplied[name](f) if name in _supplied else float(self.mean_np[i])
              for i, name in enumerate(self.channels)] for f in frames],
            dtype=torch.float32, device=self.device,
        )  # (N, C)
        xs = (raw - self.mean) / self.std  # standardized (N, C)

        # --- carried encoder state (zeros for first-seen cells) --------------
        h = torch.zeros(L, N, H, device=self.device)
        c = torch.zeros(L, N, H, device=self.device)
        for i, f in enumerate(frames):
            if f.state.h is not None:
                h[:, i : i + 1] = f.state.h
                c[:, i : i + 1] = f.state.c

        # --- ADVANCE THE ENCODER BY EXACTLY ONE STEP -------------------------
        # nn.LSTM with a length-1 sequence and carried (h, c) == encoding the
        # whole causal past (verified equivalent to pack_padded full encode).
        _, (h_new, c_new) = model.encoder.lstm(xs.unsqueeze(1), (h, c))  # (L, N, H)

        # --- control: score candidate exposures via the decoder rollout ------
        cnr_fb = xs[:, CNR : CNR + 1]  # (N,1) feedback = cnr at last real frame
        # expand each cell across the M candidates -> batch of N*M rollouts.
        h_b = h_new.repeat_interleave(M, dim=1)   # (L, N*M, H)
        c_b = c_new.repeat_interleave(M, dim=1)
        cnr_fb_b = cnr_fb.repeat_interleave(M, dim=0)              # (N*M, 1)
        flu_b = self.cand_flu_std.repeat(N).unsqueeze(-1)          # (N*M, 1)
        fut = flu_b.unsqueeze(1).expand(-1, self.horizon, -1)     # (N*M, horizon, 1) constant dose

        pred_std = self._rollout(h_b, c_b, cnr_fb_b, fut)          # (N*M, horizon) std cnr
        cost = ((pred_std - self.target_std) ** 2).mean(dim=1)     # (N*M,)
        cost = cost.view(N, M)
        best = torch.argmin(cost, dim=1)                          # (N,)
        best_ms = self.cand_ms[best]                              # (N,)

        # --- persist new encoder state + commanded fluence -------------------
        out_ms: list[float] = []
        for i, f in enumerate(frames):
            f.state.h = h_new[:, i : i + 1].detach().clone()
            f.state.c = c_new[:, i : i + 1].detach().clone()
            ms = float(best_ms[i].item())
            f.state.last_fluence = float(self.calib.ms_to_fluence(ms))
            out_ms.append(ms)
        return out_ms

    def _rollout(self, h, c, cnr_fb, fut) -> torch.Tensor:
        """Decoder-only rollout from a given (h, c); mirrors
        ``Seq2ScalarHistory.forward`` (free-running feedback, no teacher forcing).
        Returns standardized predicted CNR mean over the horizon, shape (B, F)."""
        model = self.model
        dh, dc = h, c
        means = []
        F = fut.shape[1]
        for i in range(F):
            flu_i = fut[:, i, :]                                  # (B, stim_dim=1)
            dec_in = torch.cat([cnr_fb, flu_i], dim=-1).unsqueeze(1)
            out, (dh, dc) = model.decoder(dec_in, (dh, dc))
            h_step = out[:, -1, :]
            if model.film_layer is not None:                     # default: none
                gamma, beta = model.film_layer(flu_i)
                if model.cfg.film == "output":
                    h_step = gamma * h_step + beta
                else:
                    dh = gamma.unsqueeze(0) * dh + beta.unsqueeze(0)
                    h_step = dh[-1]
            sigma_bias = (
                model.sigma_step_bias_param[i]
                if model.sigma_step_bias_param is not None else 0.0
            )
            feats = model.trunk(torch.cat([h_step, flu_i], dim=-1))
            pi, mu, _sigma = model.head(feats, sigma_bias=sigma_bias)
            pred = (pi * mu).sum(dim=-1, keepdim=True)            # (B,1) std cnr
            means.append(pred)
            cnr_fb = pred
        return torch.cat(means, dim=1)


# ---------------------------------------------------------------------------
# stub engine (no model) — deterministic, runnable before a checkpoint exists
# ---------------------------------------------------------------------------

class StubEngine:
    """Proportional placeholder: dose up cells whose CNR is below target."""

    def __init__(self, calib: FluenceCalibration, cfg: ServerConfig):
        self.calib = calib
        self.cfg = cfg
        self.optortk_fed: float | None = None  # stub ignores the optoRTK channel

    def decide(self, frames: list[CellFrame]) -> list[float]:
        cfg = self.cfg
        out: list[float] = []
        for f in frames:
            deficit = max(0.0, cfg.target_cnr - f.cnr_norm)
            ms = cfg.stub_gain_ms_per_cnr * deficit
            ms = float(np.clip(ms, cfg.min_exposure_ms, cfg.max_exposure_ms))
            f.state.last_fluence = float(self.calib.ms_to_fluence(ms))
            out.append(ms)
        return out
