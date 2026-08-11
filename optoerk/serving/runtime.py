"""Model runtime: load the checkpoint and advance the online per-cell encoder.

This module is *model plumbing*. It owns:

  * loading a checkpoint bundle (with a process-wide cache, so several FOVs
    sharing a checkpoint share one loaded model),
  * assembling the standardized input channels for a frame,
  * advancing each cell's encoder LSTM by exactly one step with the carried
    ``(h, c)`` — numerically identical to re-encoding the whole causal past,
  * the decoder rollout, exposed as a *plant* interface.

It does **not** decide what dose to command. "What are we aiming for" lives in
:mod:`optoerk.serving.objectives` and "how do we search for the dose" lives in
:mod:`optoerk.serving.control`; ``RealModelEngine.decide`` just wires the two to
the plant.

Two interchangeable engines behind a common ``decide(frames, ctx) -> [ms]`` seam:
:class:`RealModelEngine` (a trained ``Seq2ScalarHistory``) and :class:`StubEngine`
(no neural net; a deterministic proportional policy so faro can integrate before a
real checkpoint exists). All engine methods assume the caller holds the service
lock — torch models and the state tensors are not thread-safe.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from optoerk.serving.calibration import FluenceCalibration
from optoerk.serving.config import ServerConfig
from optoerk.serving.control import Controller, dose_levels
from optoerk.serving.objectives import GoalContext, Objective
from optoerk.serving.state import CellState

CNR = 0  # channel index of cnr (first in norm_channels: [cnr, u_t, ...])
FLU = 1  # channel index of u_t (fluence; second)

DEFAULT_CHANNELS = ["cnr", "u_t", "fov_density", "n_cells_200px"]


@dataclass
class CellFrame:
    """One cell's inputs for the current frame (baseline-normalized cnr).

    ``x``/``y`` are the payload's centroid — not model inputs, but objectives gate
    on them (e.g. "only stimulate the right half of the field"). They default to
    NaN so a cell with no reported position fails any position predicate rather
    than silently passing it.
    """
    state: CellState
    cnr_norm: float
    fov_density: float
    n_cells_200px: float
    x: float = float("nan")
    y: float = float("nan")
    # Session-relative optoRTK expression rank, when the server is reconstructing
    # it live. None means "not available", and the engine then falls back to the
    # channel's population mean exactly as it always has.
    optortk_expr: float | None = None
    # Nuclear area, for checkpoints whose feature set includes it. Same fallback.
    nuc_area: float | None = None


# ---------------------------------------------------------------------------
# model / norm-stats loading
# ---------------------------------------------------------------------------


def _resolve_device(device: str) -> torch.device:
    """Resolve a device string, failing LOUDLY on one that is not available.

    An explicitly requested device used to be returned unchecked, so
    ``--device cuda`` against a CPU-only torch build blew up later inside
    ``load_model``, was swallowed by the degrade-to-stub handler, and surfaced as
    ``checkpoint load failed ('Torch not compiled with CUDA enabled')`` — which
    blames the checkpoint for a problem with the install. Worse, every FOV then
    degraded to the stub, so the server came up "working" with no model at all.

    A device that does not exist is a configuration error for the whole process,
    not a per-FOV fallback, so it is raised here with the fix in the message.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        built = torch.version.cuda
        raise RuntimeError(
            f"--device cuda requested but this torch cannot use CUDA "
            f"(torch {torch.__version__}, "
            f"{'no CUDA build' if built is None else f'CUDA build {built}'}, "
            f"torch.cuda.is_available() is False).\n"
            f"  * No CUDA build: pyproject pins a bare `torch`, so uv installs "
            f"PyPI's default wheel — CPU-only on Windows. Install the CUDA wheel "
            f"for this machine, or serve from a GPU node and point faro at it "
            f"over HTTP.\n"
            f"  * CUDA build present but unavailable: the driver is missing or "
            f"the GPU is not visible — check `nvidia-smi` and "
            f"CUDA_VISIBLE_DEVICES.\n"
            f"  * To benchmark this machine as it actually is, pass --device cpu; "
            f"that is the honest number if this is what will serve."
        )
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            f"--device mps requested but MPS is unavailable (torch "
            f"{torch.__version__}). Use --device cpu."
        )
    return dev


def _resolve_checkpoint(checkpoint_dir: str) -> str:
    """Resolve a bundle name or path to a directory that exists.

    Policy files name bundles bare (``seq2scal_history_raw_cnr_f30_...``) rather
    than by absolute path, so the same file works on the cluster and on a laptop
    with the Kingston mount. A bare name is therefore resolved against
    ``results_write_path()``. An explicit path is used as given.

    Raises with BOTH attempted locations rather than letting the loader report
    only the bare name — a checkpoint that fails to resolve degrades that FOV to
    the stub, and "no .pt files found in <bare name>" does not make it obvious
    that the name was never resolved against the results directory at all.
    """
    from pathlib import Path

    from optoerk.core.utils import results_write_path

    p = Path(checkpoint_dir)
    if p.exists():
        return str(p)
    if not p.is_absolute():
        cand = Path(results_write_path()) / checkpoint_dir
        if cand.exists():
            return str(cand)
        raise FileNotFoundError(
            f"checkpoint {checkpoint_dir!r} not found: tried {p} (relative to "
            f"cwd {Path.cwd()}) and {cand} (results_write_path)"
        )
    raise FileNotFoundError(f"checkpoint {checkpoint_dir!r} not found at {p}")


def _load_norm_stats(model) -> tuple[np.ndarray, np.ndarray]:
    """Frozen standardization stats, preferring those carried on the config.

    The fallback file is keyed to the model's ``cnr_mode`` (baseline-normalized vs
    raw CNR have different cnr-channel mean/std), so a raw-CNR model never falls
    back onto the norm-CNR stats.
    """
    cfg = model.cfg
    mean = np.asarray(getattr(cfg, "norm_mean", []) or [], np.float32)
    std = np.asarray(getattr(cfg, "norm_std", []) or [], np.float32)
    if mean.size and mean.size == std.size:
        return mean, std
    # Fall back to the checked-in train-population stats file for this cnr_mode.
    from optoerk.data.history_dataset import NormStats

    stats = NormStats.load(cnr_mode=getattr(cfg, "cnr_mode", "norm"))
    return np.asarray(stats.mean, np.float32), np.asarray(stats.std, np.float32)


@dataclass
class ModelHandle:
    """A loaded checkpoint plus its frozen norm stats, shareable across FOVs."""
    model: object
    mean: np.ndarray
    std: np.ndarray
    device: torch.device
    info: dict = field(default_factory=dict)


def load_model(checkpoint_dir: str, device: str, cache: dict | None = None) -> ModelHandle:
    """Load a bundle into an eval-mode model. Raises on failure — callers decide
    whether to degrade to the stub.

    ``cache`` (keyed by ``(checkpoint_dir, device)``) lets several FOV policies
    share one loaded model and one warmup instead of paying for a copy each.
    """
    key = (checkpoint_dir, device)
    if cache is not None and key in cache:
        return cache[key]

    from optoerk.core.experiment import load_experiment

    bundle = load_experiment(_resolve_checkpoint(checkpoint_dir))
    model = bundle.reconstruct_model()
    dev = _resolve_device(device)
    model.to(dev).eval()
    mean, std = _load_norm_stats(model)
    handle = ModelHandle(
        model=model, mean=mean, std=std, device=dev,
        info={
            "model_type": bundle.model_type,
            "checkpoint_dir": checkpoint_dir,
            "device": str(dev),
            "future_len": int(model.cfg.future_len),
            "cnr_mode": getattr(model.cfg, "cnr_mode", "norm"),
            "norm_channels": list(getattr(model.cfg, "norm_channels", [])) or DEFAULT_CHANNELS,
        },
    )
    if cache is not None:
        cache[key] = handle
    return handle


def describe_cnr_convention(handle: ModelHandle, cfg: ServerConfig) -> str:
    """The load-bearing startup banner: which cnr convention this checkpoint
    expects, so a raw-CNR model served with online normalization (or vice versa)
    is obvious at a glance."""
    cnr_mode = handle.info["cnr_mode"]
    norm = (
        "ONLINE baseline-normalization ON (faro raw cnr -> cnr_median_norm)"
        if cnr_mode == "norm"
        else "ONLINE normalization OFF (feeding raw cnr_median directly)"
    )
    return (
        f"cnr_mode={cnr_mode!r} | {norm} | cnr z-score mean="
        f"{float(handle.mean[CNR]):.4f} std={float(handle.std[CNR]):.4f}"
    )


def load_engine(
    cfg: ServerConfig,
    objective: Objective | None = None,
    controller: Controller | None = None,
    checkpoint_dir: str | None = ...,  # type: ignore[assignment]
    cache: dict | None = None,
):
    """Return an ``(engine, info)`` pair. Falls back to the stub on any load failure.

    ``objective``/``controller`` default to the ones implied by ``cfg`` (a ``hold``
    at ``cfg.target_cnr``, searched by ``ConstantDoseSearch``), so the no-policy-file
    path behaves exactly as before.
    """
    from optoerk.serving.objectives import hold

    if checkpoint_dir is ...:
        checkpoint_dir = cfg.checkpoint_dir
    if objective is None:
        objective = hold(cfg.target_cnr)
    levels = dose_levels(cfg.min_exposure_ms, cfg.max_exposure_ms, cfg.n_candidates)
    if controller is None:
        from optoerk.serving.control import ConstantDoseSearch

        controller = ConstantDoseSearch(levels)

    calib = FluenceCalibration(cfg.instrument, cfg.stim_power_pct)
    if checkpoint_dir:
        # Resolve the device OUTSIDE the degrade-to-stub handler below. A bad or
        # unavailable device is a property of the process, not of one FOV, so
        # letting it be caught there would quietly turn every field into a stub
        # and report the cause as a checkpoint failure.
        _resolve_device(cfg.device)
        try:
            handle = load_model(checkpoint_dir, cfg.device, cache)
            engine = RealModelEngine(handle, calib, cfg, objective, controller)
            info = {"model_loaded": True, **handle.info,
                    "objective": objective.describe(),
                    "controller": controller.describe(),
                    "control_horizon": engine.horizon}
            print(
                f"[serving] {describe_cnr_convention(handle, cfg)} | "
                f"objective={objective.describe()} | controller={controller.name} | "
                f"horizon={engine.horizon}"
            )
            if handle.info["cnr_mode"] == "raw" and cfg.dark_baseline:
                print(
                    "[serving] NOTE: cnr_mode='raw' ignores dark_baseline/baseline_frames "
                    "(no online baseline to measure); cells are stimulated from frame 0."
                )
            if cfg.control_horizon > handle.info["future_len"]:
                print(
                    f"[serving] NOTE: control_horizon={cfg.control_horizon} exceeds the "
                    f"checkpoint's future_len={handle.info['future_len']}; clamped to "
                    f"{engine.horizon}. Rolling further is untrained — retrain with a "
                    f"larger future_len to actually look that far ahead."
                )
            if cfg.warmup and handle.device.type == "cuda":
                warmup_engine(engine)
            return engine, info
        except Exception as e:  # noqa: BLE001 - degrade gracefully to the stub
            print(f"[serving] checkpoint load failed ({e!r}); using STUB policy")
    engine = StubEngine(calib, cfg, objective)
    return engine, {"model_loaded": False, "policy": "stub",
                    "checkpoint_dir": checkpoint_dir,
                    "objective": objective.describe()}


def _bucket_to(n: int, width: int) -> int:
    """Round ``n`` up to a multiple of ``width``; a width of 0 or 1 is a no-op."""
    if width <= 1:
        return n
    return -(-n // width) * width


def _padding_frame(proto: CellFrame) -> CellFrame:
    """A throwaway cell that pads the batch to a bucket.

    Copied from a real cell rather than built from defaults, so the padded rows
    are in-distribution: objectives gate on ``x``/``y`` and read per-cell state,
    and feeding them a NaN-positioned zero-state cell would exercise branches no
    real cell takes. The state is a private copy with the encoder state cleared —
    nothing written to it can reach the cell it was copied from, and it is
    discarded when the frame returns.
    """
    st = copy.copy(proto.state)
    st.h = None
    st.c = None
    # `copy.copy` shares the mutable containers with the original; give the
    # padding cell its own so an append can never land on a real cell's history.
    for name, value in vars(st).items():
        if isinstance(value, list):
            setattr(st, name, list(value))
    return replace(proto, state=st)


def warmup_engine(engine, batch_sizes: tuple[int, ...] | None = None) -> None:
    """Prime the GPU before the first real frame by running throwaway ``decide``
    calls across a few batch sizes. Triggers CUDA context creation, cuDNN
    autotune and allocator-pool growth up front — the cold-start work that
    otherwise lands on the first predictions. Best-effort; never raises.

    Defaults to the bucket sizes the engine will actually ask for, so the
    allocator pools it builds here are the ones the run reuses; warming sizes
    that bucketing then rounds away would prime pools nothing ever hits.
    """
    if batch_sizes is None:
        width = max(int(getattr(getattr(engine, "cfg", None), "batch_bucket", 0) or 1), 1)
        batch_sizes = (
            (1, 8, 32, 64) if width == 1
            else tuple(width * m for m in (1, 2, 4, 8))
        )
    try:
        t0 = time.perf_counter()
        for n in batch_sizes:
            frames = [
                CellFrame(state=CellState(), cnr_norm=1.0, fov_density=float(n),
                          n_cells_200px=1.0, x=0.0, y=0.0)
                for _ in range(n)
            ]
            engine.decide(frames, GoalContext(fov=-1, timestep=0, cells=frames))
        dev = getattr(engine, "device", None)
        if dev is not None and dev.type == "cuda":
            torch.cuda.synchronize(dev)
        print(f"[serving] warmup: {len(batch_sizes)} batches in {time.perf_counter() - t0:.2f}s")
    except Exception as e:  # noqa: BLE001 - warmup is optional, never fatal
        print(f"[serving] warmup skipped: {e!r}")


# ---------------------------------------------------------------------------
# real model engine — also the "plant" the controllers drive
# ---------------------------------------------------------------------------


class RealModelEngine:
    def __init__(
        self,
        handle: ModelHandle,
        calib: FluenceCalibration,
        cfg: ServerConfig,
        objective: Objective,
        controller: Controller,
    ):
        self.model = handle.model
        self.calib = calib
        self.cfg = cfg
        self.objective = objective
        self.controller = controller
        self.device = handle.device
        mean, std = handle.mean, handle.std
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device)  # (C,)
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device)
        self.mean_np = np.asarray(mean, np.float32)
        # Model's channel order; unsupplied channels default to the pop. mean.
        self.channels = list(getattr(self.model.cfg, "norm_channels", []) or DEFAULT_CHANNELS)
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
        self.num_layers = self.model.cfg.num_layers
        self.hidden = self.model.cfg.hidden_dim
        # Rolling past the trained future_len is both untrained and an IndexError
        # into sigma_step_bias_param, so the horizon is hard-capped by the model.
        self.horizon = min(cfg.control_horizon, int(self.model.cfg.future_len))
        self._flu_per_ms = calib.fluence_per_ms

    # -- plant interface (what a Controller needs) -------------------------

    def std_fluence(self, ms: torch.Tensor) -> torch.Tensor:
        """exposure (ms) -> standardized ``u_t``, in torch, on the model's device."""
        flu = ms.to(self.device) * self._flu_per_ms
        return (flu - self.mean[FLU]) / self.std[FLU]

    def denorm_cnr(self, pred_std: torch.Tensor) -> torch.Tensor:
        """standardized CNR -> absolute CNR (the units objectives are written in)."""
        return pred_std * self.std[CNR] + self.mean[CNR]

    def denorm_sigma(self, sigma_std: torch.Tensor) -> torch.Tensor:
        """standardized CNR *scale* -> absolute CNR scale.

        A standard deviation transforms under the affine de-standardization by the
        scale factor only — no mean offset. Kept as its own method so a caller
        cannot reach for :meth:`denorm_cnr` and shift a width by ~0.82 CNR.
        """
        return sigma_std * self.std[CNR]

    def rollout(self, h, c, cnr_fb, fut) -> torch.Tensor:
        """Decoder-only rollout from a given (h, c); mirrors
        ``Seq2ScalarHistory.forward`` (free-running feedback, no teacher forcing).
        Returns standardized predicted CNR mean over the horizon, shape (B, F)."""
        return self._rollout(h, c, cnr_fb, fut, want_mixture=False)[0]

    def rollout_mixture(self, h, c, cnr_fb, fut):
        """As :meth:`rollout`, but also returns the per-step predictive mixture.

        ``(mean (B, F), pi (B, F, K), mu (B, F, K), sigma (B, F, K))``, all still
        **standardized** — de-standardizing is the caller's job because ``mu`` and
        ``sigma`` transform differently (see :meth:`denorm_sigma`).

        Only the distributional kernels need this; the mean-only path avoids
        materializing three (B, F, K) tensors per CEM iteration.
        """
        return self._rollout(h, c, cnr_fb, fut, want_mixture=True)

    def _rollout(self, h, c, cnr_fb, fut, want_mixture: bool):
        model = self.model
        dh, dc = h, c
        means = []
        pis, mus, sigmas = [], [], []
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
            pi, mu, sigma = model.head(feats, sigma_bias=sigma_bias)
            pred = (pi * mu).sum(dim=-1, keepdim=True)            # (B,1) std cnr
            means.append(pred)
            if want_mixture:
                pis.append(pi)
                mus.append(mu)
                sigmas.append(sigma)
            # Free-running feedback uses the mixture MEAN regardless of the kernel,
            # so a band-scored plan and an L2-scored plan see the same trajectory
            # and the arms differ only in how that trajectory is costed.
            cnr_fb = pred
        mean = torch.cat(means, dim=1)
        if not want_mixture:
            return mean, None, None, None
        return (
            mean,
            torch.stack(pis, dim=1),                              # (B, F, K)
            torch.stack(mus, dim=1),
            torch.stack(sigmas, dim=1),
        )

    # -- the frame step ----------------------------------------------------

    @torch.no_grad()
    def decide(self, frames: list[CellFrame], ctx: GoalContext) -> list[float]:
        # Per-cell diagnostics for the predict log, read back by the service via
        # getattr (the same pattern as `optortk_fed`). Cleared first so a frame
        # with no cells cannot leave the previous frame's values standing.
        self.last_plan_cost = None
        self.last_pred_cnr_h1 = None
        if not frames:
            return []
        # --- shape bucketing (see ServerConfig.batch_bucket) ------------------
        # Pad the batch out to a bucket so the tensor shapes below repeat frame
        # to frame and the CUDA caching allocator can actually reuse its blocks.
        # The padding rows are throwaway cells appended at the END: the objective
        # and the controller size themselves off `ctx.cells`, so that list is
        # padded in lockstep, and everything is sliced back to `n_real` before
        # any state is persisted or any dose returned.
        n_real = len(frames)
        n_pad = _bucket_to(n_real, self.cfg.batch_bucket) - n_real
        if n_pad:
            frames = frames + [_padding_frame(frames[-1]) for _ in range(n_pad)]
            ctx = replace(ctx, cells=frames)
        N = len(frames)
        L, H = self.num_layers, self.hidden

        # --- raw per-frame channels, assembled by NAME in the model's channel
        # order (robust to CHANNELS growing/reordering). Any channel not supplied
        # here falls back to the population mean (→ 0 after standardizing, i.e. the
        # neutral median value for that channel).
        _supplied = {
            "cnr": lambda f: f.cnr_norm,
            "u_t": lambda f: f.state.last_fluence,
            "fov_density": lambda f: f.fov_density,
            "n_cells_200px": lambda f: f.n_cells_200px,
        }
        # Channels a model may or may not use, supplied only when the payload
        # carried them. A model without the channel never asks; one WITH it would
        # otherwise be fed the population mean on a channel it actually uses —
        # `nuc_area` is the second most important channel in the area variants
        # (permutation dNLL 0.0118, behind only u_t and the expression rank).
        if any(f.nuc_area is not None for f in frames):
            _supplied["nuc_area"] = (
                lambda f: f.nuc_area if f.nuc_area is not None
                else float(self.mean_np[self.channels.index("nuc_area")])
            )
        # The optoRTK-expression channel, in precedence order:
        #   1. an operator-set constant   (--optortk-expr-value)
        #   2. the live per-cell rank     (--live-optortk-expr), when it arrived
        #   3. the channel population mean, i.e. every cell a median expresser
        # Case 2 is per-cell, so it cannot be a single closure over the frame list.
        if self._optortk_override is not None:
            _supplied[self._optortk_channel] = lambda f, v=self._optortk_override: v
        elif self._optortk_channel in self.channels:
            _mean = float(self.mean_np[self.channels.index(self._optortk_channel)])
            _supplied[self._optortk_channel] = (
                lambda f, m=_mean: m if f.optortk_expr is None else float(f.optortk_expr)
            )
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
        _, (h_new, c_new) = self.model.encoder.lstm(xs.unsqueeze(1), (h, c))  # (L, N, H)

        # --- control: the objective says what we want, the controller finds it
        cnr_fb = xs[:, CNR : CNR + 1]  # (N,1) feedback = cnr at last real frame
        # plan() rather than solve(), because the plan's cost is logged: it is what
        # separates "the controller knew it could not reach the reference" from "the
        # controller expected to reach it and was wrong". On the border-probing arms
        # that distinction IS the measurement, and a log carrying only the applied
        # dose cannot make it without a full replay.
        plan_ms, plan_cost = self.controller.plan(
            self, h_new, c_new, cnr_fb, self.objective, ctx
        )
        best_ms = self.controller.apply_gate(plan_ms, self.objective, ctx)
        self.last_plan_cost = [float(v) for v in plan_cost[:n_real]]

        # One-step prediction under the dose about to be commanded. One extra
        # decoder step for N cells, against the CEM's 512 plans x H steps — free in
        # practice, and it turns every frame into a one-step model-error
        # measurement: the drift of (achieved delta / predicted delta) is a direct
        # per-cell sensitivity readout that needs no rolling regression.
        fut1 = self.std_fluence(best_ms.reshape(N, 1, 1))
        pred1 = self.denorm_cnr(self.rollout(h_new, c_new, cnr_fb, fut1))  # (N, 1)
        self.last_pred_cnr_h1 = [float(v) for v in pred1[:n_real, 0]]

        # --- persist new encoder state + the APPLIED fluence ------------------
        # Only u[0] is applied; the rest of the optimized plan is discarded and
        # re-planned next frame (receding horizon). Persisting anything but the
        # applied dose would corrupt the u_t channel at the next encoder step.
        # Padding rows are dropped here and never touched again: no state is
        # written back for them, and no dose is returned for them.
        out_ms: list[float] = []
        for i, f in enumerate(frames[:n_real]):
            f.state.h = h_new[:, i : i + 1].detach().clone()
            f.state.c = c_new[:, i : i + 1].detach().clone()
            ms = float(best_ms[i].item())
            f.state.last_fluence = float(self.calib.ms_to_fluence(ms))
            # Kept in ms alongside the fluence: the move penalty normalizes by the
            # dose ladder's max ms, and round-tripping through the calibration to
            # recover it would silently depend on stim_power_pct.
            f.state.last_applied_ms = ms
            out_ms.append(ms)
        return out_ms


# ---------------------------------------------------------------------------
# stub engine (no model) — deterministic, runnable before a checkpoint exists
# ---------------------------------------------------------------------------


class StubEngine:
    """Proportional placeholder: dose up cells whose CNR is below target.

    It has no model, so it cannot evaluate an arbitrary objective's cost — it
    tracks ``cfg.target_cnr`` regardless of which objective is configured. It does
    honour the objective's :meth:`~optoerk.serving.objectives.Objective.allow_stim`
    gate, so "which cells may be stimulated" behaves identically to the real
    engine. This is an integration placeholder, not a research artifact.
    """

    def __init__(self, calib: FluenceCalibration, cfg: ServerConfig,
                 objective: Objective | None = None):
        self.calib = calib
        self.cfg = cfg
        self.objective = objective
        self.optortk_fed: float | None = None  # stub ignores the optoRTK channel

    def decide(self, frames: list[CellFrame], ctx: GoalContext) -> list[float]:
        cfg = self.cfg
        mask = self.objective.allow_stim(ctx) if self.objective is not None else None
        out: list[float] = []
        for i, f in enumerate(frames):
            if mask is not None and not bool(mask[i]):
                ms = 0.0
            else:
                deficit = max(0.0, cfg.target_cnr - f.cnr_norm)
                ms = float(np.clip(cfg.stub_gain_ms_per_cnr * deficit,
                                   cfg.min_exposure_ms, cfg.max_exposure_ms))
            f.state.last_fluence = float(self.calib.ms_to_fluence(ms))
            f.state.last_applied_ms = ms
            out.append(ms)
        return out
