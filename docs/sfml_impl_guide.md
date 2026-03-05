# Implementation Guide: Stochastic Flow Map Learning for Non-Autonomous Systems

## 1. What the System Does

The goal is to build a black-box predictive model for an unknown stochastic dynamical system that is being driven by external time-dependent signals. You observe short input/output data — excitation signals and corresponding system responses — and produce a model that can predict the system's stochastic behavior under entirely new excitation signals, for arbitrarily long time horizons.

Concretely: given a new excitation signal u(t) and an initial condition x₀, the trained model generates Monte Carlo sample trajectories whose statistical properties (mean, variance, full probability distributions) match what the true unknown system would produce.

---

## 2. Prerequisite Concepts

### 2.1 Stochastic Differential Equations (SDEs)

An SDE of the form `dx = a(x,t)dt + b(x,t)dW` describes a system where the state evolves under both a deterministic drift `a` and a random diffusion `b` scaled by Brownian motion increments `dW`. The key property exploited by the method is that Brownian increments are **stationary and independent**: the distribution of `W(t+Δ) - W(t)` depends only on `Δ`, not on `t`. This is what allows the method to suppress absolute time from the learned map.

You do not need to know the SDE form — the method works for general stochastic systems including non-Gaussian jump processes — but understanding why stationarity of increments matters is essential for grasping the local parameterization trick.

### 2.2 Flow Maps

A flow map Φ takes a state at time t and returns the state at time t+Δ. For a deterministic ODE, this is a single-valued function. For a stochastic system, it is a random mapping: given the same initial condition, it produces different outputs on each realization. The method learns this random one-step map from data, then iterates it to produce long trajectories.

### 2.3 Generative Models

A generative model is any neural network architecture that can produce random samples from a learned distribution. The method requires a **conditional** generative model: one that produces samples from `P(x₁ | x₀, Γ₀)`, the distribution of the next state given the current state and excitation parameters. Paper 1 uses normalizing flows, but GANs, autoencoders, or diffusion models could substitute.

### 2.4 Normalizing Flows

A normalizing flow transforms a simple base distribution (standard Gaussian) into a complex target distribution through a sequence of learned invertible transformations. The key advantages are: (1) exact likelihood computation via the change-of-variables formula, enabling clean maximum-likelihood training; (2) exact sampling by pushing Gaussian noise through the forward transformations; (3) stable training compared to adversarial methods. The paper uses Masked Autoregressive Flow (MAF), where each output dimension depends autoregressively on previous dimensions, yielding a triangular Jacobian with O(d) determinant computation.

### 2.5 Polynomial Approximation of Signals

Any smooth function over a small interval can be well-approximated by a low-degree polynomial. The method exploits this to represent excitation signals locally: over each time step [tₙ, tₙ₊₁], the excitation u(t) is approximated as a polynomial in τ = t - tₙ, and the polynomial coefficients become the parameters Γₙ that the model conditions on. Since the time step Δ is small, even linear or quadratic polynomials suffice.

---

## 3. System Components

The implementation has five major components, described below in the order they participate in the pipeline.

### 3.1 Data Ingestion and Preprocessing

**Input format.** The training data consists of N_T input/output trajectory sequences. Each sequence i provides: (a) excitation signal values u(t) at discrete time points t₀, t₁, ..., t_L; and (b) corresponding system response values x(t) at the same time points. The time step Δ = t_{k+1} - t_k is assumed uniform.

**Pair extraction.** Each trajectory of length L+1 yields L consecutive pairs. Across all trajectories, you collect M = Σ Lᵢ pairs total. Each pair consists of:

- The excitation values at two consecutive times: u(tₖ), u(tₖ₊₁)
- The state values at two consecutive times: x(tₖ), x(tₖ₊₁)

This is the fundamental reorganization: you discard the trajectory structure and treat every consecutive pair as an independent training sample. Time labels are discarded because the stochastic increments are stationary.

**Local parameterization.** For each pair, you fit a local polynomial to the excitation signal over [tₖ, tₖ₊₁]. With piecewise linear approximation (the simplest useful choice), the polynomial is:

```
ũ(τ) = u(tₖ) + (τ/Δ) · [u(tₖ₊₁) - u(tₖ)],   τ ∈ [0, Δ)
```

The parameter vector is Γ = (u(tₖ), u(tₖ₊₁)) for a scalar excitation, or the concatenation across all excitation components. For quadratic approximation (if more information about u is available between time points), Γ includes three coefficients per excitation component.

**Final training set.** After parameterization, the training data is:

```
S_M = { (Γ₀⁽ʲ⁾, x₀⁽ʲ⁾, x₁⁽ʲ⁾) }  for j = 1, ..., M
```

Each entry says: starting from state x₀, under local excitation described by Γ₀, the system arrived at state x₁ one time step later.

### 3.2 The Conditioning Network N

This is a standard fully connected feedforward DNN. It takes as input the concatenation of the current state x₀ ∈ ℝᵈ and the excitation parameters Γ₀ ∈ ℝⁿᵧ, and outputs the parameter vector θ ∈ ℝⁿᶿ that controls the normalizing flow transformations.

**Architecture specifics from the paper:** 3 hidden layers, 20 nodes each, tanh activation. The output dimension n_θ = 2d for MAF (each flow step needs a scale and shift per dimension). This network is small because the flow itself provides most of the expressive power; N just needs to smoothly adapt the flow parameters as a function of the conditioning variables.

### 3.3 The Normalizing Flow T

This is the invertible transformation that maps standard Gaussian noise z₀ ∈ ℝᵈ to the predicted next state. The flow is parameterized by θ (output of network N), so the full model is:

```
x₁ = T_{N(x₀, Γ₀)}(z₀)
```

**MAF specifics.** In Masked Autoregressive Flow, each layer transforms the input autoregressively: the k-th output depends on inputs 1 through k. This is implemented via a masked feedforward network that computes scale and shift parameters for each dimension. Multiple MAF layers are composed, with dimension permutations between layers to ensure all dimensions can influence all others.

The inverse S = T⁻¹ maps a data point x₁ back to the base space, which is needed for training (likelihood evaluation). For MAF, the inverse can be computed in a single forward pass through the masked network (this is actually the efficient direction for MAF — sampling requires sequential computation through dimensions, while density evaluation is parallelizable).

### 3.4 The Loss Function

Training minimizes the negative log-likelihood over the training set:

```
L(Θ) = -Σⱼ log p(x₁⁽ʲ⁾ | x₀⁽ʲ⁾, Γ₀⁽ʲ⁾; Θ)
```

where Θ denotes all trainable parameters (both in network N and in the flow T). For each training sample j, the conditional density is computed via the change-of-variables formula:

```
p(x₁ | x₀, Γ₀; Θ) = p_z(S(x₁)) · |det DT(S(x₁))|⁻¹
```

where S = T⁻¹ maps x₁ back to z-space, p_z is the standard Gaussian density, and the Jacobian determinant accounts for the volume distortion of the transformation.

### 3.5 The Prediction Engine

After training, prediction for a new excitation signal u(t) proceeds as follows:

1. **Parameterize the excitation.** For each time interval [tₙ, tₙ₊₁], fit a local polynomial to u(t) and extract coefficients Γₙ.
2. **Initialize.** Set x̂₀ to the given initial condition.
3. **Time-march.** For each step n = 0, 1, 2, ...:
   - Draw z_n ~ N(0, I_d)
   - Compute θₙ = N(x̂ₙ, Γₙ)
   - Compute x̂ₙ₊₁ = T_{θₙ}(zₙ)
4. **Repeat** for many independent realizations to collect ensemble statistics (mean, std, histograms, etc.).

---

## 4. Assembly: Putting the Components Together

### 4.1 Data Pipeline

```
Raw I/O trajectories
    │
    ├─► Extract consecutive pairs (x_k, x_{k+1}, u_k, u_{k+1})
    │
    ├─► Fit local polynomials to get Γ_k for each pair
    │
    └─► Assemble training tensor: [Γ, x₀, x₁]  shape (M, n_Γ + 2d)
```

Implementation notes:

- Normalize all variables (x and Γ) to zero mean, unit variance before training. Store the normalization statistics for use during prediction.
- Shuffle the training set. The original trajectory ordering carries no useful information after pair extraction.
- If excitation signals are multi-dimensional (e.g., u = (μ, ν) controlling drift and diffusion separately), concatenate all polynomial coefficients into a single Γ vector.

### 4.2 Model Architecture

```
                    ┌─────────────┐
  x₀ ──────┬──────►│             │
            │       │  Network N  │──► θ ──┐
  Γ₀ ──────┘──────►│  (3×20, tanh)│        │
                    └─────────────┘        │
                                           ▼
                    ┌──────────────────────────┐
  z₀ ~ N(0,I) ────►│  Normalizing Flow T_θ     │──► x̂₁
                    │  (K layers of MAF)        │
                    └──────────────────────────┘
```

The two sub-networks (N and the flow layers) are trained jointly end-to-end. The conditioning network N does not have a separate training phase.

### 4.3 Training Loop (Pseudocode)

```
initialize N, T with random weights Θ
for epoch in 1..max_epochs:
    for batch in shuffle_and_batch(training_data):
        Γ_batch, x0_batch, x1_batch = unpack(batch)

        # Forward through conditioning network
        θ_batch = N(x0_batch, Γ_batch)

        # Compute inverse flow: map x1 back to z-space
        z_batch = S_{θ_batch}(x1_batch)     # S = T⁻¹

        # Compute log-likelihood
        log_pz = -0.5 * sum(z_batch² + log(2π), dim=-1)
        log_det = log_jacobian_determinant(S, x1_batch, θ_batch)
        loss = -mean(log_pz + log_det)

        # Backprop and update
        loss.backward()
        optimizer.step()

    adjust_learning_rate(epoch)    # cyclic schedule
```

### 4.4 Training Hyperparameters (from the paper)

| Parameter | Value |
|-----------|-------|
| DNN layers (network N) | 3 hidden, 20 nodes each |
| Activation | tanh |
| Learning rate schedule | Cyclic: base 3×10⁻⁴, max 5×10⁻⁴ |
| Cycle period | 40,000 epochs |
| LR decay per cycle | 0.5 |
| Weight decay | 0.01 |
| Total training epochs | 200,000 – 300,000 |
| Training samples M | 120,000 (typical) |
| Time step Δ | 0.01 (typical) |

These are long training runs. The cyclic learning rate is important: it helps escape local minima in the likelihood landscape. The slow decay across cycles prevents late-stage instability.

---

## 5. Computational Tools Required

### 5.1 Core Dependencies

**PyTorch** (or JAX). The entire model — conditioning network, normalizing flow, and training loop — lives in a differentiable programming framework. PyTorch is the natural choice because the best normalizing flow libraries target it.

**nflows** (https://github.com/bayesiains/nflows) or **normflows** (https://github.com/VincentStimper/normalizing-flows). These provide production-quality MAF implementations with correct Jacobian computation, dimension permutations, and batched conditioning. Writing MAF from scratch is possible but error-prone, especially getting the masked weight matrices and Jacobian accumulation right.

**NumPy / SciPy.** For data preprocessing, polynomial fitting, and post-processing of ensemble statistics.

**Matplotlib.** For visualization of trajectories, probability densities, mean/std evolution.

### 5.2 Optional but Recommended

**Pandas / PyArrow.** For data I/O if building a general-purpose API that accepts DataFrames or Parquet files.

**Hydra or YAML configs.** For managing the many hyperparameters across different problems.

**Weights & Biases or TensorBoard.** For tracking training loss, learning rate schedules, and validation metrics across long runs.

**Joblib or multiprocessing.** For parallelizing the ensemble prediction step (each Monte Carlo trajectory is independent).

---

## 6. Detailed Implementation Walkthrough

### 6.1 Step 1: Data Loading and Pair Extraction

```python
import numpy as np

def extract_pairs(trajectories_x, trajectories_u, poly_degree=1):
    """
    Parameters
    ----------
    trajectories_x : list of arrays, each shape (L+1, d)
        System response trajectories.
    trajectories_u : list of arrays, each shape (L+1, n_u)
        Excitation signal trajectories, same time points as x.
    poly_degree : int
        Degree of local polynomial (1=linear, 2=quadratic).

    Returns
    -------
    gamma : array (M, n_gamma)
        Local polynomial coefficients for each pair.
    x0 : array (M, d)
        Starting states.
    x1 : array (M, d)
        Ending states (one step later).
    """
    pairs_gamma, pairs_x0, pairs_x1 = [], [], []

    for x_traj, u_traj in zip(trajectories_x, trajectories_u):
        L = len(x_traj) - 1
        for k in range(L):
            x0_k = x_traj[k]
            x1_k = x_traj[k + 1]

            # Local parameterization of excitation
            if poly_degree == 1:
                # Linear: Γ = [u(t_k), u(t_{k+1})]
                gamma_k = np.concatenate([u_traj[k], u_traj[k + 1]])
            elif poly_degree == 2:
                # Quadratic: need u at t_k, t_{k+1/2}, t_{k+1}
                # or fit from available data
                gamma_k = fit_quadratic_coeffs(u_traj, k)

            pairs_gamma.append(gamma_k)
            pairs_x0.append(x0_k)
            pairs_x1.append(x1_k)

    return (np.array(pairs_gamma),
            np.array(pairs_x0),
            np.array(pairs_x1))
```

For the linear case, the parameterization is trivially the endpoint values. The coefficients α⁰ = u(tₖ) and α¹ = [u(tₖ₊₁) - u(tₖ)]/Δ give the piecewise linear form ũ(τ) = α⁰ + α¹τ. Whether you store raw endpoint values or polynomial coefficients is a design choice — they carry the same information.

### 6.2 Step 2: Normalization

```python
class DataNormalizer:
    """Z-score normalization preserving statistics for prediction time."""

    def fit(self, gamma, x0, x1):
        self.gamma_mean = gamma.mean(axis=0)
        self.gamma_std = gamma.std(axis=0) + 1e-8
        self.x_mean = np.concatenate([x0, x1], axis=0).mean(axis=0)
        self.x_std = np.concatenate([x0, x1], axis=0).std(axis=0) + 1e-8

    def transform(self, gamma, x0, x1):
        return ((gamma - self.gamma_mean) / self.gamma_std,
                (x0 - self.x_mean) / self.x_std,
                (x1 - self.x_mean) / self.x_std)

    def inverse_x(self, x_normalized):
        return x_normalized * self.x_std + self.x_mean

    def transform_gamma(self, gamma):
        return (gamma - self.gamma_mean) / self.gamma_std
```

Normalization is critical. The conditioning network and flow both work best when inputs are O(1). Without it, training is slow or fails entirely.

### 6.3 Step 3: Model Definition

```python
import torch
import torch.nn as nn

class ConditioningNetwork(nn.Module):
    """Maps (x₀, Γ₀) → θ, the parameters for the normalizing flow."""

    def __init__(self, dim_x, dim_gamma, dim_theta,
                 hidden_size=20, n_layers=3):
        super().__init__()
        layers = []
        dim_in = dim_x + dim_gamma
        for _ in range(n_layers):
            layers.extend([nn.Linear(dim_in, hidden_size), nn.Tanh()])
            dim_in = hidden_size
        layers.append(nn.Linear(dim_in, dim_theta))
        self.net = nn.Sequential(*layers)

    def forward(self, x0, gamma):
        return self.net(torch.cat([x0, gamma], dim=-1))
```

For the normalizing flow itself, use a library implementation of MAF. With `nflows`:

```python
from nflows.flows import MaskedAutoregressiveFlow
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform
from nflows.distributions import StandardNormal

def build_flow(dim_x, n_flow_layers=5, hidden_features=64):
    """Build a MAF that can be conditioned on external parameters."""
    transforms = []
    for _ in range(n_flow_layers):
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=dim_x,
                hidden_features=hidden_features,
                context_features=None  # conditioning handled externally
            )
        )
    return CompositeTransform(transforms), StandardNormal([dim_x])
```

However, because the paper conditions the flow by having network N output the flow parameters directly (rather than using context features in standard MAF layers), you likely need a custom implementation. The conditioning mechanism is:

```python
class ConditionalMAFLayer(nn.Module):
    """Single MAF layer whose scale/shift params come from external θ."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, z, scale, shift):
        """
        z: (batch, dim) — input from base space
        scale, shift: (batch, dim) — from conditioning network
        Returns transformed x and log_det_jacobian
        """
        x = z * torch.exp(scale) + shift
        log_det = scale.sum(dim=-1)
        return x, log_det

    def inverse(self, x, scale, shift):
        z = (x - shift) * torch.exp(-scale)
        log_det = -scale.sum(dim=-1)
        return z, log_det
```

A full MAF typically has multiple such layers with permutations between them. For the paper's approach where θ = N(x₀, Γ₀) and n_θ = 2d, a single affine layer is used (scale and shift, each of dimension d). Multiple MAF blocks would increase n_θ accordingly.

### 6.4 Step 4: Full Model Assembly

```python
class sFML(nn.Module):
    """Complete stochastic flow map learning model."""

    def __init__(self, dim_x, dim_gamma, n_flow_layers=1,
                 hidden_size=20, n_cond_layers=3):
        super().__init__()
        self.dim_x = dim_x
        self.n_flow_layers = n_flow_layers
        dim_theta = 2 * dim_x * n_flow_layers  # scale + shift per layer

        self.cond_net = ConditioningNetwork(
            dim_x, dim_gamma, dim_theta, hidden_size, n_cond_layers
        )

    def forward(self, x0, gamma, x1):
        """Compute negative log-likelihood for training."""
        theta = self.cond_net(x0, gamma)
        # Parse theta into scale/shift pairs for each flow layer
        params = theta.reshape(-1, self.n_flow_layers, 2, self.dim_x)

        # Inverse pass: x1 → z (for likelihood computation)
        z = x1
        total_log_det = torch.zeros(x1.shape[0], device=x1.device)
        for k in reversed(range(self.n_flow_layers)):
            scale = params[:, k, 0, :]
            shift = params[:, k, 1, :]
            z = (z - shift) * torch.exp(-scale)
            total_log_det += -scale.sum(dim=-1)

        # Log-likelihood
        log_pz = -0.5 * (z**2 + np.log(2 * np.pi)).sum(dim=-1)
        log_prob = log_pz + total_log_det
        return -log_prob.mean()  # negative log-likelihood loss

    @torch.no_grad()
    def sample(self, x0, gamma, n_samples=1):
        """Generate predicted x1 samples."""
        # Expand x0, gamma for multiple samples
        x0_exp = x0.unsqueeze(0).expand(n_samples, -1)
        gamma_exp = gamma.unsqueeze(0).expand(n_samples, -1)

        theta = self.cond_net(x0_exp, gamma_exp)
        params = theta.reshape(-1, self.n_flow_layers, 2, self.dim_x)

        # Forward pass: z → x1
        z = torch.randn(n_samples, self.dim_x)
        for k in range(self.n_flow_layers):
            scale = params[:, k, 0, :]
            shift = params[:, k, 1, :]
            z = z * torch.exp(scale) + shift
        return z
```

### 6.5 Step 5: Training

```python
def train(model, gamma, x0, x1, n_epochs=200000, batch_size=1024):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=3e-4, max_lr=5e-4,
        step_size_up=10000, cycle_momentum=False,
        gamma=0.99999
    )

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(gamma, dtype=torch.float32),
        torch.tensor(x0, dtype=torch.float32),
        torch.tensor(x1, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for g_batch, x0_batch, x1_batch in loader:
            loss = model(x0_batch, g_batch, x1_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10000 == 0:
            avg = epoch_loss / len(loader)
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:>7d}  Loss: {avg:.6f}  LR: {lr:.6f}")
```

### 6.6 Step 6: Prediction Under New Excitations

```python
def predict(model, normalizer, x0_physical, u_signal, dt, n_steps,
            n_trajectories=10000):
    """
    Parameters
    ----------
    x0_physical : array (d,) — initial condition in physical units
    u_signal : callable or array — excitation signal u(t)
    dt : float — time step
    n_steps : int — number of steps to simulate
    n_trajectories : int — Monte Carlo ensemble size

    Returns
    -------
    trajectories : array (n_trajectories, n_steps+1, d)
    """
    trajectories = np.zeros((n_trajectories, n_steps + 1, x0_physical.shape[0]))
    trajectories[:, 0, :] = x0_physical

    model.eval()
    for n in range(n_steps):
        t_n = n * dt
        t_n1 = (n + 1) * dt

        # Get excitation values
        if callable(u_signal):
            u_n = np.atleast_1d(u_signal(t_n))
            u_n1 = np.atleast_1d(u_signal(t_n1))
        else:
            u_n = u_signal[n]
            u_n1 = u_signal[n + 1]

        # Build Γ (linear parameterization)
        gamma_n = np.concatenate([u_n, u_n1])

        # Normalize
        gamma_norm = normalizer.transform_gamma(gamma_n)
        x_curr_physical = trajectories[:, n, :]
        x_curr_norm = (x_curr_physical - normalizer.x_mean) / normalizer.x_std

        # Predict all trajectories at once
        g_tensor = torch.tensor(gamma_norm, dtype=torch.float32)
        g_batch = g_tensor.unsqueeze(0).expand(n_trajectories, -1)
        x_batch = torch.tensor(x_curr_norm, dtype=torch.float32)

        theta = model.cond_net(x_batch, g_batch)
        params = theta.reshape(-1, model.n_flow_layers, 2, model.dim_x)

        z = torch.randn(n_trajectories, model.dim_x)
        for k in range(model.n_flow_layers):
            scale = params[:, k, 0, :]
            shift = params[:, k, 1, :]
            z = z * torch.exp(scale) + shift

        x_next_norm = z.detach().numpy()
        trajectories[:, n+1, :] = normalizer.inverse_x(x_next_norm)

    return trajectories
```

---

## 7. Making It Generalizable: An API Design

The goal is a library where a user provides trajectory data (as a DataFrame or Parquet file) and gets back a trained model that can predict under new excitations. Below is a design for this API.

### 7.1 Data Contract

The input data should have the following columnar structure:

| Column | Type | Description |
|--------|------|-------------|
| `trajectory_id` | int | Identifies which trajectory this row belongs to |
| `time` | float | Time stamp |
| `x_0`, `x_1`, ..., `x_{d-1}` | float | State variables (response) |
| `u_0`, `u_1`, ..., `u_{p-1}` | float | Excitation signal components |

Each trajectory must have uniform time spacing. Different trajectories can cover different time spans and start at different times (absolute time does not matter).

### 7.2 Configuration Schema

```yaml
# config.yaml
data:
  state_columns: ["x_0", "x_1"]         # which columns are state variables
  excitation_columns: ["u_0"]            # which columns are excitations
  trajectory_id_column: "trajectory_id"
  time_column: "time"

preprocessing:
  poly_degree: 1                         # 1=linear, 2=quadratic
  normalize: true

model:
  n_flow_layers: 1                       # MAF layers in the flow
  cond_hidden_size: 20                   # conditioning network width
  cond_n_layers: 3                       # conditioning network depth

training:
  n_epochs: 200000
  batch_size: 1024
  base_lr: 0.0003
  max_lr: 0.0005
  weight_decay: 0.01
  cycle_steps: 40000
  checkpoint_every: 50000

prediction:
  n_trajectories: 10000                  # Monte Carlo ensemble size
```

### 7.3 API Surface

```python
from sfml import sFMLModel

# --- Training ---
model = sFMLModel.from_config("config.yaml")

# Option A: from a DataFrame
model.fit(df_train)

# Option B: from a Parquet file
model.fit_from_file("training_data.parquet")

# Save/load
model.save("trained_model.pt")
model = sFMLModel.load("trained_model.pt")

# --- Prediction ---
# Define a new excitation signal (not in training data)
import numpy as np
def new_excitation(t):
    return np.array([0.5 * np.sin(6 * t)])

result = model.predict(
    x0=np.array([2.0, 1.0]),      # initial condition
    excitation=new_excitation,      # callable: t → u(t)
    t_end=80.0,                     # prediction horizon
    n_trajectories=10000
)

# result.mean          — (n_steps+1, d) array
# result.std           — (n_steps+1, d) array
# result.trajectories  — (n_trajectories, n_steps+1, d) array
# result.times         — (n_steps+1,) array
# result.plot()        — convenience visualization

# Or from tabular excitation data
result = model.predict_from_dataframe(
    x0=np.array([2.0, 1.0]),
    excitation_df=df_new_excitation,  # DataFrame with time + u columns
    n_trajectories=10000
)
```

### 7.4 Internal Architecture of the Library

```
sfml/
├── __init__.py
├── model.py              # sFMLModel: top-level API class
├── data/
│   ├── __init__.py
│   ├── loader.py          # DataFrame/Parquet → raw trajectories
│   ├── pairs.py           # Trajectory → consecutive pairs
│   ├── parameterize.py    # Excitation → polynomial coefficients
│   └── normalize.py       # Z-score normalization
├── networks/
│   ├── __init__.py
│   ├── conditioning.py    # Feedforward DNN: (x₀, Γ₀) → θ
│   └── flow.py            # Conditional MAF implementation
├── training/
│   ├── __init__.py
│   ├── trainer.py         # Training loop with cyclic LR
│   └── scheduler.py       # Custom cyclic LR with decay
├── prediction/
│   ├── __init__.py
│   ├── engine.py          # Time-marching with ensemble sampling
│   └── result.py          # PredictionResult with stats + plotting
└── config.py              # YAML config parsing + validation
```

### 7.5 Key Design Decisions for Generalizability

**Automatic dimensionality detection.** The library should infer d (state dimension) and n_u (excitation dimension) from the column configuration, then set n_Γ = (poly_degree + 1) × n_u and n_θ = 2d × n_flow_layers automatically.

**Flexible excitation input.** Accept excitations as: (a) a callable `t → u(t)` for analytical signals, (b) a DataFrame with time and u columns for measured signals, or (c) a NumPy array for pre-discretized signals. Internally, all are converted to values at the discrete time points needed for parameterization.

**Validation split.** Hold out a fraction of trajectories (not pairs — you must keep full trajectories together) for validation. Monitor validation NLL during training to detect overfitting. This is especially important because the paper's training runs are very long.

**GPU support.** The prediction step (many Monte Carlo trajectories, many time steps) is embarrassingly parallel and benefits enormously from GPU batching. The training step also benefits. The API should auto-detect CUDA availability.

**Checkpointing and resumption.** With 200k–300k epochs, training takes hours to days. Save checkpoints periodically and support resuming from a checkpoint.

**Multiple generative backends.** The core structure — data pipeline, conditioning network, prediction engine — is independent of the generative model. Abstract the flow behind an interface so users can swap in a GAN, autoencoder, or diffusion model:

```python
class GenerativeBackend:
    def log_prob(self, x1, x0, gamma) -> Tensor: ...
    def sample(self, x0, gamma, n_samples) -> Tensor: ...

class MAFBackend(GenerativeBackend): ...
class AutoencoderBackend(GenerativeBackend): ...
```

**Diagnostics.** After training, provide methods to assess model quality without needing ground truth:

- One-step conditional distribution plots (pick a state, sample from the model, show histogram)
- Latent space diagnostics (are the z values actually Gaussian after inverse mapping?)
- Training loss curves
- Calibration checks: do the predicted confidence intervals have correct coverage on held-out data?

---

## 8. Practical Considerations

### 8.1 How Much Training Data Do You Need?

The paper uses M = 120,000 pairs in most examples. Each pair is a single time step of a single trajectory. If your trajectories have L = 100 steps each, you need about 1,200 trajectories. If trajectories are shorter (L = 10), you need about 12,000 trajectories. The pairs must cover the relevant region of (x, Γ) space with reasonable density.

### 8.2 Choosing the Time Step

The time step Δ must be small enough that: (1) the local polynomial approximation of the excitation is accurate, and (2) the one-step state transition is "small" (the state doesn't jump dramatically in one step). The paper uses Δ = 0.01 for most examples. If your system has faster dynamics, you may need a smaller Δ.

### 8.3 Training Domain vs. Prediction Domain

The model can only make reliable predictions for states x and excitation parameters Γ that fall within or near the range covered by the training data. If during a long prediction the state drifts into uncharted territory, the model will extrapolate unreliably. This is a fundamental limitation. The paper addresses it by choosing training domains (I_x and I_Γ) large enough to cover the expected prediction range.

### 8.4 Computational Cost

Training is the expensive part: 200k–300k epochs over 120k samples. On a modern GPU, this takes several hours. Prediction is cheap: each time step requires one forward pass through the conditioning network and one through the flow, applied in parallel across all Monte Carlo trajectories.

### 8.5 When This Method May Not Work Well

- **Very high-dimensional systems.** The paper demonstrates up to d = 30 (the SPDE example). For much higher dimensions, the normalizing flow may struggle, and the conditioning network output grows as O(d).
- **Non-stationary stochastic increments.** The method fundamentally relies on the stochastic driving process having stationary, independent increments. Systems driven by fractional Brownian motion or other long-memory processes violate this assumption.
- **Chaotic systems.** If the deterministic skeleton of the system is chaotic, small errors in the learned flow map compound exponentially during prediction. The stochastic component may mask this partially, but long-term accuracy will degrade faster.
- **Extremely rare events.** The model can capture rare transitions (as shown in the stochastic resonance example), but only if the training data provides sufficient coverage of the state space near the transition regions.