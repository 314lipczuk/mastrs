import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Conv1dBlock, spectral_loss


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_mlp(dims, activation):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# MLP VAE (single-timepoint state vectors)
# ---------------------------------------------------------------------------

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, activation=nn.GELU):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims, hidden_dims)
        self.net = _build_mlp([input_dim] + list(hidden_dims), activation)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, activation=nn.GELU):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims, hidden_dims)
        self.net = _build_mlp([latent_dim] + list(hidden_dims) + [output_dim], activation)

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    """MLP-based beta-VAE for single-timepoint state vectors."""

    def __init__(self, input_dim, hidden_dims=(128, 128), latent_dim=3,
                 output_dim=None, beta=1.0, activation=nn.GELU):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims, hidden_dims)
        self.beta = beta
        self.encoder = MLPEncoder(input_dim, hidden_dims, latent_dim, activation)
        self.decoder = MLPDecoder(latent_dim, tuple(reversed(hidden_dims)), output_dim, activation)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def loss(self, recon, target, mu, logvar):
        recon_loss = F.mse_loss(recon, target)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss


# ---------------------------------------------------------------------------
# Conv VAE (temporal trajectories)
# ---------------------------------------------------------------------------

class ConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, latent_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            Conv1dBlock(in_channels, hidden_dim, kernel_size=7),
            Conv1dBlock(hidden_dim, hidden_dim, kernel_size=5, dilation=2),
            Conv1dBlock(hidden_dim, hidden_dim, kernel_size=5, dilation=4),
            Conv1dBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=8),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x).mean(dim=-1)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=3, hidden_dim=64, seq_length=40):
        super().__init__()
        self.seq_length = seq_length
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.decode_net = nn.Sequential(
            Conv1dBlock(hidden_dim, hidden_dim, kernel_size=5),
            Conv1dBlock(hidden_dim, hidden_dim, kernel_size=5, dilation=2),
            Conv1dBlock(hidden_dim, hidden_dim // 2, kernel_size=5, dilation=4),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(self, z):
        z_time = self.z_proj(z).unsqueeze(-1).expand(-1, -1, self.seq_length)
        return self.decode_net(z_time)


class BetaVAE(nn.Module):
    """Conv1d-based beta-VAE for temporal trajectories."""

    def __init__(self, in_channels, hidden_dim=64, latent_dim=3,
                 seq_length=40, beta=1.0, alpha=0.3):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.encoder = ConvEncoder(in_channels, hidden_dim, latent_dim)
        self.decoder = ConvDecoder(latent_dim, hidden_dim, seq_length)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def loss(self, recon, target, mu, logvar):
        temporal = F.mse_loss(recon, target)
        spectral = spectral_loss(recon, target) if self.alpha != 0 else 0
        recon_loss = (1 - self.alpha) * temporal + self.alpha * spectral
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss


if __name__ == "__main__":
    # --- MLP VAE tests ---
    B, D = 4, 5
    for ld in (1, 2, 4):
        m = VAE(D, hidden_dims=(16, 8), latent_dim=ld, beta=0.5)
        x = torch.randn(B, D)
        recon, mu, logvar = m(x)
        assert recon.shape == (B, D), f"recon shape: {recon.shape}"
        assert mu.shape == (B, ld), f"mu shape: {mu.shape}"
        total, recon_l, kl_l = m.loss(recon, x, mu, logvar)
        assert total.shape == ()
        total.backward()
        assert all(p.grad is not None for p in m.parameters())
    print("vae (MLP): all tests passed")

    # --- Conv VAE tests ---
    B, T, C = 4, 40, 1
    x = torch.randn(B, C, T)
    tgt = torch.randn(B, 1, T)

    for ld in (1, 3, 8):
        m = BetaVAE(C, hidden_dim=32, latent_dim=ld, seq_length=T)
        recon, mu, logvar = m(x)
        assert recon.shape == (B, 1, T), f"recon shape: {recon.shape}"
        assert mu.shape == (B, ld), f"mu shape: {mu.shape}"
        assert logvar.shape == (B, ld), f"logvar shape: {logvar.shape}"

        total, recon_l, kl_l = m.loss(recon, tgt, mu, logvar)
        assert total.shape == (), "loss not scalar"
        assert not torch.isnan(total), "loss is NaN"

        total.backward()
        assert all(p.grad is not None for p in m.parameters()), "missing grads"

    # deterministic in eval
    m.eval()
    with torch.no_grad():
        mu1, _ = m.encoder(x)
        mu2, _ = m.encoder(x)
    torch.testing.assert_close(mu1, mu2)

    print("vae (Conv): all tests passed")
