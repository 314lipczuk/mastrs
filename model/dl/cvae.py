import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Conv1dBlock, Decoder, spectral_loss


class Encoder(nn.Module):
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


class ConditionalBetaVAE(nn.Module):
    def __init__(self, in_channels, stim_channels, hidden_dim=64,
                 latent_dim=3, seq_length=40, beta=1.0, alpha=0.3):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, stim_channels, hidden_dim, seq_length)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, encoder_input, stim_cond):
        mu, logvar = self.encoder(encoder_input)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, stim_cond)
        return recon, mu, logvar

    def loss(self, recon, target, mu, logvar):
        temporal = F.mse_loss(recon, target)
        spectral = spectral_loss(recon, target) if self.alpha != 0 else 0
        recon_loss = (1 - self.alpha) * temporal + self.alpha * spectral
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss


if __name__ == "__main__":
    B, T, S, L = 4, 40, 9, 3
    x = torch.randn(B, 1 + S, T)
    s = torch.randn(B, S, T)
    tgt = torch.randn(B, 1, T)

    for ld in (1, 3, 8):
        m = ConditionalBetaVAE(1 + S, S, hidden_dim=32, latent_dim=ld, seq_length=T)
        recon, mu, logvar = m(x, s)
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

    print("cvae: all tests passed")
