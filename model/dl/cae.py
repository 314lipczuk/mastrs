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
        self.fc_z = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x).mean(dim=-1)
        return self.fc_z(h)


class ConditionalAE(nn.Module):
    def __init__(self, in_channels, stim_channels, hidden_dim=64,
                 latent_dim=3, seq_length=40, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, stim_channels, hidden_dim, seq_length)

    def forward(self, encoder_input, stim_cond):
        z = self.encoder(encoder_input)
        recon = self.decoder(z, stim_cond)
        return recon, z

    def loss(self, recon, target):
        temporal = F.mse_loss(recon, target)
        spectral = spectral_loss(recon, target) if self.alpha != 0 else 0
        recon_loss = (1 - self.alpha) * temporal + self.alpha * spectral
        return recon_loss


if __name__ == "__main__":
    B, T, S, L = 4, 40, 9, 3
    x = torch.randn(B, 1 + S, T)
    s = torch.randn(B, S, T)
    tgt = torch.randn(B, 1, T)

    for ld in (1, 3, 8):
        m = ConditionalAE(1 + S, S, hidden_dim=32, latent_dim=ld, seq_length=T)
        recon, z = m(x, s)
        assert recon.shape == (B, 1, T), f"recon shape: {recon.shape}"
        assert z.shape == (B, ld), f"z shape: {z.shape}"

        loss = m.loss(recon, tgt)
        assert loss.shape == (), "loss not scalar"
        assert not torch.isnan(loss), "loss is NaN"

        loss.backward()
        assert all(p.grad is not None for p in m.parameters()), "missing grads"

    # deterministic in eval
    m.eval()
    with torch.no_grad():
        _, z1 = m(x, s)
        _, z2 = m(x, s)
    torch.testing.assert_close(z1, z2)

    print("cae: all tests passed")
