import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dBlock(nn.Module):
    """1D conv with optional dilation, batch norm, and GELU. Non-causal (symmetric padding)."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = ((kernel_size - 1) * dilation) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad))
        return F.gelu(self.bn(self.conv(x)))


class Decoder(nn.Module):
    def __init__(self, latent_dim=3, stim_channels=9, hidden_dim=64, seq_length=40):
        super().__init__()
        self.seq_length = seq_length
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.stim_net = nn.Sequential(
            Conv1dBlock(stim_channels, hidden_dim // 2, kernel_size=5),
            Conv1dBlock(hidden_dim // 2, hidden_dim, kernel_size=5),
        )
        self.decode_net = nn.Sequential(
            Conv1dBlock(hidden_dim * 2, hidden_dim, kernel_size=5),
            Conv1dBlock(hidden_dim, hidden_dim, kernel_size=5, dilation=2),
            Conv1dBlock(hidden_dim, hidden_dim // 2, kernel_size=5, dilation=4),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(self, z, stim_cond):
        z_time = self.z_proj(z).unsqueeze(-1).expand(-1, -1, self.seq_length)
        s = self.stim_net(stim_cond)
        return self.decode_net(torch.cat([z_time, s], dim=1))


def spectral_loss(recon, target):
    return F.mse_loss(torch.abs(torch.fft.rfft(recon, dim=-1)),
                      torch.abs(torch.fft.rfft(target, dim=-1)))
