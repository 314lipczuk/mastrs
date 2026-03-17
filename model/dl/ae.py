import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(dims, activation):
    """Build a sequential MLP from a list of layer dimensions."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, activation=nn.GELU):
        super().__init__()
        self.net = _build_mlp([input_dim] + list(hidden_dims) + [latent_dim], activation)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, activation=nn.GELU):
        super().__init__()
        self.net = _build_mlp([latent_dim] + list(hidden_dims) + [output_dim], activation)

    def forward(self, z):
        return self.net(z)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), latent_dim=3,
                 output_dim=None, activation=nn.GELU):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims, hidden_dims)
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim, activation)
        self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)), output_dim, activation)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def loss(self, recon, target):
        return F.mse_loss(recon, target)


if __name__ == "__main__":
    B, D_in = 4, 40

    configs = [
        dict(hidden_dims=64, latent_dim=3),           # int shorthand
        dict(hidden_dims=(128, 64), latent_dim=3),     # tapering
        dict(hidden_dims=(32,), latent_dim=2),         # single hidden layer
        dict(hidden_dims=(64, 32, 16), latent_dim=1),  # deep narrow
    ]

    for cfg in configs:
        x = torch.randn(B, D_in)
        tgt = torch.randn(B, D_in)

        m = AutoEncoder(D_in, **cfg)
        recon, z = m(x)
        assert recon.shape == (B, D_in), f"recon shape: {recon.shape}"
        assert z.shape == (B, cfg["latent_dim"]), f"z shape: {z.shape}"

        loss = m.loss(recon, tgt)
        assert loss.shape == (), "loss not scalar"
        assert not torch.isnan(loss), "loss is NaN"

        loss.backward()
        assert all(p.grad is not None for p in m.parameters()), "missing grads"
        print(f"  {cfg} -> params={sum(p.numel() for p in m.parameters()):,}")

    # deterministic in eval
    m.eval()
    x = torch.randn(B, D_in)
    with torch.no_grad():
        _, z1 = m(x)
        _, z2 = m(x)
    torch.testing.assert_close(z1, z2)

    print("ae: all tests passed")
