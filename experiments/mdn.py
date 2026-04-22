import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    from hastyplot import qplot
    import torch
    from torch import tensor as T
    import torch.functional as F
    import torch.nn as nn
    import pandas as pd
    from pandas import DataFrame as DF
    from sklearn.model_selection import train_test_split
    import torch.optim as optim

    return DF, mo, nn, np, optim, qplot, torch


@app.cell
def _(DF, mo, np, qplot, torch):
    N = 2000
    _y = np.random.uniform(-1, 1, N)          
    _x = _y + 0.3 * np.sin(1.8 * np.pi * _y)     
    _x += np.random.normal(0, 0.02, N)        

    x = torch.FloatTensor(_x).unsqueeze(-1)
    y = torch.FloatTensor(_y).unsqueeze(-1)
    dataset = torch.utils.data.TensorDataset(x,y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    mo.ui.altair_chart(qplot(DF(dict(x=_x, y=_y)), 'x', 'y'))
    return (loader,)


@app.cell
def _(nn, torch):
    class MDN(nn.Module):
        def __init__(self, in_feat, out_feat, n_gaussians):
            super().__init__()
            self.in_feat = in_feat
            self.out_feat = out_feat
            self.n_gaussians = n_gaussians
            self.pi = nn.Sequential( # mixing weights (sum to 1)
                nn.Linear(in_feat, n_gaussians),
                nn.Softmax(dim=1)
            )
            self.sigma = nn.Linear(in_feat, out_feat * n_gaussians) # variances (>0),
            # so we treat this as log(sigma), and exp in the forward;
            self.mu = nn.Linear(in_feat, out_feat * n_gaussians)  # means (unconstrained)

        def forward(self, x):
            pi = self.pi(x)
            sigma = torch.exp(self.sigma(x))
            sigma = sigma.view(-1, self.n_gaussians, self.out_feat) 
            # shaping it into (datapoints, gaussians, features)
            mu = self.mu(x)
            mu = mu.view(-1, self.n_gaussians, self.out_feat)
            return pi,mu, sigma


    def mdn_loss(pi_logits, mu, sigma, y):
        y = y.unsqueeze(1) # (B, 1, D)
        log_gaussian = ( # one logLik per gaussian component
            -0.5 * torch.log(torch.tensor(2*torch.pi))
            - torch.log(sigma)
            - 0.5 * ( ( y - mu ) / sigma) ** 2 
        ).sum(dim=-1)  # sum them
        log_components = torch.log(pi_logits) + log_gaussian
        loss = - torch.logsumexp(log_components, dim=-1).mean()
        return loss


    return MDN, mdn_loss


@app.cell
def _(MDN, loader, mdn_loss, nn, optim, torch):
    model     = nn.Sequential(
        nn.Linear(1,64),
        nn.Tanh(),
        nn.Linear(64,64),
        nn.Tanh(),
        MDN(in_feat=64, out_feat=1, n_gaussians=3)
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 800
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            _pi, _mu, _sigma = model(xb)
            _loss = mdn_loss(_pi, _mu, _sigma, yb)
            _loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # safety
            optimizer.step()
            epoch_loss += _loss.item()
        avg = epoch_loss / len(loader)
        losses.append(avg)
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | loss {avg:.4f}")


    return (model,)


@app.cell
def _(DF, model, qplot, torch):
    model.eval()
    pts = 50
    with torch.no_grad():
        x_test = torch.linspace(-1, 1,pts).unsqueeze(1)
        _pi, _mu, _sigma = model(x_test)

        rows = []
        for draw in range(10):
            idx       = torch.multinomial(_pi, 1).squeeze(1)
            mu_sel    = _mu[torch.arange(pts), idx, 0]
            sigma_sel = _sigma[torch.arange(pts), idx, 0]
            s         = torch.normal(mu_sel, sigma_sel).numpy()

            for i, (t_val, x_val) in enumerate(zip(x_test.squeeze().numpy(), s)):
                rows.append(dict(t=t_val, x=x_val, draw=draw))

    df_samples = DF(rows)

    # overlay: data in one color, MDN samples in another
    qplot(df_samples, 't', 'x', color='draw')
    return


@app.cell
def _(model, torch):
    model(torch.tensor([0.8]).unsqueeze(1))
    return


if __name__ == "__main__":
    app.run()
