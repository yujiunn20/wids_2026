import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# 1. Dataset
# =========================
class WildfireDataset(Dataset):
    def __init__(self, df, feature_cols, time_col="time_to_hit_hours"):
        self.x = df[feature_cols].values.astype(np.float32)
        self.t = df[time_col].values.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx], dtype=torch.float32),
            "t": torch.tensor(self.t[idx], dtype=torch.float32),
        }


# =========================
# 2. Model: g(x,t) -> lambda(t|x)=exp(g)
# =========================
class HazardNet(nn.Module):
    def __init__(self, in_features, hidden_dim=128):
        super().__init__()

        # feature encoder
        self.feature_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # combine encoded x with time t
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, t):
        """
        x: [B, D]
        t: [B, 1]
        return g(x,t): [B, 1]
        """
        hx = self.feature_net(x)
        z = torch.cat([hx, t], dim=1)
        g = self.head(z)
        return g

    def hazard(self, x, t):
        """
        lambda(t|x) = exp(g(x,t))
        x: [B, D]
        t: [B, 1]
        """
        g = self.forward(x, t)
        lam = torch.exp(g)
        return lam


# =========================
# 3. Numerical integration
# =========================
def cumulative_hazard(model, x, T, n_steps=100):
    """
    Approximate integral_0^T lambda(s|x) ds by trapezoidal rule.

    x: [B, D]
    T: [B] or [B, 1]
    return: [B, 1]
    """
    if T.ndim == 1:
        T = T.unsqueeze(1)  # [B,1]

    batch_size = x.size(0)
    device = x.device

    # u in [0,1], then s = u * T
    u = torch.linspace(0.0, 1.0, steps=n_steps, device=device).view(1, n_steps, 1)  # [1,N,1]
    t_grid = u * T.unsqueeze(1)  # [B,N,1]

    # expand x to match time grid
    x_grid = x.unsqueeze(1).expand(batch_size, n_steps, x.size(1))  # [B,N,D]

    # flatten for model forward
    x_flat = x_grid.reshape(batch_size * n_steps, x.size(1))
    t_flat = t_grid.reshape(batch_size * n_steps, 1)

    lam_flat = model.hazard(x_flat, t_flat)  # [B*N,1]
    lam = lam_flat.reshape(batch_size, n_steps, 1)  # [B,N,1]

    # trapezoidal integration over actual time axis
    # dt for each sample = T / (n_steps - 1)
    dt = T / (n_steps - 1)  # [B,1]

    trap = 0.5 * (lam[:, :-1, :] + lam[:, 1:, :]).sum(dim=1) * dt  # [B,1]
    return trap


# =========================
# 4. NLL loss for uncensored event times
# =========================
def survival_nll_loss(model, x, T, n_steps=100):
    """
    For observed event time T:
    loss = -log f(T|x)
         = -log lambda(T|x) + integral_0^T lambda(s|x) ds
    """
    if T.ndim == 1:
        T_input = T.unsqueeze(1)  # [B,1]
    else:
        T_input = T

    lam_T = model.hazard(x, T_input)               # [B,1]
    H_T = cumulative_hazard(model, x, T, n_steps) # [B,1]

    # avoid log(0)
    eps = 1e-8
    log_lam_T = torch.log(lam_T + eps)

    loss = -(log_lam_T - H_T)  # [B,1]
    return loss.mean()


# =========================
# 5. Survival / CDF prediction
# =========================
@torch.no_grad()
def predict_cdf(model, x, eval_times, n_steps=200):
    """
    Return P(T <= t | x) for each t in eval_times
    x: [B,D]
    eval_times: list like [12,24,48,72]
    return: [B, len(eval_times)]
    """
    preds = []
    for t_val in eval_times:
        T = torch.full((x.size(0),), float(t_val), device=x.device)
        H_t = cumulative_hazard(model, x, T, n_steps=n_steps)  # [B,1]
        S_t = torch.exp(-H_t)                                  # [B,1]
        cdf_t = 1.0 - S_t                                      # [B,1]
        preds.append(cdf_t)

    return torch.cat(preds, dim=1)  # [B,4]


# =========================
# 6. Example training loop
# =========================
def train_one_epoch(model, loader, optimizer, device, n_steps=100):
    model.train()
    total_loss = 0.0

    for batch in loader:
        x = batch["x"].to(device)
        t = batch["t"].to(device)

        optimizer.zero_grad()
        loss = survival_nll_loss(model, x, t, n_steps=n_steps)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# =========================
# 7. Example validation / inference
# =========================
@torch.no_grad()
def predict_dataframe(model, df, feature_cols, device, eval_times=(12, 24, 48, 72), batch_size=256):
    model.eval()

    x_all = torch.tensor(df[feature_cols].values.astype(np.float32), device=device)
    outputs = []

    for start in range(0, len(df), batch_size):
        x = x_all[start:start+batch_size]
        prob = predict_cdf(model, x, eval_times=eval_times, n_steps=200)
        outputs.append(prob.cpu().numpy())

    prob = np.concatenate(outputs, axis=0)

    sub = pd.DataFrame({
        "event_id": df["event_id"].values,
        "prob_12h": prob[:, 0],
        "prob_24h": prob[:, 1],
        "prob_48h": prob[:, 2],
        "prob_72h": prob[:, 3],
    })
    return sub


# =========================
# 8. Minimal usage example
# =========================
if __name__ == "__main__":
    # train_df = pd.read_csv("train.csv")
    # test_df  = pd.read_csv("test.csv")

    # 你之後自己改
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    exclude_cols = ["event_id", "time_to_hit_hours", "event"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    dataset = WildfireDataset(train_df, feature_cols, time_col="time_to_hit_hours")
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HazardNet(in_features=len(feature_cols), hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        train_loss = train_one_epoch(model, loader, optimizer, device, n_steps=100)
        print(f"Epoch {epoch+1:02d} | train_loss = {train_loss:.6f}")

    submission = predict_dataframe(
        model,
        test_df,
        feature_cols=feature_cols,
        device=device,
        eval_times=(12, 24, 48, 72),
        batch_size=256,
    )

    submission.to_csv("submission_survival.csv", index=False)
    print(submission.head())