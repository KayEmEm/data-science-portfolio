"""A dense autoencoder in PyTorch: the anomaly score of a flow is its mean squared reconstruction error.

Trained without labels on the training sample; a random tenth of that sample is held out for early
stopping. Every epoch is saved, so a stopped run resumes at the next epoch.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flow_anomaly.config import SEED, AutoencoderSettings


class Autoencoder(nn.Module):
    def __init__(self, n_features: int, settings: AutoencoderSettings):
        super().__init__()
        sizes = [n_features, *settings.hidden, settings.code]
        encoder, decoder = [], []
        for a, b in zip(sizes[:-1], sizes[1:], strict=True):
            encoder += [nn.Linear(a, b), nn.ReLU()]
        encoder.pop()  # linear code layer
        reverse = sizes[::-1]
        for a, b in zip(reverse[:-1], reverse[1:], strict=True):
            decoder += [nn.Linear(a, b), nn.ReLU()]
        decoder.pop()  # linear output layer
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _loss(model: Autoencoder, x: np.ndarray, batch: int = 8192) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunk = torch.from_numpy(x[start : start + batch])
            total += float(((model(chunk) - chunk) ** 2).mean(dim=1).sum())
    return total / len(x)


def fit(x: np.ndarray, settings: AutoencoderSettings, folder: Path, seed: int = SEED) -> dict:
    """Train with early stopping; returns the history. Resumes from ``folder/last.pt``."""
    torch.set_num_threads(settings.threads)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    n_hold = max(1, int(len(x) * settings.holdout))
    hold, train = x[order[:n_hold]], x[order[n_hold:]]
    torch.manual_seed(seed)
    model = Autoencoder(x.shape[1], settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    state = {"epoch": 0, "history": [], "best_loss": float("inf"), "best_epoch": 0, "stopped": False}
    last = folder / "last.pt"
    if last.exists():
        stored = torch.load(last, weights_only=False)
        model.load_state_dict(stored["model"])
        optimizer.load_state_dict(stored["optimizer"])
        state = stored["state"]
        torch.set_rng_state(stored["torch_rng"])
        print(f"autoencoder: resumed after epoch {state['epoch']}")
    data = torch.from_numpy(train)
    while state["epoch"] < settings.epochs and not state["stopped"]:
        model.train()
        epoch = state["epoch"] + 1
        perm = torch.randperm(len(data))
        total = 0.0
        for start in range(0, len(data), settings.batch_size):
            batch = data[perm[start : start + settings.batch_size]]
            optimizer.zero_grad()
            loss = ((model(batch) - batch) ** 2).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        hold_loss = _loss(model, hold)
        state["history"].append({"epoch": epoch, "train_loss": total / len(data), "holdout_loss": hold_loss})
        if hold_loss < state["best_loss"]:
            state["best_loss"], state["best_epoch"] = hold_loss, epoch
            torch.save(model.state_dict(), folder / "best.pt")
        state["stopped"] = epoch - state["best_epoch"] >= settings.patience
        state["epoch"] = epoch
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "state": state,
                "torch_rng": torch.get_rng_state(),
            },
            last,
        )
        print(f"autoencoder epoch {epoch}: train {total / len(data):.4f}, holdout {hold_loss:.4f}")
    (folder / "history.json").write_text(json.dumps(state, indent=2))
    return state


def load(folder: Path, n_features: int, settings: AutoencoderSettings) -> Autoencoder:
    model = Autoencoder(n_features, settings)
    model.load_state_dict(torch.load(folder / "best.pt", weights_only=True))
    model.eval()
    return model


def feature_errors(model: Autoencoder, x: np.ndarray, batch: int = 8192) -> np.ndarray:
    """Squared reconstruction error per flow and feature."""
    out = np.empty_like(x)
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunk = torch.from_numpy(x[start : start + batch])
            out[start : start + batch] = ((model(chunk) - chunk) ** 2).numpy()
    return out
