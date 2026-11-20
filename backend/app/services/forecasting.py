"""Next-day close forecasting: naive baseline, Linear Regression and an LSTM, evaluated on a chronological split."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torch import nn

from ..config import settings

torch.set_num_threads(settings.torch_threads)
_train_lock = threading.Lock()


class TrainingBusy(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainConfig:
    lookback: int = 30
    test_fraction: float = 0.2
    epochs: int = 40
    hidden_size: int = 32
    horizon: int = 10
    seed: int = 42


class MinMax:
    """Scaler fitted on the training segment only, so the test period cannot leak into training."""

    def __init__(self, values: np.ndarray):
        self.lo = float(values.min())
        self.span = float(values.max() - values.min()) or 1.0

    def fwd(self, x: np.ndarray) -> np.ndarray:
        return (x - self.lo) / self.span

    def inv(self, x: np.ndarray) -> np.ndarray:
        return x * self.span + self.lo


def windows(series: np.ndarray, lookback: int, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Windows whose target index t lies in [start, end); inputs are the `lookback` values before t."""
    idx = np.arange(max(start, lookback), end)
    x = np.stack([series[t - lookback:t] for t in idx]) if len(idx) else np.empty((0, lookback))
    return x, series[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.head(out[:, -1]).squeeze(-1)


def _fit_lstm(x: np.ndarray, y: np.ndarray, cfg: TrainConfig) -> tuple[LSTMRegressor, list[float]]:
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = LSTMRegressor(cfg.hidden_size)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    xt, yt = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    losses = []
    for _ in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        total = 0.0
        for i in range(0, len(xt), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        losses.append(total / len(xt))
    model.eval()
    return model, losses


def _lstm_predict(model: LSTMRegressor, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy().astype(float)


def metrics(actual: np.ndarray, pred: np.ndarray, previous: np.ndarray) -> dict:
    err = pred - actual
    # Days with no actual move, or where the model predicts no move, carry no direction to score.
    scored = (actual != previous) & (pred != previous)
    direction = np.sign(pred - previous)[scored] == np.sign(actual - previous)[scored]
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err) / np.abs(actual)) * 100),
        "directional_accuracy": float(direction.mean() * 100) if direction.size else None,
    }


def _recursive_forecast(predict_one, history: np.ndarray, lookback: int, steps: int) -> np.ndarray:
    buf = list(history[-lookback:])
    out = []
    for _ in range(steps):
        nxt = float(predict_one(np.array(buf[-lookback:])[None, :])[0])
        out.append(nxt)
        buf.append(nxt)
    return np.array(out)


def train_and_evaluate(dates: list[str], close: np.ndarray, cfg: TrainConfig) -> dict:
    n = len(close)
    split = int(round(n * (1 - cfg.test_fraction)))
    if split - cfg.lookback < 50 or n - split < 20:
        raise ValueError("Not enough data for this lookback and test size.")
    if not _train_lock.acquire(blocking=False):
        raise TrainingBusy("Another model is training; try again in a few seconds.")
    try:
        started = time.perf_counter()
        scaler = MinMax(close[:split])
        scaled = scaler.fwd(close)
        x_train, y_train = windows(scaled, cfg.lookback, 0, split)
        x_test, _ = windows(scaled, cfg.lookback, split, n)
        actual = close[split:]
        previous = close[split - 1:n - 1]

        linear = LinearRegression().fit(x_train, y_train)
        lstm, losses = _fit_lstm(x_train, y_train, cfg)

        preds = {
            "naive": previous.copy(),
            "linear_regression": scaler.inv(linear.predict(x_test)),
            "lstm": scaler.inv(_lstm_predict(lstm, x_test)),
        }
        forecast = {
            "naive": np.full(cfg.horizon, close[-1]),
            "linear_regression": scaler.inv(_recursive_forecast(linear.predict, scaled, cfg.lookback, cfg.horizon)),
            "lstm": scaler.inv(_recursive_forecast(lambda w: _lstm_predict(lstm, w), scaled, cfg.lookback, cfg.horizon)),
        }
        future = np.busday_offset(np.datetime64(dates[-1]), np.arange(1, cfg.horizon + 1), roll="forward")
        return {
            "config": cfg.__dict__,
            "train_rows": split,
            "test_rows": n - split,
            "split_date": dates[split],
            "metrics": {k: metrics(actual, v, previous) for k, v in preds.items()},
            "test": {"dates": dates[split:], "actual": actual.round(4).tolist(),
                     **{k: np.round(v, 4).tolist() for k, v in preds.items()}},
            "forecast": {"dates": [str(d) for d in future], **{k: np.round(v, 4).tolist() for k, v in forecast.items()}},
            "lstm_loss": [round(v, 6) for v in losses],
            "seconds": round(time.perf_counter() - started, 2),
        }
    finally:
        _train_lock.release()
