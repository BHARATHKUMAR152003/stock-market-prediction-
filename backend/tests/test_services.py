import numpy as np
import pytest

from app.services import datasets
from app.services.forecasting import MinMax, TrainConfig, metrics, train_and_evaluate, windows


def test_windows_never_use_the_target_or_later_values():
    series = np.arange(10, dtype=float)
    x, y = windows(series, 3, 5, 8)
    assert y.tolist() == [5, 6, 7]
    assert x.tolist() == [[2, 3, 4], [3, 4, 5], [4, 5, 6]]


def test_scaler_is_fitted_on_training_values_only():
    s = MinMax(np.array([10.0, 20.0]))
    assert s.fwd(np.array([30.0]))[0] == pytest.approx(2.0)
    assert s.inv(s.fwd(np.array([12.5])))[0] == pytest.approx(12.5)


def test_metrics_and_directional_accuracy():
    actual = np.array([11.0, 9.0, 10.0])
    previous = np.array([10.0, 11.0, 9.0])
    pred = np.array([10.5, 10.0, 8.5])
    m = metrics(actual, pred, previous)
    assert m["mae"] == pytest.approx(1.0)
    assert m["directional_accuracy"] == pytest.approx(100 * 2 / 3)


def test_parse_csv_accepts_adj_close_and_sorts_dates():
    raw = b"date,Adj Close\n" + b"\n".join(
        f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d},{200 - i}".encode() for i in reversed(range(130)))
    df = datasets.parse_csv(raw)
    assert list(df.columns) == ["date", "close"]
    assert df["date"].is_monotonic_increasing
    assert len(df) == 130


@pytest.mark.parametrize("raw, message", [
    (b"Date,Open\n2024-01-01,1", "'Close'"),
    (b"\x00\x01binary", "text CSV"),
    (b"Date,Close\n2024-01-01,1\n2024-01-02,2", "at least"),
])
def test_parse_csv_rejects_bad_files(raw, message):
    with pytest.raises(datasets.DatasetError, match=message):
        datasets.parse_csv(raw)


def test_training_is_deterministic_and_reports_all_models():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 300)))
    dates = [str(d) for d in np.busday_offset(np.datetime64("2023-01-02"), np.arange(300), roll="forward")]
    cfg = TrainConfig(lookback=20, epochs=5, horizon=5)
    a = train_and_evaluate(dates, close, cfg)
    b = train_and_evaluate(dates, close, cfg)
    assert a["metrics"] == b["metrics"]
    assert set(a["metrics"]) == {"naive", "linear_regression", "lstm"}
    assert len(a["test"]["actual"]) == a["test_rows"] == 60
    assert len(a["forecast"]["lstm"]) == 5
    assert a["forecast"]["dates"][0] > dates[-1]
    assert len(a["lstm_loss"]) == 5


def test_training_rejects_too_little_data():
    close = np.linspace(100, 120, 60)
    dates = [f"d{i}" for i in range(60)]
    with pytest.raises(ValueError, match="Not enough data"):
        train_and_evaluate(dates, close, TrainConfig(lookback=30))
