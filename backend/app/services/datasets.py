"""Price datasets: seeded synthetic samples, CSV upload/validation and on-disk storage."""
from __future__ import annotations

import io
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import settings

MIN_ROWS = 120


class DatasetError(ValueError):
    """Invalid upload or unknown dataset; the message is safe to show to users."""


@dataclass(frozen=True)
class Sample:
    slug: str
    name: str
    description: str
    seed: int
    drift: float
    volatility: float
    mean_reversion: float


SAMPLES = [
    Sample("synth-trend", "SYNTH-TREND (synthetic)", "Steady upward drift with moderate noise.", 6, 0.0006, 0.012, 0.0),
    Sample("synth-volatile", "SYNTH-VOL (synthetic)", "Little drift, high day-to-day volatility and a sharp drawdown.", 11, 0.0001, 0.026, 0.0),
    Sample("synth-cycle", "SYNTH-CYCLE (synthetic)", "Mean-reverting around a slow seasonal cycle.", 23, 0.0, 0.010, 0.04),
]


def _dir():
    path = settings.data_dir / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _synthetic_prices(s: Sample, days: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(s.seed)
    dates = pd.bdate_range("2021-01-04", periods=days)
    log_price = np.empty(days)
    log_price[0] = np.log(100.0)
    t = np.arange(days)
    anchor = np.log(100.0) + 0.15 * np.sin(2 * np.pi * t / 250)
    for i in range(1, days):
        shock = rng.normal(s.drift, s.volatility)
        if s.slug == "synth-volatile" and 600 <= i < 640:
            shock -= 0.01
        pull = s.mean_reversion * (anchor[i - 1] - log_price[i - 1])
        log_price[i] = log_price[i - 1] + shock + pull
    close = np.exp(log_price)
    spread = np.abs(rng.normal(0, s.volatility * 0.6, days)) * close
    open_ = close * (1 + rng.normal(0, s.volatility * 0.3, days))
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": open_.round(2),
        "high": (np.maximum(open_, close) + spread).round(2),
        "low": (np.minimum(open_, close) - spread).round(2),
        "close": close.round(2),
        "volume": rng.integers(800_000, 3_000_000, days),
    })


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: re.sub(r"[^a-z]", "", str(c).lower()) for c in df.columns}
    df = df.rename(columns=rename)
    if "close" not in df.columns and "adjclose" in df.columns:
        df = df.rename(columns={"adjclose": "close"})
    if "date" not in df.columns or "close" not in df.columns:
        raise DatasetError("The CSV needs a 'Date' column and a 'Close' (or 'Adj Close') column.")
    return df


def parse_csv(raw: bytes) -> pd.DataFrame:
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise DatasetError(f"File is larger than {settings.max_upload_mb:g} MB.")
    if b"\x00" in raw[:4096]:
        raise DatasetError("This does not look like a text CSV file.")
    try:
        df = pd.read_csv(io.BytesIO(raw), nrows=settings.max_rows + 1)
    except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
        raise DatasetError("Could not read the file as CSV.") from exc
    if len(df) > settings.max_rows:
        raise DatasetError(f"Too many rows (limit {settings.max_rows}).")
    df = _normalise_columns(df)

    dates = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    close = pd.to_numeric(df["close"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    clean = pd.DataFrame({"date": dates, "close": close}).dropna()
    clean = clean[clean["close"] > 0]
    clean = clean.sort_values("date").drop_duplicates("date", keep="last")
    if len(clean) < MIN_ROWS:
        raise DatasetError(f"Need at least {MIN_ROWS} rows with a valid date and positive close; found {len(clean)}.")
    clean["date"] = clean["date"].dt.strftime("%Y-%m-%d")
    return clean.reset_index(drop=True)


def _save(dataset_id: str, df: pd.DataFrame, meta: dict) -> dict:
    df[["date", "close"]].to_csv(_dir() / f"{dataset_id}.csv", index=False)
    meta = {**meta, "id": dataset_id, "rows": int(len(df)), "start": df["date"].iloc[0], "end": df["date"].iloc[-1]}
    (_dir() / f"{dataset_id}.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def list_datasets() -> list[dict]:
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in _dir().glob("*.json")]
    return sorted(metas, key=lambda m: (not m.get("synthetic", False), m["created_at"]))


def load_samples() -> list[dict]:
    existing = {m["id"] for m in list_datasets()}
    for s in SAMPLES:
        if s.slug not in existing:
            df = _synthetic_prices(s)
            _save(s.slug, df, {"name": s.name, "description": s.description, "synthetic": True,
                               "created_at": datetime.now(timezone.utc).isoformat()})
    return list_datasets()


def create_from_upload(filename: str, raw: bytes) -> dict:
    if len(list_datasets()) >= settings.max_datasets:
        raise DatasetError("Dataset limit reached; delete one first.")
    if not filename.lower().endswith(".csv"):
        raise DatasetError("Only .csv files are supported.")
    df = parse_csv(raw)
    name = re.sub(r"[^\w .()-]", "", filename.rsplit(".", 1)[0])[:60] or "Uploaded data"
    return _save(uuid.uuid4().hex[:12], df, {"name": name, "description": "Uploaded CSV", "synthetic": False,
                                              "created_at": datetime.now(timezone.utc).isoformat()})


def _check_id(dataset_id: str) -> None:
    if not re.fullmatch(r"[a-z0-9-]{1,40}", dataset_id) or not (_dir() / f"{dataset_id}.json").exists():
        raise DatasetError("Dataset not found.")


def get_meta(dataset_id: str) -> dict:
    _check_id(dataset_id)
    return json.loads((_dir() / f"{dataset_id}.json").read_text(encoding="utf-8"))


def load_series(dataset_id: str) -> pd.DataFrame:
    _check_id(dataset_id)
    return pd.read_csv(_dir() / f"{dataset_id}.csv")


def delete(dataset_id: str) -> None:
    _check_id(dataset_id)
    for ext in ("csv", "json"):
        (_dir() / f"{dataset_id}.{ext}").unlink(missing_ok=True)
