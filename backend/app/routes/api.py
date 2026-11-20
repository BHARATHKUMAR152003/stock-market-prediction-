import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ..config import settings
from ..services import datasets
from ..services.forecasting import TrainConfig, TrainingBusy, train_and_evaluate

router = APIRouter(prefix="/api")


class TrainRequest(BaseModel):
    dataset_id: str = Field(max_length=40)
    lookback: int = Field(30, ge=5, le=120)
    test_fraction: float = Field(0.2, ge=0.1, le=0.4)
    epochs: int = Field(40, ge=1, le=200)
    hidden_size: int = Field(32, ge=4, le=128)
    horizon: int = Field(10, ge=1, le=30)
    seed: int = Field(42, ge=0, le=1_000_000)


def _not_found(exc: datasets.DatasetError) -> HTTPException:
    return HTTPException(404 if "not found" in str(exc) else 400, str(exc))


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "datasets": len(datasets.list_datasets()), "max_upload_mb": settings.max_upload_mb}


@router.get("/datasets")
def list_datasets() -> list[dict]:
    return datasets.list_datasets()


@router.post("/datasets/samples")
def load_samples() -> list[dict]:
    return datasets.load_samples()


@router.post("/datasets", status_code=201)
async def upload(file: UploadFile = File(...)) -> dict:
    raw = await file.read(int(settings.max_upload_mb * 1024 * 1024) + 1)
    try:
        return datasets.create_from_upload(file.filename or "upload.csv", raw)
    except datasets.DatasetError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict:
    try:
        meta = datasets.get_meta(dataset_id)
        df = datasets.load_series(dataset_id)
    except datasets.DatasetError as exc:
        raise _not_found(exc) from exc
    close = df["close"].to_numpy(float)
    returns = np.diff(np.log(close))
    return {
        **meta,
        "dates": df["date"].tolist(),
        "close": close.round(4).tolist(),
        "stats": {
            "first": float(close[0]),
            "last": float(close[-1]),
            "min": float(close.min()),
            "max": float(close.max()),
            "total_return_pct": float((close[-1] / close[0] - 1) * 100),
            "daily_volatility_pct": float(returns.std() * 100),
        },
    }


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str) -> None:
    try:
        datasets.delete(dataset_id)
    except datasets.DatasetError as exc:
        raise _not_found(exc) from exc


@router.post("/train")
async def train(req: TrainRequest) -> dict:
    try:
        df = datasets.load_series(req.dataset_id)
    except datasets.DatasetError as exc:
        raise _not_found(exc) from exc
    cfg = TrainConfig(**req.model_dump(exclude={"dataset_id"}))
    try:
        result = await run_in_threadpool(train_and_evaluate, df["date"].tolist(), df["close"].to_numpy(float), cfg)
    except TrainingBusy as exc:
        raise HTTPException(429, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"dataset_id": req.dataset_id, **result}
