# Stock Market Prediction

Forecast the next day's closing price with **Linear Regression** and an **LSTM** (PyTorch), and compare both against
a **naive baseline** (tomorrow = today) on a chronological hold-out period. Rebuilt from the project on my resume.

| Part | Port | Tech |
| --- | --- | --- |
| Frontend | 5113 | React 18, TypeScript, Vite, Tailwind CSS, SVG charts |
| Backend | 8113 | FastAPI, pandas, scikit-learn, PyTorch (CPU) |

## How it works

1. **Data** — three seeded *synthetic* price series (trend, volatile with drawdown, mean-reverting cycle) or your own
   CSV with a `Date` and `Close`/`Adj Close` column (≥ 120 rows, ≤ 5 MB). Rows are sorted by date, duplicates and
   non-positive prices dropped. No market data is downloaded or redistributed.
2. **Split** — the last 20 % (configurable) of days is the test period. The Min-Max scaler is fitted on the training
   days only, and every input window contains only values *before* the day being predicted.
3. **Models** — Linear Regression on the last *N* scaled closes; a 1-layer LSTM (configurable hidden units/epochs,
   Adam, MSE, fixed seed → reproducible); naive persistence baseline.
4. **Metrics** — RMSE, MAE, MAPE and directional accuracy (did the model call up/down correctly?).
5. **Forecast** — a recursive multi-day forecast after the last date; each prediction is fed back as input, so errors
   compound with the horizon.

On random-walk-like prices the naive baseline is very hard to beat and directional accuracy stays near 50 %. The app
shows this rather than hiding it. **Educational demo — not financial advice.**

## Run locally

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --port 8113
```

```bash
cd frontend
npm install
npm run dev        # http://localhost:5113
```

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/health` | Status |
| GET / POST | `/api/datasets` | List / upload CSV (multipart `file`) |
| POST | `/api/datasets/samples` | Create the synthetic samples (idempotent) |
| GET / DELETE | `/api/datasets/{id}` | Series + summary stats / delete |
| POST | `/api/train` | `{dataset_id, lookback, test_fraction, epochs, hidden_size, horizon, seed}` → metrics, test predictions, forecast, LSTM loss |

## Tests

```bash
cd backend && pip install -r requirements-dev.txt && pytest
cd frontend && npx vitest run && npm run build
```

Covered: leakage-free windows, train-only scaling, metrics, CSV validation (columns, binary, size, row count),
deterministic training, API validation, upload/delete and path-safe dataset ids.

## Deployment

- **Backend**: any Python host (e.g. Render, Railway, Azure App Service). Install the CPU PyTorch wheel first, then
  `requirements.txt`; start with `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. Set `CORS_ORIGINS` to the
  frontend URL. PyTorch needs ~1 GB RAM; on small free tiers lower `TORCH_THREADS` and expect slow training.
  `backend/data/` (uploads) is lost on redeploy unless a persistent disk is attached; samples are recreated on demand.
- **Frontend**: static site (Vercel, Netlify, GitHub Pages) — build with `VITE_API_URL=<backend URL> npm run build`
  and publish `frontend/dist`.

## Limitations

- Uses closing prices only (no volume, news or fundamentals); one training run at a time.
- The synthetic samples are generated, not real tickers. Uploaded data is stored in `backend/data/` until deleted.
