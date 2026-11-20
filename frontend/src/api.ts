export interface DatasetMeta {
  id: string;
  name: string;
  description: string;
  synthetic: boolean;
  rows: number;
  start: string;
  end: string;
  created_at: string;
}

export interface DatasetDetail extends DatasetMeta {
  dates: string[];
  close: number[];
  stats: { first: number; last: number; min: number; max: number; total_return_pct: number; daily_volatility_pct: number };
}

export type ModelKey = 'naive' | 'linear_regression' | 'lstm';

export interface Metrics {
  rmse: number;
  mae: number;
  mape: number;
  directional_accuracy: number | null;
}

export interface TrainParams {
  lookback: number;
  test_fraction: number;
  epochs: number;
  hidden_size: number;
  horizon: number;
  seed: number;
}

export interface TrainResult {
  dataset_id: string;
  config: TrainParams;
  train_rows: number;
  test_rows: number;
  split_date: string;
  metrics: Record<ModelKey, Metrics>;
  test: { dates: string[]; actual: number[] } & Record<ModelKey, number[]>;
  forecast: { dates: string[] } & Record<ModelKey, number[]>;
  lstm_loss: number[];
  seconds: number;
}

const BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '');

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new Error('Cannot reach the API. Is the backend running on port 8113?');
  }
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === 'string') msg = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(msg);
  }
  return (res.status === 204 ? undefined : await res.json()) as T;
}

export const api = {
  datasets: () => call<DatasetMeta[]>('/api/datasets'),
  loadSamples: () => call<DatasetMeta[]>('/api/datasets/samples', { method: 'POST' }),
  upload: (file: File) => {
    const form = new FormData();
    form.append('file', file);
    return call<DatasetMeta>('/api/datasets', { method: 'POST', body: form });
  },
  dataset: (id: string) => call<DatasetDetail>(`/api/datasets/${encodeURIComponent(id)}`),
  remove: (id: string) => call<void>(`/api/datasets/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  train: (dataset_id: string, params: TrainParams) =>
    call<TrainResult>('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_id, ...params }),
    }),
};
