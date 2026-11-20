import { useCallback, useEffect, useRef, useState } from 'react';
import { AlertTriangle, Database, LineChart as ChartIcon, Loader2, Play, Trash2, Upload } from 'lucide-react';
import { api, type DatasetDetail, type DatasetMeta, type ModelKey, type TrainParams, type TrainResult } from './api';
import LineChart from './components/LineChart';

const MODELS: { key: ModelKey; label: string; color: string }[] = [
  { key: 'naive', label: 'Naive (yesterday’s close)', color: '#8b97a6' },
  { key: 'linear_regression', label: 'Linear Regression', color: '#fbbf24' },
  { key: 'lstm', label: 'LSTM', color: '#2dd4bf' },
];

const DEFAULTS: TrainParams = { lookback: 30, test_fraction: 0.2, epochs: 40, hidden_size: 32, horizon: 10, seed: 42 };

const FIELDS: { key: keyof TrainParams; label: string; min: number; max: number; step: number; hint: string }[] = [
  { key: 'lookback', label: 'Lookback window (days)', min: 5, max: 120, step: 1, hint: 'Past closes used for each prediction' },
  { key: 'test_fraction', label: 'Test share', min: 0.1, max: 0.4, step: 0.05, hint: 'Most recent part held out for evaluation' },
  { key: 'epochs', label: 'LSTM epochs', min: 1, max: 200, step: 1, hint: 'More epochs take longer' },
  { key: 'hidden_size', label: 'LSTM hidden units', min: 4, max: 128, step: 4, hint: 'Model capacity' },
  { key: 'horizon', label: 'Forecast horizon (days)', min: 1, max: 30, step: 1, hint: 'Recursive forecast after the last date' },
  { key: 'seed', label: 'Random seed', min: 0, max: 1000000, step: 1, hint: 'Same seed, same result' },
];

const money = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 });

export default function App() {
  const [list, setList] = useState<DatasetMeta[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [params, setParams] = useState<TrainParams>(DEFAULTS);
  const [result, setResult] = useState<TrainResult | null>(null);
  const [busy, setBusy] = useState<'list' | 'upload' | 'train' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const run = useCallback(async <T,>(kind: typeof busy, fn: () => Promise<T>) => {
    setBusy(kind);
    setError(null);
    try {
      return await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return undefined;
    } finally {
      setBusy(null);
    }
  }, []);

  const refresh = useCallback(async () => {
    const items = await run('list', api.datasets);
    if (items) {
      setList(items);
      setSelected((cur) => (cur && items.some((d) => d.id === cur) ? cur : items[0]?.id ?? null));
    }
  }, [run]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    setResult(null);
    setDetail(null);
    if (!selected) return;
    let live = true;
    api.dataset(selected).then(
      (d) => live && setDetail(d),
      (e) => live && setError(e instanceof Error ? e.message : String(e)),
    );
    return () => {
      live = false;
    };
  }, [selected]);

  const loadSamples = async () => {
    const items = await run('list', api.loadSamples);
    if (items) {
      setList(items);
      setSelected((cur) => cur ?? items[0]?.id ?? null);
    }
  };

  const onUpload = async (file: File | undefined) => {
    if (!file) return;
    const created = await run('upload', () => api.upload(file));
    if (fileRef.current) fileRef.current.value = '';
    if (created) {
      await refresh();
      setSelected(created.id);
    }
  };

  const onDelete = async (id: string) => {
    if (!window.confirm('Delete this dataset?')) return;
    await run('list', () => api.remove(id));
    if (selected === id) setSelected(null);
    await refresh();
  };

  const train = async () => {
    if (!selected) return;
    const r = await run('train', () => api.train(selected, params));
    if (r) setResult(r);
  };

  const best = result
    ? (Object.entries(result.metrics) as [ModelKey, TrainResult['metrics'][ModelKey]][]).reduce((a, b) => (b[1].rmse < a[1].rmse ? b : a))[0]
    : null;

  return (
    <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
      <header className="mb-6">
        <p className="text-xs font-semibold uppercase tracking-wider text-accent-400">Stock market prediction</p>
        <h1 className="mt-1 text-2xl font-semibold text-fog-100 sm:text-3xl">Linear Regression vs LSTM, measured honestly</h1>
        <p className="mt-2 max-w-3xl text-sm text-fog-300">
          Next-day closing-price forecasts evaluated on a chronological hold-out period against a naive baseline. Educational
          demo — not financial advice.
        </p>
      </header>

      {error && (
        <div role="alert" className="mb-4 flex items-start gap-2 rounded-xl border border-red-400/40 bg-red-400/10 px-4 py-3 text-sm text-red-200">
          <AlertTriangle size={16} className="mt-0.5 shrink-0" aria-hidden="true" />
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,280px)_minmax(0,1fr)]">
        <aside className="panel h-fit" aria-label="Datasets">
          <h2 className="panel-title flex items-center gap-2">
            <Database size={16} aria-hidden="true" /> Datasets
          </h2>
          <div className="mt-3 flex flex-wrap gap-2">
            <button type="button" className="btn btn-secondary" onClick={loadSamples} disabled={busy !== null}>
              Load samples
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => fileRef.current?.click()} disabled={busy !== null}>
              <Upload size={15} aria-hidden="true" /> Upload CSV
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".csv,text/csv"
              className="sr-only"
              aria-label="Upload price CSV"
              onChange={(e) => void onUpload(e.target.files?.[0])}
            />
          </div>
          <p className="mt-2 text-xs text-fog-400">CSV with a Date column and a Close (or Adj Close) column, at least 120 rows.</p>
          {list.length === 0 ? (
            <p className="mt-4 text-sm text-fog-400">No datasets yet. Load the synthetic samples or upload a CSV.</p>
          ) : (
            <ul className="mt-4 space-y-1.5">
              {list.map((d) => (
                <li key={d.id} className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={() => setSelected(d.id)}
                    aria-current={selected === d.id}
                    className={`min-w-0 flex-1 rounded-lg border px-3 py-2 text-left text-sm ${
                      selected === d.id ? 'border-accent-400/60 bg-accent-400/10 text-fog-100' : 'border-ink-700 hover:bg-ink-800'
                    }`}
                  >
                    <span className="block truncate font-medium">{d.name}</span>
                    <span className="block text-xs text-fog-400">
                      {d.rows} rows · {d.start} → {d.end}
                    </span>
                  </button>
                  <button type="button" className="btn btn-danger min-h-9 px-2" aria-label={`Delete ${d.name}`} onClick={() => void onDelete(d.id)}>
                    <Trash2 size={14} aria-hidden="true" />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </aside>

        <main className="min-w-0 space-y-4">
          {detail ? (
            <section className="panel" aria-label="Price history">
              <div className="mb-3 flex flex-wrap gap-x-6 gap-y-1 text-sm">
                <span className="font-semibold text-fog-100">{detail.name}</span>
                {detail.synthetic && <span className="chip border-amber-400/40 text-amber-200">Synthetic data</span>}
                <span>Last close {money(detail.stats.last)}</span>
                <span>Total return {detail.stats.total_return_pct.toFixed(1)}%</span>
                <span>Daily volatility {detail.stats.daily_volatility_pct.toFixed(2)}%</span>
              </div>
              <LineChart title="Closing price" labels={detail.dates} series={[{ name: 'Close', color: '#cfd6de', values: detail.close }]} />
            </section>
          ) : (
            <section className="panel text-sm text-fog-400">{selected ? 'Loading dataset…' : 'Select or load a dataset to begin.'}</section>
          )}

          <section className="panel" aria-label="Train models">
            <h2 className="panel-title flex items-center gap-2">
              <ChartIcon size={16} aria-hidden="true" /> Train and evaluate
            </h2>
            <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {FIELDS.map((f) => (
                <label key={f.key} className="block">
                  <span className="label">{f.label}</span>
                  <input
                    type="number"
                    className="input"
                    min={f.min}
                    max={f.max}
                    step={f.step}
                    value={params[f.key]}
                    onChange={(e) => setParams((p) => ({ ...p, [f.key]: Number(e.target.value) }))}
                  />
                  <span className="mt-1 block text-xs text-fog-400">{f.hint}</span>
                </label>
              ))}
            </div>
            <button type="button" className="btn btn-primary mt-4" onClick={train} disabled={!detail || busy !== null}>
              {busy === 'train' ? <Loader2 size={16} className="animate-spin" aria-hidden="true" /> : <Play size={16} aria-hidden="true" />}
              {busy === 'train' ? 'Training…' : 'Train models'}
            </button>
          </section>

          {result && (
            <>
              <section className="panel" aria-label="Results">
                <h2 className="panel-title">
                  Test-period accuracy ({result.test_rows} days from {result.split_date}; trained on {result.train_rows} days in {result.seconds}s)
                </h2>
                <div className="mt-3 overflow-x-auto">
                  <table className="w-full min-w-[520px] text-left text-sm">
                    <caption className="sr-only">Model comparison</caption>
                    <thead className="text-xs text-fog-400">
                      <tr>
                        <th className="py-2 pr-3 font-medium">Model</th>
                        <th className="py-2 pr-3 font-medium">RMSE</th>
                        <th className="py-2 pr-3 font-medium">MAE</th>
                        <th className="py-2 pr-3 font-medium">MAPE</th>
                        <th className="py-2 font-medium">Direction correct</th>
                      </tr>
                    </thead>
                    <tbody>
                      {MODELS.map((m) => {
                        const r = result.metrics[m.key];
                        return (
                          <tr key={m.key} className="border-t border-ink-700">
                            <td className="py-2 pr-3">
                              <span className="inline-flex items-center gap-2">
                                <span aria-hidden="true" className="inline-block h-2 w-2 rounded-full" style={{ background: m.color }} />
                                {m.label}
                                {best === m.key && <span className="chip border-accent-400/50 text-accent-300">lowest error</span>}
                              </span>
                            </td>
                            <td className="py-2 pr-3 tabular-nums">{money(r.rmse)}</td>
                            <td className="py-2 pr-3 tabular-nums">{money(r.mae)}</td>
                            <td className="py-2 pr-3 tabular-nums">{r.mape.toFixed(2)}%</td>
                            <td className="py-2 tabular-nums">{r.directional_accuracy === null ? '—' : `${r.directional_accuracy.toFixed(1)}%`}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="mt-3 text-xs text-fog-400">
                  The naive baseline simply repeats the previous close. Beating it on RMSE is hard for daily prices; “direction correct”
                  near 50% means no more skill than a coin flip.
                </p>
              </section>

              <section className="panel" aria-label="Predictions and forecast">
                <LineChart
                  title="Test period and forecast"
                  labels={[...result.test.dates, ...result.forecast.dates]}
                  marker={{ index: result.test.dates.length, label: 'forecast' }}
                  series={[
                    { name: 'Actual', color: '#e8ecf1', values: [...result.test.actual, ...result.forecast.dates.map(() => null)] },
                    ...MODELS.filter((m) => m.key !== 'naive').map((m) => ({
                      name: m.label,
                      color: m.color,
                      values: [...result.test[m.key], ...result.forecast[m.key]],
                    })),
                  ]}
                />
                <p className="mt-2 text-xs text-fog-400">
                  After the marker each model feeds its own predictions back in (recursive forecast), so errors compound with the horizon.
                </p>
              </section>

              <section className="panel" aria-label="LSTM training loss">
                <LineChart
                  title="LSTM training loss (MSE on scaled prices)"
                  labels={result.lstm_loss.map((_, i) => `epoch ${i + 1}`)}
                  series={[{ name: 'loss', color: '#2dd4bf', values: result.lstm_loss }]}
                  height={180}
                  format={(v) => v.toFixed(4)}
                />
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
