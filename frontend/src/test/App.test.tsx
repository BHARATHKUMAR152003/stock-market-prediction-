import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../App';

const meta = { id: 'synth-trend', name: 'SYNTH-TREND (synthetic)', description: '', synthetic: true, rows: 3, start: '2024-01-01', end: '2024-01-03', created_at: '' };
const detail = {
  ...meta,
  dates: ['2024-01-01', '2024-01-02', '2024-01-03'],
  close: [100, 101, 102],
  stats: { first: 100, last: 102, min: 100, max: 102, total_return_pct: 2, daily_volatility_pct: 1 },
};
const m = (rmse: number) => ({ rmse, mae: rmse, mape: 1, directional_accuracy: 50 });
const result = {
  dataset_id: 'synth-trend',
  config: { lookback: 30, test_fraction: 0.2, epochs: 40, hidden_size: 32, horizon: 2, seed: 42 },
  train_rows: 2,
  test_rows: 1,
  split_date: '2024-01-03',
  metrics: { naive: m(2), linear_regression: m(1.5), lstm: m(1.8) },
  test: { dates: ['2024-01-03'], actual: [102], naive: [101], linear_regression: [101.5], lstm: [101.8] },
  forecast: { dates: ['2024-01-04', '2024-01-05'], naive: [102, 102], linear_regression: [102.4, 102.8], lstm: [102.2, 102.3] },
  lstm_loss: [0.1, 0.05],
  seconds: 0.4,
};

let calls: { url: string; init?: RequestInit }[];

beforeEach(() => {
  calls = [];
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, init });
    const respond = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
    if (url === '/api/datasets') return respond([meta]);
    if (url === '/api/datasets/synth-trend') return respond(detail);
    if (url === '/api/train') return respond(result);
    return respond({ detail: 'not found' }, 404);
  }));
});

afterEach(() => vi.unstubAllGlobals());

it('loads datasets, trains and highlights the model with the lowest error', async () => {
  const user = userEvent.setup();
  render(<App />);
  expect(await screen.findByText('Last close 102.00')).toBeInTheDocument();

  const lookback = screen.getByLabelText(/lookback window/i);
  await user.clear(lookback);
  await user.type(lookback, '20');
  await user.click(screen.getByRole('button', { name: /train models/i }));

  expect(await screen.findByText(/lowest error/i)).toBeInTheDocument();
  const row = screen.getByText('lowest error').closest('tr')!;
  expect(row).toHaveTextContent('Linear Regression');
  const body = JSON.parse(String(calls.find((c) => c.url === '/api/train')!.init!.body));
  expect(body).toMatchObject({ dataset_id: 'synth-trend', lookback: 20, epochs: 40 });
});

it('shows API errors to the user', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ detail: 'Only .csv files are supported.' }), { status: 400 })));
  render(<App />);
  await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Only .csv files are supported.'));
});
