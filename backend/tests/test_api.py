def test_samples_are_idempotent_and_listed(client):
    first = client.post("/api/datasets/samples").json()
    second = client.post("/api/datasets/samples").json()
    assert [d["id"] for d in first] == [d["id"] for d in second]
    assert {"synth-trend", "synth-volatile", "synth-cycle"} <= {d["id"] for d in first}
    assert all(d["synthetic"] for d in first)


def test_dataset_detail_has_series_and_stats(client):
    client.post("/api/datasets/samples")
    body = client.get("/api/datasets/synth-trend").json()
    assert len(body["dates"]) == len(body["close"]) == body["rows"] == 1000
    assert body["stats"]["last"] == body["close"][-1]


def test_upload_validation_and_delete(client):
    bad = client.post("/api/datasets", files={"file": ("prices.txt", b"x", "text/plain")})
    assert bad.status_code == 400 and "csv" in bad.json()["detail"].lower()

    too_big = client.post("/api/datasets", files={"file": ("big.csv", b"Date,Close\n" + b"0" * 600_000, "text/csv")})
    assert too_big.status_code == 400 and "larger" in too_big.json()["detail"]

    rows = "\n".join(f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d},{100 + i * 0.5}" for i in range(150))
    ok = client.post("/api/datasets", files={"file": ("my stock.csv", f"Date,Close\n{rows}".encode(), "text/csv")})
    assert ok.status_code == 201
    created = ok.json()
    assert created["rows"] == 150 and created["synthetic"] is False

    assert client.delete(f"/api/datasets/{created['id']}").status_code == 204
    assert client.get(f"/api/datasets/{created['id']}").status_code == 404


def test_path_traversal_ids_are_rejected(client):
    assert client.get("/api/datasets/..%2F..%2Fsecret").status_code == 404


def test_train_endpoint(client):
    client.post("/api/datasets/samples")
    res = client.post("/api/train", json={"dataset_id": "synth-cycle", "lookback": 20, "epochs": 3, "horizon": 3})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["metrics"]["naive"]["rmse"] > 0
    assert body["metrics"]["naive"]["directional_accuracy"] is None
    assert len(body["forecast"]["linear_regression"]) == 3


def test_train_validation(client):
    res = client.post("/api/train", json={"dataset_id": "synth-cycle", "lookback": 2})
    assert res.status_code == 422 and "lookback" in res.json()["detail"]
    assert client.post("/api/train", json={"dataset_id": "missing"}).status_code == 404
