import os
import sys
import tempfile
from pathlib import Path

import pytest

_tmp = Path(tempfile.mkdtemp(prefix="stocks-test-"))
os.environ["DATA_DIR"] = str(_tmp)
os.environ["MAX_UPLOAD_MB"] = "0.5"
os.environ["TORCH_THREADS"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
