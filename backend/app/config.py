import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", str(BACKEND_DIR / "data")))
    max_upload_mb: float = float(os.getenv("MAX_UPLOAD_MB", "5"))
    max_rows: int = int(os.getenv("MAX_ROWS", "20000"))
    max_datasets: int = int(os.getenv("MAX_DATASETS", "50"))
    torch_threads: int = int(os.getenv("TORCH_THREADS", "4"))
    cors_origins: list[str] = field(
        default_factory=lambda: [o.strip() for o in os.getenv(
            "CORS_ORIGINS", "http://localhost:5113,http://127.0.0.1:5113").split(",") if o.strip()]
    )


settings = Settings()
