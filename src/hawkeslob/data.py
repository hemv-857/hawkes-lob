"""Trade-arrival data: Binance aggTrades fetcher with local cache + synthetic fallback."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import httpx
import numpy as np

__all__ = ["fetch_binance_trades", "load_times_csv", "save_times_csv"]

_BASE = "https://api.binance.com/api/v3/aggTrades"


def fetch_binance_trades(
    symbol: str = "BTCUSDT",
    minutes: int = 60,
    end_ms: int | None = None,
    cache_dir: str = "data",
) -> tuple[np.ndarray, Path]:
    """Fetch aggregated trades for a window ending at end_ms (default: now).

    ponytail: single-window fetch, ~1 req/sec pagination; multi-day parallel
    backfill is the obvious upgrade when the study needs it.
    Returns (trade timestamps in seconds, cached file path).
    """
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    start_ms = end_ms - minutes * 60_000
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{symbol}_{end_ms}_{minutes}m.csv"

    stamps: list[float] = []
    if path.exists():
        return load_times_csv(str(path)), path

    from_id: int | None = None
    with httpx.Client(timeout=30) as client:
        while True:
            params: dict = {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000}
            if from_id is not None:
                params = {"symbol": symbol, "fromId": from_id, "limit": 1000}
            r = client.get(_BASE, params=params)
            r.raise_for_status()
            rows = r.json()
            if not rows:
                break
            for row in rows:
                stamps.append(row["T"] / 1000.0)
            if len(rows) < 1000 or rows[-1]["T"] >= end_ms:
                break
            from_id = rows[-1]["a"] + 1
            time.sleep(0.05)

    arr = np.asarray(sorted(stamps))
    save_times_csv(arr, str(path))
    return arr, path


def load_times_csv(path: str) -> np.ndarray:
    vals = []
    with open(path) as fh:
        for row in csv.reader(fh):
            try:
                vals.append(float(row[0]))
            except (ValueError, IndexError):  # header or blank line
                continue
    return np.asarray(vals)


def save_times_csv(stamps: np.ndarray, path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_sec"])
        for v in stamps:
            w.writerow([f"{v:.6f}"])
