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
    start_ms: int | None = None,
    with_prices: bool = False,
) -> tuple[np.ndarray, Path] | tuple[np.ndarray, np.ndarray, Path]:
    """Fetch aggregated trades for [start, end) — defaults to trailing `minutes` from now.

    ponytail: ~1 req/sec pagination; multi-day parallel backfill is the obvious
    upgrade when the study needs it.
    Returns (times_sec[, prices], cached_file_path).
    """
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    if start_ms is None:
        start_ms = end_ms - minutes * 60_000
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{symbol}_{start_ms}_{end_ms}.csv"

    stamps: list[float] = []
    prices: list[float] = []
    if path.exists():
        times, pxs = load_times_and_prices(str(path))
        if with_prices:
            return times, pxs, path
        return times, path

    from_id: int | None = None
    with httpx.Client(timeout=30) as client:
        while True:
            params: dict = {"symbol": symbol, "startTime": start_ms,
                            "endTime": end_ms, "limit": 1000}
            if from_id is not None:
                params = {"symbol": symbol, "fromId": from_id, "limit": 1000}
            r = client.get(_BASE, params=params)
            r.raise_for_status()
            rows = r.json()
            if not rows:
                break
            for row in rows:
                stamps.append(row["T"] / 1000.0)
                prices.append(float(row["p"]))
            if len(rows) < 1000 or rows[-1]["T"] >= end_ms:
                break
            from_id = rows[-1]["a"] + 1
            time.sleep(0.05)

    np.asarray(sorted(zip(stamps, prices))) if False else None
    order = np.argsort(stamps)
    stamps = np.asarray(stamps)[order]
    prices = np.asarray(prices)[order]
    save_times_and_prices(stamps, prices, str(path))
    if with_prices:
        return stamps, prices, path
    return stamps, path


def load_times_and_prices(path: str) -> tuple[np.ndarray, np.ndarray]:
    times, pxs = [], []
    with open(path) as fh:
        for row in csv.reader(fh):
            try:
                times.append(float(row[0]))
                pxs.append(float(row[1]))
            except (ValueError, IndexError):
                continue  # header/blank/legacy single-column
    return np.asarray(times), np.asarray(pxs)


def save_times_and_prices(times: np.ndarray, prices: np.ndarray, path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_sec", "price"])
        for t, p in zip(times, prices):
            w.writerow([f"{t:.6f}", f"{p:.2f}"])


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
