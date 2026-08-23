"""Empirical study: Hawkes fits on N historical BTCUSDT windows.

    python scripts/run_study.py --windows 10 --minutes 60

For each window: MLE (mu, alpha, beta), branching ratio +/- SE, realized vol
from trade prices, likelihood-ratio test against the Poisson null, KS GOF.
Outputs docs/study_results.json, docs/branching_vs_vol.png, docs/table.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, kstest

from hawkeslob.data import fetch_binance_trades
from hawkeslob.estimate import fit_mle, time_rescaling
from hawkeslob.process import cluster_events, log_likelihood

WINDOW_MS = 60 * 60 * 1000


def realized_vol_bps(prices: np.ndarray) -> float:
    """Sum of squared 1-trade log returns, expressed in bps of price move."""
    if prices.size < 10:
        return float("nan")
    r = np.diff(np.log(prices))
    return float(np.sqrt(np.sum(r * r)) * 1e4)


def analyze_window(times: np.ndarray, prices: np.ndarray) -> dict:
    fit = fit_mle(times)
    T = fit["T"]

    ll_hawkes = fit["log_likelihood"]
    ll_poisson = log_likelihood(times, fit["mu"], 0.0, 1.0, T)
    lr_stat = 2.0 * (ll_hawkes - ll_poisson)
    lr_p = float(chi2.sf(lr_stat, df=2))  # alpha, beta free under H1

    taus = time_rescaling(times, fit["mu"], fit["alpha"], fit["beta"])
    ks_p = float(kstest(taus, "expon").pvalue) if taus.size > 20 else float("nan")

    return {
        "n_trades": fit["n_events"],
        "mu": round(fit["mu"], 4), "alpha": round(fit["alpha"], 4),
        "beta": round(fit["beta"], 4),
        "branching": round(fit["branching_ratio"], 4),
        "branching_se": round(fit["branching_ratio"] *
                              ((fit.get("se_alpha", 0) / max(fit["alpha"], 1e-9)) ** 2 +
                               (fit.get("se_beta", 0) / max(fit["beta"], 1e-9)) ** 2) ** 0.5, 4),
        "realized_vol_bps": round(realized_vol_bps(prices), 1),
        "lr_vs_poisson_stat": round(lr_stat, 1),
        "lr_p_value": float(f"{lr_p:.3g}"),
        "ks_p_value": float(f"{ks_p:.3g}"),
        "window_end_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime(times[-1])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--minutes", type=int, default=60)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--gap_hours", type=float, default=9.0,
                    help="spacing between window starts")
    ap.add_argument("--eps", type=float, default=0.1,
                    help="burst-clustering threshold in seconds (0 = raw tape)")
    ap.add_argument("--observable", choices=["price-change", "raw"], default="price-change",
                    help="event type: price-change events (default) or raw prints")
    args = ap.parse_args()

    now_ms = int(time.time() * 1000)
    results = []
    for i in range(args.windows):
        end_ms = now_ms - int(i * args.gap_hours * 3600 * 1000) - 3600_000
        start_ms = end_ms - args.minutes * 60_000
        times, prices, _path = fetch_binance_trades(
            args.symbol, start_ms=start_ms, end_ms=end_ms, with_prices=True)
        if times.size < 500:
            print(f"window {i}: only {times.size} trades, skipping")
            continue
        n_raw = times.size
        if args.observable == "price-change":
            events = times[np.r_[True, np.diff(prices) != 0]]
        else:
            events = cluster_events(times, eps=args.eps)
        res = analyze_window(events, prices)
        res["n_raw"] = n_raw
        res["observable"] = args.observable
        results.append(res)
        print(f"[{i+1}/{args.windows}] {res['window_end_utc']}UTC n={res['n_trades']} "
              f"(raw {res['n_raw']}) n*={res['branching']:.3f}+-{res['branching_se']:.3f} "
              f"rv={res['realized_vol_bps']:.0f}bps lr_p={res['lr_p_value']:.1e} "
              f"ks_p={res['ks_p_value']:.2f}")

    Path("docs").mkdir(exist_ok=True)
    with open("docs/study_results.json", "w") as fh:
        json.dump({"symbol": args.symbol, "window_minutes": args.minutes,
                   "results": results}, fh, indent=2)

    # figure: branching ratio vs realized vol
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rv = np.array([r["realized_vol_bps"] for r in results])
    nb = np.array([r["branching"] for r in results])
    se = np.array([r["branching_se"] for r in results])
    ax.errorbar(rv, nb, yerr=se, fmt="o", capsize=3, ms=5)
    ax.set_xlabel("realized vol over window (bps, 1-trade returns)")
    ax.set_ylabel("branching ratio $n^*=\\alpha/\\beta$")
    ax.set_title(f"{args.symbol} {args.minutes}min windows: self-excitation vs activity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("docs/branching_vs_vol.png", dpi=130)

    # markdown table for the paper
    lines = [("| window (UTC) | trades | $\\hat\\mu$ | $\\hat\\alpha$ | $\\hat\\beta$ | "
             "$n^*$ | RV (bps) | LR $p$ | KS $p$ |"),
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['window_end_utc']} | {r['n_trades']} | {r['mu']} | "
                     f"{r['alpha']} | {r['beta']} | {r['branching']}±{r['branching_se']} | "
                     f"{r['realized_vol_bps']} | {r['lr_p_value']:.1e} | {r['ks_p_value']:.2f} |")
    Path("docs/table.md").write_text("\n".join(lines) + "\n")
    print(f"\nsaved docs/study_results.json, docs/branching_vs_vol.png, docs/table.md "
          f"({len(results)} windows)")


if __name__ == "__main__":
    main()
