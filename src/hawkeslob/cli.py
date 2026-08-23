"""CLI: simulate, fetch, fit, diagnose Hawkes fits on trade arrivals."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .data import fetch_binance_trades, load_times_csv
from .estimate import fit_mle, ks_gof, time_rescaling
from .process import simulate_hawkes


def _sim(args):
    ev = simulate_hawkes(args.mu, args.alpha, args.beta, args.T, seed=args.seed)
    print(f"simulated {ev.size} events on [0, {args.T}] "
          f"(branching ratio {args.alpha / args.beta:.3f})")
    np.savetxt(f"{args.out}", ev, header="t", comments="")
    print(f"saved -> {args.out}")


def _fetch(args):
    stamps, path = fetch_binance_trades(args.symbol, minutes=args.minutes)
    print(f"{stamps.size} trades for {args.symbol} -> {path}")
    if stamps.size:
        rates = np.diff(stamps)
        print(f"median inter-arrival: {np.median(rates):.4f}s")


def _fit(args):
    times = load_times_csv(args.csv) if args.csv else simulate_hawkes(
        0.6, 0.8, 1.4, 300.0, seed=1
    )
    fit = fit_mle(times)
    taus = time_rescaling(times, fit["mu"], fit["alpha"], fit["beta"])
    gof = ks_gof(taus)
    out = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in fit.items()}
    out["gof"] = {k: round(v, 5) for k, v in gof.items()}
    print(json.dumps(out, indent=2))
    verdict = "consistent" if gof["p_value"] > 0.05 else "REJECTED"
    print(f"time-rescaling GOF: p={gof['p_value']:.4f} -> model {verdict} at 5% level")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="hawkeslob", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sim", help="simulate a Hawkes path")
    s.add_argument("--mu", type=float, default=0.6)
    s.add_argument("--alpha", type=float, default=0.8)
    s.add_argument("--beta", type=float, default=1.4)
    s.add_argument("--T", type=float, default=300.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/sim.csv")
    s.set_defaults(func=_sim)

    f = sub.add_parser("fetch", help="pull Binance aggTrades into data/")
    f.add_argument("--symbol", default="BTCUSDT")
    f.add_argument("--minutes", type=int, default=60)
    f.set_defaults(func=_fetch)

    t = sub.add_parser("fit", help="fit MLE + GOF to csv of timestamps")
    t.add_argument("--csv", help="csv with column of event times; omit for synthetic demo")
    t.set_defaults(func=_fit)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
