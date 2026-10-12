# hawkes-lob

<a href="https://github.com/hemv-857/hawkes-lob/actions/workflows/ci.yml"><img src="https://github.com/hemv-857/hawkes-lob/actions/workflows/ci.yml/badge.svg" alt="hawkes-lob CI"></a> <a href="https://github.com/hemv-857/hawkes-lob/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a> <img src="https://img.shields.io/badge/Python-3776AB.svg" alt="Python">

Univariate **exponential Hawkes processes** for limit-order-book trade arrivals:
simulation (Ogata thinning), exact O(N) maximum likelihood, standard errors from
the observed information, and goodness-of-fit via the time-rescaling theorem.

```
lambda*(t) = mu + alpha * sum_{t_j < t} exp(-beta (t - t_j))
branching ratio n = alpha/beta   (self-excitation share of trades; < 1 for stationarity)
```

## Quickstart

```bash
pip install -e ".[dev]"
make test
python -m hawkeslob.cli sim                      # synthetic path
python -m hawkeslob.cli fetch --symbol BTCUSDT --minutes 120
python -m hawkeslob.cli fit --csv data/BTCUSDT_*.csv
python -m hawkeslob.cli fit                      # synthetic demo fit
```

## Correctness gates (enforced in tests)

- O(N) recursive log-likelihood == naive O(N²) computation to 1e-9
- Thinning simulator recovers theoretical stationary rate μ/(1−n) within 15%
- MLE recovers synthetic (μ, α, β) within ~3 standard errors
- Time-rescaling residuals are Exp(1) under the true model (KS p > 0.01)
- Misspecified model is rejected by KS or gross mean distortion
- CSV roundtrip exact

## Study design (what to run for the writeup)

1. Fetch N sessions of BTCUSDT aggTrades (`fetch` in a loop over days).
2. Fit per session; tabulate branching ratio vs session realized volatility.
3. Report LR test against the Poisson null (α=0): `2(ll - ll_poisson) ~ chi2(2)`.
4. Honest GOF: report KS p-values including failures, not just means.

## Honest scope

- `ponytail:` univariate exponential kernel only. Multivariate (bid/ask/mid)
  and power-law kernels are the natural extensions.
- Standard errors assume regularity of the observed information; check with
  parametric bootstrap before quoting them in the paper.

## License

[LICENSE](https://github.com/hemv-857/hawkes-lob/blob/master/LICENSE) — see the file for the full text.
