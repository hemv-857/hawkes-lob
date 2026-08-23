"""Maximum-likelihood estimation and goodness-of-fit for exponential Hawkes."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import kstest

__all__ = ["fit_mle", "ks_gof", "time_rescaling"]


def fit_mle(times: np.ndarray, T: float | None = None) -> dict:
    """MLE of (mu, alpha, beta) on log-scale parameters.

    Returns estimates, standard errors (numerical Hessian), branching ratio,
    max log-likelihood and convergence flag.
    """
    times = np.sort(np.asarray(times, dtype=float))
    if T is None:
        T = float(times[-1])
    if times.size < 10:
        raise ValueError("need >= 10 events for meaningful estimation")

    def neg_ll(z):
        mu, alpha, beta = np.exp(z)
        ll = _loglik(times, mu, alpha, beta, T)
        return -ll if np.isfinite(ll) else 1e18

    n = times.size
    x0 = np.log([max(0.2 * n / T, 1e-4), max(0.5 * n / T, 1e-3), max(1.5 * n / T, 1e-2)])
    res = minimize(neg_ll, x0, method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000, "maxfev": 6000})
    mu, alpha, beta = np.exp(res.x)

    out = {
        "mu": float(mu),
        "alpha": float(alpha),
        "beta": float(beta),
        "branching_ratio": float(alpha / beta),
        "n_events": int(n),
        "T": float(T),
        "log_likelihood": float(-res.fun),
        "converged": bool(res.success),
    }
    se = _std_err(res.x, times, T)
    if se is not None:
        out["se_mu"], out["se_alpha"], out["se_beta"] = (float(v) for v in se)
    return out


def _loglik(times: np.ndarray, mu: float, alpha: float, beta: float, T: float) -> float:
    from .process import log_likelihood

    return log_likelihood(times, mu, alpha, beta, T)


def _std_err(zhat: np.ndarray, times: np.ndarray, T: float, h: float = 1e-4):
    """Inverse observed information from central-difference Hessian on z-scale."""
    def f(z):
        mu, alpha, beta = np.exp(z)
        return -neg_ll_z(z, times, mu, alpha, beta, T)

    m = zhat.size
    H = np.empty((m, m))
    for i in range(m):
        for j in range(m):
            zpp = zhat.copy(); zpp[i] += h; zpp[j] += h
            zpm = zhat.copy(); zpm[i] += h; zpm[j] -= h
            zmp = zhat.copy(); zmp[i] -= h; zmp[j] += h
            zmm = zhat.copy(); zmm[i] -= h; zmm[j] -= h
            H[i, j] = (f(zpp) - f(zpm) - f(zmp) + f(zmm)) / (4 * h * h)
    try:
        cov = np.linalg.inv(H)
        return np.sqrt(np.diag(np.abs(cov)))
    except np.linalg.LinAlgError:
        return None


def neg_ll_z(z, times, mu, alpha, beta, T):
    ll = _loglik(times, mu, alpha, beta, T)
    return -ll if np.isfinite(ll) else 1e18


def time_rescaling(times: np.ndarray, mu: float, alpha: float, beta: float) -> np.ndarray:
    """Time-rescaling residuals tau_i = Lambda(t_i) - Lambda(t_{i-1}), Exp(1) if correct.

    Stable O(N) recursion over inter-event gaps:
      int_gap lambda*(u) du = mu*Delta + (alpha/beta) * A * (1 - e^{-beta*Delta}),
    where A is the decayed excitation carried across the gap.
    """
    times = np.sort(np.asarray(times, dtype=float))
    taus = np.empty(times.size)
    A, prev = 0.0, 0.0
    for i, ti in enumerate(times):
        d = ti - prev
        decay = np.exp(-beta * d)
        taus[i] = mu * d + (alpha / beta) * A * (1.0 - decay)
        A = A * decay + 1.0  # cross the event at t_i
        prev = ti
    return taus


def ks_gof(taus: np.ndarray) -> dict:
    """KS test of Exp(1) on compensated inter-arrival residuals."""
    stat, p = kstest(np.asarray(taus), "expon")
    return {"ks_stat": float(stat), "p_value": float(p)}
