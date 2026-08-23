"""Simulation and O(N) log-likelihood for univariate exponential Hawkes processes.

Intensity:      lambda*(t) = mu + alpha * sum_{t_j < t} exp(-beta (t - t_j))
Log-likelihood: sum_i log lambda*(t_i) - mu T - (alpha/beta) sum_i (1 - exp(-beta (T - t_i)))

The recursive statistic A_i = sum_{t_j < t_i} exp(-beta (t_i - t_j)) satisfies
A_i = exp(-beta (t_i - t_{i-1})) * (1 + A_{i-1}), making the likelihood O(N).
"""

from __future__ import annotations

import numpy as np

__all__ = ["branching_ratio", "log_likelihood", "simulate_hawkes"]


def simulate_hawkes(mu: float, alpha: float, beta: float, T: float, seed: int) -> np.ndarray:
    """Ogata thinning. Requires alpha < beta (stationary branching ratio < 1).

    State invariant between iterations: A == sum_j exp(-beta*(t - t_j)) over
    accepted events. Candidates evolve A deterministically (pure decay); an
    acceptance adds the new event at lag 0 (i.e. +1).
    """
    if not 0 <= alpha < beta:
        raise ValueError(f"need 0 <= alpha < beta for stability, got alpha={alpha}, beta={beta}")
    rng = np.random.default_rng(seed)
    events: list[float] = []
    t, A = 0.0, 0.0
    while True:
        lam = mu + alpha * A
        s = t + rng.exponential(1.0 / max(lam, 1e-300))
        if s > T:
            break
        decay = np.exp(-beta * (s - t))
        A_prop = A * decay
        lam_star = mu + alpha * A_prop
        if rng.uniform() < lam_star / lam:
            events.append(s)
            A = A_prop + 1.0
        else:
            A = A_prop
        t = s
    return np.asarray(events)


def log_likelihood(
    times: np.ndarray, mu: float, alpha: float, beta: float, T: float | None = None
) -> float:
    """O(N) exact log-likelihood on observed event times in [0, T]."""
    times = np.sort(np.asarray(times, dtype=float))
    n = times.size
    if T is None:
        T = float(times[-1]) if n else 0.0
    if n == 0:
        return -mu * T
    dt = np.diff(times, prepend=times[0])
    dt[0] = times[0]  # from horizon start 0
    A = np.empty(n)
    A[0] = 0.0
    acc = 0.0
    for i in range(1, n):
        acc = np.exp(-beta * (times[i] - times[i - 1])) * (1.0 + acc)
        A[i] = acc
    lam = mu + alpha * A
    if np.any(lam <= 0):
        return -np.inf
    comp = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - times)))
    return float(np.sum(np.log(lam)) - comp)


def branching_ratio(alpha: float, beta: float) -> float:
    """n = alpha/beta: expected children per event; < 1 required for stationarity."""
    return alpha / beta


def cluster_events(times: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """Merge near-simultaneous prints (gap < eps sec) into one event.

    Raw exchange tapes contain bursts of split-order prints with identical
    timestamps; without clustering the MLE pins n* at the critical boundary
    chasing this mechanical burstiness. eps=0.1s is the standard choice for
    crypto aggTrades.
    """
    times = np.sort(np.asarray(times, dtype=float))
    if times.size == 0:
        return times
    keep = [times[0]]
    for t in times[1:]:
        if t - keep[-1] >= eps:
            keep.append(t)
    return np.asarray(keep)
