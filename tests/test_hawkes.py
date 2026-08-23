import numpy as np
import pytest

from hawkeslob.data import load_times_csv, save_times_csv
from hawkeslob.estimate import fit_mle, ks_gof, time_rescaling
from hawkeslob.process import log_likelihood, simulate_hawkes


# ------------------------------------------------------------------ likelihood
def test_recursive_likelihood_matches_bruteforce():
    """O(N) recursion must equal the naive O(N^2) sum."""
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 50, size=120))
    mu, alpha, beta = 0.7, 0.9, 1.7

    A = np.array([
        np.exp(-beta * (ti - t[t < ti])).sum() for ti in t
    ])
    lam = mu + alpha * A
    T = 50.0
    ref = np.log(lam).sum() - mu * T - (alpha / beta) * np.sum(1 - np.exp(-beta * (T - t)))
    assert abs(log_likelihood(t, mu, alpha, beta, T) - ref) < 1e-9


# ------------------------------------------------------------------ simulation
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_thinning_recovers_baseline_rate(seed):
    """Near-Poisson regime: total count ~ Poisson(mu*T/(1-n))."""
    mu, alpha, beta, T = 1.0, 0.3, 1.5, 500.0
    ev = simulate_hawkes(mu, alpha, beta, T, seed=seed)
    expected_rate = mu / (1 - alpha / beta)
    lo, hi = 0.85 * expected_rate * T, 1.15 * expected_rate * T
    assert lo < ev.size < hi


def test_thinning_rejects_unstable_params():
    with pytest.raises(ValueError):
        simulate_hawkes(1.0, 1.5, 1.0, 10.0, seed=0)


def test_events_sorted_strictly_increasing():
    ev = simulate_hawkes(0.5, 0.9, 1.2, 200.0, seed=7)
    assert np.all(np.diff(ev) > 0)


# ------------------------------------------------------------------ estimation
def test_mle_recovers_synthetic_params():
    true = {"mu": 0.6, "alpha": 0.8, "beta": 1.4}
    ev = simulate_hawkes(true["mu"], true["alpha"], true["beta"], 4000.0, seed=42)
    fit = fit_mle(ev, T=4000.0)
    se = fit.get("se_alpha", np.inf)
    assert abs(fit["mu"] - true["mu"]) < max(3 * fit.get("se_mu", 0.1), 0.12)
    assert abs(fit["alpha"] - true["alpha"]) < max(3 * se, 0.15)
    assert abs(fit["beta"] - true["beta"]) / true["beta"] < 0.35
    assert abs(fit["branching_ratio"] - true["alpha"] / true["beta"]) < 0.12


def test_time_rescaling_gof_passes_for_true_model():
    ev = simulate_hawkes(0.6, 0.8, 1.4, 3000.0, seed=11)
    taus = time_rescaling(ev, 0.6, 0.8, 1.4)
    gof = ks_gof(taus)
    assert gof["p_value"] > 0.01
    assert abs(taus.mean() - 1.0) < 0.08  # Exp(1) mean


def test_time_rescaling_rejects_wrong_model():
    # fit Poisson data with an excitation-free intensity but test against wrong beta
    ev = simulate_hawkes(1.0, 0.0, 5.0, 2000.0, seed=3)
    taus = time_rescaling(ev, 1.0, 0.9, 0.2)  # badly misspecified excitation
    assert ks_gof(taus)["p_value"] < 0.05 or taus.mean() > 1.5


# ------------------------------------------------------------------ data io
def test_csv_roundtrip(tmp_path):
    arr = np.arange(1, 10) * 0.5
    p = tmp_path / "x.csv"
    save_times_csv(arr, str(p))
    assert np.allclose(load_times_csv(str(p)), arr)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
