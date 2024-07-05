import numpy as np
from choosi.distr import WeightedNormal
import pytest
from scipy.stats import norm


@pytest.mark.parametrize("mu, sigma, n_grid", [
    [0, 1, 500],
    [0, 2, 500],
    [1, 3, 500],
    [-4, 10, 500],
])
def test_cdf_normal(mu, sigma, n_grid):
    wn = WeightedNormal(
        mu,
        sigma, 
        np.linspace(-10*sigma,10*sigma,n_grid), 
        weights=np.ones(n_grid),
    )
    wn_est = wn.cdf(mu, 1.96*sigma + mu)
    n_est = norm.cdf(1.96)

    assert np.allclose(wn_est, n_est, atol=.005)


@pytest.mark.parametrize("mu, sigma, level, n_grid", [
    [0, 1, .95, 2000],
    [0, 2, .95, 2000],
    [1, 3, .95, 2000],
    [-4, 10, .95, 2000],
])
def test_ci_normal(mu, sigma, level, n_grid):
    wn = WeightedNormal(
        mu=mu,
        sigma=sigma, 
        grid=np.linspace(-20*sigma,20*sigma,n_grid), 
        weights=np.ones(n_grid),
    )
    L, U = wn.find_ci(0, level)

    assert np.allclose(L, sigma*norm.ppf(1 - (1+level)/2), atol=.005)
    assert np.allclose(U, sigma*norm.ppf(1 - (1-level)/2), atol=.005)


@pytest.mark.parametrize("mu, sigma, level, n_grid", [
    [0, 1, .95, 2000],
    [0, .5, .95, 2000],
    [0, .25, .95, 2000],
    [0, 2, .95, 5000],
    [1, 3, .95, 5000],
])
def test_ci_trunc_norm(mu, sigma, level, n_grid, lt=0, ut=5):
    grid = np.linspace(-20*sigma,20*sigma,n_grid)
    weights = (grid >= lt) * (grid <= ut)
    wn = WeightedNormal(
        mu=mu,
        sigma=sigma, 
        grid=grid, 
        weights=weights,
    )
    mu0 =(ut - lt)/2
    L, U = wn.find_ci(mu0, level)

    assert np.allclose(
        (1+level)/2, 
        (
            (norm.cdf((mu0-L)/sigma) - norm.cdf((lt-L)/sigma))
            / (norm.cdf((ut - L)/sigma)-norm.cdf((lt-L)/sigma))
        ), 
        atol=.005
    )
    assert np.allclose(
        (1-level)/2, 
        (
            (norm.cdf((mu0-U)/sigma) - norm.cdf((lt-U)/sigma))
            / (norm.cdf((ut - U)/sigma)-norm.cdf((lt-U)/sigma))
        ), 
        atol=.005
    )




















