from choosi.choosi_core import distr as core_distr
from choosi.distr import WeightedNormal
from scipy.stats import norm
import numpy as np
import scipy
import pytest


@pytest.mark.parametrize("s", [1e-4, 1, 1e4])
@pytest.mark.parametrize("l", [-1e4, -5, -1, -1e-4])
@pytest.mark.parametrize("u", [1e-4, 1, 5, 1e4])
@pytest.mark.parametrize("seed", [0, 10])
def test_compute_weights(s, l, u, seed):
    np.random.seed(seed)
    z = np.random.normal(0, 1, 1)
    actual = core_distr.compute_weights(z, s, l, u)
    expected = (
        scipy.stats.norm.cdf((u-z) / s) -
        scipy.stats.norm.cdf((l-z) / s)
    )
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("mu", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("seed", [0, 5])
def test_compute_cdf(mu, seed):
    np.random.seed(seed)
    mu_z = mu / np.sqrt(2)
    a = np.random.normal(0, 1, 1)[0] 
    a_z = a / np.sqrt(2)
    z_L, xi_L = scipy.special.roots_laguerre(100)
    z_H, xi_H = scipy.special.roots_hermite(10)
    s_z = 1
    l_z = -np.inf
    u_z = np.inf

    actual = core_distr.compute_cdf(
        mu_z, a_z, z_L, xi_L, z_H, xi_H, s_z, l_z, u_z
    )

    expected = scipy.stats.norm.cdf(a, loc=mu)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("level", [1e-2, 1e-1, 0.5, 1-1e-1, 1-1e-2])
@pytest.mark.parametrize("seed", [0, 5])
def test_compute_cdf_root(level, seed):
    np.random.seed(seed)
    lower = -5
    upper =  5
    a = np.random.normal(0, 1, 1)[0]
    a_z = a / np.sqrt(2)
    z_L, xi_L = scipy.special.roots_laguerre(100)
    z_H, xi_H = scipy.special.roots_hermite(100)
    s = 1
    l = -1
    u = 1
    s_z, l_z, u_z = s / np.sqrt(2), l / np.sqrt(2), u / np.sqrt(2)
    w_pool_L = xi_L * core_distr.compute_weights(a_z - z_L, s_z, l_z, u_z)

    mu_z = core_distr.compute_cdf_root(
        level, lower, upper, a_z, z_L, w_pool_L, z_H, xi_H, s_z, l_z, u_z, 1e-14,
    )[0]
    assert mu_z > lower and mu_z < upper

    def _w(x):
        return scipy.stats.norm.cdf((u-x)/s) - scipy.stats.norm.cdf((l-x)/s)

    def _cdf(mu):
        return scipy.stats.norm.expect(
            _w,
            loc=mu,
            scale=1,
            lb=-np.inf,
            ub=a,
        ) / scipy.stats.norm.expect(
            _w,
            loc=mu,
            scale=1,
            lb=-np.inf,
            ub=np.inf,
        )

    mu = mu_z * np.sqrt(2)
    assert np.allclose(_cdf(mu), level)


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




















