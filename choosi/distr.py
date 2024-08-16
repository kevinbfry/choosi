from .choosi_core.distr import (
    compute_cdf,
    compute_cdf_root,
)
from scipy.optimize import root_scalar
from scipy.stats import norm, rv_discrete
import numpy as np

class WeightedNormal(object):
    def __init__(
        self,
        mu,
        sigma,
        grid,
        weights,
    ):
        self._sigma = sigma
        self.grid = grid
        self.weights = np.maximum(weights, 1e-40)
        largest = self.weights.max()
        c = largest + np.log(np.sum(np.exp(self.weights - largest)))
        self.weights = np.exp(self.weights - c) 
        self._mu = None
        self.mu = mu


    @property
    def mu(self):
        return self._mu


    @property
    def sigma(self):
        return self._sigma


    @mu.setter
    def mu(
        self,
        mu,
    ):
        if self._mu != mu:
            self._mu = mu
            sigma = self._sigma
            # unw_pmf = norm.pdf((self.grid - mu)/sigma)
            # self.pmf = unw_pmf * self.weights
            # self.pmf /= self.pmf.sum()

            unw_log_pmf = norm.logpdf((self.grid - mu) / sigma)
            self.log_pmf = unw_log_pmf + np.log(self.weights)
            largest = self.log_pmf.max()# - 10
            c = largest + np.log(np.sum(np.exp(self.log_pmf - largest)))
            self.pmf = np.exp(self.log_pmf - c)

            # self.distr = rv_discrete(values=(self.grid, self.pmf))

    
    def cdf(
        self,
        mu,
        observed,
    ):
        self.mu = mu
        if observed < self.grid[0]:
            return 0
        elif observed > self.grid[-1]:
            return 1
        # return self.distr.cdf(observed)
        return self.pmf[self.grid <= observed].sum()
    
    
    def find_ci(
        self,
        obs_val,
        level,
    ):
        mu = (self.grid * self.pmf).sum()
        sigma = np.sqrt(((self.grid - mu)**2 * self.pmf).sum())

        lb = mu - 20 * sigma
        ct = 0
        while self.cdf(lb, obs_val) < (1+level)/2:
            if ct == 1000:
                assert 0==1
            lb -= 20*sigma
            ct += 1

        ct = 0
        ub = mu + 20 * sigma
        while self.cdf(ub, obs_val) > (1-level)/2:
            if ct == 1000:
                assert 0==1
            ub += 20*sigma
            ct += 1

        U = root_scalar(
            lambda x: self.cdf(x, obs_val) - (1-level)/2, 
            method='bisect', 
            bracket=[lb, ub], 

            # method='secant', 
            # x0=ub,
            # x1=lb,
        ).root

        L = root_scalar(
            lambda x: self.cdf(x, obs_val) - (1+level)/2, 
            method='bisect', 
            bracket=[lb, ub], 

            # method='secant', 
            # x0=ub,
            # x1=lb,
        ).root

        return L, U

    
class WeightedNormal2(object):
    """

    The CDF is given by 
    ..math::
        \\begin{align*}
            F(z; \mu) := \\frac{
                \\int_{-\\infty}^z w(x) \\phi\\left(\\frac{x-\\mu}{\\sigma}\\right) dx
            }{
                \\int_{-\\infty}^{\\infty} w(x) \\phi\\left(\\frac{x-\\mu}{\\sigma}\\right) dx
            }
        \\end{align*}

    Parameters
    ----------
    sigma : float
        The scale parameter :math:`\\sigma`.
    hermite_roots : ndarray
        The Hermite roots as given by :func:`scipy.special.roots_hermite`
        multiplied by ``np.sqrt(2) * sigma``.
    hermite_weights : ndarray
        The Hermite weights as given by :func:`scipy.special.roots_hermite`.
    weights : ndarray, optional
        Any non-negative weights :math:`w(x)` evaluated at ``hermite_roots``.
        If ``None``, it is assumed that :math:`w(x) \\equiv 1``.
        Default is ``None``.
    """
    def __init__(
        self,
        sigma: float,
        hermite_roots: np.ndarray,
        hermite_weights: np.ndarray,
        weights: np.ndarray =None,
    ):
        self.sigma = sigma
        self.var = sigma ** 2
        self.hermite_roots = hermite_roots
        self.weights = hermite_weights
        if not (weights is None):
            self.weights = self.weights * weights

    def cdf(
        self,
        obs_val,
        mu,
    ):
        # compute the number of roots <= obs_val
        # TODO: binary search can make it slightly faster.
        s = np.sum(self.hermite_roots <= obs_val)
        return compute_cdf(mu / self.var, s, self.hermite_roots, self.weights)
        
    def find_ci(
        self,
        obs_val: float,
        level: float,
    ):
        # compute the number of roots <= obs_val
        # TODO: binary search can make it slightly faster.
        s = np.sum(self.hermite_roots <= obs_val)

        # construct a grid of nu = mu / sigma ** 2
        # TODO: if obs_val / sigma is very large in magnitude,
        # the gridding of nu seems to be very bad!
        # Needs a more stable initial choice of nus.
        center = obs_val / self.var
        width = 3 / self.sigma
        n_gridpts = 1000
        nus_orig = np.linspace(center - width, center + width, n_gridpts)

        # compute lower bound
        lower_level = (1-level) / 2
        nus = np.copy(nus_orig)
        Ui = compute_cdf_root(lower_level, nus, s, self.hermite_roots, self.weights)
        while (Ui <= 0 or Ui >= nus.size):
            if Ui <= 0:
                nus -= width
            else:
                nus += width
            Ui = compute_cdf_root(lower_level, nus, s, self.hermite_roots, self.weights)
        U = nus[Ui]

        # compute upper bound
        upper_level = (1+level) / 2
        nus = np.copy(nus_orig)
        Li = compute_cdf_root(upper_level, nus, s, self.hermite_roots, self.weights)
        while (Li <= 0 or Li >= nus.size):
            if Li <= 0:
                nus -= width
            else:
                nus += width
            Li = compute_cdf_root(upper_level, nus, s, self.hermite_roots, self.weights)
        L = nus[Li]

        # return the CI on the correct scale
        return self.var * L, self.var * U