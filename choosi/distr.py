import numpy as np
from scipy.stats import norm, rv_discrete
from scipy.optimize import root_scalar

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
        self.weights = weights
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
            unw_pmf = norm.pdf((self.grid - mu)/sigma)
            self.pmf = unw_pmf * self.weights
            self.pmf /= self.pmf.sum()
            self.distr = rv_discrete(values=(self.grid, self.pmf))

    
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
        return self.distr.cdf(observed)
    
    
    def find_ci(
        self,
        obs_val,
        level,
    ):
        mu = (self.grid * self.pmf).sum()
        sigma = np.sqrt(((self.grid - mu)**2 * self.pmf).sum())

        lb = mu - 20 * sigma
        while self.cdf(lb, obs_val) < (1+level)/2:
            lb -= 10*sigma
        ub = mu + 20 * sigma
        while self.cdf(ub, obs_val) > (1-level)/2:
            ub += 10*sigma

        U = root_scalar(
            lambda x: self.cdf(x, obs_val) - (1-level)/2, 
            method='bisect', 
            bracket=[lb, ub], 

        ).root

        L = root_scalar(
            lambda x: self.cdf(x, obs_val) - (1+level)/2, 
            method='bisect', 
            bracket=[lb, ub], 

        ).root

        return L, U

    
