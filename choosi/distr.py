from .choosi_core.distr import (
    compute_cdf,
    compute_cdf_root,
    compute_weights,
)
from scipy.special import (
    roots_hermite,
    roots_laguerre,
)
from scipy.optimize import root_scalar
from scipy.stats import norm, rv_discrete
from typing import Union
import numpy as np
import warnings

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
    """TODO

    The CDF is given by 

    ..math::
        \\begin{align*}
            F(z; \\mu) := \\frac{
                \\int_{-\\infty}^z w(x) \\phi\\left(\\frac{x-\\mu}{\\sigma}\\right) dx
            }{
                \\int_{-\\infty}^{\\infty} w(x) \\phi\\left(\\frac{x-\\mu}{\\sigma}\\right) dx
            }
        \\end{align*}

    The weight function :math:`w` is given by

    ..math::
        \\begin{align*}
            w(x) = \Phi((u-x)/s) - \Phi((\\ell-x)/s)
        \\end{align*}

    where :math:`\Phi` is the standard normal CDF.

    Parameters
    ----------
    observed : float
        The observed value :math:`z`.
    sigma : float
        The scale parameter :math:`\\sigma`.
    w_s : float
        The scale argument :math:`s` to :math:`w(\cdot)`.
    w_l : float
        The lower argument :math:`\\ell` to :math:`w(\cdot)`.
    w_u : float
        The upper argument :math:`u` to :math:`w(\cdot)`.
    laguerre : ndarray, optional
        The Laguerre roots and weights as given by :func:`scipy.special.roots_laguerre`.
        If ``None``, it is initialized internally in an implementation-specific way.
        Default is ``None``.
    hermite : ndarray, optional
        The Hermite roots and weights as given by :func:`scipy.special.roots_hermite`.
        If ``None``, it is initialized internally in an implementation-specific way.
        Default is ``None``.
    """
    def __init__(
        self,
        observed: float,
        sigma: float,
        w_s: float,
        w_l: float,
        w_u: float,
        laguerre: Union[tuple, None] =None,
        hermite: Union[tuple, None] =None,
    ):
        self.observed = observed
        self.sigma = sigma
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0.")
        self.scale = sigma * np.sqrt(2)
        self.a_z = observed / self.scale
        self.w_args = (w_s / self.scale, w_l / self.scale, w_u / self.scale)
        if self.w_args[0] <= 0:
            raise ValueError("w_s must be > 0.")
        if self.w_args[1] > self.w_args[2]:
            raise ValueError("w_l must be <= w_u.")
        self.laguerre = laguerre
        self.hermite = hermite
        if laguerre is None:
            self.laguerre = roots_laguerre(100)
            subset = self.laguerre[1] >= 1e-24
            self.laguerre = (
                self.laguerre[0][subset],
                self.laguerre[1][subset],
            )
        if hermite is None:
            self.hermite = roots_hermite(20)
        self.w_pool_L = self._compute_w_pool_L()

    def _compute_weights(self, z):
        return compute_weights(z, *self.w_args)

    def _compute_w_pool_L(self):
        return self.laguerre[1] * self._compute_weights(self.a_z - self.laguerre[0])

    def _compute_numerator(self, mu):
        """Debug purposes."""
        mu_z = mu / self.scale
        z_L = self.laguerre[0]
        diff = self.a_z - mu_z
        return np.sum(self.w_pool_L * np.exp(-np.square(z_L - (diff + 0.5)) + (0.25 + diff)))

    def _compute_denominator(self, mu):
        """Debug purposes."""
        mu_z = mu / self.scale
        z_H = self.hermite[0]
        xi_H = self.hermite[1]
        return np.sum(xi_H * self._compute_weights(mu_z + z_H))

    def _compute_cdf(self, mu_z):
        return compute_cdf(
            mu_z, 
            self.a_z,
            self.laguerre[0],
            self.w_pool_L,
            self.hermite[0], 
            self.hermite[1],
            *self.w_args,
        )

    def _compute_cdf_root(self, level, lower, upper, tol):
        return compute_cdf_root(
            level, 
            lower, 
            upper, 
            self.a_z, 
            self.laguerre[0],
            self.w_pool_L,
            self.hermite[0], 
            self.hermite[1],
            *self.w_args,
            tol,
        )

    def cdf(self, mu):
        return self._compute_cdf(mu / self.scale)
        
    def find_ci(
        self,
        level: float,
        verbose: bool =False,
        tol: float =1e-9,
    ):
        a_z = self.a_z

        # construct an initial grid of nu = mu / self.scale
        center = a_z
        width = 3

        # compute the range of nu containing lower_level
        def _compute_range(level, direction):
            nu_lower = center - width
            nu_upper = center + width
            iters = 0

            # set the initial moving and following values
            nu_moving, nu_following = (
                (nu_upper, -np.inf)
                if direction == 1 else
                (nu_lower, np.inf)
            )

            # keep sliding the window until it contains the desired level
            cdf = self._compute_cdf(nu_moving)
            while direction * (cdf - level) > 0:
                nu_dist = abs(nu_moving - nu_following) if iters else width
                nu_following = nu_moving
                nu_moving += direction * 2 * nu_dist
                cdf = self._compute_cdf(nu_moving)
                iters += 1

            # order the window in increasing order
            nu_lower, nu_upper = (
                (nu_following, nu_moving)
                if direction == 1 else
                (nu_moving, nu_following)
            )

            return nu_lower, nu_upper, iters
            
        # compute desired symmetric levels
        lower_level = (1-level) / 2
        upper_level = (1+level) / 2

        # find range containing lower_level
        U_lower, U_upper, U_search_iters = _compute_range(lower_level,  1)
        L_lower, L_upper, L_search_iters = _compute_range(upper_level, -1)
        L_upper, U_lower = (
            min(L_upper, U_upper),
            max(L_lower, U_lower),
        )

        # compute CI bounds
        U, U_iters = self._compute_cdf_root(lower_level, U_lower, U_upper, tol)
        L, L_iters = self._compute_cdf_root(upper_level, L_lower, L_upper, tol)

        # return CI on original scale
        L, U = self.scale * L, self.scale * U

        # Check that the CDFs at the bounds are close to the desired levels.
        # _compute_cdf_root may have exited just because the range was getting too small.
        cdf_L = self.cdf(L)
        cdf_U = self.cdf(U)
        if not (
            np.allclose(cdf_L, upper_level, atol=1e-2) &
            np.allclose(cdf_U, lower_level, atol=1e-2) 
        ):
            warnings.warn("CDF bounds may be inaccurate")

        if verbose:
            return L, U, L_search_iters, U_search_iters, L_iters, U_iters,

        return L, U