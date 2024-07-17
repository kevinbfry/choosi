import os
import numpy as np
import adelie as ad
from scipy.stats import norm

from .distr import WeightedNormal
from .optimizer import CQNMOptimizer
from .choosir import Choosir
from .choosi_core.matrix import hessian

INF_METHODS = ["mle", "exact"]


## TODO
class Lasso(Choosir):
    def __init__(self):
        pass

## Implements split lasso with lambda chosen by validation set performance
class SplitLasso(Choosir):
    def __init__(
        self, 
        X, 
        y,
        tr_idx, 
        val_idx, 
        penalty,
        family,
        lmda_choice=None,
        fit_intercept=True,
        n_threads=None,
    ):
        X, y = self._check_Xy(X, y)

        self.X_full = X
        self.y_full = y

        self.n_full = len(self.y_full)
        tr_idx, val_idx = self._check_idxs(tr_idx, val_idx)

        self.X_tr = self.X_full[tr_idx,:] 
        self.X_val = self.X_full[val_idx,:] 
 
        self.y_tr = self.y_full[tr_idx]   
        self.y_val = self.y_full[val_idx]
        self.n_val = len(self.y_val)
        
        self.n_tr = len(self.y_tr)

        self.tr_idx = tr_idx
        self.val_idx = val_idx
        self.ts_idx = ~(self.tr_idx | self.val_idx)

        self.y_ts = self.y_full[self.ts_idx]
        self.n_ts = len(self.y_ts)

        self.lmda_choice = lmda_choice

        self.pi = self.n_tr / self.n_full

        exec(f"self.glm_full = ad.glm.{family}(self.y_full)")
        exec(f"self.glm_tr = ad.glm.{family}(self.y_tr)")
        exec(f"self.glm_val = ad.glm.{family}(self.y_val)")
        exec(f"self.glm_ts = ad.glm.{family}(self.y_ts)")

        if fit_intercept:
            self.penalty=np.concatenate(([0],penalty)) ## TODO: add unpenalized indices optionality
        else:
            self.penalty = np.array(penalty)

        self.fit_intercept = fit_intercept

        self.n_threads=os.cpu_count() // 2 if n_threads is None else n_threads


    def _check_Xy(
        self,
        X,
        y,
    ):
        if not isinstance(X, np.ndarray) or X.squeeze().ndim != 2:
            raise TypeError("X matrix must be a 2D numpy array")

        if not isinstance(y, np.ndarray) or y.squeeze().ndim != 1:
            raise TypeError("y must be a 1D numpy array")

        if y.squeeze().shape[0] != X.squeeze().shape[0]:
            raise ValueError("X and y must have same number of rows")

        return X.squeeze(), y.squeeze()


    def _check_idxs(
        self,
        tr_idx,
        val_idx,
    ):
        if not isinstance(tr_idx, np.ndarray) or tr_idx.squeeze().ndim != 1:
            raise TypeError("tr_idx must be a 1D numpy array")
        
        if not isinstance(val_idx, np.ndarray) or val_idx.squeeze().ndim != 1:
            raise TypeError("val_idx must be a 1D numpy array")

        if tr_idx.dtype == bool:
            tr_idxs = np.where(tr_idx)[0]
            tr_bool = tr_idx
        else:
            tr_idxs = tr_idx
            tr_bool = np.zeros(self.n_full, dtype=bool)
            tr_bool[tr_idx] = True
        if val_idx.dtype == bool:
            val_idxs = np.where(val_idx)[0]
            val_bool = val_idx
        else:
            val_idxs = val_idx
            val_bool = np.zeros(self.n_full, dtype=bool)
            val_bool[val_idx] = True
            
        if len(np.intersect1d(tr_idxs, val_idxs)) > 0:
            raise ValueError("tr_idx and val_idx must not share indices")

        return tr_bool, val_bool


    def fit(self):
        X = self.X_tr
        y = self.y_tr

        glm = self.glm_tr

        prev_val_obj = 1.
        twice = False
        self.observed_beta = None
        self.observed_intercept = None

        ## TODO: right now if lmda is None just picks 50th lmda in path. Makes testing easy
        ## Really should do opposite. If lmda is None, should dynamically choose. Else it is provided,
        ## and so should make path with lmda as endpt and fit full path.
        if self.lmda_choice is None:
            print("picking by Rsq")
            lmda_path = None
            def val_exit_cond(state):

                nonlocal prev_val_obj
                nonlocal twice

                etas = ad.diagnostic.predict(
                    X=self.X_val,
                    betas=state.betas[-1:],
                    intercepts=state.intercepts[-1:],
                )
                resids = ad.diagnostic.residuals(
                    glm=self.glm_val,
                    etas=etas,
                )
                val_obj = (self.n_val * resids**2).sum() / np.var(self.y_val) - 1 ## objective is -R^2
                if prev_val_obj < val_obj:
                    if twice:
                        return True
                    else:
                        twice = True
                else:
                    twice = False

                prev_val_obj = val_obj
            
                self.observed_beta = state.betas[-1]
                if self.fit_intercept:
                    self.observed_intercept = state.intercepts[-1:]
                else:
                    self.observed_intercept = np.zeros((1))

                self.lmda = state.lmdas[-1]

                return False
            
        elif isinstance(self.lmda_choice, (int, float)):
            self.lmda = self.lmda_choice

            tmp_state = ad.solver.grpnet(
                X=self.X_tr,
                glm=glm,
                adev_tol=.8,
                early_exit=False,
                intercept=self.fit_intercept,
                n_threads=self.n_threads,
                progress_bar=False,
            )
            lmda_path = tmp_state.lmda_path
            lmda_path = np.concatenate((lmda_path[lmda_path > self.lmda], [self.lmda]))

            def val_exit_cond(state):
                nonlocal prev_val_obj
                
                self.observed_beta = state.betas[-1]
                if self.fit_intercept:
                    self.observed_intercept = state.intercepts[-1:]
                else:
                    self.observed_intercept = np.zeros((1))
                return False
            
        elif self.lmda_choice == "mid":
            ## TODO: think I don't need val_exit_cond here.
            ##       can just take last beta, intercept
            tmp_state = ad.solver.grpnet(
                X=self.X_tr,
                glm=glm,
                adev_tol=.8,
                early_exit=False,
                intercept=self.fit_intercept,
                n_threads=self.n_threads,
                progress_bar=False,
            )
            lmda_path = tmp_state.lmda_path
            lmda_path = lmda_path[:len(lmda_path)//2]
            self.lmda = lmda_path[-1]

            def val_exit_cond(state):
                nonlocal prev_val_obj
                
                self.observed_beta = state.betas[-1]
                if self.fit_intercept:
                    self.observed_intercept = state.intercepts[-1:]
                else:
                    self.observed_intercept = np.zeros((1))
                return False

        state = ad.solver.grpnet(
            X=self.X_tr,
            glm=glm,
            adev_tol=.8,
            early_exit=False,
            lmda_path=lmda_path,
            intercept=self.fit_intercept,
            exit_cond=val_exit_cond,
            n_threads=self.n_threads,
            progress_bar=False,
        )
        
        self.observed_beta = self.observed_beta.toarray()
        if (self.observed_beta != 0).sum() == 0:
            assert 0==1
        if self.fit_intercept:
            self.observed_soln = np.concatenate((
                self.observed_intercept,
                self.observed_beta.flatten(),
            ))
        else:
            self.observed_soln = self.observed_beta.flatten()

        self.state = state
        return state

    
    def _selected_estimator(self, X, glm):
        features = self.overall
        if self.fit_intercept:
            features = features[1:]
        penalty = np.ones(X.shape[1])
        penalty[features] = 0

        state = ad.solver.grpnet(
            X=X,
            glm=glm,
            penalty=penalty,
            lmda_path_size=1,
            intercept=self.fit_intercept,
            n_threads=self.n_threads,
            progress_bar=False,
            tol=1e-12,
        )

        beta = state.betas[0].toarray().flatten()[features]
        if self.fit_intercept:
            intercept = state.intercepts[0]
            beta = np.concatenate([[intercept], beta])

        return beta


    def extract_event(self):
        self.unpen = (self.penalty == 0)
        self.overall = (self.observed_soln != 0)
        self.p_sel = self.overall.sum()
        self.active = ~self.unpen & self.overall
        self.active_signs = self.observed_soln[self.active]

        """
        right now code sets signs of unpenalized to 1, 
        while still including them in constraints.
        don't know if this is correct. Seems should drop
        unpenalized from constraints so that unpenalized
        parameters can be negative.
        """
        self.signs = np.sign(self.observed_soln)
        self.signs = self.signs[self.overall]

        etas = ad.diagnostic.predict(
            X=self.X_tr,
            betas=self.observed_beta,
            intercepts=self.observed_intercept,
        )
        resids = ad.diagnostic.residuals(
            glm=self.glm_tr,
            etas=etas,
        )
        observed_subgrad = ad.diagnostic.gradients(
            X=self.X_tr,
            resids=resids,
            n_threads=self.n_threads,
        )[0]
        self.observed_subgrad = observed_subgrad
        if self.fit_intercept:
            self.observed_subgrad = np.concatenate(([resids.sum()],self.observed_subgrad))

        eta = etas[0]
        grad = np.empty_like(eta)
        self.glm_tr.gradient(eta, grad)
        self.W = np.empty_like(eta)
        self.glm_tr.hessian(eta, grad, self.W)# includes 1/n

        self.beta_unpen = self._selected_estimator(self.X_full, self.glm_full)

        ## opt conditional mean, prec
        self.alpha = (1 - self.pi) / self.pi
        self.H_E, self.H_E_inv = self._compute_hessian()

        return self


    def _compute_hessian(self):
        if self.fit_intercept:
            self.X_E = X_E = np.hstack((np.ones((self.n_full,1)), self.X_full))[:, self.overall]
        else:
            self.X_E = X_E = self.X_full[:, self.overall]

        H_E = X_E.T @ X_E / self.n_full

        try:
            H_E_inv = np.linalg.inv(H_E)
        except:
            H_E[np.diag_indices_from(H_E)] += 1e-12 ## TODO: do this dynamically, add enough to get PDness
            H_E_inv = np.linalg.inv(H_E)

        return H_E, H_E_inv
    

    def _compute_dispersion(self):
        ## TODO: add logic for when X_ts provided vs not
        X_ts = self.X_full[self.ts_idx,:]

        beta = np.zeros_like(self.observed_soln)
        beta[self.overall] = self.beta_unpen

        etas = ad.diagnostic.predict(
            X=X_ts,
            betas=beta[None,1:],
            intercepts=beta[:1],
        )
        resids = ad.diagnostic.residuals(
            glm=self.glm_ts,
            etas=etas,
        ) ## is (y- Xb) / n_ts

        return np.sum((self.n_ts*resids)**2) / (self.n_ts - self.p_sel)
        

    ## TODO: add level parameter
    def infer(self, method="mle", dispersion=None, level=.95):
        if method not in INF_METHODS:
            raise ValueError("Inference only valid for 'method' in f{INF_METHODS}")

        if dispersion is None:
            self.dispersion = self._compute_dispersion()
        else:
            self.dispersion = dispersion

        if method == "mle":
            return self._infer_MLE(level=level)
        elif method == "exact":
            return self._infer_exact(level=level)
        
    
    def _infer_exact(self, level):

        ## constraints
        self.con_linear = -np.diag(self.signs)[self.active[self.overall]]
        self.con_offset = self.con_linear @ self.H_E_inv @ (self.lmda * self.penalty[self.overall] * self.signs)# / self.n_full

        ## compute unrandomized intervals
        L, U = self._compute_unrand_trunc()

        ## convolve with randomization noise
        grids, weights = self._convolve_with_rand(L, U)

        ## use convolution as weights in conditional distribution of \hat\beta \mid selection
        cond_distrs = self._form_cond_distr(grids, weights)

        ## 1D-root finding for each target
        lower, upper = self._find_roots(cond_distrs, level) ## TODO: update with level parameter

        return lower, upper
        

    def _convolve_with_rand(
        self,
        L,
        U,
        n_grid=4000,
    ):
        sigmas = np.sqrt(
            np.diag(self.H_E_inv) * self.alpha * self.dispersion / self.n_full
        )
        opts = self.beta_unpen
        weights = np.zeros((self.p_sel,n_grid))
        grids = np.zeros((self.p_sel,n_grid))
        for j in range(self.p_sel):
            t = np.linspace(opts[j] - 10*sigmas[j]/self.alpha, opts[j] + 10*sigmas[j]/self.alpha, n_grid)
            zu = (U[j] - t)/sigmas[j]
            zl = (L[j] - t)/sigmas[j]

            grids[j,:] = t
            weights[j,:] = norm.cdf(zu) - norm.cdf(zl)

        return grids, weights
    

    def _form_cond_distr(
        self,
        grids,
        weights,
    ):
        sigmas = np.sqrt(np.diag(self.H_E_inv) * self.dispersion / self.n_full)
        return [
            WeightedNormal(
                mu=self.observed_soln[self.overall][i],
                sigma=sigmas[i],
                grid=grids[i,:],
                weights=weights[i,:],
            ) 
            for i in range(weights.shape[0])
        ]


    def _find_roots(
        self,
        cond_distrs,
        level,
    ):
        lower = np.zeros(self.p_sel)
        upper = np.zeros(self.p_sel)
        opts = self.beta_unpen
        for j in range(self.p_sel):
            cond_distr = cond_distrs[j]

            lower[j], upper[j] = cond_distr.find_ci(opts[j], level)

        return lower, upper


    def _compute_unrand_trunc(
        self
    ):
        A = self.con_linear ## don't think this is ever needed as matrix. should restructure to store as vector
        b = self.con_offset

        cov = self.H_E_inv * self.dispersion / self.n_full ## (X'X)^{-1}
        opt = self.observed_soln[self.overall] + self.H_E_inv @ (self.lmda * self.penalty[self.overall] * self.signs)# / self.n_full ## TODO: should this be observed_soln?

        L = np.ones(self.p_sel) * (-np.inf)
        U = np.ones(self.p_sel) * np.inf

        if len(A) > 0:
            for j in range(self.p_sel):
                ## regress target out of opt variables
                gamma = cov[:,j] / cov[j,j]
                nuis = opt - gamma * opt[j]
                Agamma = A.dot(gamma)#A @ gamma ## TODO: should be able to do this as element-wise.
                Anuis = A.dot(nuis)#A @ nuis
                bmAnuis = b - Anuis

                tol = 1.e-4 * (np.fabs(Agamma).max() if len(Agamma) > 0 else 1.)

                L[j] = np.amax((bmAnuis / Agamma)[Agamma < -tol]) if (Agamma < -tol).sum() > 0 else -np.inf
                U[j] = np.amin((bmAnuis / Agamma)[Agamma > tol]) if (Agamma > tol).sum() > 0 else np.inf

                assert L[j] < U[j] ##TODO: sometimes this gets triggered...

        return L, U


    def _infer_MLE(self, level):
        dispersion = self.dispersion
        H_E = self.H_E
        H_E_inv = self.H_E_inv
        alpha = self.alpha
        self.cond_cov = cond_cov = H_E_inv * dispersion / self.n_full
        self.cond_prec = cond_prec = H_E * self.n_full / (dispersion * alpha)
        self.cond_mean = cond_mean = self.beta_unpen - H_E_inv @ (self.lmda * self.penalty[self.overall] * self.signs) ## \bar\beta_E - \lambda(X_E.T X_E)^{-1}z_E

        ## call optimizer
        self.optimizer = optimizer = CQNMOptimizer(
            linear_term=-cond_prec @ cond_mean,
            quad_form=cond_prec,
            signs=self.signs,
            pen_idxs=self.penalty[self.overall] != 0,
            lmda=1,
        )

        soln = optimizer.optimize()
        hess_barrier = cond_prec.copy()
        hess_barrier[np.diag_indices_from(hess_barrier)] += optimizer.barrier.get_hess_diag(soln)

        sel_MLE = self.beta_unpen + (cond_mean - soln) / alpha
        sel_MLE_std = np.sqrt(np.diag((1 + 1./alpha) * cond_cov - np.linalg.inv(hess_barrier) / alpha**2))

        sigma = norm.ppf((1+level)/2)

        return sel_MLE - sigma * sel_MLE_std, sel_MLE + sigma * sel_MLE_std



class SplitLassoSNPUnphased(SplitLasso):
    def _compute_dispersion(self):
        return super()._compute_dispersion()

    ## TODO: add covs_ts, X_ts_fnames optional argument
    def __init__(
        self,
        X_fnames,
        y,
        tr_idx, 
        val_idx,
        penalty,
        family,
        covs_full=None,
        X_tr_fnames=None, ## NOTE: passing in tr, val fnames may be faster than just idxs
        covs_tr=None,
        X_val_fnames=None,
        covs_val=None,
        lmda_choice=None,
        fit_intercept=True,
        n_threads=None,
    ):

        self.covs_full = covs_full
        self.covs_tr = covs_tr
        self.covs_val = covs_val

        super().__init__(
            X=X_fnames,
            y=y,
            tr_idx=tr_idx,
            val_idx=val_idx,
            penalty=penalty,
            family=family,
            lmda_choice=lmda_choice,
            fit_intercept=fit_intercept,
            n_threads=n_threads,
        )

        if X_tr_fnames is not None:
            self.X_tr_fnames = [os.path.join(fname) for fname in X_tr_fnames]
            self.X_tr_handlers = [ad.io.snp_unphased(fname) for fname in self.X_tr_fnames]
            self.X_tr = ad.matrix.concatenate([ad.matrix.snp_unphased(handler) for handler in self.X_tr_handlers], axis=1)
            if self.covs_tr is not None:
                if not isinstance(self.covs_tr, np.ndarray):
                    raise TypeError("'covs_tr' must be an ndarray.")
                self.X_tr = ad.matrix.concatenate(
                    [
                        ad.matrix.dense(self.covs_tr),
                        self.X_tr,
                    ],
                    axis=1,
                )

        if X_val_fnames is not None:
            self.X_val_fnames = [os.path.join(fname) for fname in X_val_fnames]
            self.X_val_handlers = [ad.io.snp_unphased(fname) for fname in self.X_val_fnames]
            self.X_val = ad.matrix.concatenate([ad.matrix.snp_unphased(handler) for handler in self.X_val_handlers], axis=1)
            if self.covs_val is not None:
                if not isinstance(self.covs_val, np.ndarray):
                    raise TypeError("'covs_val' must be an ndarray.")
                self.X_val = ad.matrix.concatenate(
                    [
                        ad.matrix.dense(self.covs_val),
                        self.X_val,
                    ],
                    axis=1,
                )


    def _check_Xy(self, X_fnames, y):

        if not isinstance(y, np.ndarray) or y.squeeze().ndim != 1:
            raise TypeError("y must be a 1D numpy array")

        self.X_full_fnames = [os.path.join(fname) for fname in X_fnames]
        self.X_full_handlers = [ad.io.snp_unphased(fname) for fname in self.X_full_fnames]
        X = ad.matrix.concatenate([ad.matrix.snp_unphased(handler) for handler in self.X_full_handlers], axis=1)
        if self.covs_full is not None:
            if not isinstance(self.covs_full, np.ndarray):
                raise TypeError("'covs' must be an ndarray.")
            X = ad.matrix.concatenate(
                [
                    ad.matrix.dense(self.covs_full),
                    X,
                ],
                axis=1,
            )

        if y.squeeze().shape[0] != X.shape[0]:
            raise ValueError("X and y must have same number of rows")

        return X, y


    def _compute_hessian(self):
        # subset = np.where(self.overall[1:])[0] if self.fit_intercept else np.where(self.overall)[0]
        if self.covs_full is not None:
            n_covs = self.covs_full.shape[1] + self.fit_intercept
        else:
            n_covs = int(self.fit_intercept)

        subset = np.where(self.overall[n_covs:])[0]
        E = len(subset)
        H_snps = np.zeros((E, E))
        weights = np.ones(self.n_full)/self.n_full
        if E > 0:
            hessian(self.X_full_handlers, subset, weights, H_snps, self.n_threads)
        if self.fit_intercept or self.covs_full is not None:

            covs_subset = np.where(self.overall[:n_covs])[0]
            covs_E = len(covs_subset)
            covs = np.hstack((np.ones((self.n_full,1)),self.covs_full))[:,covs_subset] if self.fit_intercept else self.covs_full[:,covs_subset]

            H_covs = covs.T @ covs / self.n_full

            H_E = np.zeros((E + covs_E, E + covs_E))
            H_E[:covs_E,:][:,:covs_E] = H_covs
            H_E[covs_E:,:][:,covs_E:] = H_snps

            subset = subset + n_covs - self.fit_intercept
            for i in range(len(subset)):
                for j in range(covs_E):
                    H_E[j,covs_E+i] = H_E[covs_E+i,j] = self.X_full.cmul(subset[i], covs[:,j], weights)

        H_E_inv = np.linalg.inv(H_E)

        return H_E, H_E_inv


    



        






