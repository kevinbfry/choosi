import os
import numpy as np
import adelie as ad
from .optimizer import CQNMOptimizer
from .choosir import Choosir
from .choosi_core.matrix import hessian

INF_METHODS = ["MLE"]


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
        tr_idx, ## TODO: just pass in tr, val, idxs
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

        # if not isinstance(X, ad.matrix.MatrixNaiveBase64):
        #     ad_X = ad.matrix.dense(X)
        # else:
        #     ad_X = X
        
        # self.glm = glm = ad.glm.gaussian(y) ## TODO: rn just gaussian
        glm = self.glm_tr

        prev_val_obj = np.inf
        self.observed_beta = None
        self.observed_intercept = None

        ## TODO: right now if lmda is None just picks 50th lmda in path. Makes testing easy
        ## Really should do opposite. If lmda is None, should dynamically choose. Else it is provided,
        ## and so should make path with lmda as endpt and fit full path.
        if self.lmda_choice is None:
            lmda_path = None
            def val_exit_cond(state):
                # if state.betas.shape[0] < 2:
                #     return False

                nonlocal prev_val_obj

                ## TODO: change to Rsq
                val_obj = ad.diagnostic.objective(
                    X=self.X_val,
                    glm=self.glm_val,
                    penalty=self.penalty,
                    intercepts=state.intercepts[-1:],
                    betas=state.betas[-1:],
                    lmdas=state.lmdas[-1:],
                    add_penalty=False,
                    n_threads=self.n_threads,
                )[0]
                if prev_val_obj < val_obj:
                    return True
                else:
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
            )
            lmda_path = tmp_state.lmda_path
            lmda_path = np.concatenate((lmda_path[lmda_path > self.lmda], [self.lmda]))

            def val_exit_cond(state):
                # if state.betas.shape[0] < 2:
                #     return False

                nonlocal prev_val_obj
                
                self.observed_beta = state.betas[-1]
                if self.fit_intercept:
                    self.observed_intercept = state.intercepts[-1:]
                else:
                    self.observed_intercept = np.zeros((1))
                return False
            
        elif self.lmda_choice == "mid":
            tmp_state = ad.solver.grpnet(
                X=self.X_tr,
                glm=glm,
                adev_tol=.8,
                early_exit=False,
                intercept=self.fit_intercept,
                n_threads=self.n_threads,
            )
            lmda_path = tmp_state.lmda_path
            lmda_path = lmda_path[:len(lmda_path)//2]
            self.lmda = lmda_path[-1]

            def val_exit_cond(state):
                # if state.betas.shape[0] < 2:
                #     return False

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
        )

        # if len(lmda_path) == 1:
        #     self.observed_beta = state.betas[-1]
        #     if self.fit_intercept:
        #         self.observed_intercept = state.intercepts[-1:]
        #     else:
        #         self.observed_intercept = np.zeros((1))

        self.observed_beta = self.observed_beta.toarray()
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
            progress_bar=False,
            intercept=self.fit_intercept,
            n_threads=self.n_threads,
        )

        beta = state.betas[0].toarray().flatten()[features]
        if self.fit_intercept:
            intercept = state.intercepts[0]
            beta = np.concatenate([[intercept], beta])

        return beta


    def extract_event(self):
        # if not isinstance(self.X_full, ad.matrix.MatrixNaiveBase64):
        #     self.X_isnumpy = True
        #     ad_X_full = ad.matrix.dense(self.X_full)
        #     ad_X = ad.matrix.dense(self.X_tr)
        # else:
        #     self.X_isnumpy = False
        #     ad_X_full = self.X_full
        #     ad_X = self.X_tr

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
        self.signs = np.sign(self.observed_soln)#[self.overall]
        # self.signs[self.unpen] = 1.
        self.signs = self.signs[self.overall]

        etas = ad.diagnostic.predict(
            X=self.X_tr,
            betas=self.observed_beta,
            intercepts=self.observed_intercept,
        )
        resids = ad.diagnostic.residuals(
            # glm=self.glm_full,
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
        # self.glm_full.gradient(eta, grad)
        self.glm_tr.gradient(eta, grad)
        self.W = np.empty_like(eta)
        # self.glm_full.hessian(eta, grad, self.W)# includes 1/n
        self.glm_tr.hessian(eta, grad, self.W)# includes 1/n

        ## constraints
        self.con_linear = -np.eye(self.p_sel)
        self.con_offset = np.zeros(self.p_sel)


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

        H_E = X_E.T @ X_E / len(self.y_full)

        try:
            H_E_inv = np.linalg.inv(H_E)
        except:
            H_E[np.diag_indices_from(H_E)] += 1e-12 ## TODO: do this dynamically, add enough to get PDness
            H_E_inv = np.linalg.inv(H_E)

        return H_E, H_E_inv
    

    ## SCRATCH THIS?? NOTE: assumes self._compute_hessian() called first
    def _compute_dispersion(self):
        # if not hasattr(self, "X_E"):
        #     raise AttributeError("'X_E' has not been computed. Usually due to not running extract_event() before infer()")

        X_ts = self.X_full[self.ts_idx,:]

        etas = ad.diagnostic.predict(
            X=X_ts,
            betas=self.observed_beta,
            intercepts=self.observed_intercept,
        )
        resids = ad.diagnostic.residuals(
            glm=self.glm_ts,
            etas=etas,
        ) ## is (y- Xb) / n_ts

        return np.sum((self.n_ts*resids)**2)/ (self.n_ts - self.p_sel)
        

    def infer(self, method="MLE", dispersion=None):
        if method not in INF_METHODS:
            raise ValueError("Inference only valid for 'method' in f{INF_METHODS}")

        if dispersion is None:
            self.dispersion = dispersion = self._compute_dispersion()
        else:
            self.dispersion = dispersion

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
        # H_E_barrier[np.diag_indices_from(H_E_barrier)] += optimizer.get_barrier_hess(soln)
        hess_barrier[np.diag_indices_from(hess_barrier)] += optimizer.barrier.get_hess_diag(soln)

        sel_MLE = self.beta_unpen + (cond_mean - soln) / alpha
        sel_MLE_std = np.sqrt(np.diag((1 + 1./alpha) * cond_cov - np.linalg.inv(hess_barrier) / alpha**2))# / np.sqrt(dispersion)

        return (sel_MLE, sel_MLE_std)



class SplitLassoSNPUnphased(SplitLasso):
    def __init__(
        self,
        X_fnames,
        y,
        penalty,
        family,
        X_tr_fnames=None,
        X_val_fnames=None,
        y_tr=None,
        y_val=None,
        tr_idx=None, ## NOTE: passing in tr_idx, val_idx, may be slower than fnames
        val_idx=None,
        lmda_choice=None,
        fit_intercept=True,
        n_threads=None,
    ):

        self.X_full_fnames = [os.path.join(fname) for fname in X_fnames]
        self.X_full_handlers = [ad.io.snp_unphased(fname) for fname in self.X_full_fnames]
        self.X_full = ad.matrix.concatenate([ad.matrix.snp_unphased(handler) for handler in self.X_full_handlers], axis=1)
        self.y_full = y
        self.n_full = len(self.y_full)

        if X_tr_fnames is None and X_val_fnames is None:
            tr_idx, val_idx = self._check_idxs(tr_idx, val_idx)
            self.tr_idx = tr_idx
            self.val_idx = val_idx
            self.ts_idx = ~(tr_idx | val_idx)

        if X_tr_fnames is None:
            self.X_tr_fnames = None
            self.X_tr_handlers = None
            self.X_tr = self.X_full[tr_idx,:]
        else:
            self.X_tr_fnames = [os.path.join(fname) for fname in X_tr_fnames]
            self.X_tr_handlers = [ad.io.snp_unphased(fname) for fname in self.X_tr_fnames]
            self.X_tr = ad.matrix.concatenate([ad.matrix.snp_unphased(handler) for handler in self.X_tr_handlers], axis=1)

        if X_val_fnames is None:
            self.X_val_fnames = None
            self.X_val_handlers = None
            self.X_val = self.X_full[val_idx,:]
        else:
            self.X_val_fnames = [os.path.join(fname) for fname in X_val_fnames]
            self.X_val_handlers = [ad.io.snp_unphased(fname) for fname in self.X_val_fnames]
            self.X_val = ad.matrix.concatenate([ad.matrix.snp_unphased(handler) for handler in self.X_val_handlers], axis=1)


        if y_tr is None:
            self.y_tr = self.y_full[tr_idx]
        else:
            self.y_tr = y_tr
        self.n_tr = len(self.y_tr)

        if y_val is None:
            self.y_val = self.y_full[val_idx]
        else:
            self.y_val = y_val

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


    def _compute_hessian(self):
        E = self.p_sel - self.fit_intercept
        subset = np.where(self.overall[1:])[0] if self.fit_intercept else np.where(self.overall)[0]
        H_E = np.zeros((E, E))
        hessian(self.X_full_handlers, subset, np.ones(self.n_full)/self.n_full, H_E, self.n_threads)
        if self.fit_intercept:
            int_col = np.zeros((1,E))
            for i in range(len(subset)):
                int_col[:,i] = self.X_full.cmul(subset[i], np.ones(self.n_full), np.ones(self.n_full) / self.n_full)
            H_E = np.hstack((int_col.T, H_E))
            H_E = np.vstack((np.hstack(([[1]],int_col)), H_E))

        H_E_inv = np.linalg.inv(H_E)
        return H_E, H_E_inv


    



        






