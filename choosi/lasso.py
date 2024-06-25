import os
import numpy as np
import adelie as ad
from .optimizer import CQNMOptimizer
from .choosir import Choosir

INF_METHODS = ["MLE"]


## TODO
class Lasso(Choosir):
    def __init__(self):
        pass

## Implements split lasso with lambda chosen by validation set performance
class SplitLasso(Choosir):
    def __init__(
        self, 
        X_full, 
        X_tr, 
        X_val, 
        X_ts,
        y_full,
        y_tr, 
        y_val, 
        y_ts,
        penalty,
        family,
        lmda_choice=None,
        fit_intercept=True,
        n_threads=None,
    ):
        self.X_full=X_full
        self.X_tr=X_tr 
        self.X_val=X_val 
        self.X_ts=X_ts

        self.y_full=y_full
        self.y_tr=y_tr   
        self.y_val=y_val 
        self.y_ts=y_ts

        self.lmda_choice = lmda_choice
        
        self.pi = len(self.y_tr) / len(self.y_full)

        exec(f"self.glm_full = ad.glm.{family}(self.y_full)")
        exec(f"self.glm_tr = ad.glm.{family}(self.y_tr)")

        if fit_intercept:
            self.penalty=np.concatenate(([0],penalty)) ## TODO: add unpenalized indices optionality
        else:
            self.penalty = np.array(penalty)

        self.fit_intercept = fit_intercept

        self.n_threads=os.cpu_count() // 2 if n_threads is None else n_threads


    def fit(self):
        X = self.X_tr
        y = self.y_tr

        if not isinstance(X, ad.matrix.MatrixNaiveBase64):
            ad_X = ad.matrix.dense(X)
        
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
                if state.betas.shape[0] < 2:
                    return False

                nonlocal prev_val_obj
                
                self.observed_beta = state.betas[-1]
                if self.fit_intercept:
                    self.observed_intercept = state.intercepts[-1:]
                else:
                    self.observed_intercept = np.zeros((1))

                self.lmda = state.lmdas[-1]

                ## TODO: change to Rsq
                val_obj = ad.diagnostic.objective(
                    X=self.X_val,
                    glm=ad.glm.gaussian(y=self.y_val),
                    penalty=self.penalty,
                    n_threads=self.n_threads,
                    intercepts=state.intercepts[-1:],
                    betas=state.betas[-1:],
                    lmdas=state.lmdas[-1:],
                    add_penalty=False,
                )[0]
                if prev_val_obj < val_obj:
                    return True
                else:
                    prev_val_obj = val_obj
                    return Fals
            
        elif isinstance(self.lmda_choice, (int, float)):
            self.lmda = self.lmda_choice

            tmp_state = ad.solver.grpnet(
                X=ad_X,
                glm=glm,
                adev_tol=.8,
                early_exit=False,
                intercept=self.fit_intercept,
            )
            lmda_path = tmp_state.lmda_path
            lmda_path = np.concatenate((lmda_path[lmda_path > self.lmda], [self.lmda]))

            def val_exit_cond(state):
                if state.betas.shape[0] < 2:
                    return False

                nonlocal prev_val_obj
                
                self.observed_beta = state.betas[-1]
                if self.fit_intercept:
                    self.observed_intercept = state.intercepts[-1:]
                else:
                    self.observed_intercept = np.zeros((1))
                return False
            
        elif self.lmda_choice == "mid":
            # self.lmda = self.lmda_choice

            tmp_state = ad.solver.grpnet(
                X=ad_X,
                glm=glm,
                adev_tol=.8,
                early_exit=False,
                intercept=self.fit_intercept,
            )
            lmda_path = tmp_state.lmda_path
            lmda_path = lmda_path[:len(lmda_path)//2]
            self.lmda = lmda_path[-1]

            def val_exit_cond(state):
                if state.betas.shape[0] < 2:
                    return False

                nonlocal prev_val_obj
                
                self.observed_beta = state.betas[-1]
                if self.fit_intercept:
                    self.observed_intercept = state.intercepts[-1:]
                else:
                    self.observed_intercept = np.zeros((1))
                return False

        state = ad.solver.grpnet(
            X=ad_X,
            glm=glm,
            adev_tol=.8,
            early_exit=False,
            lmda_path=lmda_path,
            intercept=self.fit_intercept,
            exit_cond=val_exit_cond,
        )

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
            n_threads=self.n_threads,
            lmda_path_size=1,
            progress_bar=False,
            intercept=self.fit_intercept,
        )

        beta = state.betas[0].toarray().flatten()[features]
        if self.fit_intercept:
            intercept = state.intercepts[0]
            beta = np.concatenate([[intercept], beta])

        return beta


    def extract_event(self):
        if not isinstance(self.X_full, ad.matrix.MatrixNaiveBase64):
            ad_X_full = ad.matrix.dense(self.X_full)
            ad_X = ad.matrix.dense(self.X_tr)
        else:
            ad_X_full = self.X_full
            ad_X = self.X_tr

        self.unpen = (self.penalty == 0)
        self.overall = (self.observed_soln != 0)
        self.n_opt_var = self.overall.sum()
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
            X=ad_X,
            betas=self.observed_beta,
            intercepts=self.observed_intercept,
        )
        resids = ad.diagnostic.residuals(
            # glm=self.glm_full,
            glm=self.glm_tr,
            etas=etas,
        )
        observed_subgrad = ad.diagnostic.gradients(
            X=ad_X,
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
        self.con_linear = -np.eye(self.n_opt_var)
        self.con_offset = np.zeros(self.n_opt_var)


        self.beta_unpen = self._selected_estimator(ad_X_full, self.glm_full)

        ## opt conditional mean, prec
        self.alpha = (1 - self.pi) / self.pi
        if self.fit_intercept:
            # X_E = np.hstack((np.ones((self.X_tr.shape[0],1)), self.X_tr))[:, self.overall]#[:, self.active]
            X_E = np.hstack((np.ones((self.X_full.shape[0],1)), self.X_full))[:, self.overall]#[:, self.active]
        else:
            # X_E = self.X_tr[:, self.overall]#[:, self.active]
            X_E = self.X_full[:, self.overall]#[:, self.active]
        # self.H_E = X_E.T @ X_E / len(self.y_tr)
        self.H_E = X_E.T @ X_E / len(self.y_full)

        return self

    def infer(self, method="MLE"):
        if method not in INF_METHODS:
            raise ValueError("Inference only valid for 'method' in f{INF_METHODS}")


        H_E = self.H_E
        alpha = self.alpha
        self.H_E_inv = H_E_inv = np.linalg.inv(H_E)
        self.cond_cov = cond_cov = H_E_inv / len(self.y_full)
        self.cond_prec = cond_prec = H_E * len(self.y_full) / alpha ## need to undo division by n ## needs to scale by (1 - pi) / pi
        self.cond_mean = cond_mean = self.beta_unpen - H_E_inv @ (self.lmda * self.penalty[self.overall] * self.signs) ## \bar\beta_E - \lambda(X_E.T X_E)^{-1}z_E

        ## call optimizer
        self.optimizer = optimizer = CQNMOptimizer(
            linear_term=-cond_prec @ cond_mean,
            quad_form=cond_prec,
            signs=self.signs, ## TODO: update Optimizer class to disregard signs for unpenalized variables (affects obj, grad, hess)
            # scaling=np.ones_like(self.signs),#np.sqrt(np.diag(cond_mean.dot(cond_prec).dot(cond_mean.T))), ## TODO: scaling needs to be dynamic, should be integrated into Optimizer class. Prob need to change hessians/gradients
            pen_idxs=self.penalty[self.overall] != 0,
            lmda=1,
        )

        soln = optimizer.optimize()
        hess_barrier = cond_prec.copy()
        # H_E_barrier[np.diag_indices_from(H_E_barrier)] += optimizer.get_barrier_hess(soln)
        hess_barrier[np.diag_indices_from(hess_barrier)] += optimizer.barrier.get_hess_diag(soln)

        sel_MLE = self.beta_unpen + (cond_mean - soln) / alpha
        sel_MLE_std = np.sqrt(np.diag((1 + 1./alpha) * cond_cov - np.linalg.inv(hess_barrier) / alpha**2))

        return (sel_MLE, sel_MLE_std)


        






