import os
import numpy as np
import adelie as ad
from .optimizer import EQNMOptimizer
from .choosir import Choosir

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

        exec(f"self.glm_full = ad.glm.{family}(self.y_full)")
        exec(f"self.glm_tr = ad.glm.{family}(self.y_tr)")

        self.penalty=np.concatenate(([0],penalty)) ## TODO: add intercept optionality

        self.n_threads=os.cpu_count() // 2 if n_threads is None else n_threads


    def fit(self):
        X = self.X_tr
        y = self.y_tr

        if not isinstance(X, ad.matrix.MatrixNaiveBase64):
            ad_X = ad.matrix.dense(X)
        
        glm = ad.glm.gaussian(y) ## TODO: rn just gaussian

        prev_val_obj = np.inf
        self.observed_beta = None
        self.observed_intercept = None

        def val_exit_cond(state):
            if state.betas.shape[0] < 2:
                return False

            nonlocal prev_val_obj
            
            self.observed_beta = state.betas[-1]
            self.observed_intercept = state.intercepts[-1:]
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
                return False

        state = ad.solver.grpnet(
            X=ad_X,
            glm=glm,
            adev_tol=.8,
            early_exit=False,
            exit_cond=val_exit_cond,
        )

        self.observed_beta = self.observed_beta.toarray()
        self.observed_soln = np.concatenate((
            self.observed_intercept,
            self.observed_beta.flatten(),
        ))

        self.state = state
        return state

    
    def extract_event(self):
        if not isinstance(self.X_full, ad.matrix.MatrixNaiveBase64):
            ad_X = ad.matrix.dense(self.X_full)

        self.unpen = (self.penalty == 0)
        self.overall = (self.observed_soln != 0)
        self.n_opt_var = self.overall.sum()
        self.active = ~self.unpen & self.overall

        etas = ad.diagnostic.predict(
            X=ad_X,
            betas=self.observed_beta,
            intercepts=self.observed_intercept,
        )
        resids = ad.diagnostic.residuals(
            glm=self.glm_full,
            etas=etas,
        )
        observed_subgrad = ad.diagnostic.gradients(
            X=ad_X,
            resids=resids,
            n_threads=self.n_threads,
        )[0]
        self.observed_subgrad = observed_subgrad

        eta = etas[0]
        grad = np.empty_like(eta)
        self.glm_full.gradient(eta, grad)
        self.W = np.empty_like(eta)
        self.glm_full.hessian(eta, grad, self.W)# includes 1/n

        ## constraints
        self.con_linear = -np.eye(self.n_opt_var)
        self.con_offset = np.zeros(self.n_opt_var)

        ## opt conditional mean, prec
        # cond_mean = self.observed_soln[self.active] ## \bar\beta_E - \lambda(XE.T XE)^{-1}z_E = \hat\beta_E
        # cond_prec = self.hess_E ## XE.T XE

        


    def infer(self):
        ## solve optimization problem


        ## compute selective MLE from solution


        ## compute fisher info from solution
        pass


        






