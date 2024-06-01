import numpy as np
from scipy.sparse.linalg import LinearOperator
from abc import ABC, abstractmethod


class Optimizer(object):
    @abstractmethod
    def optimize(
        self,
    ):
        pass

class AbstractNMOptimizer(Optimizer):
    def __init__(
        self,
        linear_term, # v
        quad_form, # H
        signs, # S
        scaling, # s
        lmda=1., # penalty param
    ):
        self.v = self.linear_term = linear_term
        self.H = quad_form
        self.signs = signs
        self.scaling = scaling
        self.lmda=lmda
    

    @abstractmethod
    def _get_hessinv(self, z):
        pass


    @abstractmethod
    def _get_obj(self, z):
        return .5*z.dot(self.H @ z) + self.v.dot(z) - self.lmda * np.log((self.signs * z) / self.scaling).sum()


    @abstractmethod
    def _get_grad(self, z):
        return self.H @ z + self.v - self.lmda / z


    def optimize(
        self,
        max_iters=100,
        c=.5,
        tau=.5,
        tol=1e-8,
    ):
        return self._optimize(
            max_iters=max_iters,
            c=c,
            tau=tau,
            tol=tol,
        )


    def _optimize(
        self,
        max_iters=100,
        c=.5,
        tau=.5,
        tol=1e-8,
    ):
        z_new = 1. / self.scaling 
        grad_new = self._get_grad(z_new)
        for i in range(max_iters):
            step_size = 1.

            z_prev = z_new
            grad_prev = grad_new

            H_inv = self._get_hessinv(z_prev)
            p = -H_inv @ grad_prev
            z_new = z_prev + step_size * p
            t = -c*grad_prev.dot(p)
            obj_prev = self._get_obj(z_prev)
            obj_new = self._get_obj(z_new)
            ct = 0
            while np.isnan(obj_new) or obj_prev - obj_new < step_size * t:
                if ct == 20: break
                ct += 1
                step_size = tau * step_size
                z_new = z_prev + step_size * p
                obj_new = self._get_obj(z_new)

            if ct == 20:
                break

            grad_new = self._get_grad(z_new)
            if i > 0 and (grad_new - grad_prev).dot(z_new - z_prev) < tol:
                break

        return z_new

class QNMOptimizer(AbstractNMOptimizer):
    def __init__(
        self,
        linear_term, # v
        quad_form, # H
        quad_form_approx, # (Q,\Lambda) where H \approx Q\LambdaQ'
        signs, # S
        scaling, # s
        lmda=1., # penalty param
    ):
        super().__init__(
            linear_term,
            quad_form,
            signs,
            scaling,
            lmda,
        )
        self.Q, self.Lambda = quad_form_approx
        self.v = self.Q.T @ self.linear_term
        self.SQ = signs[:,None] * self.Q
    

    ## invert using woodbury matrix identity
    def _get_hessinv(self, z):
        Lambda_inv = 1/self.Lambda
        SQz = self.SQ @ z
        Lambda_inv_SQ = Lambda_inv[:,None] * self.SQ
        M = self.Q.T @ (Lambda_inv[:,None] * self.Q)
        for i in range(SQz.shape[0]):
            M[i,i] += SQz[i]**2 / self.lmda 
        soln = np.linalg.solve(M, Lambda_inv_SQ.T)

        woodbury_inv = np.diag(Lambda_inv) - Lambda_inv_SQ @ soln

        return woodbury_inv


    def _get_obj(self, z):
        return .5*(z**2 * self.Lambda).sum() + self.v.dot(z) - self.lmda * np.log((self.SQ @ z) / self.scaling).sum()


    def _get_grad(self, z):
        return z * self.Lambda + self.v - self.lmda * (self.SQ / (self.SQ @ z)).sum(0)

        
    def optimize(
        self,
        max_iters=100,
        c=.5,
        tau=.5,
        tol=1e-8,
    ):
        ## TODO: wrap and solve multiple times for path of weighting of obj and barrier

        ## solves:
        ## min_x .5 x' H x + x'v - sum_i log(Sx_i/s_i)
        ## log barrier approximates constraint S x_i \ge 0, scaled by s_i to give equal weight
        ## by approximating by low rank letting z = Q'x. Then
        ## min_z .5 z' \Lambda z + w'z - sum_i ((SQ)_i z / s_i)
        ## where w = Q'v

        ## gradient is \Lambda z + w - sum_i ((SQ)_i / s_i) / ((SQ)_i z / s_i) = \Lambda z + w - sum_i (SQ)_i / ((SQ)_i z) 
        ## hessian is \Lambda + sum_i ((SQ)_i / s_i)((SQ)_i / s_i)' / ((SQ)_i z / s_i)^2 = \Lambda + SQ diag(1/SQz^2) Q'S'
        ## hess_inv computed with Sherman-Morrison-Woodbury
        ## H_inv = \Lambda^{-1} - \Lambda^{-1} SQ[diag(1/SQz^2) + Q'S' \Lambda^{-1} SQ]^{-1} Q'S' \Lambda^{-1}
        ##       = \Lambda^{-1} - \Lambda^{-1} SQ[diag(1/SQz^2) + Q' \Lambda^{-1} Q]^{-1} Q'S' \Lambda^{-1}

        ## update is z_new = z_prev - H_inv(z_prev) grad(z_old)
        z_opt = self._optimize(
            max_iters=max_iters,
            c=c,
            tau=tau,
            tol=tol,
        )

        return self.Q @ z_opt


class GDOptimizer(AbstractNMOptimizer):
    def _get_hessinv(self, z):
        identity = lambda x: x
        return LinearOperator(
            shape=(z.shape[0],z.shape[0]), 
            matvec=identity,
            rmatvec=identity, 
            matmat=identity,
            rmatmat=identity,
        )
    

class CDOptimizer(Optimizer):
    ## UNSURE if this will be fast. Can use as warm start for barrier problem
    ## barrier problem has better inference properties
    def optimize(
        self,
        linear_term, ## v
        quad_form, ## Q
    ):
        ## Assumes solving problem:
        ## min_x .5 x'Qx + x'v
        ## s.t. x \ge 0
        ## IDEA: solve path with ridge to exploit warm starts

        ## outer loop: iterate over lambda path
        for lmda in lmda_path:
        #### while KKT not satisfied: add offending coords to active set, iterate active set only til convergence
        
        #### in this case update is
        #### x_i^* = max{ 0, (v_i - Q_{i,-i}x_{-i}) / Q_{i,i} }
            while active_set_changing():
                viol_idx = kkt_check(x, quad_form, linear_term)
                active_set = np.union1d(active_set, viol_idx)
                idx = 0
                x_old
                while True:
                    if idx == 0 and converged(x, x_old):
                        break
                    else:
                        x_old = x
                    i = active_set[idx]

                    xi_new = (linear_term[i] - quad_form[i,:].dot(x) + quad_form[i,i] * x[i]) / (quad_form[i,i] + lmda)
                    x[i] = max(0, xi_new)

                    idx += 1
                    if idx == len(active_set):
                        idx = 0
                        ## update vector x_old (use to test convergence)

