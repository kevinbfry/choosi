import numpy as np
from abc import ABC, abstractmethod


class Optimizer(object):
    @abstractmethod
    def optimize(
        self,
    ):
        pass

class QNM(Optimizer):
    def __init__(
        self,
        linear_term, # v
        quad_form, # H
        quad_form_approx, # (Q,\Lambda) where H \approx Q\LambdaQ'
        signs, # S
        scaling, # s
        max_iters=100,
        c=.5,
        tau=.5,
        tol=1e-4,
    ):
        self.v = linear_term
        self.H = quad_form
        self.Q, self.Lambda = quad_form_approx
        self.w = self.Q.T @ self.v
        self.signs = signs
        self.SQ = signs[:,None] * self.Q
        self.scaling = scaling
    

    ## invert using woodbury matrix identity
    def _get_hessinv(self, z):
        Lambda_inv = 1/self.Lambda
        SQz = self.SQ @ z
        Lambda_inv_SQ = Lambda_inv[:,None] * self.SQ
        M = self.Q.T @ (Lambda_inv[:,None] * self.Q)
        for i in range(SQz.shape[0]):
            M[i,i] += SQz[i]**2 
        soln = np.linalg.solve(M, Lambda_inv_SQ.T)

        woodbury_inv = np.diag(Lambda_inv) - Lambda_inv_SQ @ soln

        return woodbury_inv


    def _get_obj(self, z):
        return .5*(z**2 * self.Lambda).sum() + self.w.dot(z) - np.log((self.SQ @ z) / self.scaling).sum()


    def _get_grad(self, z):
        return z * self.Lambda + self.w - (self.SQ / (self.SQ @ z)).sum(0)

        
    def optimize(
        self,
        max_iters=100,
        c=.5,
        tau=.5,
        tol=1e-4,
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
        Q, Lambda = self.Q, self.Lambda
        SQ = self.SQ
        w = self.w

        ## update is z_new = z_prev - H_inv(z_prev) grad(z_old)
        grad_new = np.zeros(Q.shape[1])
        z_new = 1. / self.scaling # np.zeros(Q.shape[1])
        for i in range(max_iters):
            step_size = 1.

            z_prev = z_new
            H_inv = self._get_hessinv(z_prev)
            grad_prev = grad_new
            grad_new = self._get_grad(z_prev)
            p = -H_inv @ grad_new
            z_new = z_prev + step_size * p
            if i > 0 and (grad_new-grad_prev).dot(z_new - z_prev) < tol:
                break
            t = -c*grad_new.dot(p / (p**2).sum())
            while self._get_obj(z_prev) - self._get_obj(z_new) > step_size * t:
                step_size = tau * step_size
                z_new = z_prev + step_size * p


        return Q @ z_new


class GD(Optimizer):
    def optimize(
        self,
        linear_term,
        quad_form,
    ):
        pass
        
class CD(Optimizer):
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

