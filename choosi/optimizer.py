import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import solve_triangular
from abc import ABC, abstractmethod


class Optimizer(object):
    @abstractmethod
    def optimize(
        self,
    ):
        pass

class NMOptimizer(Optimizer):
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
        self.hess = self.H
        self.hess[np.diag_indices_from(self.hess)] += self.lmda / z**2
        # self.H_inv = np.linalg.inv(self.hess)
        def solve_matvec(x):
            return np.linalg.solve(self.hess, x)

        self.H_inv = LinearOperator(
            shape=self.hess.shape,
            matvec=solve_matvec,
        )
        return self.H_inv


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
        maxiters=100,
    ):
        z_new = self.signs / self.scaling 
        grad_new = self._get_grad(z_new)
        for i in range(max_iters):
            # print(f"iter {i}")
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
            # assert(0==1)
            while np.isnan(obj_new) or obj_prev - obj_new < step_size * t:
                if ct == maxiters: break
                ct += 1
                step_size = tau * step_size
                z_new = z_prev + step_size * p
                obj_new = self._get_obj(z_new)
                # print(z_prev, z_new, obj_prev, obj_new)

            if ct == maxiters:
                print(f"hit {maxiters}, giving up")
                break

            grad_new = self._get_grad(z_new)
            if i > 0 and (grad_new - grad_prev).dot(z_new - z_prev) < tol:
                # print("close enough, returning")
                break

        return z_new

class EQNMOptimizer(NMOptimizer):
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
        M[np.diag_indices_from(M)] += SQz**2 / self.lmda
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


class CQNMOptimizer(NMOptimizer):
    def _get_hessinv(self, z):
        if not hasattr(self, "H_chol"):
            self.H_chol = np.linalg.cholesky(self.H)
        self.H_barrier_diag = np.sqrt(self.lmda) / np.fabs(z)
        def trisolve_matvec(b):
            self.H_chol[np.diag_indices_from(self.H_chol)] += self.H_barrier_diag

            y = solve_triangular(self.H_chol, b, lower=True, check_finite=False)
            x = solve_triangular(self.H_chol.T, y, lower=False, check_finite=False)

            self.H_chol[np.diag_indices_from(self.H_chol)] -= self.H_barrier_diag

            return x

        self.H_inv = LinearOperator(
            shape=self.H_chol.shape,
            matvec=trisolve_matvec,
        )

        return self.H_inv
    
    
class DQNMOptimizer(NMOptimizer):
    def _get_hessinv(self, z):
        if not hasattr(self, "H_inv"):
            # self.H_inv = np.diag(1. / np.diag(self.H))
            self.H_diag = np.diag(self.H)
            h_inv_matvec = lambda x: x / self.H_diag
            h_inv_matmat = lambda x: x / self.H_diag[:,None]
            self.H_inv = LinearOperator(
                shape=self.H.shape,
                matvec=h_inv_matvec,
                rmatvec=h_inv_matvec,
                matmat=h_inv_matmat,
                rmatmat=h_inv_matmat,
            )
            
        return self.H_inv
    

# class TBDQNMOptimizer(NMOptimizer):
#     def _get_hessinv(self, z):
#         if not hasattr(self, "H_inv"):
#             self.blocks = self._findblocks()

#         return self.H_inv
    

class GDOptimizer(NMOptimizer):
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

