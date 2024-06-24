import numpy as np
from abc import ABC, abstractmethod


class Barrier(object):
    def __init__(
        self,
        con_linear,
        pen_idxs,
        lmda=1.,
    ):
        self.D = con_linear[pen_idxs] ## vector of diagonal of constraint matrix -A (Ax <= b)
        self.pen_idxs = pen_idxs
        self.lmda = lmda
        

    @abstractmethod
    def get_value(
        self,
        z,
    ):
        pass


    @abstractmethod
    def get_grad(
        self,
        z,
    ):
        pass

    @abstractmethod
    def get_hess(
        self,
        z,
    ):
        pass


## -log(b - Ax)
class LogBarrier(Barrier):
    def get_value(self, z):
        return - self.lmda * np.log(self.D * z[self.pen_idxs]).sum()


    def get_grad(self, z):
        grad = np.zeros_like(z)
        grad[self.pen_idxs] = - self.lmda / z[self.pen_idxs]
        return grad
    

    def get_hess_diag(self, z):
        hess_diag = np.zeros_like(z)
        hess_diag[self.pen_idxs] = self.lmda / z[self.pen_idxs]**2
        return hess_diag


## log(1 + 1/(b - Ax))
class InvLogBarrier(Barrier):
    def get_value(self, z):
        return self.lmda * np.log(1. + 1. / (self.D * z[self.pen_idxs])).sum()


    def get_grad(self, z):
        # return self.lmda / (1. + 1. / (self.D * z)) * (-1. / (self.D * z)**2) * (self.D)
        grad = np.zeros_like(z)
        grad[self.pen_idxs] = - self.lmda * self.D / ((self.D * z[self.pen_idxs]) * (self.D*z[self.pen_idxs] + 1.)) 
        return grad
    

    def get_hess_diag(self, z):
        ## -lmda * D * { -D/(D*z)^{2} + -D/(D*z + 1)^2 } = lmda * D**2 * ( 1/(D*z)^2 + 1/(D*z + 1)^2 )
        # return self.lmda * self.D *(- (self.D * (self.D * z + 1) + self.D**2 * z)) / ((self.D * z) * (self.D*z + 1.))**2  
        hess_diag = np.zeros_like(z)
        hess_diag[self.pen_idxs] = self.lmda * self.D**2 *(2 * self.D * z[self.pen_idxs] + 1) / ((self.D * z[self.pen_idxs]) * (self.D*z[self.pen_idxs] + 1.))**2  
        return hess_diag


