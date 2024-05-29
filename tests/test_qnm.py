import adelie as ad
import choosi as cs
from scipy.sparse.linalg import eigsh
import numpy as np
import pytest


@pytest.mark.parametrize("n, p", [
    [100, 10],
    [200, 100],
])
def test_hessinv(n, p):
    X = np.asfortranarray(np.random.randn(n,p))
    H = X.T @ X

    scaling = np.ones(p)
    signs = np.random.choice([-1,1], size=p, replace=True)

    eigvals, eigvecs = eigsh(H, k=p)

    Lambda = eigvals
    Q = eigvecs
    SQ = signs[:,None] * Q
    # print(Q[:,:5] @ Q[:,:5].T, Q[:,:5].T @ Q[:,:5])
    # assert(0==1)

    z = np.random.randn(p)
    inv = np.linalg.inv(np.diag(Lambda) + SQ @ np.diag(1/(SQ @ z)**2) @ SQ.T)
    qnm = cs.optimizer.QNM(np.zeros(p), H, (eigvecs, eigvals), signs, scaling)
    assert np.allclose(qnm._get_hessinv(z), inv)


@pytest.mark.parametrize("n, p", [
    [100, 10],
    [200, 100],
])
def test_qnm(n, p):
    X = np.asfortranarray(np.random.randn(n,p))
    H = X.T @ X

    # beta = np.random.choice([3,4,5], size=p, replace=True)
    beta = np.random.choice([-1,1], size=p, replace=True) * np.random.choice([3,4,5], size=p, replace=True)
    y = X @ beta + np.random.randn(n) / 10

    v = -X.T @ y
    scaling = np.ones(p)
    signs = np.sign(beta) 

    x_soln = np.linalg.pinv(X) @ y

    eigvals, eigvecs = eigsh(H, k=p)

    qnm = cs.optimizer.QNM(100*v, 100*H, (eigvecs, eigvals), signs, scaling)
    x_opt = qnm.optimize()
    assert np.allclose(x_opt / 100, x_soln)