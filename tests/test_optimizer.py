import adelie as ad
import choosi as cs
from scipy.sparse.linalg import eigsh
# TODO: selectinf cannot be installed properly through both pip and poetry.
# Until it can be integrated properly as a dependency, remove all usage of it.
#from selectinf.algorithms.barrier_affine import solve_barrier_affine_py
import numpy as np
import pytest


@pytest.mark.parametrize("n, p", [
    [100, 10],
    [200, 100],
])
def test_nm(n, p):
    X = np.asfortranarray(np.random.randn(n,p))
    H = X.T @ X / n

    beta = np.random.choice([-1,1], size=p, replace=True) * np.random.choice([3,4,5], size=p, replace=True)
    y = X @ beta + np.random.randn(n)

    v = -X.T @ y / n
    scaling = np.ones(p)
    x_soln = np.linalg.pinv(X) @ y

    signs = np.sign(x_soln) 

    qnm = cs.optimizer.NMOptimizer(v, H, signs, lmda=1e-10)
    x_opt = qnm.optimize(max_steps=1000000, tau=.5, c=.5, tol=1e-20)

    assert np.allclose(x_opt, x_soln)


@pytest.mark.parametrize("n, p", [
    [100, 10],
    [200, 100],
])
def test_dqnm(n, p):
    X = np.asfortranarray(np.random.randn(n,p))
    H = X.T @ X / n

    beta = np.random.choice([-1,1], size=p, replace=True) * np.random.choice([3,4,5], size=p, replace=True)
    y = X @ beta + np.random.randn(n)

    v = -X.T @ y / n
    scaling = np.ones(p)
    x_soln = np.linalg.pinv(X) @ y

    signs = np.sign(x_soln) 
    
    qnm = cs.optimizer.DQNMOptimizer(v, H, signs, lmda=1e-10)
    x_opt = qnm.optimize(max_steps=1000000, tau=.5, c=.5, tol=1e-20)

    assert np.allclose(x_opt, x_soln)


@pytest.mark.parametrize("n, p", [
    [100, 10],
    [200, 100],
])
def test_cqnm(n, p):
    X = np.asfortranarray(np.random.randn(n,p))
    H = X.T @ X / n

    beta = np.random.choice([-1,1], size=p, replace=True) * np.random.choice([3,4,5], size=p, replace=True)
    y = X @ beta + np.random.randn(n)

    v = -X.T @ y / n
    scaling = np.ones(p)
    x_soln = np.linalg.pinv(X) @ y

    signs = np.sign(x_soln) 
    
    qnm = cs.optimizer.CQNMOptimizer(v, H, signs, lmda=1e-6)
    x_opt = qnm.optimize(max_steps=100, tau=.5, c=.5, tol=1e-10)

    assert np.allclose(x_opt, x_soln)


# @pytest.mark.parametrize("n, p", [
#     [100, 10],
#     [200, 100],
# ])
# def test_eqnm(n, p):
#     X = np.asfortranarray(np.random.randn(n,p))
#     H = X.T @ X / n

#     beta = np.random.choice([-1,1], size=p, replace=True) * np.random.choice([3,4,5], size=p, replace=True)
#     y = X @ beta + np.random.randn(n)

#     v = -X.T @ y / n
#     scaling = np.ones(p)
#     x_soln = np.linalg.pinv(X) @ y

#     signs = np.sign(x_soln) 
    
#     eigvals, eigvecs = eigsh(H, k=p)

#     qnm = cs.optimizer.EQNMOptimizer(v, H, (eigvecs, eigvals), signs, lmda=1e-6)
#     x_opt = qnm.optimize(max_steps=100, tau=.5, c=.5, tol=1e-10)

#     assert np.allclose(x_opt, x_soln)


@pytest.mark.parametrize("n, p", [
    [100, 10],
    [200, 100],
])
def test_gd(n, p):
    X = np.asfortranarray(np.random.randn(n,p))
    H = X.T @ X / n

    # beta = np.random.choice([3,4,5], size=p, replace=True)
    beta = np.random.choice([-1,1], size=p, replace=True) * np.random.choice([3,4,5], size=p, replace=True)
    y = X @ beta + np.random.randn(n)

    v = -X.T @ y / n
    scaling = np.ones(p)
    x_soln = np.linalg.pinv(X) @ y

    signs = np.sign(x_soln) 

    gd = cs.optimizer.GDOptimizer(v, H, signs, lmda=1e-10)
    x_opt = gd.optimize(max_steps=1000000, tau=.5, tol=1e-20)

    assert np.allclose(x_opt, x_soln)


#@pytest.mark.parametrize("n, p", [
#    [100, 10],
#    # [200, 100],
#])
#def test_match_selinf(n, p):
#    X = np.asfortranarray(np.random.randn(n,p))
#    H = X.T @ X / n
#
#    beta = np.random.choice([-1,1], size=p, replace=True) * np.random.choice([3,4,5], size=p, replace=True)
#    y = X @ beta + np.random.randn(n)
#
#    v = -X.T @ y / n
#    scaling = np.ones(p)
#    x_soln = np.linalg.pinv(X) @ y
#
#    signs = np.sign(x_soln) 
#    
#    qnm = cs.optimizer.CQNMOptimizer(v, H, signs, lmda=1)
#    x_opt = qnm.optimize(max_steps=100, tau=.5, c=.5, tol=1e-10)
#
#    sl_val, x_sl, _ = solve_barrier_affine_py(-v, H, signs, -np.diag(signs), np.zeros_like(signs))
#
#    assert np.allclose(qnm._get_obj(x_opt), sl_val)
#    assert np.allclose(x_opt, x_sl)
