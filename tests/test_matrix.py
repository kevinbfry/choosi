import numpy as np
import adelie as ad
import choosi as cs

def test_matrix_hessian():
    X = np.random.randn(100,10)
    X = np.asfortranarray(X)
    aX = ad.matrix.dense(X)

    subset = np.random.choice(10,size=5,replace=False).astype(int)
    weights = np.random.randn(100)
    out = np.empty((5,5), order='F')

    cs.choosi_core.matrix.hessian(aX, subset, weights, out)

    assert np.allclose(X[:,subset].T @ (weights[:,None] * X[:,subset]), out)