import adelie as ad
import choosi as cs
import numpy as np
import pytest


@pytest.mark.parametrize("n, p, E", [
    [100, 10, 5],
    [100, 100, 10],
])
def test_hessian(n, p, E, n_threads=2):
    data = ad.data.snp_unphased(n, p)
    X = data["X"]
    filename = "/tmp/test_hessian_snp_unphased.snpdat"
    handler = ad.io.snp_unphased(filename)
    handler.write(X)

    handler.read()
    subset = np.random.choice(p, size=E, replace=False).astype(int)
    weights = np.random.uniform(0, 1, n)
    out = np.empty((E, E))
    cs.choosi_core.matrix.hessian(handler, subset, weights, out, n_threads)

    X = X.astype(float)
    X = np.where(X == -9, handler.impute[None], X)
    assert np.allclose(X[:,subset].T @ (weights[:,None] * X[:,subset]), out)
