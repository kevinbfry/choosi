import os
import pytest
import numpy as np
import pandas as pd
from selectinf.algorithms.barrier_affine import solve_barrier_affine_py
from selectinf.randomized.snp_lasso import split_lasso_solved, selected_targets
from selectinf.randomized.lasso import split_lasso
import adelie as ad
import pgenlib as pg
from choosi.lasso import SplitLasso, SplitLassoSNPUnphased


def gen_X(n, p, X_type, seed=0, n_threads=1):
    if X_type == "normal":
        np.random.seed(seed)
        X = np.asfortranarray(np.random.randn(n,p))
        return X

    elif X_type == "snp":
        X_full_fnames = [
            f"/tmp/X_full_su_chr_{i}.snpdat"
            for i in range(len(p))
        ]
        X_tr_fnames = [
            f"/tmp/X_tr_su_chr_{i}.snpdat"
            for i in range(len(p))
        ]
        X_val_fnames = [
            f"/tmp/X_val_su_chr_{i}.snpdat"
            for i in range(len(p))
        ]

        for i, s in enumerate(p):
            data = ad.data.snp_unphased(n, s, seed=i + seed, missing_ratio=0.)
            handler = ad.io.snp_unphased(X_full_fnames[i])
            handler.write(data["X"], n_threads=n_threads)
            handler = ad.io.snp_unphased(X_tr_fnames[i])
            handler.write(np.asfortranarray(data["X"][:n//3,:]), n_threads=n_threads)
            handler = ad.io.snp_unphased(X_val_fnames[i])
            handler.write(np.asfortranarray(data["X"][n//3:2*n//3,:]), n_threads=n_threads)

        covs = np.random.randn(n,p[0])

        X_full = ad.matrix.concatenate(
            [ad.matrix.dense(covs)] +
            [
                ad.matrix.snp_unphased(
                    ad.io.snp_unphased(filename),
                    n_threads=n_threads,
                )
                for filename in X_full_fnames
            ],
            axis=1,
            n_threads=n_threads,
        )

        return X_full, covs, X_full_fnames, X_tr_fnames, X_val_fnames
    else:
        raise ValueError("X_type must be in ['normal', 'snp']")
    

def gen_beta(X, s, snr, dispersion, beta_gen, fit_intercept):
    p = X.shape[1]
    beta = np.zeros(p + fit_intercept)
    if beta_gen == "norm":
        beta = np.random.randn(p+fit_intercept)
    elif beta_gen == "sparse":
        nz_idx = np.concatenate(([0],np.random.choice(p, size=s, replace=False) + fit_intercept))
        beta[nz_idx] = np.random.choice([-1,1],size=s+fit_intercept,replace=True) * np.random.uniform(0,1,size=s+fit_intercept)

    Xb = X @ beta[1:] + beta[0] if fit_intercept else X @ beta
    ratio = np.sqrt(snr * dispersion / np.var(Xb))

    Xb *= ratio
    beta *= ratio

    return beta, Xb


def gen_response(Xb, dispersion):
    n = Xb.shape[0]

    y = Xb + np.random.randn(n) * np.sqrt(dispersion)
    y = np.asfortranarray(y)

    return y


def run_lasso_snp_pipeline(
    X, 
    covs,
    X_full_fnames, 
    X_tr_fnames, 
    X_val_fnames, 
    y, 
    beta, 
    dispersion, 
    fit_intercept, 
    method, 
    level=.9,
):
    n, p = X.shape
    sl = SplitLassoSNPUnphased(
        X_fnames=X_full_fnames,
        X_tr_fnames=X_tr_fnames,
        X_val_fnames=X_val_fnames,
        covs_full=covs,
        covs_tr=covs[:n//3],
        covs_val=covs[n//3:2*n//3],
        y=y,
        tr_idx=np.arange(n//3),
        val_idx=np.arange(n//3,2*n//3),
        fit_intercept=fit_intercept,
        lmda_choice=None,
        penalty=np.concatenate((np.zeros(5),np.ones(p-5))),
        # penalty=np.ones(p),
        family="gaussian",
    )

    sl.fit()
    sl.extract_event()
    L, U = sl.infer(
        dispersion=dispersion, 
        method=method,
        level=level,
    )

    X_np = X @ np.eye(p)
    X_np = np.hstack((np.ones((n,1)), X_np))
    targets = np.linalg.pinv(X_np[:, sl.overall]) @ X_np @ beta

    return np.sum((L <= targets) & (targets <= U)), sl.overall.sum()


def run_lasso_arr_pipeline(X, y, beta, dispersion, fit_intercept, method, level=.9,):
    n, p = X.shape
    sl = SplitLasso(
        X=X,
        y=y,
        tr_idx=np.arange(n//3),
        val_idx=np.arange(n//3,2*n//3),
        fit_intercept=fit_intercept,
        lmda_choice=None,
        penalty=np.ones(p),
        family="gaussian",
    )

    sl.fit()
    sl.extract_event()
    L, U = sl.infer(
        dispersion=dispersion, 
        method=method,
        level=level,
    )

    X_int = np.hstack((np.ones((X.shape[0],1)), X)) if fit_intercept else X
    targets = np.linalg.pinv(X_int[:, sl.overall]) @ X_int @ beta

    return np.sum((L <= targets) & (targets <= U)), sl.overall.sum()



@pytest.mark.parametrize("n, p, s, dispersion, est_dispersion, beta_gen, method, seed", [
    [1000, 100, 30, 1, False, "sparse", "exact", 314],#14],
    [1000, 100, 30, 1, False, "sparse", "mle", 123],#32],
    # [300, 10, None, 1, False, "norm", 11],
    # [600, 10, 3, 1, True, "sparse", 317],
    # [600, 10, None, 1, True, "norm", 314],
    # [300, 10, 3, 3, False, "sparse", 64],
    # [300, 10, None, 3, False, "norm", 172],
    # [600, 10, 3, 3, True, "sparse", 316],
    # [600, 10, None, 3, True, "norm", 420],
])
def test_lasso_arr_coverage(
    n, 
    p, 
    s, 
    dispersion, 
    est_dispersion, 
    beta_gen, 
    method,
    seed, 
    snr=.4,
    level=.9,
    fit_intercept=True,
    niter=100
):
    n_sel = np.zeros(niter)
    n_cov = np.zeros(niter)
    
    for i in range(niter):
        X = gen_X(n, p, "normal", seed + i)
        beta, Xb = gen_beta(X, s, snr, dispersion, beta_gen, fit_intercept)
        y = gen_response(Xb, dispersion)

        n_cov[i], n_sel[i] = run_lasso_arr_pipeline(
            X, 
            y, 
            beta, 
            dispersion=None if est_dispersion else dispersion, 
            method=method,
            fit_intercept=fit_intercept, 
            level=level,
        )

    assert np.allclose(n_cov.sum() / n_sel.sum(), .9, atol=.02)



@pytest.mark.parametrize("n, p, s, dispersion, est_dispersion, beta_gen, method, seed", [
    # [1_000, [5_00,3_00,2_00], 1_00, 1, False, "sparse", "mle", 32],
    # [1_000, [5_00,3_00,2_00], 1_00, 1, False, "sparse", "exact", 32],
    [300, [10,10,10], 10, 1, False, "sparse", "mle", 32],
    [300, [10,10,10], 10, 1, False, "sparse", "exact", 14],
    # [300, 10, None, 1, False, "norm", 11],
    # [600, 10, 3, 1, True, "sparse", 317],
    # [600, 10, None, 1, True, "norm", 314],
    # [300, 10, 3, 3, False, "sparse", 64],
    # [300, 10, None, 3, False, "norm", 172],
    # [600, 10, 3, 3, True, "sparse", 316],
    # [600, 10, None, 3, True, "norm", 420],
])
def test_lasso_snp_coverage(
    n, 
    p, 
    s, 
    dispersion, 
    est_dispersion, 
    beta_gen, 
    method,
    seed, 
    snr=.4,
    level=.9,
    fit_intercept=True,
    niter=100
):
    n_sel = np.zeros(niter)
    n_cov = np.zeros(niter)
    
    for j in range(niter):
        (
            X_full,
            covs,
            X_full_fnames,
            X_tr_fnames,
            X_val_fnames,
        ) = gen_X(n, p, "snp", seed + j)
        beta, Xb = gen_beta(X_full, s, snr, dispersion, beta_gen, fit_intercept)
        y_full = gen_response(Xb, dispersion)

        n_cov[j], n_sel[j] = run_lasso_snp_pipeline(
            X_full, 
            covs,
            X_full_fnames,
            X_tr_fnames,
            X_val_fnames,
            y_full, 
            beta, 
            dispersion=None if est_dispersion else dispersion, 
            method=method,
            fit_intercept=fit_intercept, 
            level=level,
        )

    assert np.allclose(n_cov.sum() / n_sel.sum(), .9, atol=.02)

    for fname in X_full_fnames:
        os.remove(fname)
    for fname in X_tr_fnames:
        os.remove(fname)
    for fname in X_val_fnames:
        os.remove(fname)




