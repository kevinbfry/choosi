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




@pytest.mark.parametrize("n, p, s, dispersion, beta_gen", [
    [100, 30, 10, 1, "norm"],
    [100, 30, 10, 1, "sparse"],
    [100, 30, 10, 2, "norm"],
    [100, 30, 10, 2, "sparse"],
])
def test_lasso_match_si(n, p, s, dispersion, beta_gen):
    X_tr = np.asfortranarray(np.random.randn(n,p))
    X_val = np.asfortranarray(np.random.randn(n,p))
    X_ts = np.asfortranarray(np.random.randn(n,p))
    X_full = np.asfortranarray(np.vstack((X_tr, X_val, X_ts)))
    beta = np.zeros(p+1)
    if beta_gen == "sparse":
        nz_idx = np.concatenate(([0],np.random.choice(p, size=s, replace=False) + 1))
        beta[nz_idx] = np.random.choice([-1,1],size=s+1,replace=True) * np.random.uniform(3,5,size=s+1)
    elif beta_gen == "norm":
        beta = np.random.randn(p+1)
    else:
        raise ValueError("'beta_gen' must be in ['sparse', 'norm']")
    y_tr = X_tr @ beta[1:] + beta[0] + np.random.randn(n) * np.sqrt(dispersion)
    y_val = X_val @ beta[1:] + beta[0] + np.random.randn(n) * np.sqrt(dispersion)
    y_ts = X_ts @ beta[1:] + beta[0] + np.random.randn(n) * np.sqrt(dispersion)
    y_full = np.asfortranarray(np.concatenate((y_tr, y_val, y_ts)))

    sl = SplitLasso(
        X=X_full,
        y=y_full,
        tr_idx=np.arange(n),
        val_idx=np.arange(n,2*n),
        fit_intercept=True,
        lmda_choice=1.,
        penalty=np.ones(p),
        family="gaussian",
    )

    sl.fit()
    sl.extract_event()
    sel_MLE, sel_MLE_std = sl.infer(dispersion=dispersion)
    L = sel_MLE - 1.645 * sel_MLE_std
    U = sel_MLE + 1.645 * sel_MLE_std
    
    si_sl = split_lasso_solved.gaussian(
        np.hstack((np.ones((3*n,1)),X_full)), 
        y_full, 
        feature_weights=np.concatenate((np.zeros(1), np.ones(p) * sl.lmda*3*n)), 
        proportion=1./3,
    )
    si_sl.fit(
        observed_soln=sl.observed_soln, 
        observed_subgrad=sl.observed_subgrad * 3 * n,
        perturb=np.concatenate((np.ones(n), np.zeros(2*n))).astype(bool),
    )
    si_sl.setup_inference(dispersion=dispersion)
    TS = selected_targets(si_sl.loglike, si_sl.observed_soln, dispersion=1)
    out = si_sl.inference(target_spec=TS, method='selective_MLE')
    
    si_sl_selfsolve = split_lasso.gaussian(
        np.hstack((np.ones((3*n,1)),X_full)), 
        y_full, 
        feature_weights=np.concatenate((np.zeros(1), np.ones(p) * sl.lmda*3*n)), 
        proportion=1./3,
    )
    si_sl_selfsolve.fit(
        perturb=np.concatenate((np.ones(n), np.zeros(2*n))).astype(bool),
    )
    si_sl_selfsolve.setup_inference(dispersion=dispersion)
    TS_selfsolve = selected_targets(si_sl_selfsolve.loglike, si_sl_selfsolve.observed_soln, dispersion=1)
    out_selfsolve = si_sl_selfsolve.inference(target_spec=TS_selfsolve, method='selective_MLE')

    assert np.allclose(np.sort(np.fabs(sl.cond_mean)), np.sort(np.fabs(si_sl_selfsolve.cond_mean)))
    assert np.allclose(np.sort(np.fabs(np.diag(sl.cond_cov))), np.sort(np.fabs(np.diag(si_sl_selfsolve.cond_cov)/2)))
    assert np.allclose(sel_MLE_std, out_selfsolve['SE'])
    assert np.allclose(sel_MLE, out_selfsolve['MLE'])

    assert np.allclose(
        np.mean((L <= beta[sl.overall]) & (beta[sl.overall] <= U)), 
        np.mean(
            (out_selfsolve['MLE'] - 1.645* out_selfsolve['SE'] <= beta[sl.overall]) & 
            (beta[sl.overall] <= out_selfsolve['MLE'] + 1.645*out_selfsolve['SE'])
        )
    )


@pytest.mark.parametrize("n, p, s, dispersion, est_dispersion, beta_gen, seed", [
    [100, 10, 3, 1, False, "sparse", 15],
    [100, 10, None, 1, False, "norm", 11],
    [200, 10, 3, 1, True, "sparse", 317],
    [200, 10, None, 1, True, "norm", 314],
    [100, 10, 3, 3, False, "sparse", 64],
    [100, 10, None, 3, False, "norm", 172],
    [200, 10, 3, 3, True, "sparse", 316],
    [200, 10, None, 3, True, "norm", 420],
])
def test_lasso_exact_coverage(n, p, s, dispersion, est_dispersion, beta_gen, seed, niter=100):
    if seed is not None:
        np.random.seed(seed)
    n_sel = np.zeros(niter)
    n_cov = np.zeros(niter)
    # n_cov_MLE = np.zeros(niter)

    beta = np.zeros(p+1)
    if beta_gen == "sparse":
        nz_idx = np.concatenate(([0],np.random.choice(p, size=s, replace=False) + 1))
        beta[nz_idx] = np.random.choice([-1,1],size=s+1,replace=True) * np.random.uniform(3,5,size=s+1)
    elif beta_gen == "norm":
        beta = np.random.randn(p+1)
    else:
        raise ValueError("'beta_gen' must be in ['sparse', 'norm']")
    
    for i in range(niter):
        X_tr = np.asfortranarray(np.random.randn(n,p))
        X_val = np.asfortranarray(np.random.randn(n,p))
        X_ts = np.asfortranarray(np.random.randn(n,p))
        X_full = np.asfortranarray(np.vstack((X_tr, X_val, X_ts)))
        
        # beta[0] = np.fabs(beta[0])
        y_tr = X_tr @ beta[1:] + beta[0] + np.random.randn(n) * np.sqrt(dispersion)
        y_val = X_val @ beta[1:] + beta[0] + np.random.randn(n) * np.sqrt(dispersion)
        y_ts = X_ts @ beta[1:] + beta[0] + np.random.randn(n) * np.sqrt(dispersion)
        y_full = np.asfortranarray(np.concatenate((y_tr, y_val, y_ts)))

        sl = SplitLasso(
            X=X_full,
            y=y_full,
            tr_idx=np.arange(n),
            val_idx=np.arange(n,2*n),
            fit_intercept=True,
            lmda_choice="mid",
            penalty=np.ones(p),
            family="gaussian",
        )

        sl.fit()
        sl.extract_event()
        # sel_MLE, sel_MLE_std = sl.infer(
        #     dispersion=None if est_dispersion else dispersion, 
        #     method="mle",
        # )
        # L_MLE = sel_MLE - 1.645 * sel_MLE_std
        # U_MLE = sel_MLE + 1.645 * sel_MLE_std
        L, U = sl.infer(
            dispersion=None if est_dispersion else dispersion, 
            method="exact",
            level=.9,
        )
        # L = sel_MLE - 1.645 * sel_MLE_std
        # U = sel_MLE + 1.645 * sel_MLE_std

        X_full_int = np.hstack((np.ones((X_full.shape[0],1)), X_full))
        targets = np.linalg.pinv(X_full_int[:, sl.overall]) @ X_full_int @ beta

        n_sel[i] = sl.overall.sum()
        # n_cov[i] = np.sum((L <= beta[sl.overall]) & (beta[sl.overall] <= U))
        # n_cov_MLE[i] = np.sum((L_MLE <= beta[sl.overall]) & (beta[sl.overall] <= U_MLE))
        n_cov[i] = np.sum((L <= targets) & (targets <= U))
        # n_cov_MLE[i] = np.sum((L_MLE <= targets) & (targets <= U_MLE))

    # assert np.allclose(n_cov_MLE.sum() / n_sel.sum(), .9, atol=.02)
    assert np.allclose(n_cov.sum() / n_sel.sum(), .9, atol=.02)


@pytest.mark.parametrize("n, p, s, dispersion, est_dispersion, beta_gen", [
    [100, 30, 10, 1, True, "sparse"],
    [100, 30, 10, 1, False, "norm"],
    [100, 30, 10, 1, False, "sparse"],
    [100, 30, 10, 2, False, "norm"],
    [100, 30, 10, 2, False, "sparse"],
    [100, 30, 10, 2, True, "sparse"],
    [100, 30, 10, 5, False, "sparse"],
    # [100, 30, 10, 5, False, "norm"],
    [100, 30, 10, 5, True, "sparse"],
    [100, 30, 10, 1, True, "norm"],
    [100, 30, 10, 2, True, "norm"],
    # [100, 30, 10, 5, True, "norm"],
    # [33, 30, 10, 1, False, "sparse"],
])
def test_lasso_coverage(n, p, s, dispersion, est_dispersion, beta_gen):
    n_sel = np.zeros(100)
    n_cov = np.zeros(100)

    beta = np.zeros(p+1)
    if beta_gen == "sparse":
        nz_idx = np.concatenate(([0],np.random.choice(p, size=s, replace=False) + 1))
        beta[nz_idx] = np.random.choice([-1,1],size=s+1,replace=True) * np.random.uniform(3,5,size=s+1)
    elif beta_gen == "norm":
        beta = np.random.randn(p+1)
    else:
        raise ValueError("'beta_gen' must be in ['sparse', 'norm']")
    
    for i in range(100):
        X_tr = np.asfortranarray(np.random.randn(n,p))
        X_val = np.asfortranarray(np.random.randn(n,p))
        X_ts = np.asfortranarray(np.random.randn(n,p))
        X_full = np.asfortranarray(np.vstack((X_tr, X_val, X_ts)))
        
        # beta[0] = np.fabs(beta[0])
        y_tr = X_tr @ beta[1:] + beta[0] + np.random.randn(n) * np.sqrt(dispersion)
        y_val = X_val @ beta[1:] + beta[0] + np.random.randn(n) * np.sqrt(dispersion)
        y_ts = X_ts @ beta[1:] + beta[0] + np.random.randn(n) * np.sqrt(dispersion)
        y_full = np.asfortranarray(np.concatenate((y_tr, y_val, y_ts)))

        sl = SplitLasso(
            X=X_full,
            y=y_full,
            tr_idx=np.arange(n),
            val_idx=np.arange(n,2*n),
            fit_intercept=True,
            lmda_choice=None,
            penalty=np.ones(p),
            family="gaussian",
        )

        sl.fit()
        sl.extract_event()
        sel_MLE, sel_MLE_std = sl.infer(dispersion=None if est_dispersion else dispersion)
        L = sel_MLE - 1.645 * sel_MLE_std
        U = sel_MLE + 1.645 * sel_MLE_std

        X_full_int = np.hstack((np.ones((X_full.shape[0],1)), X_full))
        targets = np.linalg.pinv(X_full_int[:, sl.overall]) @ X_full_int @ beta

        n_sel[i] = sl.overall.sum()
        # n_cov[i] = np.sum((L <= beta[sl.overall]) & (beta[sl.overall] <= U))
        n_cov[i] = np.sum((L <= targets) & (targets <= U))

    assert np.allclose(n_cov.sum() / n_sel.sum(), .9, atol=.0875)
    assert np.allclose(n_cov.sum() / n_sel.sum(), .9, atol=.075)
    assert np.allclose(n_cov.sum() / n_sel.sum(), .9, atol=.05)
    assert np.allclose(n_cov.sum() / n_sel.sum(), .9, atol=.025)


@pytest.mark.parametrize("n, p, s, dispersion, beta_gen", [
    [100, 30, 10, 1, "sparse"],
    [100, 30, 10, 1, "norm"],
    [100, 30, 10, 2, "norm"],
    [100, 30, 10, 2, "sparse"],
])
def test_si_coverage(n, p, s, dispersion, beta_gen):
    n_sel = np.zeros(100)
    n_cov = np.zeros(100)
    for i in range(100):
        X_tr = np.asfortranarray(np.random.randn(n,p))
        X_val = np.asfortranarray(np.random.randn(n,p))
        X_ts = np.asfortranarray(np.random.randn(n,p))
        X_full = np.asfortranarray(np.vstack((X_tr, X_val, X_ts)))
        beta = np.zeros(p+1)
        if beta_gen == "sparse":
            nz_idx = np.concatenate(([0],np.random.choice(p, size=s, replace=False) + 1))
            beta[nz_idx] = np.random.choice([-1,1],size=s+1,replace=True) * np.random.uniform(3,5,size=s+1)
        elif beta_gen == "norm":
            beta = np.random.randn(p+1)
        else:
            raise ValueError("'beta_gen' must be in ['sparse', 'norm']")
        y_tr = X_tr @ beta[1:] + beta[0] + np.random.randn(n) * np.sqrt(dispersion)
        y_val = X_val @ beta[1:] + beta[0] + np.random.randn(n) * np.sqrt(dispersion)
        y_ts = X_ts @ beta[1:] + beta[0] + np.random.randn(n) * np.sqrt(dispersion)
        y_full = np.asfortranarray(np.concatenate((y_tr, y_val, y_ts)))

        si_sl_selfsolve = split_lasso.gaussian(
            np.hstack((np.ones((3*n,1)),X_full)), 
            y_full, 
            feature_weights=np.concatenate((np.zeros(1), np.ones(p) * 3 * n)), 
            proportion=1./3,
        )
        si_sl_selfsolve.fit(
            perturb=np.concatenate((np.ones(n), np.zeros(2*n))).astype(bool),
        )
        si_sl_selfsolve.setup_inference(dispersion=dispersion)
        TS_selfsolve = selected_targets(si_sl_selfsolve.loglike, si_sl_selfsolve.observed_soln, dispersion=1)
        out_selfsolve = si_sl_selfsolve.inference(target_spec=TS_selfsolve, method='selective_MLE')
        L = out_selfsolve['MLE'] - 1.645 * out_selfsolve['SE']
        U = out_selfsolve['MLE'] + 1.645 * out_selfsolve['SE']

        X_full_int = np.hstack((np.ones((X_full.shape[0],1)), X_full))
        targets = np.linalg.pinv(X_full_int[:, si_sl_selfsolve._overall]) @ X_full_int @ beta
        
        n_sel[i] = si_sl_selfsolve._overall.sum()
        # n_cov[i] = np.sum((L <= beta[si_sl_selfsolve._overall]) & (beta[si_sl_selfsolve._overall] <= U))
        n_cov[i] = np.sum((L <= targets) & (targets <= U))

    assert np.allclose(n_cov.sum() / n_sel.sum(), .9, atol=.04)


@pytest.mark.parametrize("n, ss, s, dispersion, est_dispersion, beta_gen", [
    # [300, [10,10,10], 10, 1, True, "norm"],
    # [300, [10,10,10], 10, 1, True, "sparse"],
    # [300, [10,10,10], 10, 1, False, "sparse"],
    # [300, [10,10,10], 10, 1, False, "norm"],
    [300, [10,10,10], 10, 5, False, "sparse"],
    [300, [10,10,10], 10, 5, False, "norm"],
    [300, [10,10,10], 10, 5, True, "sparse"],
    [300, [10,10,10], 10, 5, True, "norm"],
])
def test_lasso_snp_coverage(n, ss, s, dispersion, est_dispersion, beta_gen):
    n_threads = os.cpu_count() // 2 # number of threads
    
    X_full_fnames = [
        f"/tmp/X_full_su_chr_{i}.snpdat"
        for i in range(len(ss))
    ]
    X_tr_fnames = [
        f"/tmp/X_tr_su_chr_{i}.snpdat"
        for i in range(len(ss))
    ]
    X_val_fnames = [
        f"/tmp/X_val_su_chr_{i}.snpdat"
        for i in range(len(ss))
    ]

    n_sel = np.zeros(100)
    n_cov = np.zeros(100)
    # n_sel_si = np.zeros(100)
    # n_cov_si = np.zeros(100)

    p = int(np.sum(ss))

    beta = np.zeros(p+1)
    if beta_gen == "sparse":
        nz_idx = np.concatenate(([0],np.random.choice(p, size=s, replace=False) + 1))
        beta[nz_idx] = np.random.choice([-1,1],size=s+1,replace=True) * np.random.uniform(3,5,size=s+1)
    elif beta_gen == "norm":
        beta = np.random.randn(p+1)
    else:
        raise ValueError("'beta_gen' must be in ['sparse', 'norm']")

    for j in range(100): 
        for i, s in enumerate(ss):
            data = ad.data.snp_unphased(n, s, seed=j*len(ss) + i + 314, missing_ratio=0.)
            handler = ad.io.snp_unphased(X_full_fnames[i])
            handler.write(data["X"], n_threads=n_threads)
            handler = ad.io.snp_unphased(X_tr_fnames[i])
            handler.write(np.asfortranarray(data["X"][:n//3,:]), n_threads=n_threads)
            handler = ad.io.snp_unphased(X_val_fnames[i])
            handler.write(np.asfortranarray(data["X"][n//3:2*n//3,:]), n_threads=n_threads)


        X_full = ad.matrix.concatenate(
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

        Xb = np.zeros(n)
        X_full.btmul(0, p, beta[1:], Xb)

        y_full = Xb + beta[0] + np.random.normal(0, np.sqrt(dispersion), n)
        y_tr = y_full[:n//3]
        y_val = y_full[n//3:2*n//3]

        penalty = np.concatenate([
            np.ones(X_full.shape[1]),
        ])



        sl = SplitLassoSNPUnphased(
            X_fnames=X_full_fnames,
            # X_tr_fnames=X_tr_fnames,
            # X_val_fnames=X_val_fnames,
            y=y_full,
            # y_tr=y_tr,
            # y_val=y_val,
            tr_idx=np.arange(n//3),
            val_idx=np.arange(n//3,2*n//3),
            fit_intercept=True,
            lmda_choice="mid",#.15,
            penalty=penalty,
            family="gaussian",
        )

        X_full_np = X_full @ np.eye(p)
        X_full_np = np.hstack((np.ones((n,1)), X_full_np))

        sl.fit()
        sl.extract_event()
        sel_MLE, sel_MLE_std = sl.infer(dispersion=None if est_dispersion else dispersion)
        L = sel_MLE - 1.645 * sel_MLE_std
        U = sel_MLE + 1.645 * sel_MLE_std

        # si_sl_selfsolve = split_lasso.gaussian(
        #     X_full_np, 
        #     y_full, 
        #     feature_weights=np.concatenate((np.zeros(1), np.ones(p) * sl.lmda*n)), 
        #     proportion=1./3,
        # )
        # si_sl_selfsolve.fit(
        #     perturb=np.concatenate((np.ones(n//3), np.zeros(2*n//3))).astype(bool),
        # )
        # si_sl_selfsolve.setup_inference(dispersion=1)
        # TS_selfsolve = selected_targets(si_sl_selfsolve.loglike, si_sl_selfsolve.observed_soln, dispersion=1)
        # out = si_sl_selfsolve.inference(target_spec=TS_selfsolve, method='selective_MLE')

        n_sel[j] = sl.overall.sum()
        targets = beta[sl.overall]#np.linalg.pinv(X_full_np[:, sl.overall]) @ X_full_np @ beta
        n_cov[j] = np.sum((L <= targets) & (targets <= U))

        # n_sel_si[i] = si_sl_selfsolve._overall.sum()
        # targets = np.linalg.pinv(X_full_np[:, si_sl_selfsolve.overall]) @ X_full_np @ beta
        # n_cov_si[i] = np.sum((out['MLE'] - 1.645*out['SE'] <= targets) & (out['MLE'] + 1.645*out['SE'] >= targets))
 
    
    for fname in X_full_fnames:
        os.remove(fname)
    for fname in X_tr_fnames:
        os.remove(fname)
    for fname in X_val_fnames:
        os.remove(fname)


    assert np.allclose(n_cov.sum() / n_sel.sum(), .9, atol=.05)
    assert np.allclose(n_cov.sum() / n_sel.sum(), .9, atol=.02)
    # assert np.allclose(n_cov_si.sum() / n_sel_si.sum(), .9, atol=.05)



@pytest.mark.parametrize("n, ss, s, dispersion, est_dispersion, beta_gen, seed", [
    # [300, [10,10,10], 10, 1, True, "norm"],
    # [300, [10,10,10], 10, 1, True, "sparse"],
    # [300, [10,10,10], 10, 1, False, "sparse"],
    # [300, [10,10,10], 10, 1, False, "norm"],
    [300, [10,10,10], 10, 1, False, "sparse", 1214],
    [300, [10,10,10], 10, 2, False, "norm", 314], ## this fails
    [300, [10,10,10], 10, 5, False, "sparse", 123],
    # [300, [10,10,10], 10, 5, False, "norm", 123], ## this fails
    # [300, [10,10,10], 10, 5, True, "sparse", 543],
    # [300, [10,10,10], 10, 5, True, "norm", 420],
])
def test_lasso_exact_snp_coverage(n, ss, s, dispersion, est_dispersion, beta_gen, seed):
    np.random.seed(seed)
    n_threads = os.cpu_count() // 2 # number of threads
    
    X_full_fnames = [
        f"/tmp/X_full_su_chr_{i}.snpdat"
        for i in range(len(ss))
    ]
    X_tr_fnames = [
        f"/tmp/X_tr_su_chr_{i}.snpdat"
        for i in range(len(ss))
    ]
    X_val_fnames = [
        f"/tmp/X_val_su_chr_{i}.snpdat"
        for i in range(len(ss))
    ]

    n_sel = np.zeros(100)
    n_cov = np.zeros(100)

    p = int(np.sum(ss))

    beta = np.zeros(p+1)
    if beta_gen == "sparse":
        nz_idx = np.concatenate(([0],np.random.choice(p, size=s, replace=False) + 1))
        beta[nz_idx] = np.random.choice([-1,1],size=s+1,replace=True) * np.random.uniform(3,5,size=s+1)
    elif beta_gen == "norm":
        beta = np.random.randn(p+1)
    else:
        raise ValueError("'beta_gen' must be in ['sparse', 'norm']")

    for j in range(100): 
        for i, s in enumerate(ss):
            data = ad.data.snp_unphased(n, s, seed=j*len(ss) + i + seed, missing_ratio=0.)
            handler = ad.io.snp_unphased(X_full_fnames[i])
            handler.write(data["X"], n_threads=n_threads)
            handler = ad.io.snp_unphased(X_tr_fnames[i])
            handler.write(np.asfortranarray(data["X"][:n//3,:]), n_threads=n_threads)
            handler = ad.io.snp_unphased(X_val_fnames[i])
            handler.write(np.asfortranarray(data["X"][n//3:2*n//3,:]), n_threads=n_threads)


        X_full = ad.matrix.concatenate(
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

        Xb = np.zeros(n)
        X_full.btmul(0, p, beta[1:], Xb)

        y_full = Xb + beta[0] + np.random.normal(0, np.sqrt(dispersion), n)
        y_tr = y_full[:n//3]
        y_val = y_full[n//3:2*n//3]

        penalty = np.concatenate([
            np.ones(X_full.shape[1]),
        ])



        sl = SplitLassoSNPUnphased(
            X_fnames=X_full_fnames,
            X_tr_fnames=X_tr_fnames,
            # X_val_fnames=X_val_fnames,
            y=y_full,
            # y_tr=y_tr,
            # y_val=y_val,
            tr_idx=np.arange(n//3),
            val_idx=np.arange(n//3,2*n//3),
            fit_intercept=True,
            lmda_choice="mid",#.15,
            penalty=penalty,
            family="gaussian",
        )

        X_full_np = X_full @ np.eye(p)
        X_full_np = np.hstack((np.ones((n,1)), X_full_np))

        sl.fit()
        sl.extract_event()
        L, U = sl.infer(dispersion=None if est_dispersion else dispersion, method='exact', level=.9)

        # si_sl_selfsolve = split_lasso.gaussian(
        #     X_full_np, 
        #     y_full, 
        #     feature_weights=np.concatenate((np.zeros(1), np.ones(p) * sl.lmda*n)), 
        #     proportion=1./3,
        # )
        # si_sl_selfsolve.fit(
        #     perturb=np.concatenate((np.ones(n//3), np.zeros(2*n//3))).astype(bool),
        # )
        # si_sl_selfsolve.setup_inference(dispersion=1)
        # TS_selfsolve = selected_targets(si_sl_selfsolve.loglike, si_sl_selfsolve.observed_soln, dispersion=1)
        # out = si_sl_selfsolve.inference(target_spec=TS_selfsolve, method='selective_MLE')

        n_sel[j] = sl.overall.sum()
        targets = np.linalg.pinv(X_full_np[:, sl.overall]) @ X_full_np @ beta # beta[sl.overall]
        n_cov[j] = np.sum((L <= targets) & (targets <= U))

        # n_sel_si[i] = si_sl_selfsolve._overall.sum()
        # targets = np.linalg.pinv(X_full_np[:, si_sl_selfsolve.overall]) @ X_full_np @ beta
        # n_cov_si[i] = np.sum((out['MLE'] - 1.645*out['SE'] <= targets) & (out['MLE'] + 1.645*out['SE'] >= targets))
 
    
    for fname in X_full_fnames:
        os.remove(fname)
    for fname in X_tr_fnames:
        os.remove(fname)
    for fname in X_val_fnames:
        os.remove(fname)


    assert np.allclose(n_cov.sum() / n_sel.sum(), .9, atol=.05)
    assert np.allclose(n_cov.sum() / n_sel.sum(), .9, atol=.02)
    # assert np.allclose(n_cov_si.sum() / n_sel_si.sum(), .9, atol=.05)














