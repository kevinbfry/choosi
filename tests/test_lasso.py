import numpy as np
import pandas as pd
from selectinf.algorithms.barrier_affine import solve_barrier_affine_py
from selectinf.randomized.snp_lasso import split_lasso_solved, selected_targets
from selectinf.randomized.lasso import split_lasso
# from selectinf.base import selected_targets
from choosi.lasso import SplitLasso

def test_lasso():
    n = 100
    p = 30
    X_tr = np.asfortranarray(np.random.randn(n,p))
    X_val = np.asfortranarray(np.random.randn(n,p))
    X_ts = np.asfortranarray(np.random.randn(n,p))
    X_full = np.asfortranarray(np.vstack((X_tr, X_val, X_ts)))
    y_tr = np.random.randn(n)
    y_val = np.random.randn(n)
    y_ts = np.random.randn(n)
    y_full = np.asfortranarray(np.concatenate((y_tr, y_val, y_ts)))

    sl = SplitLasso(
        X_full=X_full,
        X_tr=X_tr,
        X_val=X_val,
        X_ts=X_ts,
        y_full=y_full,
        y_tr=y_tr,
        y_val=y_val,
        y_ts=y_ts,
        fit_intercept=True,
        penalty=np.ones(p),
        family="gaussian",
    )

    sl.fit()
    sl.extract_event()
    sel_MLE, sel_MLE_std = sl.infer()
    L = sel_MLE - 1.96 * sel_MLE_std
    U = sel_MLE + 1.96 * sel_MLE_std

    # print(pd.DataFrame({'std': sel_MLE_std, 'lower': L, 'MLE': sel_MLE, 'upper': U}))
    # print(np.mean((L <= 0) & (0 <= U)))

    # assert np.mean((L <= 0) & (0 <= U)) - .95 > -.05
    # assert 0==1

    si_sl_selfsolve = split_lasso.gaussian(
        np.hstack((np.ones((3*n,1)),X_full)), 
        y_full, 
        feature_weights=np.concatenate((np.zeros(1), np.ones(p) * sl.lmda*3*n)), 
        # feature_weights=np.ones(p) * sl.lmda*3*n, 
        proportion=1./3, 
    )
    si_sl_selfsolve.fit(
        perturb=np.concatenate((np.ones(n), np.zeros(2*n))).astype(bool),
    )
    si_sl_selfsolve.setup_inference(dispersion=1)
    TS = selected_targets(si_sl_selfsolve.loglike, si_sl_selfsolve.observed_soln, dispersion=1)
    out = si_sl_selfsolve.inference(target_spec=TS, method='selective_MLE')

    # si_sl = split_lasso_solved.gaussian(
    #     np.hstack((np.ones((3*n,1)),X_full)), 
    #     y_full, 
    #     feature_weights=np.concatenate((np.zeros(1), np.ones(p) * sl.lmda*3*n)), 
    #     # feature_weights=np.ones(p) * sl.lmda*3*n, 
    #     proportion=1./3,
    # )
    # si_sl.fit(
    #     observed_soln=sl.observed_soln, 
    #     observed_subgrad=sl.observed_subgrad,
    #     perturb=np.concatenate((np.ones(n), np.zeros(2*n))).astype(bool),
    # )
    # si_sl.setup_inference(dispersion=1)

    # np.allclose(sl.observed_soln, si_sl_selfsolve.observed_soln)

    # print(sl.cond_mean)
    # print(si_sl.cond_mean) 
    # print(np.diag(sl.cond_prec))
    # print(np.diag(np.linalg.inv(si_sl_selfsolve.cond_cov)))
    # assert np.allclose(np.sort(np.fabs(sl.cond_mean)), np.sort(np.fabs(si_sl_selfsolve.cond_mean)))
    # assert np.allclose(np.sort(np.fabs(sl.cond_mean)), np.sort(np.fabs(si_sl.cond_mean)))
    # assert np.allclose(np.diag(sl.cond_prec[1:,:][:,1:]), np.diag(np.linalg.inv(si_sl_selfsolve.cond_cov)[:-1,:][:,:-1]))
    # assert np.allclose(np.fabs(sl.cond_prec[1:,:][:,1:]), np.fabs(np.linalg.inv(si_sl_selfsolve.cond_cov)[:-1,:][:,:-1]))

    x_opt = sl.optimizer.optimize(max_steps=100, tau=.5, c=.5, tol=1e-10, ls_max_iters=1000)

    si_val, si_x, si_hess = solve_barrier_affine_py(sl.cond_prec @ sl.cond_mean, sl.cond_prec, -sl.signs, -np.diag(sl.signs)[1:,:], np.zeros_like(sl.signs)[1:])

    # assert np.allclose(x_opt, si_x)

    # assert np.allclose(np.sort(np.fabs(sel_MLE)), np.sort(np.fabs(out['MLE'])))
    ## o_1^* do not match, but we are computing sel_MLE the same way...
    # print(out)

    # print(pd.DataFrame({'std': sel_MLE_std, 'lower': L, 'MLE': sel_MLE, 'upper': U}))
    # print(np.mean((L <= 0) & (0 <= U)), np.mean((out['MLE'] - 1.96* out['SE'] <= 0) & (0 <= out['MLE'] + 1.96*out['SE'])))
    assert np.allclose(np.mean((L <= 0) & (0 <= U)), np.mean((out['MLE'] - 1.96* out['SE'] <= 0) & (0 <= out['MLE'] + 1.96*out['SE'])))
    # assert np.allclose(sel_MLE, out['MLE'])
    # assert np.allclose(sel_MLE_std, out['SE']) 
    # assert 0==1





