import numpy as np
import pandas as pd
from choosi.lasso import SplitLasso

def test_lasso():
    X_tr = np.asfortranarray(np.random.randn(100,200))
    X_val = np.asfortranarray(np.random.randn(100,200))
    X_ts = np.asfortranarray(np.random.randn(100,200))
    X_full = np.asfortranarray(np.vstack((X_tr, X_val, X_ts)))
    y_tr = np.random.randn(100)
    y_val = np.random.randn(100)
    y_ts = np.random.randn(100)
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
        fit_intercept=False,
        penalty=np.ones(200),
        family="gaussian",
    )

    sl.fit()
    sl.extract_event()
    sel_MLE, sel_MLE_std = sl.infer()
    L = sel_MLE - 1.96 * sel_MLE_std
    U = sel_MLE + 1.96 * sel_MLE_std

    print(pd.DataFrame({'std': sel_MLE_std, 'lower': L, 'MLE': sel_MLE, 'upper': U}))
    print(np.mean((L <= 0) & (0 <= U)))
    assert np.fabs(np.mean((L <= 0) & (0 <= U)) - .95) < .05
    assert 0==1