import numpy as np
from choosi.lasso import SplitLasso

def test_lasso():
    sl = SplitLasso(
        X_full=np.asfortranarray(np.random.randn(300,200)),
        X_tr=np.asfortranarray(np.random.randn(100,200)),
        X_val=np.asfortranarray(np.random.randn(100,200)),
        X_ts=np.asfortranarray(np.random.randn(100,200)),
        y_full=np.random.randn(300),
        y_tr=np.random.randn(100),
        y_val=np.random.randn(100),
        y_ts=np.random.randn(100),
        penalty=np.ones(200),
        family="gaussian",
    )

    sl.fit()
    sl.extract_event()