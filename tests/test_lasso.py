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

@pytest.mark.parametrize("n, p", [
    [100, 30],
    # [200, 100],
])
def test_lasso_null(n, p):
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
        y_full=y_full,
        y_tr=y_tr,
        y_val=y_val,
        fit_intercept=True,
        lmda_choice="mid",
        penalty=np.ones(p),
        family="gaussian",
    )

    sl.fit()
    sl.extract_event()
    sel_MLE, sel_MLE_std = sl.infer()
    L = sel_MLE - 1.96 * sel_MLE_std
    U = sel_MLE + 1.96 * sel_MLE_std

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
    si_sl.setup_inference(dispersion=1)
    TS = selected_targets(si_sl.loglike, si_sl.observed_soln, dispersion=1)
    out = si_sl.inference(target_spec=TS, method='selective_MLE')
    
    assert np.allclose(np.mean((L <= 0) & (0 <= U)), np.mean((out['MLE'] - 1.96* out['SE'] <= 0) & (0 <= out['MLE'] + 1.96*out['SE'])))


@pytest.mark.parametrize("n, p", [
    [100, 30],
    # [200, 100],
])
def test_lasso_alt(n, p):
    X_tr = np.asfortranarray(np.random.randn(n,p))
    X_val = np.asfortranarray(np.random.randn(n,p))
    X_ts = np.asfortranarray(np.random.randn(n,p))
    X_full = np.asfortranarray(np.vstack((X_tr, X_val, X_ts)))
    beta = np.random.choice([-1,1],size=p+1,replace=True) * np.random.uniform(3,5,size=p+1)
    # beta[0] = np.fabs(beta[0])
    y_tr = X_tr @ beta[1:] + beta[0] + np.random.randn(n)
    y_val = X_val @ beta[1:] + beta[0] + np.random.randn(n)
    y_ts = X_ts @ beta[1:] + beta[0] + np.random.randn(n)
    y_full = np.asfortranarray(np.concatenate((y_tr, y_val, y_ts)))

    sl = SplitLasso(
        X_full=X_full,
        X_tr=X_tr,
        X_val=X_val,
        y_full=y_full,
        y_tr=y_tr,
        y_val=y_val,
        fit_intercept=True,
        lmda_choice="mid",
        penalty=np.ones(p),
        family="gaussian",
    )

    sl.fit()
    sl.extract_event()
    sel_MLE, sel_MLE_std = sl.infer()
    L = sel_MLE - 1.96 * sel_MLE_std
    U = sel_MLE + 1.96 * sel_MLE_std
    
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
    si_sl.setup_inference(dispersion=1)
    TS = selected_targets(si_sl.loglike, si_sl.observed_soln, dispersion=1)
    out = si_sl.inference(target_spec=TS, method='selective_MLE')

    assert np.allclose(
        np.mean((L <= beta[sl.overall]) & (beta[sl.overall] <= U)), 
        np.mean((out['MLE'] - 1.96* out['SE'] <= beta[sl.overall]) & (beta[sl.overall] <= out['MLE'] + 1.96*out['SE']))
    )


@pytest.mark.parametrize("n, p, s", [
    [100, 30, 10],
    # [200, 100],
])
def test_lasso(n, p, s):
    X_tr = np.asfortranarray(np.random.randn(n,p))
    X_val = np.asfortranarray(np.random.randn(n,p))
    X_ts = np.asfortranarray(np.random.randn(n,p))
    X_full = np.asfortranarray(np.vstack((X_tr, X_val, X_ts)))
    beta = np.zeros(p+1)
    nz_idx = np.concatenate(([0],np.random.choice(p, size=s, replace=False) + 1))
    beta[nz_idx] = np.random.choice([-1,1],size=s+1,replace=True) * np.random.uniform(3,5,size=s+1)
    # beta[0] = np.fabs(beta[0])
    y_tr = X_tr @ beta[1:] + beta[0] + np.random.randn(n)
    y_val = X_val @ beta[1:] + beta[0] + np.random.randn(n)
    y_ts = X_ts @ beta[1:] + beta[0] + np.random.randn(n)
    y_full = np.asfortranarray(np.concatenate((y_tr, y_val, y_ts)))

    sl = SplitLasso(
        X_full=X_full,
        X_tr=X_tr,
        X_val=X_val,
        y_full=y_full,
        y_tr=y_tr,
        y_val=y_val,
        fit_intercept=True,
        lmda_choice="mid",
        penalty=np.ones(p),
        family="gaussian",
    )

    sl.fit()
    sl.extract_event()
    sel_MLE, sel_MLE_std = sl.infer()
    L = sel_MLE - 1.96 * sel_MLE_std
    U = sel_MLE + 1.96 * sel_MLE_std
    
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
    si_sl.setup_inference(dispersion=1)
    TS = selected_targets(si_sl.loglike, si_sl.observed_soln, dispersion=1)
    out = si_sl.inference(target_spec=TS, method='selective_MLE')

    assert np.allclose(
        np.mean((L <= beta[sl.overall]) & (beta[sl.overall] <= U)), 
        np.mean((out['MLE'] - 1.96* out['SE'] <= beta[sl.overall]) & (beta[sl.overall] <= out['MLE'] + 1.96*out['SE']))
    )


@pytest.mark.parametrize("n, p", [
    [100, 30],
    # [200, 100],
])
def test_lasso_snp(n, p):

    ## from JamesYang007/adelie GWAS user guide
    data_dir = "./data/"
    bedname = os.path.join(data_dir, "EUR_subset.bed")
    bimname = os.path.join(data_dir, "EUR_subset.bim")
    famname = os.path.join(data_dir, "EUR_subset.fam")

    df_fam = pd.read_csv(
        famname,
        sep=" ",
        header=None,
        names=["FID", "IID", "Father", "Mother", "Sex", "Phenotype"],
    )
    n_samples = df_fam.shape[0]

    df_bim = pd.read_csv(
        bimname,
        sep="\t",
        header=None,
        names=["chr", "variant", "pos", "base", "a1", "a2"],
    )
    n_snps = df_bim.shape[0]

    # create bed reader
    reader = pg.PgenReader(
        str.encode(bedname),
        raw_sample_ct=n_samples,
    )

    # get 0-indexed indices for current chromosome
    df_bim_chr = df_bim[df_bim["chr"] == 17]
    variant_idxs = df_bim_chr.index.to_numpy().astype(np.uint32)

    # read the SNP matrix
    geno_out_chr = np.empty((variant_idxs.shape[0], n_samples), dtype=np.int8)
    reader.read_list(variant_idxs, geno_out_chr)

    # convert to sample-major
    geno_out_chr = np.asfortranarray(geno_out_chr.T)

    # define cache directory and snpdat filename
    cache_dir = "/tmp"
    snpdat_name = os.path.join(cache_dir, "EUR_subset_chr17.snpdat")

    # create handler to convert the SNP matrix to .snpdat
    handler = ad.io.snp_unphased(snpdat_name)
    _ = handler.write(geno_out_chr)

    chromosomes = df_bim["chr"].unique()

    # create bed reader
    reader = pg.PgenReader(
        str.encode(bedname),
        raw_sample_ct=n_samples,
    )

    for chr in chromosomes:
        # get 0-indexed indices for current chromosome
        df_bim_chr = df_bim[df_bim["chr"] == chr]
        variant_idxs = df_bim_chr.index.to_numpy().astype(np.uint32)

        # read the SNP matrix
        geno_out = np.empty((variant_idxs.shape[0], n_samples), dtype=np.int8)
        reader.read_list(variant_idxs, geno_out)

        # define snpdat filename
        snpdat_name = os.path.join(cache_dir, f"EUR_subset_chr{chr}.snpdat")

        # create handler to convert the SNP matrix to .snpdat
        handler = ad.io.snp_unphased(snpdat_name)
        _ = handler.write(geno_out_chr)

    df = pd.read_csv(os.path.join(data_dir, "master_phe.csv"), sep="\t", index_col=0)
    covars_dense = df.iloc[:, :-1].to_numpy()
    y_full = df.iloc[:, -1].to_numpy()
    y_tr = y_full[:len(y_full)//3]
    y_val = y_full[len(y_full)//3:2*len(y_full)//3]

    X_full = ad.matrix.concatenate(
        # [ad.matrix.dense(covars_dense)]
        # + [
        [
            ad.matrix.snp_unphased(
                ad.io.snp_unphased(
                    os.path.join(cache_dir, f"EUR_subset_chr{chr}.snpdat"),
                )
            )
            for chr in chromosomes
        ],
        axis=1,
    )
    X_tr = X_full[:len(y_full)//3,:]
    X_val = X_full[len(y_full)//3:2*len(y_full)//3,:]

    X_full_fnames = []
    X_tr_fnames = []
    X_val_fnames = []
    for chr in chromosomes:
        X_full_chr = ad.matrix.snp_unphased(
            ad.io.snp_unphased(
                os.path.join(cache_dir, f"EUR_subset_chr{chr}.snpdat"),
            )
        )
        X_tr_chr = X_full_chr[:len(y_full)//3,:]
        X_val_chr = X_full_chr[len(y_full)//3:2*len(y_full)//3,:]
        X_full_fname = os.path.join(cache_dir, f"X_full_chr{chr}.snpdat")
        X_full_fnames.append(X_full_fname)
        handler = ad.io.snp_unphased(X_full_fname)
        X_full_chr = np.asfortranarray((X_full_chr.T @ np.eye(X_full_chr.rows())).T.astype(np.int8))
        _ = handler.write(X_full_chr)
        
        X_tr_fname = os.path.join(cache_dir, f"X_tr_chr{chr}.snpdat")
        X_tr_fnames.append(X_tr_fname)
        handler = ad.io.snp_unphased(X_tr_fname)
        X_tr_chr = np.asfortranarray((X_tr_chr.T @ np.eye(X_tr_chr.rows())).T.astype(np.int8))
        _ = handler.write(X_tr_chr)

        X_val_fname = os.path.join(cache_dir, f"X_val_chr{chr}.snpdat")
        X_val_fnames.append(X_val_fname)
        handler = ad.io.snp_unphased(X_val_fname)
        X_val_chr = np.asfortranarray((X_val_chr.T @ np.eye(X_val_chr.rows())).T.astype(np.int8))
        _ = handler.write(X_val_chr)

    penalty = np.concatenate([
        np.zeros(covars_dense.shape[-1]),
        np.ones(X_full.shape[-1] - covars_dense.shape[-1]),
    ])

    sl = SplitLassoSNPUnphased(
        X_full_fnames=X_full_fnames,
        X_tr_fnames=X_tr_fnames,
        X_val_fnames=X_val_fnames,
        y_full=y_full,
        y_tr=y_tr,
        y_val=y_val,
        fit_intercept=True,
        lmda_choice="mid",
        penalty=penalty,
        family="gaussian",
    )

    sl.fit()
    sl.extract_event()
    sel_MLE, sel_MLE_std = sl.infer()
    L = sel_MLE - 1.96 * sel_MLE_std
    U = sel_MLE + 1.96 * sel_MLE_std
    
    for chr in chromosomes:
        os.remove(os.path.join(cache_dir, f"EUR_subset_chr{chr}.snpdat"))
        os.remove(os.path.join(cache_dir, f"X_full_chr{chr}.snpdat"))
        os.remove(os.path.join(cache_dir, f"X_tr_chr{chr}.snpdat"))
        os.remove(os.path.join(cache_dir, f"X_val_chr{chr}.snpdat"))









