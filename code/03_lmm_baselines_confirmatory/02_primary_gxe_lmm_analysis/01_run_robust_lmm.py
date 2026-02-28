#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust LMM (numeric-only) — no formulas, no object dtypes.

- Y kept exactly as saved (already log1p in your .pt).
- MixedLM fitted with numeric exog & integer group codes.
- OLS FE fallback also numeric (accession/env one-hot + PCs + SNPs).
- NaN-safe aggregation. No zero padding.
"""

import os, sys, json, platform, warnings, time
import argparse, subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import trim_mean
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Thread hygiene
torch.set_num_threads(8)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="Robust LMM Analysis with Environment-Conditional eQTL")
parser.add_argument("--smoke", action="store_true", help="3-fold, single-seed quick run")
parser.add_argument("--dump-fold-sentinels", action="store_true", help="write per-fold train/test IDs")
parser.add_argument(
    "--permute-within-fold",
    action="store_true",
    help="Permute genotypes within the training fold for null model comparison.",
)
parser.add_argument(
    "--cis-window-scale",
    type=float,
    default=1.0,
    help="Scaling factor for cis-window size (NOTE: not used in this script as cis-map is precomputed).",
)
parser.add_argument("--per-env-lmm", action="store_true", default=True, help="Run per-environment LMM analysis.")
parser.add_argument("--n-pcs", type=int, default=10, help="Number of principal components to use.")
parser.add_argument("--verbose", action="store_true", help="Enable detailed per-gene logging.")
parser.add_argument("--gene-cap", type=int, default=None,
                    help="Limit the number of genes processed (useful for smoke tests).")
parser.add_argument("--permutations", type=int, default=1000, help="Number of permutations for empirical p-value calculation.")
parser.add_argument("--seed", type=int, default=2025, help="Master seed for full reproducibility of folds, bootstraps, and permutations.")
# NEW ARGUMENTS - Add these to support flexible data paths
parser.add_argument(
    "--data-file", 
    type=str, 
    default=None,
    help="Path to transformer data .pt file (auto-detects if not specified)"
)
parser.add_argument(
    "--output-dir", 
    type=str, 
    default=r"C:\Users\ms\Desktop\gwas\output\robust_lmm_analysis",
    help="Output directory for LMM results"
)
parser.add_argument(
    "--gene-selector", 
    type=str, 
    default=None,
    help="Gene selector mode (baseline/drought) - auto-detects if not specified"
)

args, _ = parser.parse_known_args()


def r2_safe(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    if ok.sum() < 3: return np.nan
    ys = np.std(y[ok])
    if ys < 0.05:
        return np.nan
    ssr = np.sum((y[ok] - yhat[ok]) ** 2)
    sst = np.sum((y[ok] - np.mean(y[ok])) ** 2)
    return 1.0 - ssr / sst


def nrmse_safe(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    if ok.sum() < 3: return np.nan
    rmse = np.sqrt(np.mean((y[ok] - yhat[ok]) ** 2))
    denom = np.std(y[ok])
    return rmse / denom if denom > 0 else np.nan


class RobustLMMAnalyzer:
    def __init__(self, data_path: str, output_dir: str, device: torch.device, args: argparse.Namespace):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        self.device = device
        self.args = args
        self._load_data()

        # logs - will be populated in the main process
        self.fit_audit: List[Dict[str, Any]] = []
        self.permutation_log: List[Dict[str, Any]] = []
        self._written_folds_lmm: Set[int] = set()
        
        # RNG
        self.master_seed = int(args.seed)
        self.perm_seed_base = np.random.SeedSequence(self.master_seed).generate_state(1, dtype=np.uint32)[0]
        self.gene_seeds = [int(s.generate_state(1, dtype=np.uint32)[0]) 
                           for s in np.random.SeedSequence(self.master_seed).spawn(self.Y.shape[1])]

        # OOF predictions - will be allocated in run()
        self.oof_null: Optional[np.ndarray] = None
        self.oof_full: Optional[np.ndarray] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # don’t ship large parent-only arrays to workers
        state["oof_null"] = None
        state["oof_full"] = None
        return state

    # ------------------------ IO ------------------------

    def _load_data(self):
        """Enhanced data loading with validation and error handling."""
        try:
            print(f"Loading data from: {self.data_path}")
            data = torch.load(self.data_path, map_location="cpu")
            
            # Validate required keys
            required_keys = ["G", "Y", "PCs", "E"]
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise KeyError(f"Missing required keys in data file: {missing_keys}")
            
            print("Data file validation passed")
            
        except Exception as e:
            print(f"Error loading data file: {e}")
            print(f"File path: {self.data_path}")
            print(f"File exists: {self.data_path.exists()}")
            if self.data_path.exists():
                print(f"File size: {self.data_path.stat().st_size} bytes")
            raise

        # Load tensors with proper type conversion
        self.G = (data["G"] if torch.is_tensor(data["G"]) else torch.from_numpy(np.asarray(data["G"]))).to(
            self.device, dtype=torch.float32
        )
        self.Y_raw = (data["Y"] if torch.is_tensor(data["Y"]) else torch.from_numpy(np.asarray(data["Y"]))).to(
            self.device, dtype=torch.float32
        )
        self.PCs = (data["PCs"] if torch.is_tensor(data["PCs"]) else torch.from_numpy(np.asarray(data["PCs"]))).to(
            self.device, dtype=torch.float32
        )
        self.PCs = self.PCs[:, : self.args.n_pcs]

        # Environment labels (numeric 0/1/2)
        self.E = (data["E"] if torch.is_tensor(data["E"]) else torch.from_numpy(np.asarray(data["E"]))).to(self.device)
        
        # Accession groups - handle both new and legacy formats
        if "sample_names" in data:
            # New format from prepare_data_5e.py
            groups = data["sample_names"]
            if torch.is_tensor(groups): 
                groups = groups.cpu().numpy()
            # Extract accession from sample names (assumes sample_name format)
            self.groups = np.array([str(s).split('_')[0] if '_' in str(s) else str(s) for s in groups])
        elif "accession_ids" in data:
            # Legacy format
            groups = data["accession_ids"]
            if torch.is_tensor(groups): 
                groups = groups.cpu().numpy()
            self.groups = np.asarray(groups).astype(str)
        else:
            # Fallback - create dummy groups
            print("Warning: No accession groups found, using sample indices")
            self.groups = np.array([f"Sample_{i}" for i in range(self.Y_raw.shape[0])])

        # Cis-regulatory mapping - handle both tensor and dict formats
        if "gene_snp_mask" in data and torch.is_tensor(data["gene_snp_mask"]):
            mask = data["gene_snp_mask"].to(self.device).bool()
            m = {}
            for i in range(mask.shape[0]):
                m[i] = torch.where(mask[i])[0].cpu().tolist()
            self.cis_map = m
        elif "gene_snp_indices" in data:
            # Alternative format from prepare_data_5e.py
            self.cis_map = {i: indices for i, indices in enumerate(data["gene_snp_indices"])}
        else:
            self.cis_map = data.get("gene_snp_mask", {})

        # Gene names
        if "gene_names" in data:
            self.gene_names = data["gene_names"]
        else:
            self.gene_names = [f"Gene_{i}" for i in range(self.Y_raw.shape[1])]

        # Y is already log-transformed in prepare_data_5e.py
        self.Y = self.Y_raw.clone()

        # Cleanup NaNs
        self.G = torch.nan_to_num(self.G)
        self.PCs = torch.nan_to_num(self.PCs)

        print(f"Data loaded successfully:")
        print(f"  Samples: {self.G.shape[0]}")
        print(f"  SNPs: {self.G.shape[1]}")
        print(f"  Genes: {self.Y.shape[1]}")
        print(f"  PCs: {self.PCs.shape[1]}")
        print(f"  Unique accessions: {len(np.unique(self.groups))}")
        print(f"  Environments: {sorted(np.unique(self.E.cpu().numpy()))}")
        print(f"  Cis-map coverage: {len(self.cis_map)} genes")

    def _record_fit(self, *, gene_id, fold_id, model_name, method, converged,
                    n_snps, n_train, n_test, r2, nrmse, time_s, fail_reason=None):
        return {
            "gene_id": str(gene_id), "fold": int(fold_id), "model": model_name,  # "NULL"/"FULL"
            "method": method,                    # "MixedLM" | "OLS_FE" | "MEAN_BACKOFF" | "NULL_BACKOFF"
            "converged": bool(converged), "n_snps": int(n_snps),
            "n_train": int(n_train), "n_test": int(n_test),
            "r2": float(r2) if r2 is not None else np.nan,
            "nrmse": float(nrmse) if nrmse is not None else np.nan,
            "time_s": float(time_s), "fail_reason": (str(fail_reason) if fail_reason else "")
        }

    # ------------------------ helpers ------------------------

    def get_top_cis_snps(self, gene_idx: int, train_idx: np.ndarray, top_k: int = 5) -> List[int]:
        if self.args.verbose or gene_idx < 5:
            print(f"Debug Gene {gene_idx}: ")
            print(f"  Gene name: {self.gene_names[gene_idx] if hasattr(self, 'gene_names') else 'unknown'}")
        
        cis_snps = self.cis_map.get(gene_idx, []) if isinstance(self.cis_map, dict) else []
        
        if self.args.verbose or gene_idx < 5:
            print(f"  Cis SNPs available: {len(cis_snps)}")
            print(f"  Genotype matrix shape: {self.G.shape}")
            print(f"  Train samples: {len(train_idx)}")

        if not cis_snps:
            if self.args.verbose or gene_idx < 5:
                print(f"  No cis SNPs for gene {gene_idx}")
            return []

        cis_snps = [s for s in cis_snps if s < self.G.shape[1]]
        if not cis_snps: return []

        ti = torch.from_numpy(train_idx).to(self.device)
        y = self.Y[ti, gene_idx]
        if float(y.std()) == 0.0:
            return []

        g = self.G[ti][:, torch.tensor(cis_snps, device=self.device)]
        gstd = g.std(dim=0)
        keep = gstd > 0
        if not torch.any(keep): return []

        g = g[:, keep]
        cis_kept = torch.tensor(cis_snps, device=self.device)[keep]
        ystd = (y - y.mean()) / y.std()
        gzs = (g - g.mean(dim=0)) / gstd[keep]
        corr = torch.abs((ystd.unsqueeze(0) @ gzs) / len(y)).squeeze()
        if hasattr(corr, 'shape') and len(corr.shape) > 0:
            k = int(min(top_k, corr.shape[0]))
        else:
            # Handle empty/scalar correlation case
            if self.args.verbose or gene_idx < 5:
                print(f"Warning: Gene {gene_idx} has no valid correlations")
            return []
        if k <= 0: return []

        _, idx = torch.topk(corr, k)
        return cis_kept[idx].cpu().tolist()

    def _make_df(
        self, idx: np.ndarray, gene_idx: int, top_snps: List[int], permute_snps: bool = False
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "expression": self.Y[idx, gene_idx].cpu().numpy().astype("float64"),
                "accession": self.groups[idx],
                "env": self.E[idx].cpu().numpy().astype(int),
            }
        )
        # PCs
        for i in range(self.PCs.shape[1]):
            df[f"PC{i+1}"] = self.PCs[idx, i].cpu().numpy().astype("float64")
        # SNPs
        if permute_snps:
            rng = np.random.RandomState(123)
            if top_snps:
                G_train_snps = self.G[idx][:, top_snps].cpu().numpy()
                G_train_snps_permuted = G_train_snps.copy()
                for j in range(G_train_snps_permuted.shape[1]):
                    rng.shuffle(G_train_snps_permuted[:, j])

                for j, snp in enumerate(top_snps):
                    df[f"SNP{j+1}"] = G_train_snps_permuted[:, j].astype("float64")
        else:
            for j, snp in enumerate(top_snps):
                df[f"SNP{j+1}"] = self.G[idx, snp].cpu().numpy().astype("float64")
        return df

    def _build_numeric_design(
        self, df: pd.DataFrame, include_snps: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Build numeric (float64) exog and endog, and integer group codes for MixedLM.
        exog = [const | PCs | env one-hot (drop_first) | SNPs?]
        """
        y = pd.to_numeric(df["expression"], errors="coerce").astype("float64").values

        # PCs
        pc_cols = [c for c in df.columns if c.startswith("PC")]
        X_parts = [df[pc_cols].astype("float64")] if pc_cols else []

        # env one-hot (drop_first)
        env_oh = pd.get_dummies(df["env"].astype(int), prefix="env", drop_first=True)
        if env_oh.shape[1] > 0:
            X_parts.append(env_oh.astype("float64"))

        # SNPs
        if include_snps:
            snp_cols = [c for c in df.columns if c.startswith("SNP")]
            if snp_cols:
                X_parts.append(df[snp_cols].astype("float64"))

        X = pd.concat(X_parts, axis=1) if len(X_parts) else pd.DataFrame(index=df.index)
        X = sm.add_constant(X, has_constant="add")
        X = X.astype("float64").values  # <— NUMERIC

        # groups for random intercept
        acc = pd.Categorical(df["accession"])
        group_codes = acc.codes.astype(int)  # <— integer array; not part of X

        return y, X, group_codes, pc_cols

    def _mixedlm_fit_predict(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame, include_snps: bool
    ) -> Tuple[Optional[Any], Optional[np.ndarray]]:
        """
        Pure numeric MixedLM: endog=float64, exog=float64, groups=int codes.
        Predicts on test using fixed effects (random effects default to 0 if unseen groups).
        """
        try:
            y_tr, X_tr, g_tr, _ = self._build_numeric_design(df_train, include_snps)
            _, X_te, _, _ = self._build_numeric_design(df_test, include_snps)

            model = sm.MixedLM(endog=y_tr, exog=X_tr, groups=g_tr).fit(
                reml=False, method="lbfgs", maxiter=100, disp=False
            )
            yhat = model.predict(exog=X_te)
            yhat = np.asarray(yhat, dtype=float)
            return model, yhat
        except Exception:
            return None, None

    def _ols_fe_fit_predict(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame, include_snps: bool
    ) -> Tuple[Optional[Any], Optional[np.ndarray]]:
        """
        Numeric OLS with fixed effects for accession + env (one-hot), plus PCs + SNPs.
        """
        try:
            # design: accession one-hot (drop_first)
            Xtr_acc = pd.get_dummies(df_train["accession"], drop_first=True)
            Xte_acc = pd.get_dummies(df_test["accession"], drop_first=True).reindex(
                columns=Xtr_acc.columns, fill_value=0
            )

            # env one-hot
            Xtr_env = pd.get_dummies(df_train["env"].astype(int), prefix="env", drop_first=True)
            Xte_env = pd.get_dummies(df_test["env"].astype(int), prefix="env", drop_first=True).reindex(
                columns=Xtr_env.columns, fill_value=0
            )

            # PCs
            pc_cols = [c for c in df_train.columns if c.startswith("PC")]
            Xtr_pc = df_train[pc_cols].astype("float64") if pc_cols else pd.DataFrame(index=df_train.index)
            Xte_pc = df_test[pc_cols].astype("float64") if pc_cols else pd.DataFrame(index=df_test.index)

            X_parts_tr = [Xtr_acc, Xtr_env, Xtr_pc]
            X_parts_te = [Xte_acc, Xte_env, Xte_pc]

            # SNPs
            if include_snps:
                snp_cols = [c for c in df_train.columns if c.startswith("SNP")]
                if snp_cols:
                    X_parts_tr.append(df_train[snp_cols].astype("float64"))
                    X_parts_te.append(df_test[snp_cols].astype("float64"))

            Xtr = pd.concat(X_parts_tr, axis=1)
            Xte = pd.concat(X_parts_te, axis=1)
            Xtr = sm.add_constant(Xtr, has_constant="add").astype("float64")
            Xte = sm.add_constant(Xte, has_constant="add").astype("float64")

            ytr = pd.to_numeric(df_train["expression"], errors="coerce").astype("float64").values

            model = sm.OLS(ytr, Xtr.values, hasconst=True).fit()
            yhat = np.asarray(model.predict(Xte.values), dtype=float)
            return model, yhat
        except Exception:
            return None, None

    # ------------------------ core per-fold ------------------------

    def fit_gene_fold(
        self, gene_idx: int, train_idx: np.ndarray, test_idx: np.ndarray, fold_id: int, top_k: int = 5
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        top_snps = self.get_top_cis_snps(gene_idx, train_idx, top_k=top_k)

        # Prepare frames
        df_tr = self._make_df(train_idx, gene_idx, top_snps, permute_snps=self.args.permute_within_fold)
        df_te = self._make_df(test_idx, gene_idx, top_snps, permute_snps=False)
        y_te = df_te["expression"].values

        # Low-variance test fold: still emit predictions to keep OOF complete
        if np.std(y_te) < 0.05:
            mu = float(np.nanmean(df_tr["expression"].values))
            yhat_null = np.full(len(df_te), mu, float)
            yhat_full = yhat_null.copy()
            return {
                "gene_id": gene_idx,
                "gene_name": self.gene_names[gene_idx] if gene_idx < len(self.gene_names) else f"Gene_{gene_idx}",
                "fold": fold_id,
                "n_snps": len(top_snps),
                "is_low_variance": True,
                "r2_null": np.nan, "nrmse_null": np.nan,
                "r2_full": np.nan, "nrmse_full": np.nan,
                "delta_r2": np.nan, "delta_nrmse": np.nan,
                "converged_null": False, "converged_full": False,
                "lr_pval": np.nan,
                "yhat_null_vec": yhat_null,
                "yhat_full_vec": yhat_full,
                "te_idx": np.asarray(test_idx, dtype=int),
            }, []

        audit_rows = []

        # --- NULL model ---
        t0 = time.time()
        null_method, null_conv, null_reason = "MixedLM", False, None
        yhat_null = None
        try:
            model_null, yhat_null = self._mixedlm_fit_predict(df_tr, df_te, include_snps=False)
            if model_null is not None:
                null_conv = bool(getattr(model_null, "converged", False))
        except Exception as e:
            null_reason = f"exception:{e.__class__.__name__}"

        if (yhat_null is None) or (not np.isfinite(yhat_null).all()) or (not null_conv):
            null_method = "OLS_FE"
            try:
                _, yhat_null = self._ols_fe_fit_predict(df_tr, df_te, include_snps=False)
                null_conv = True
                if null_reason is None: null_reason = "nonconverged_or_nan"
            except Exception as e:
                null_reason = f"ols_exception:{e.__class__.__name__}"
                yhat_null = np.full(len(df_te), float(np.nanmean(df_tr["expression"].values)))
                null_method = "MEAN_BACKOFF"; null_conv = False
        
        t1 = time.time()
        r2_null = r2_safe(y_te, yhat_null); nrmse_null = nrmse_safe(y_te, yhat_null)
        audit_rows.append(self._record_fit(gene_id=gene_idx, fold_id=fold_id, model_name="NULL",
            method=null_method, converged=null_conv, n_snps=0,
            n_train=len(df_tr), n_test=len(df_te),
            r2=r2_null, nrmse=nrmse_null, time_s=(t1-t0), fail_reason=null_reason))
        
        # --- FULL model (cis SNPs) ---
        t2 = time.time()
        full_method, full_conv, full_reason = "MixedLM", False, None
        yhat_full = None
        try:
            model_full, yhat_full = self._mixedlm_fit_predict(df_tr, df_te, include_snps=True)
            if model_full is not None:
                full_conv = bool(getattr(model_full, "converged", False))
        except Exception as e:
            full_reason = f"exception:{e.__class__.__name__}"
        
        if (yhat_full is None) or (not np.isfinite(yhat_full).all()) or (not full_conv):
            full_method = "OLS_FE"
            try:
                _, yhat_full = self._ols_fe_fit_predict(df_tr, df_te, include_snps=True)
                full_conv = True
                if full_reason is None: full_reason = "nonconverged_or_nan"
            except Exception as e:
                full_reason = f"ols_exception:{e.__class__.__name__}"
                yhat_full = yhat_null
                full_method = "NULL_BACKOFF"; full_conv = False
                
        t3 = time.time()
        r2_full = r2_safe(y_te, yhat_full); nrmse_full = nrmse_safe(y_te, yhat_full)
        audit_rows.append(self._record_fit(gene_id=gene_idx, fold_id=fold_id, model_name="FULL",
            method=full_method, converged=full_conv, n_snps=len(top_snps),
            n_train=len(df_tr), n_test=len(df_te),
            r2=r2_full, nrmse=nrmse_full, time_s=(t3-t2), fail_reason=full_reason))

        result = {
            "gene_id": gene_idx,
            "gene_name": self.gene_names[gene_idx] if gene_idx < len(self.gene_names) else f"Gene_{gene_idx}",
            "fold": fold_id,
            "n_snps": len(top_snps),
            "is_low_variance": False,
            "r2_null": r2_null, "nrmse_null": nrmse_null,
            "r2_full": r2_full, "nrmse_full": nrmse_full,
            "delta_r2": r2_full - r2_null if np.isfinite(r2_full) and np.isfinite(r2_null) else np.nan,
            "delta_nrmse": nrmse_null - nrmse_full if np.isfinite(nrmse_null) and np.isfinite(nrmse_full) else np.nan,
            "converged_null": null_conv, 
            "converged_full": full_conv,
            "lr_pval": np.nan,
            "te_idx": np.asarray(test_idx, dtype=int),
            "yhat_null_vec": yhat_null.astype(float),
            "yhat_full_vec": yhat_full.astype(float),
        }

        # LR p-value (use actual fixed-effect dof; guard for collinearity/drops)
        if (full_method == "MixedLM") and (null_method == "MixedLM") and full_conv and null_conv:
            if (model_full is not None) and (model_null is not None):
                try:
                    lr_stat = 2.0 * (model_full.llf - model_null.llf)
                    
                    # More robust introspection for number of fixed-effect parameters
                    k_full = int(getattr(model_full, "k_fe", None) 
                                 or getattr(getattr(model_full, "fe_params", None), "size", 0) 
                                 or len(model_full.params))
                    k_null = int(getattr(model_null, "k_fe", None) 
                                 or getattr(getattr(model_null, "fe_params", None), "size", 0) 
                                 or len(model_null.params))
                    
                    df_lr = max(0, k_full - k_null) # Use the difference in actual degrees of freedom
                    
                    if lr_stat > 0 and df_lr > 0:
                        result["lr_pval"] = float(stats.chi2.sf(lr_stat, df_lr))
                    else:
                        result["lr_pval"] = np.nan
                except Exception:
                    result["lr_pval"] = np.nan
        return result, audit_rows

    def _fit_predict_consistent(self, df_tr: pd.DataFrame, df_te: pd.DataFrame, include_snps: bool):
        """Try MixedLM; if it fails or is non-finite, fall back to OLS-FE."""
        model, yhat = self._mixedlm_fit_predict(df_tr, df_te, include_snps=include_snps)
        
        # Check for failure (yhat is None) or non-convergence/NaN predictions
        converged = getattr(model, "converged", False) if model is not None else False
        if (yhat is None) or (not np.isfinite(yhat).all()) or not converged:
            _, yhat = self._ols_fe_fit_predict(df_tr, df_te, include_snps=include_snps)
            
        # Final backstop for rare failures in both methods
        if yhat is None:
            mu = np.nanmean(df_tr["expression"].values)
            yhat = np.full(len(df_te), mu, dtype=float)
            
        return yhat

    # ------------------------ permutation (numeric OLS FE) ------------------------

    def run_perm(
        self, gene_idx: int, train_idx: np.ndarray, test_idx: np.ndarray, fold_id: int, orig_delta: float, n_perm: int = 50
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        if not np.isfinite(orig_delta): return (np.nan, None)

        top_snps = self.get_top_cis_snps(gene_idx, train_idx, top_k=5)
        if not top_snps: return (np.nan, None)

        df_tr = self._make_df(train_idx, gene_idx, top_snps)
        df_te = self._make_df(test_idx, gene_idx, top_snps)
        y_te = df_te["expression"].values

        # Use the consistent model path for the null prediction
        yhat_null = self._fit_predict_consistent(df_tr, df_te, include_snps=False)
        if yhat_null is None: return (np.nan, None)
        r2_null = r2_safe(y_te, yhat_null)

        snp_cols = [c for c in df_tr.columns if c.startswith("SNP")]
        rng = np.random.default_rng(self.perm_seed_base + gene_idx * 131 + fold_id)
        deltas = []

        for b in range(n_perm):
            dfx = df_tr.copy()
            perm = rng.permutation(len(dfx))
            for c in snp_cols:
                dfx[c] = dfx[c].values[perm]

            # Use the consistent model path for the permuted full model
            yhat_full = self._fit_predict_consistent(dfx, df_te, include_snps=True)
            if yhat_full is None:
                deltas.append(np.nan)
            else:
                r2_full = r2_safe(y_te, yhat_full)
                deltas.append(r2_full - r2_null)

            # Early stopping logic
            valid = np.isfinite(deltas)
            if valid.sum() >= 30: # Check after a reasonable number of permutations
                hits = np.sum(np.array(deltas)[valid] >= orig_delta)
                lo, hi = proportion_confint(hits, valid.sum(), alpha=0.05, method="beta")
                if hi < 0.05 or lo > 0.05:
                    break

        valid = np.isfinite(deltas)
        if valid.sum() == 0:
            return (np.nan, None)
        hits = np.sum(np.array(deltas)[valid] >= orig_delta)
        p_emp = (hits + 1) / (valid.sum() + 1)

        log = {
            "gene_idx": gene_idx, "fold_id": fold_id, "original_delta": float(orig_delta),
            "n_permutations_run": int(valid.sum()), "empirical_p": float(p_emp),
        }
        return float(p_emp), log

    # ------------------------ transfer ------------------------

    def env_transfer(self, gene_idx: int, KFOLDS: int = 3) -> Dict[str, List[float]]:
        out = {"WW_to_WS1": [], "WW_to_WS2": [], "WS1_to_WW": [], "WS1_to_WS2": [], "WS2_to_WW": [], "WS2_to_WS1": []}
        env_map = {0: "WW", 1: "WS1", 2: "WS2"}

        for tr_env in [0, 1, 2]:
            for te_env in [0, 1, 2]:
                if tr_env == te_env: continue

                tr_mask = self.E.cpu().numpy() == tr_env
                te_mask = self.E.cpu().numpy() == te_env
                tr_idx = np.where(tr_mask)[0]
                te_idx = np.where(te_mask)[0]
                if len(tr_idx) < 10 or len(te_idx) < 5:
                    continue

                gkf = GroupKFold(n_splits=min(KFOLDS, len(np.unique(self.groups[tr_idx]))))
                for tr_cv, _ in gkf.split(tr_idx, groups=self.groups[tr_idx]):
                    fold_tr = tr_idx[tr_cv]
                    top_snps = self.get_top_cis_snps(gene_idx, fold_tr, top_k=5)
                    if not top_snps: 
                        out[f"{env_map[tr_env]}_to_{env_map[te_env]}"].append(np.nan)
                        continue

                    df_tr = self._make_df(fold_tr, gene_idx, top_snps)
                    df_te = self._make_df(te_idx, gene_idx, top_snps)
                    y_te = df_te["expression"].values

                    # cache null
                    _, yhat_null = self._mixedlm_fit_predict(df_tr, df_te, include_snps=False)
                    if yhat_null is None:
                        out[f"{env_map[tr_env]}_to_{env_map[te_env]}"].append(np.nan)
                        continue
                    r2_null = r2_safe(y_te, yhat_null)

                    # full with fallback
                    _, yhat_full = self._mixedlm_fit_predict(df_tr, df_te, include_snps=True)
                    if yhat_full is None:
                        _, yhat_full = self._ols_fe_fit_predict(df_tr, df_te, include_snps=True)
                    if yhat_full is None:
                        out[f"{env_map[tr_env]}_to_{env_map[te_env]}"].append(np.nan)
                        continue

                    r2_full = r2_safe(y_te, yhat_full)
                    out[f"{env_map[tr_env]}_to_{env_map[te_env]}"].append(r2_full - r2_null)

        print("\n=== TRANSFER SANITY ===")
        for k, arr in out.items():
            a = np.asarray(arr, float)
            nfin = np.isfinite(a).sum()
            mean = float(np.nanmean(a)) if nfin else np.nan
            med = float(np.nanmedian(a)) if nfin else np.nan
            pos = float(np.nanmean(a > 1e-3) * 100) if nfin else 0.0
            print(f"{k}: n_fin={nfin}/{a.size}, mean={mean:.4f}, median={med:.4f}, pos%={pos:.1f}%")
        
        # Write per-gene transfer deltas for debugging/outlier triage
        debug_rows = []
        for k, arr in out.items():
            for v in arr:
                debug_rows.append({"gene_idx": gene_idx, "gene_name": self.gene_names[gene_idx], "direction": k, "delta_r2": v})
        dbg = pd.DataFrame(debug_rows)
        dbg.to_csv(self.output_dir / "logs" / f"transfer_by_gene_{gene_idx}.csv", index=False)
        
        return out

    # ------------------------ whole-gene analysis ------------------------

    def analyze_gene(
        self, gene_idx: int, KFOLDS: int, n_boot: int, n_perm: int, var_thr: float
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(self.gene_seeds[gene_idx])
        gkf = GroupKFold(n_splits=KFOLDS)
        cv_rows, emp_pvals, test_sds = [], [], []
        perm_logs: List[Dict[str, Any]] = []
        fold_audit_rows: List[Dict[str, Any]] = []

        n = self.Y.shape[0]
        oof_null = np.full(n, np.nan, float)
        oof_full = np.full(n, np.nan, float)

        for fold_id, (tr, te) in enumerate(gkf.split(self.G.cpu().numpy(), groups=self.groups)):
            if self.args.dump_fold_sentinels and fold_id not in self._written_folds_lmm:
                outd = Path(r"C:\Users\ms\Desktop\gwas\output\fold_sentinels\lmm")
                outd.mkdir(parents=True, exist_ok=True)
                train_vals = np.asarray(self.groups)[tr]
                test_vals = np.asarray(self.groups)[te]
                np.savetxt(outd / f"fold_{fold_id}_train.txt", train_vals, fmt="%s")
                np.savetxt(outd / f"fold_{fold_id}_test.txt", test_vals, fmt="%s")
                print(f"[LMM] wrote sentinels: {outd / f'fold_{fold_id}_train.txt'} , {outd / f'fold_{fold_id}_test.txt'}")
                self._written_folds_lmm.add(fold_id)

            row, audit = self.fit_gene_fold(gene_idx, tr, te, fold_id)
            cv_rows.append(row)
            fold_audit_rows.extend(audit)
            test_sds.append(np.std(self.Y[te, gene_idx].cpu().numpy()))
            
            if row.get("yhat_null_vec") is not None: oof_null[row["te_idx"]] = row["yhat_null_vec"]
            if row.get("yhat_full_vec") is not None: oof_full[row["te_idx"]] = row["yhat_full_vec"]

            if fold_id % 2 == 0:
                p_emp, plog = self.run_perm(gene_idx, tr, te, fold_id, row.get("delta_r2", np.nan), n_perm)
                emp_pvals.append(p_emp)
                if plog: perm_logs.append(plog)

        transfer = self.env_transfer(gene_idx)
        per_env_results = self._run_per_env_lmm_cv(gene_idx)

        valid = [r for r in cv_rows if np.isfinite(r.get("delta_r2", np.nan))]
        if not valid:
            summary_row = {
                "gene_id": gene_idx, "gene_name": self.gene_names[gene_idx],
                "r2_null": np.nan, "r2_full": np.nan, "delta_r2": np.nan,
                "nrmse_null": np.nan, "nrmse_full": np.nan, "delta_nrmse": np.nan,
                "delta_r2_ci_lower": np.nan, "delta_r2_ci_upper": np.nan,
                "pvalue": np.nan, "empirical_pvalue": np.nan, "empirical_pvalue_combined": np.nan,
                "n_cis_snps": np.nan, "mean_snp_effect": np.nan,
                "is_low_variance": np.nan, "test_sd_mean": np.nan,
                "n_low_variance_folds": sum(r.get("is_low_variance", False) for r in cv_rows),
                "convergence_rate_null": np.nan, "convergence_rate_full": np.nan,
                "transfer_WW_to_WS1": np.nan, "transfer_WS1_to_WS2": np.nan, "transfer_WS2_to_WW": np.nan,
            }
            summary_row.update(per_env_results)
        else:
            deltas = np.array([r["delta_r2"] for r in valid], float)
            
            # bootstrap CI around the MEDIAN ΔR²
            finite = np.isfinite(deltas)
            if finite.any():
                boots = [np.nanmedian(
                    rng.choice(deltas[finite], finite.sum(), replace=True)
                ) for _ in range(n_boot)]
                ci_lo = np.percentile(boots, 2.5)
                ci_hi = np.percentile(boots, 97.5)
            else:
                ci_lo = ci_hi = np.nan

            summary_row = {
                "gene_id": gene_idx,
                "gene_name": self.gene_names[gene_idx],
                "r2_null": np.nanmedian([r["r2_null"] for r in valid]),
                "r2_full": np.nanmedian([r["r2_full"] for r in valid]),
                "delta_r2": np.nanmedian(deltas),
                "nrmse_null": np.nanmedian([r["nrmse_null"] for r in valid]),
                "nrmse_full": np.nanmedian([r["nrmse_full"] for r in valid]),
                "delta_nrmse": np.nanmedian([r["delta_nrmse"] for r in valid]),
                "delta_r2_ci_lower": ci_lo,
                "delta_r2_ci_upper": ci_hi,
                "pvalue": np.nanmedian([r["lr_pval"] for r in valid]),
                "empirical_pvalue": np.nanmedian(emp_pvals) if len(emp_pvals) else np.nan,
                "empirical_pvalue_combined": self._combine_empirical(emp_pvals),
                "n_cis_snps": np.nanmean([r["n_snps"] for r in valid]),
                "mean_snp_effect": np.nan,  # (not computed with numeric MixedLM path)
                "is_low_variance": float(np.mean(test_sds) < var_thr),
                "test_sd_mean": float(np.mean(test_sds)),
                "n_low_variance_folds": sum(r.get("is_low_variance", False) for r in cv_rows),
                "convergence_rate_null": float(np.nanmean([r["converged_null"] for r in cv_rows])),
                "convergence_rate_full": float(np.nanmean([r["converged_full"] for r in cv_rows])),
                "transfer_WW_to_WS1": float(np.nanmean(transfer.get("WW_to_WS1", [np.nan]))),
                "transfer_WS1_to_WS2": float(np.nanmean(transfer.get("WS1_to_WS2", [np.nan]))),
                "transfer_WS2_to_WW": float(np.nanmean(transfer.get("WS2_to_WW", [np.nan]))),
            }
            summary_row.update(per_env_results)

        return {
            "gene_idx": int(gene_idx),
            "summary_row": summary_row,
            "oof_null_vec": np.asarray(oof_null, dtype=np.float32),
            "oof_full_vec": np.asarray(oof_full, dtype=np.float32),
            "audit_rows": fold_audit_rows,
            "perm_logs": perm_logs,
        }

    @staticmethod
    def _combine_empirical(pvals: List[float]) -> float:
        if not pvals: return np.nan
        p = np.clip([x for x in pvals if np.isfinite(x)], 1e-300, 1.0)
        if len(p) == 0: return np.nan
        stat = -2 * np.sum(np.log(p))
        return float(stats.chi2.sf(stat, 2 * len(p)))

    def _run_per_env_lmm_cv(self, gene_idx: int, KFOLDS: int = 3) -> Dict[str, float]:
        out = {"delta_r2_WW": np.nan, "delta_r2_WS1": np.nan, "delta_r2_WS2": np.nan,
               "lr_p_WW": np.nan, "lr_p_WS1": np.nan, "lr_p_WS2": np.nan}
        env_codes = {0:"WW", 1:"WS1", 2:"WS2"}
        E = self.E.cpu().numpy(); groups = np.asarray(self.groups)
        for ecode, ename in env_codes.items():
            idx_all = np.where(E == ecode)[0]
            if idx_all.size < 50:
                continue
            
            try:
                gkf = GroupKFold(n_splits=min(KFOLDS, len(np.unique(groups[idx_all]))))
            except ValueError:
                continue

            deltas, lrp = [], []
            for tr_loc, te_loc in gkf.split(idx_all, groups=groups[idx_all]):
                tr = idx_all[tr_loc]; te = idx_all[te_loc]
                top_snps = self.get_top_cis_snps(gene_idx, tr, top_k=5)
                if not top_snps:
                    deltas.append(np.nan); lrp.append(np.nan); continue
                df_tr = self._make_df(tr, gene_idx, top_snps)
                df_te = self._make_df(te, gene_idx, top_snps)
                y_te = df_te["expression"].values
                # fast/stable OLS-FE on CV splits
                _, yhat_null = self._ols_fe_fit_predict(df_tr, df_te, include_snps=False)
                _, yhat_full = self._ols_fe_fit_predict(df_tr, df_te, include_snps=True)
                r2n = r2_safe(y_te, yhat_null); r2f = r2_safe(y_te, yhat_full)
                deltas.append(r2f - r2n)
                # optional: LR p on TRAIN ONLY (diagnostic)
                try:
                    y_tr_fit, Xn, g_tr_fit, _ = self._build_numeric_design(df_tr, include_snps=False)
                    _, Xf, _, _ = self._build_numeric_design(df_tr, include_snps=True)
                    md0 = sm.MixedLM(y_tr_fit, Xn, groups=g_tr_fit)
                    md1 = sm.MixedLM(y_tr_fit, Xf, groups=g_tr_fit)
                    f0 = md0.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
                    f1 = md1.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
                    if f0.converged and f1.converged and (Xf.shape[1] > Xn.shape[1]):
                        lr = 2*(f1.llf - f0.llf)
                        lrp.append(float(stats.chi2.sf(max(lr,0.0), Xf.shape[1]-Xn.shape[1])))
                    else:
                        lrp.append(np.nan)
                except Exception:
                    lrp.append(np.nan)
            out[f"delta_r2_{ename}"] = float(np.nanmedian(deltas))
            out[f"lr_p_{ename}"] = float(np.nanmedian(lrp)) if len(lrp) else np.nan
        return out

    # ------------------------ driver ------------------------

    def run(
        self,
        KFOLDS: int = 5,
        n_bootstrap: int = 200,
        n_permutations: int = 1000,
        variance_threshold: float = 0.05,
        n_jobs: int = 32,
    ) -> pd.DataFrame:
        n_jobs = min(n_jobs, os.cpu_count() or 1)

        # build the list of gene indices to process
        total_genes = self.Y.shape[1]
        gene_indices = list(range(total_genes))

        cap = None
        if getattr(self, "args", None) is not None:
            if self.args.gene_cap is not None:
                cap = int(self.args.gene_cap)
            elif getattr(self.args, "smoke", False):
                # env var fallback if --gene-cap not provided
                cap = int(os.environ.get("SMOKE_MAX_GENES", "20"))

        if cap is not None and cap > 0 and cap < total_genes:
            gene_indices = gene_indices[:cap]

        print(f"Running robust LMM on {len(gene_indices)}/{total_genes} genes using {n_jobs} workers...")

        # Pre-allocate OOF arrays in parent
        n, g = self.Y.shape
        self.oof_null = np.full((n, g), np.nan, dtype=float)
        self.oof_full = np.full((n, g), np.nan, dtype=float)

        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
            delayed(self.analyze_gene)(g, KFOLDS, n_bootstrap, n_permutations, variance_threshold)
            for g in tqdm(gene_indices, desc="Genes")
        )

        summary_rows = []
        self.fit_audit, self.permutation_log = [], []
        for res in results:
            if res:
                j = res["gene_idx"]
                self.oof_null[:, j] = res["oof_null_vec"]
                self.oof_full[:, j] = res["oof_full_vec"]
                summary_rows.append(res["summary_row"])
                self.fit_audit.extend(res.get("audit_rows", []))
                self.permutation_log.extend(res.get("perm_logs", []))

        df = pd.DataFrame(summary_rows)
        if df.empty:
            print("No results produced.")
            return df

        # FDRs
        for col, out in [
            ("pvalue", "pvalue_fdr"),
            ("empirical_pvalue", "empirical_pvalue_fdr"),
            ("empirical_pvalue_combined", "empirical_pvalue_combined_fdr"),
        ]:
            if col in df.columns:
                _, df[out], _, _ = multipletests(df[col].fillna(1.0), method="fdr_bh")

        df["significant_lr"] = df.get("pvalue_fdr", pd.Series(np.nan, index=df.index)) < 0.05
        df["significant_empirical"] = df.get("empirical_pvalue_fdr", pd.Series(np.nan, index=df.index)) < 0.05
        df["significant_empirical_combined"] = df.get("empirical_pvalue_combined_fdr", pd.Series(np.nan, index=df.index)) < 0.05
        df["improved"] = df["delta_r2"] > 1e-3

        # Save
        out_tables = self.output_dir / "tables"
        out_logs = self.output_dir / "logs"
        
        # Atomic save
        temp_file = out_tables / f"robust_lmm_comprehensive_results_{os.getpid()}.tmp"
        df.to_csv(temp_file, index=False)
        os.replace(temp_file, out_tables / "robust_lmm_comprehensive_results.csv")

        if self.permutation_log:
            pd.DataFrame(self.permutation_log).to_csv(out_logs / "permutation_log.csv", index=False)

        # Residual and audit analysis
        out_res = self.output_dir / "residual_analysis"; out_res.mkdir(exist_ok=True)
        Y = self.Y.cpu().numpy(); genes = [str(x) for x in self.gene_names]

        pd.DataFrame(self.oof_null, columns=genes).to_csv(out_res/"lmm_oof_pred_null_500g.csv", index=False)
        pd.DataFrame(Y - self.oof_null, columns=genes).to_csv(out_res/"lmm_residuals_null_500g.csv", index=False)

        pd.DataFrame(self.oof_full, columns=genes).to_csv(out_res/"lmm_oof_pred_full_500g.csv", index=False)
        pd.DataFrame(Y - self.oof_full, columns=genes).to_csv(out_res/"lmm_residuals_full_500g.csv", index=False)

        if self.fit_audit:
            audit_df = pd.DataFrame(self.fit_audit)
            audit_df.to_csv(out_res/"lmm_fit_audit.csv", index=False)
            audit_df.to_csv(out_logs / "convergence_log.csv", index=False)
            
            summary = (audit_df.groupby(["model","method"]).size().rename("n").reset_index())
            summary.to_csv(out_res/"lmm_fit_audit_summary.csv", index=False)
            
            per_gene_null = (audit_df[audit_df["model"]=="NULL"]
                             .assign(ols=lambda d: (d["method"]=="OLS_FE").astype(int))
                             .groupby("gene_id")["ols"].mean().reset_index(name="null_ols_frac"))
            per_gene_null.to_csv(out_res/"lmm_per_gene_null_fallback.csv", index=False)
            
            null_total = (audit_df["model"]=="NULL").sum()
            null_ols = ((audit_df["model"]=="NULL") & (audit_df["method"]=="OLS_FE")).sum()
            rate = (null_ols/null_total) if null_total else 0.0
            print(f"[AUDIT] NULL MixedLM→OLS_FE fallbacks: {null_ols}/{null_total} ({rate:.1%})")

        self._print_summary(df)
        self._print_transfer(df)
        return df

    # ------------------------ reporting ------------------------

    @staticmethod
    def _print_summary(df: pd.DataFrame):
        print("\n" + "=" * 80)
        print("ROBUST ANALYSIS SUMMARY (numeric-only)")
        print("=" * 80)
        
        d = df["delta_r2"].to_numpy(float)
        finite = np.isfinite(d)
        print(f"  Median ΔR²:      {float(np.nanmedian(d)):.4f}")
        
        if finite.sum() >= 30:
            print(f"  10%-trim ΔR²:    {float(trim_mean(d[finite], 0.10)):.4f}")
        else:
            print(f"  10%-trim ΔR²:    {float(np.nanmedian(d)):.4f}  (N too small for trimmed mean)")
            
        print(f"  Median nRMSE(Null): {float(np.nanmedian(df['nrmse_null'])):.4f}")
        print(f"  Median nRMSE(Full): {float(np.nanmedian(df['nrmse_full'])):.4f}")
        print(f"  Median ΔnRMSE:      {float(np.nanmedian(df['delta_nrmse'])):.4f}")
        
        n_lowvar = df["n_low_variance_folds"].sum()
        print(f"\n  Low-variance test folds skipped: {int(n_lowvar)}")
        print(f"  LR FDR<0.05:     {int(np.nansum(df['significant_lr']))}/{len(df)}")
        print(f"  Emp FDR<0.05:    {int(np.nansum(df['significant_empirical']))}/{len(df)}")
        if 'significant_empirical_combined' in df.columns:
            print(f"  Emp (combined) FDR<0.05: {int(np.nansum(df['significant_empirical_combined']))}/{len(df)}")
        print(f"  Improved (>0.001): {int(np.nansum(df['improved']))}/{len(df)}")

    @staticmethod
    def _print_transfer(df: pd.DataFrame):
        print("\n" + "=" * 80)
        print("TRANSFER ANALYSIS (gene-level means)")
        print("=" * 80)
        for col in ["transfer_WW_to_WS1", "transfer_WS1_to_WS2", "transfer_WS2_to_WW"]:
            if col in df.columns:
                arr = df[col].values.astype(float)
                # 10% trimmed mean (robust)
                a = arr[np.isfinite(arr)]
                if a.size:
                    med   = float(np.nanmedian(arr))
                    pos   = float(np.nanmean(arr > 1e-3) * 100) if np.isfinite(arr).any() else 0.0
                    
                    if a.size >= 30:
                        lo = np.quantile(a, 0.10); hi = np.quantile(a, 0.90)
                        a_trim = a[(a >= lo) & (a <= hi)]
                        tmean = float(np.mean(a_trim)) if a_trim.size else float(np.nan)
                        tmean_str = f"trimmed-mean {tmean:.4f}"
                    else:
                        tmean_str = f"trimmed-mean {med:.4f} (N<{30})"

                    print(
                        f"  {col.replace('transfer_','').replace('_to_',' → ')}: "
                        f"median {med:.4f}, {tmean_str}, pos% {pos:.1f}%"
                    )
                else:
                    print(
                        f"  {col.replace('transfer_','').replace('_to_',' → ')}: median nan, trimmed-mean nan, pos% 0.0%"
                    )


def get_dynamic_gene_count_from_metadata() -> Optional[int]:
    """
    Extract gene count from prepare_data_5e.py output metadata.
    Returns the actual number of genes selected by the data preparation pipeline.
    """
    base_dir = Path(r"C:\Users\ms\Desktop\gwas\output")

    # Check preparation manifest first (most reliable)
    manifest_file = base_dir / "preparation_manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, "r") as f:
                manifest = json.load(f)
            gene_count = manifest.get("n_genes")
            if gene_count and isinstance(gene_count, int):
                print(f"Found {gene_count} genes from preparation manifest")
                return gene_count
        except Exception as e:
            print(f"Warning: Could not read preparation manifest: {e}")

    # Fallback: Check gene selection summary from use_working_genes_2.py
    summary_file = base_dir / "working_gene_selection_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r") as f:
                summary = json.load(f)
            gene_count = summary.get("n_genes_selected")
            if gene_count and isinstance(gene_count, int):
                print(f"Found {gene_count} genes from working gene selection summary")
                return gene_count
        except Exception as e:
            print(f"Warning: Could not read gene selection summary: {e}")

    # Fallback: Count lines in gene list file
    gene_list_patterns = ["transformer_gene_set_*.csv", "working_gene_selection_scores.csv"]

    for pattern in gene_list_patterns:
        gene_files = list(base_dir.glob(pattern))
        if gene_files:
            try:
                import pandas as pd

                gene_df = pd.read_csv(gene_files[0])
                gene_count = len(gene_df)
                if gene_count > 0:
                    print(f"Detected {gene_count} genes from {gene_files[0].name}")
                    return gene_count
            except Exception as e:
                print(f"Warning: Could not read gene file {gene_files[0]}: {e}")

    return None


def find_data_file():
    """
    Enhanced data file detection with dynamic gene count support.
    Automatically detects the correct number of genes from prepare_data_5e.py output.
    """
    base_dir = Path(r"C:\Users\ms\Desktop\gwas\output")

    # Get gene selector from environment or args
    gene_selector = os.getenv("GENE_SELECTOR", getattr(args, "gene_selector", None) or "drought")

    # Dynamically detect gene count
    detected_gene_count = get_dynamic_gene_count_from_metadata()

    # Build candidate file list with dynamic gene count
    candidates = []

    if detected_gene_count:
        # Primary candidates with detected gene count
        candidates.extend(
            [
                base_dir / "transformer_data" / f"transformer_data_{detected_gene_count}_genes_{gene_selector}.pt",
                base_dir / "transformer_data" / f"transformer_data_{detected_gene_count}_genes.pt",
            ]
        )

    # Fallback candidates with common gene counts
    for gene_count in [100, 50, 75, 150, 200]:  # Common gene counts
        if gene_count != detected_gene_count:  # Avoid duplicates
            candidates.extend(
                [
                    base_dir / "transformer_data" / f"transformer_data_{gene_count}_genes_{gene_selector}.pt",
                    base_dir / "transformer_data" / f"transformer_data_{gene_count}_genes.pt",
                ]
            )

    # Legacy hardcoded patterns for backward compatibility
    candidates.extend(
        [
            base_dir / "transformer_data" / "transformer_data_100_genes.pt",
            base_dir / "transformer_data" / "transformer_data_100_genes_drought.pt",
            base_dir / "transformer_data" / "transformer_data_100_genes_baseline.pt",
        ]
    )

    # Search for existing files
    for candidate in candidates:
        if candidate.exists():
            print(f"Found data file: {candidate}")

            # Validate gene count matches detected count
            if detected_gene_count:
                try:
                    import torch

                    data = torch.load(candidate, map_location="cpu")
                    actual_genes = data["Y"].shape[1] if "Y" in data else None

                    if actual_genes and actual_genes != detected_gene_count:
                        print(f"Warning: File contains {actual_genes} genes but expected {detected_gene_count}")
                        print(f"This may indicate a version mismatch. Proceeding with {candidate}")
                    else:
                        print(f"Validated: File contains {actual_genes} genes as expected")

                except Exception as e:
                    print(f"Warning: Could not validate gene count in {candidate}: {e}")

            return str(candidate)

    # Enhanced error reporting
    available_files = []
    if (base_dir / "transformer_data").exists():
        available_files = list((base_dir / "transformer_data").glob("*.pt"))

    error_msg = f"No valid data file found.\n"
    error_msg += f"Detected gene count: {detected_gene_count or 'Unknown'}\n"
    error_msg += f"Gene selector: {gene_selector}\n\n"
    error_msg += f"Searched locations:\n"

    for candidate in candidates[:5]:  # Show first 5 candidates
        error_msg += f"  - {candidate} {'(exists)' if candidate.exists() else '(missing)'}\n"

    if len(candidates) > 5:
        error_msg += f"  ... and {len(candidates) - 5} other candidates\n"

    if available_files:
        error_msg += f"\nAvailable .pt files in transformer_data:\n"
        for f in available_files:
            error_msg += f"  - {f}\n"
        error_msg += f"\nUse --data-file to specify a custom path."
    else:
        error_msg += f"\nNo .pt files found in {base_dir / 'transformer_data'}"
        error_msg += f"\nRun prepare_data_5e.py first to generate data files."

    raise FileNotFoundError(error_msg)


def setup_output_directory(output_dir: str):
    """Setup output directory structure."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "tables").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)
    return output_path

def main():
    print("=" * 80)
    print("ROBUST LMM — NUMERIC PATH (Updated for prepare_data_5e.py)")
    print("=" * 80)
    print("Python:", sys.version)
    print("Platform:", platform.platform())
    
    # Determine data file path
    if args.data_file:
        data_path = args.data_file
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Specified data file not found: {data_path}")
        print(f"Using specified data file: {data_path}")
    else:
        data_path = find_data_file()
        print(f"Auto-detected data file: {data_path}")
    
    # Setup output directory
    out_dir = setup_output_directory(args.output_dir)
    print(f"Output directory: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Create analyzer with robust data loading
    try:
        analyzer = RobustLMMAnalyzer(data_path, str(out_dir), device, args)
    except Exception as e:
        print(f"\nERROR: Failed to load data from {data_path}")
        print(f"Error details: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Ensure prepare_data_5e.py has been run successfully")
        print(f"2. Check that the .pt file contains required keys: Y, G, PCs, E, gene_snp_mask")
        print(f"3. Try specifying a different file with --data-file")
        raise

    # Analysis parameters
    KFOLDS = 5
    SEEDS = [42, 101, 202]
    n_bootstrap = 50  # 200
    n_permutations = args.permutations
    variance_threshold = 0.05
    n_jobs = 32

    if args.smoke:
        KFOLDS = 3
        SEEDS = [42]
        n_bootstrap = 20
        n_permutations = 10
        print("\n*** SMOKE TEST MODE ACTIVATED ***")
        # log overrides:
        smoke_out = Path("output")
        smoke_out.mkdir(parents=True, exist_ok=True)
        with open(smoke_out / "smoke_overrides.json", "w") as f:
            log_data = {
                "when": datetime.now().isoformat(timespec="seconds"),
                "script": "lmm_9.py",
                "data_file": data_path,
                "output_dir": str(out_dir),
                "kfolds": KFOLDS,
                "n_bootstrap": n_bootstrap,
                "n_permutations": n_permutations,
                "cis_window_scale": args.cis_window_scale,
                "permute_within_fold": args.permute_within_fold,
            }
            json.dump(log_data, f, indent=2)
            print("Smoke test overrides logged to output/smoke_overrides.json")

    print(f"\nRunning analysis with:")
    print(f"  Data file: {data_path}")
    print(f"  Output: {out_dir}")
    print(f"  K-folds: {KFOLDS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Permutations: {n_permutations}")

    # Run analysis
    df = analyzer.run(
        KFOLDS=KFOLDS,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        variance_threshold=variance_threshold,
        n_jobs=n_jobs,
    )

    if args.per_env_lmm and not df.empty:
        print("\nApplying stress-triggered target filter...")
        primary_gate = (df["delta_r2_WS2"] >= 0.02) | (df["delta_r2_WS1"] >= 0.02)
        baseline_gate = df["delta_r2_WW"] <= 0.01
        
        df_filtered = df[primary_gate & baseline_gate].copy()
        
        print(f"  Initial genes: {len(df)}")
        print(f"  Genes passing primary gate (stress delta_r2 >= 0.02): {primary_gate.sum()}")
        print(f"  Genes passing baseline gate (WW delta_r2 < 0.01): {baseline_gate.sum()}")
        print(f"  Genes passing both gates: {len(df_filtered)}")
        
        out_file = out_dir / "tables" / "robust_lmm_stress_triggered_hits.csv"
        df_filtered.to_csv(out_file, index=False)
        print(f"  Filtered results saved to: {out_file}")

    print("\nDone. Outputs:")
    print(f"  - {out_dir}/tables/robust_lmm_comprehensive_results.csv")
    if args.per_env_lmm and not df.empty:
        print(f"  - {out_dir}/tables/robust_lmm_stress_triggered_hits.csv")
    print(f"  - {out_dir}/logs/convergence_log.csv")
    print(f"  - {out_dir}/logs/permutation_log.csv")

    # Save package versions for reproducibility
    ver = Path(r"C:\Users\ms\Desktop\gwas\output\versions.txt")
    if not ver.exists():
        try:
            import subprocess
            ver.write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
        except Exception as e:
            ver.write_text(f"pip freeze failed: {e}\n")

    return df


if __name__ == "__main__":
    main()



