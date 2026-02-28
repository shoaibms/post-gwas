#!/usr/bin/env python3
"""
Environment-Conditional Transformer for eQTL Discovery
======================================================
Purpose: Implement environment-conditional attention mechanism for regulatory
         discovery across drought stress conditions (WW, WS1, WS2)
Author: Environment-Conditional eQTL Analysis Pipeline
Date: 2025-01-09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import math
from pathlib import Path
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from joblib import Parallel, delayed
import argparse
from itertools import combinations
import warnings
from os import cpu_count
import json
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import os
import time
import shutil


PENALTY_SCORE = -1e6  # bad score to penalize failed/missing-output trials

def nanmean_or(arr, axis=0, fallback=None):
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return fallback
    finite = np.isfinite(arr)
    if not finite.any():
        return fallback
    with np.errstate(all="ignore"):
        return np.nanmean(arr, axis=axis)


def _safe_alpha(uplift_val, k=12.0):
    if not np.isfinite(uplift_val) or uplift_val <= 0.0:
        return 0.0
    return 1.0 / (1.0 + np.exp(-k * uplift_val))


def _spearman(a, b):
    # SciPy-free Spearman
    ax = pd.Series(a).rank()
    bx = pd.Series(b).rank()
    return float(ax.corr(bx, method="pearson"))

def _load_lmm_table(path):
    df = pd.read_csv(path)
    # prefer gene_name, else gene_id, else first col
    join_key = "gene_name" if "gene_name" in df.columns else ("gene_id" if "gene_id" in df.columns else df.columns[0])
    df[join_key] = df[join_key].astype(str).str.upper()
    # ensure per-env columns exist or set NaN
    for c in ["delta_r2_WW","delta_r2_WS1","delta_r2_WS2"]:
        if c not in df.columns: df[c] = np.nan
    return df, join_key

def _score_trial(out_dir, lmm_csv=None):
    """
    Score a single trial directory.
    Looks for ect_oof_r2_by_gene.csv. If missing, tries a few fallbacks.
    Returns (score, details_dict) without raising to keep Optuna running.
    """
    from pathlib import Path
    import json, shutil
    import pandas as pd
    import numpy as np
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_csv = out_dir / "ect_oof_r2_by_gene.csv"

    # Fallback 1: sometimes outputs are written to the trial parent (buggy trainers).
    if not oof_csv.exists():
        parent_oof = out_dir.parent / "ect_oof_r2_by_gene.csv"
        if parent_oof.exists():
            try:
                shutil.copy(parent_oof, oof_csv)
            except Exception:
                pass

    # Fallback 2: synthesize from a summary json if present
    if not oof_csv.exists():
        summary = out_dir / "ect_summary.json"
        if summary.exists():
            try:
                js = json.loads(summary.read_text())
                # Accept either list-of-dicts or dict-of-lists
                if "oof_r2_by_gene" in js:
                    df = pd.DataFrame(js["oof_r2_by_gene"])
                elif "oof" in js and isinstance(js["oof"], dict):
                    df = pd.DataFrame(js["oof"])
                else:
                    df = None
                if df is not None and not df.empty:
                    df.to_csv(oof_csv, index=False)
            except Exception:
                pass

    # If still missing, return a penalty (do not crash the study)
    if not oof_csv.exists():
        return PENALTY_SCORE, {
            "status": "missing_oof_csv",
            "expected": str(oof_csv),
            "hint": "Ensure training writes ect_oof_r2_by_gene.csv into each trial directory."
        }

    # Read and compute a robust score
    df = pd.read_csv(oof_csv)
    cols = {c.lower(): c for c in df.columns}

    # case A: full & null columns => median ΔR²
    if "r2_full" in cols and "r2_null" in cols:
        d = df[cols["r2_full"]] - df[cols["r2_null"]]
        score = np.nanmedian(d.values)

    # case B: just r2 column => median R²
    elif "r2" in cols:
        score = np.nanmedian(df[cols["r2"]].values)

    else:
        # No recognizable columns; penalize but don't crash
        return PENALTY_SCORE, {
            "status": "bad_oof_schema",
            "columns": list(df.columns),
            "hint": "Expect r2_full/r2_null or r2."
        }

    details = {
        "status": "ok",
        "oof_path": str(oof_csv),
        "n_genes": int(len(df)),
        "score_metric": "median_delta_r2" if ("r2_full" in cols and "r2_null" in cols) else "median_r2"
    }

    # (Optional) If you want to fold LMM reference here, join by gene_id and recompute
    # if lmm_csv and Path(lmm_csv).exists() and "gene_id" in cols:
    #     ...

    return float(score), details


class CisAttentionLayer(nn.Module):
    """
    Cis-constrained attention layer with environment conditioning
    """
    def __init__(self, 
                 d_snp: int,
                 d_model: int,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 temperature: float = 1.0):
        super().__init__()
        
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.temperature = temperature
        
        # Multi-head projections
        self.w_q = nn.Linear(d_snp, d_model)
        self.w_k = nn.Linear(d_snp, d_model)
        self.w_v = nn.Linear(d_snp, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, 
                queries: torch.Tensor,
                keys_values: torch.Tensor,
                dq: torch.Tensor,
                dk: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                cis_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            queries: [batch_size, n_genes, d_model] gene query embeddings
            keys_values: [batch_size, n_snps, d_model] SNP key/value embeddings
            dq: [batch, 1, d_model] environment shift for queries
            dk: [batch, 1, d_model] environment shift for keys
            mask: [batch_size, n_snps] binary mask for valid SNPs (1=valid, 0=invalid)
            cis_mask: [n_genes, n_snps] boolean mask for cis-SNPs (True=cis)
            return_attention: Whether to return attention weights
        Returns:
            output: [batch_size, n_genes, d_model] aggregated features for each gene
            attention_weights: [batch_size, n_heads, n_genes, n_snps] if return_attention=True
        """
        batch_size, n_genes, _ = queries.shape
        _, n_snps, _ = keys_values.shape
        
        # Project to Q, K, V
        Q = self.w_q(queries)
        K = self.w_k(keys_values)
        V = self.w_v(keys_values)
        
        # Additive environment shifts
        Q = Q + dq
        K = K + dk
        
        Q = Q.view(batch_size, n_genes, self.n_heads, self.d_k)
        K = K.view(batch_size, n_snps, self.n_heads, self.d_k)
        V = V.view(batch_size, n_snps, self.n_heads, self.d_k)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, n_heads, n_genes, d_k]
        K = K.transpose(1, 2)  # [batch, n_heads, n_snps, d_k]
        V = V.transpose(1, 2)  # [batch, n_heads, n_snps, d_k]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.temperature * math.sqrt(self.d_k))
        # Shape: [batch, n_heads, n_genes, n_snps]
        
        # Apply cis mask if provided
        if cis_mask is not None:
            # broadcast [1, 1, n_genes, n_snps] to batch and heads
            attn_mask = (~cis_mask).unsqueeze(0).unsqueeze(0).to(scores.device) # True where we want to block
            scores = scores.masked_fill(attn_mask, -1e9)

        # Apply SNP validity mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, n_snps]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [batch, n_heads, n_genes, d_k]
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, n_genes, self.d_model)
        output = self.w_o(attn_output)
        output = self.layer_norm(output)
        
        if return_attention:
            return output, attn_weights
        return output, None


class EnvironmentConditionalTransformer(nn.Module):
    """
    Main transformer model for environment-conditional eQTL discovery
    """
    def __init__(self,
                 n_snps: int,
                 n_genes: int,
                 n_pcs: int = 20,
                 n_envs: int = 3,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 mc_dropout: bool = True,
                 residual_mode: bool = True,
                 use_film: bool = True):
        super().__init__()
        
        self.n_snps = n_snps
        self.n_genes = n_genes
        self.n_pcs = n_pcs
        self.n_envs = n_envs
        self.mc_dropout = mc_dropout
        self.d_model = d_model
        self.residual_mode = residual_mode
        self.use_film = use_film
        
        # --- cis mask ---
        # This buffer will store a boolean mask of shape [n_genes, n_snps]
        # where mask[i, j] is True if SNP j is in the cis-window of gene i.
        self.register_buffer("gene_snp_mask", torch.ones(n_genes, n_snps, dtype=torch.bool))

        # SNP encoder: projects genotypes + PCs to feature space
        self.snp_encoder = nn.Sequential(
            nn.Linear(1 + n_pcs, 64),  # 1 for genotype dosage + n_pcs
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout if not mc_dropout else 0),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Environment embedding and conditioning
        self.env_emb   = nn.Embedding(n_envs, d_model)
        self.env_to_q  = nn.Linear(d_model, d_model, bias=False)
        self.env_to_k  = nn.Linear(d_model, d_model, bias=False)
        
        # optional baseline (unused in residual_mode)
        self.baseline_head = nn.Linear(self.n_pcs + d_model, self.n_genes) if not residual_mode else None
        if self.baseline_head is not None:
            nn.init.zeros_(self.baseline_head.weight)
            nn.init.zeros_(self.baseline_head.bias)

        # FiLM layers
        self.env_gamma = nn.Linear(d_model, self.d_model, bias=True)
        self.env_beta  = nn.Linear(d_model, self.d_model, bias=True)

        # Stacked attention layers
        self.attention_layers = nn.ModuleList([
            CisAttentionLayer(
                d_snp=d_model,
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout if not mc_dropout else 0,
                temperature=1.0
            ) for _ in range(n_layers)
        ])
        
        # Gene-specific prediction head (unified for efficiency)
        self.gene_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),  # Always apply dropout here for MC-dropout
            nn.Linear(64, 1)
        )
        
        # Optional: learnable gene embeddings for gene-specific processing
        self.gene_embeds = nn.Embedding(n_genes, d_model)
        
        # Initialize weights
        self._init_weights()
        
    def set_cis_mask(self, mask: torch.Tensor):
        """
        Set the gene-SNP cis mask.

        Args:
            mask: A boolean tensor of shape [n_genes, n_snps] where True indicates
                  a SNP is in the cis-window of a gene.
        """
        if mask.shape != (self.n_genes, self.n_snps):
            raise ValueError(f"Mask shape must be [{self.n_genes}, {self.n_snps}]")
        self.gene_snp_mask = mask.to(self.gene_snp_mask.device)

    def _init_weights(self):
        """Initialize model weights using Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                
    def encode_snps(self, 
                    genotypes: torch.Tensor, 
                    pcs: torch.Tensor) -> torch.Tensor:
        """
        Encode SNP features combining genotypes and PCs
        
        Args:
            genotypes: [batch_size, n_snps] genotype dosages
            pcs: [batch_size, n_pcs] principal components
        Returns:
            [batch_size, n_snps, d_model] encoded SNP features
        """
        batch_size = genotypes.shape[0]
        
        # Expand PCs to match each SNP
        pcs_expanded = pcs.unsqueeze(1).expand(-1, self.n_snps, -1)
        
        # Combine genotype and PCs for each SNP
        genotypes_expanded = genotypes.unsqueeze(-1)
        snp_input = torch.cat([genotypes_expanded, pcs_expanded], dim=-1)
        
        # Encode each SNP
        snp_features = self.snp_encoder(snp_input)
        
        return snp_features
    
    def forward(self, 
                genotypes: torch.Tensor,
                pcs: torch.Tensor,
                env_indices: torch.Tensor,
                return_attention: bool = False,
                cis_mask: Optional[torch.Tensor] = "use_internal") -> Dict[str, torch.Tensor]:
        """
        Forward pass of the transformer
        
        Args:
            genotypes: [batch_size, n_snps] genotype dosages
            pcs: [batch_size, n_pcs] principal components
            env_indices: [batch_size] environment indices (0=WW, 1=WS1, 2=WS2)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - predictions: [batch_size, n_genes] expression predictions
                - attention_weights: List of attention weights from each layer (if requested)
        """
        batch_size = genotypes.shape[0]
        
        # Encode SNPs
        snp_features = self.encode_snps(genotypes, pcs)
        
        # Get environment embeddings
        env = self.env_emb(env_indices)
        
        # compute baseline only if not residual mode
        baseline = (self.baseline_head(torch.cat([pcs, env], dim=-1))
                    if self.baseline_head is not None else None)
        
        # Environment-specific shifts for Q and K
        dq = self.env_to_q(env).unsqueeze(1)
        dk = self.env_to_k(env).unsqueeze(1)
        
        # Create mask for valid SNPs (non-zero genotypes)
        # No per-SNP mask after scaling; union selection has already cleaned inputs
        mask = None
        
        # Get gene queries
        gene_queries = self.gene_embeds.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # Process through attention layers
        hidden = gene_queries
        attention_weights_all = []
        
        mask_to_use = self.gene_snp_mask if cis_mask == "use_internal" else cis_mask

        for layer in self.attention_layers:
            layer_output, attn_weights = layer(
                hidden, snp_features, dq, dk, mask, mask_to_use, return_attention
            )
            
            # Residual connection
            hidden = hidden + layer_output
            
            if return_attention and attn_weights is not None:
                attention_weights_all.append(attn_weights)
        
        # Final representation is now per-gene
        final_repr = hidden  # [batch_size, n_genes, d_model]

        if self.use_film:
            gamma = self.env_gamma(env).unsqueeze(1)  # [B,1,D]
            beta  = self.env_beta(env).unsqueeze(1)   # [B,1,D]
            final_repr = final_repr * (1.0 + torch.tanh(gamma)) + beta
        
        # Gene-specific predictions (batched)
        # Reshape for efficient processing with a single head
        final_repr_flat = final_repr.view(-1, self.d_model) # [batch_size * n_genes, d_model]
        
        # Apply the unified gene head
        gene_preds_flat = self.gene_head(final_repr_flat) # [batch_size * n_genes, 1]
        
        # Reshape back to per-gene predictions
        predictions = gene_preds_flat.view(batch_size, self.n_genes) # [batch_size, n_genes]

        if baseline is not None:
            predictions = predictions + baseline
        
        return {"predictions": predictions, "attention_weights": attention_weights_all} if return_attention \
               else {"predictions": predictions}
    
# ---------- CONFIGURATION ----------
DEFAULTS = dict(
    KFOLDS=5,
    SEEDS=[42],  # Single seed for testing
    EPOCHS=250,
    PATIENCE=30,
    MC_SAMPLES=16,
    TOPK=5,
    UNION_CAP=1200,  # CRITICAL: Change from 800 to 1200
    D_MODEL=128,
    LAYERS=2,
    HEADS=4,
    DROPOUT=0.1,
    LR=1e-3,
    WD=0.005,
    LOSS_CF_WEIGHT=0.3  # Change from 0.5 to 0.3
)

# Keys in the .pt file (maps script's expected names -> stored names)
KEYS = {
    "G": "G",
    "Y": "Y",
    "E": "E",
    "PCs": "PCs",
    "groups": "sample_names",   # <- IMPORTANT: your bundle uses 'accession_ids'
    "cis_map": "gene_snp_mask"
}

def resolve_config(args):
    cfg = DEFAULTS.copy()
    if args.smoke:
        cfg.update(KFOLDS=3, SEEDS=[42], EPOCHS=50, PATIENCE=10, MC_SAMPLES=8)
        if hasattr(args, 'keep_genes'):
             # If keep_genes is part of args, we might want to override it in smoke mode.
             # note.txt doesn't specify this, using a smaller value for smoke.
             args.keep_genes = 20
    if hasattr(args, 'n_pcs') and args.n_pcs is not None:
        cfg['N_PCS'] = args.n_pcs
    if hasattr(args, 'top_k_cis') and args.top_k_cis is not None:
        cfg['TOPK'] = args.top_k_cis
    if hasattr(args, 'union_cap') and args.union_cap is not None:
        cfg['UNION_CAP'] = args.union_cap
    # ----- CLI overrides (dropout, weight decay) -----
    # We set multiple common keys for robustness; your training/init code will pick up one of them.
    if hasattr(args, "dropout") and args.dropout is not None:
        try:
            dval = float(args.dropout)
        except Exception:
            raise SystemExit("[ARG ERROR] --dropout must be a float.")
        # Common keys people use
        cfg["DROPOUT"] = dval
        cfg["dropout"] = dval
        cfg.setdefault("model_dropout", dval)

    if hasattr(args, "weight_decay") and args.weight_decay is not None:
        try:
            wdval = float(args.weight_decay)
        except Exception:
            raise SystemExit("[ARG ERROR] --weight-decay must be a float.")
        # Common keys people use
        cfg["WD"] = wdval
        cfg["WEIGHT_DECAY"] = wdval
        cfg["weight_decay"] = wdval
    # NEW: Loss CF weight handling
    if hasattr(args, "loss_cf_weight") and args.loss_cf_weight is not None:
        try:
            cfval = float(args.loss_cf_weight)
        except Exception:
            raise SystemExit("[ARG ERROR] --loss-cf-weight must be a float.")
        cfg["LOSS_CF_WEIGHT"] = cfval
        cfg["loss_cf_weight"] = cfval
    # --- NEW: fold/seed overrides to align with LMM ---
    if hasattr(args, "kfolds") and args.kfolds is not None:
        if args.kfolds < 2:
            raise SystemExit("[ARG ERROR] --kfolds must be >= 2.")
        cfg["KFOLDS"] = int(args.kfolds)
    if hasattr(args, "seed") and args.seed is not None:
        cfg["SEEDS"] = [int(args.seed)]
        
    # PATCH: Consume new CLI arguments for model capacity and training
    if getattr(args, "layers", None) is not None:
        cfg["LAYERS"] = int(args.layers)
    if getattr(args, "d_model", None) is not None:
        cfg["D_MODEL"] = int(args.d_model)
    if getattr(args, "mc_samples", None) is not None:
        cfg["MC_SAMPLES"] = int(args.mc_samples)
    if getattr(args, "patience", None) is not None:
        cfg["PATIENCE"] = int(args.patience)
        
    return cfg
# -----------------------------------

def load_data(p, keys):
    obj = torch.load(Path(p), map_location="cpu")
    k = lambda n: keys.get(n, n)

    G   = obj[k("G")].float().numpy()
    Y   = obj[k("Y")].float().numpy()
    E   = obj[k("E")].long().numpy()
    PCs = obj.get(k("PCs"), None)
    PCs = None if PCs is None else PCs.float().numpy()

    # --- robust 'groups' pull (supports both 'accession_ids' and 'groups') ---
    grp_raw = obj.get(k("groups"), obj.get("accession_ids", obj.get("groups")))
    if grp_raw is None:
        raise KeyError(
            f"Could not find accession/group IDs. Keys present: {list(obj.keys())}. "
            f"Tried {k('groups')}, 'accession_ids', and 'groups'."
        )
    groups = grp_raw.numpy() if isinstance(grp_raw, torch.Tensor) else np.asarray(grp_raw)

    # cis map (tensor mask [genes x snps] or dict)
    cis = obj[k("cis_map")] if k("cis_map") in obj else obj.get("gene_snp_mask")
    if isinstance(cis, torch.Tensor):
        cis_map = {i: torch.where(cis[i].bool())[0].tolist() for i in range(cis.shape[0])}
    else:
        cis_map = {int(gi): [int(x) for x in v] for gi, v in dict(cis).items()}

    # *** NEW: pull gene identifiers from the bundle ***
    meta = obj.get("metadata", {}) or {}
    gene_ids = (meta.get("gene_ids")
                or obj.get("gene_ids")
                or obj.get("gene_names"))
    if gene_ids is not None:
        gene_ids = [str(g) for g in gene_ids]
        # also keep in meta for upstream lookups
        meta = {**meta, "gene_ids": gene_ids}
    # *** END NEW ***

    return {
        "G": G, "Y": Y, "E": E, "PCs": PCs,
        "groups": groups, "cis_map": cis_map,
        # *** NEW: surface gene IDs and meta ***
        "gene_ids": gene_ids, "meta": meta
    }

def _r2_safe(y, yhat, eps=1e-12):
    y = np.asarray(y); yhat = np.asarray(yhat)
    m = np.isfinite(yhat)
    if m.sum() < 2 or np.var(y[m]) < eps: return np.nan
    return float(r2_score(y[m], yhat[m]))

def _fit_null_and_predict(y_tr, E_tr, PCs_tr, acc_tr, E_te, PCs_te, acc_te, n_pcs):
    """
    Robust null fit with strict dtypes:
      - y + PCs cast to float
      - Env is categorical (int-coded)
      - OLS fallback if MixedLM fails or returns non-finite values
    Returns: (yhat_train, yhat_test) as float np.ndarrays
    """
    """
    Robust null fit... returns (yhat_train, yhat_test, used_fallback_bool)
    """
    # ---- coerce arrays to correct shape/dtype ----
    y_tr  = np.asarray(y_tr, dtype=float).ravel()
    E_tr  = np.asarray(E_tr, dtype=int).ravel()
    E_te  = np.asarray(E_te, dtype=int).ravel()
    acc_tr = np.asarray(acc_tr)  # labels OK as object
    acc_te = np.asarray(acc_te)

    if n_pcs:
        PCs_tr = np.asarray(PCs_tr, dtype=float)
        PCs_te = np.asarray(PCs_te, dtype=float)

    # ---- build data frames with clean dtypes ----
    df_tr = pd.DataFrame({"y": y_tr, "Acc": acc_tr, "Env": pd.Categorical(E_tr)})
    df_te = pd.DataFrame({"y": np.zeros_like(E_te, dtype=float), "Acc": acc_te, "Env": pd.Categorical(E_te)})

    if n_pcs:
        for i in range(n_pcs):
            # ensure float columns, no objects
            df_tr[f"PC{i+1}"] = PCs_tr[:, i].astype(float)
            df_te[f"PC{i+1}"] = PCs_te[:, i].astype(float)

    # replace inf with nan; drop rows with nan in y or PCs (train only)
    df_tr.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_te.replace([np.inf, -np.inf], np.nan, inplace=True)
    drop_cols = ["y"] + ([f"PC{i+1}"] * 0 if not n_pcs else [f"PC{i+1}" for i in range(n_pcs)])
    df_tr = df_tr.dropna(subset=drop_cols)

    # ---- model formula ----
    terms = ["C(Env)"] + ([f"PC{i+1}" for i in range(n_pcs)] if n_pcs else [])
    base = " + ".join(terms) if terms else "1"

    def _ols_fallback(df_tr, df_te, n_pcs):
        """
        OLS fixed-effects fallback:
          Env dummies + PCs (+ Accession FE) → strictly float matrices
          If lstsq still fails (e.g. rank deficiency with too many FE),
          drop accession FE and refit with Env + PCs only.
        """
        # --- Build parts with explicit float dtype ---
        X_env_tr = pd.get_dummies(df_tr["Env"], prefix="Env", drop_first=True, dtype=float)
        X_env_te = pd.get_dummies(df_te["Env"], prefix="Env", drop_first=True, dtype=float)

        X_parts_tr = [X_env_tr]
        X_parts_te = [X_env_te]

        if n_pcs:
            pc_cols = [f"PC{i+1}" for i in range(n_pcs)]
            X_parts_tr.append(df_tr[pc_cols].astype(float))
            X_parts_te.append(df_te[pc_cols].astype(float))

        # Keep base parts before adding accession FE
        X_parts_tr_base = list(X_parts_tr)
        X_parts_te_base = list(X_parts_te)

        # Accession fixed effects (can be wide; keep but enforce float)
        X_acc_tr = pd.get_dummies(df_tr["Acc"], prefix="Acc", drop_first=True, dtype=float)
        X_acc_te = pd.get_dummies(df_te["Acc"], prefix="Acc", drop_first=True, dtype=float)
        X_parts_tr.append(X_acc_tr)
        X_parts_te.append(X_acc_te)

        # --- Concatenate and align columns, force float everywhere ---
        X_tr = pd.concat(X_parts_tr, axis=1)
        X_te = pd.concat(X_parts_te, axis=1).reindex(columns=X_tr.columns, fill_value=0.0)

        # Add constant and convert to pure float NumPy arrays
        X_tr = sm.add_constant(X_tr, has_constant='add')
        X_te = sm.add_constant(X_te, has_constant='add')
        Xtr = X_tr.to_numpy(dtype=float, copy=False)
        Xte = X_te.to_numpy(dtype=float, copy=False)
        y   = df_tr["y"].to_numpy(dtype=float, copy=False)

        # debug (one-time): print when FE are dropped
        if Xtr.shape[1] >= Xtr.shape[0]:
            print(f"[OLS FE] warning: p={Xtr.shape[1]} >= n={Xtr.shape[0]} — may drop Acc FE")

        # Try least squares; if it fails, drop accession FE and retry
        try:
            XtX = Xtr.T @ Xtr
            Xty = Xtr.T @ y
            coef = np.linalg.solve(XtX + 1e-4 * np.eye(XtX.shape[0]), Xty)
        except Exception:
            # Rebuild without accession FE
            X_tr2 = pd.concat(X_parts_tr_base, axis=1)
            X_te2 = pd.concat(X_parts_te_base, axis=1).reindex(columns=X_tr2.columns, fill_value=0.0)
            X_tr2 = sm.add_constant(X_tr2, has_constant='add')
            X_te2 = sm.add_constant(X_te2, has_constant='add')
            Xtr, Xte = X_tr2.to_numpy(dtype=float, copy=False), X_te2.to_numpy(dtype=float, copy=False)
            XtX = Xtr.T @ Xtr
            Xty = Xtr.T @ y
            coef = np.linalg.solve(XtX + 1e-4 * np.eye(XtX.shape[0]), Xty)

        yhat_tr = Xtr @ coef
        yhat_te = Xte @ coef
        return yhat_tr, yhat_te

    used_fallback = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message=".*covariance matrix is singular.*")
            warnings.filterwarnings("ignore", message=".*on the boundary of the parameter space.*")
            res = sm.MixedLM.from_formula(f"y ~ {base}", groups=df_tr["Acc"], data=df_tr)\
                            .fit(reml=False, method="lbfgs", maxiter=200, disp=False)
        yhat_tr = np.asarray(res.predict(df_tr), dtype=float)
        yhat_te = np.asarray(res.predict(df_te), dtype=float)
        # Fallback if not converged or any non-finite predictions
        if (not getattr(res, "converged", False)
            or not np.isfinite(yhat_tr).all()
            or not np.isfinite(yhat_te).all()):
            raise RuntimeError("MixedLM unstable; use OLS FE fallback.")
        return yhat_tr, yhat_te, used_fallback
    except Exception:
        used_fallback = True
        yhat_tr_fe, yhat_te_fe = _ols_fallback(df_tr, df_te, n_pcs)
        return yhat_tr_fe, yhat_te_fe, used_fallback

def _fit_gene_null_model(gi, Y_tr_all_genes, E_tr, PCs_tr, grp_tr, E_te, PCs_te, grp_te, n_pcs):
    tr_hat, te_hat, fallback_used = _fit_null_and_predict(
        Y_tr_all_genes[:, gi], E_tr, PCs_tr, grp_tr,
        E_te, PCs_te, grp_te, n_pcs
    )
    return gi, tr_hat, te_hat, fallback_used

def ect_counterfactual_alignment(runner, G_te, PCs_te, E_te, Y_te, accessions_te, env_ids=(0,1,2)):
    # build pairs within test fold: same accession with two envs
    df = pd.DataFrame({"acc": accessions_te, "env": E_te, "idx": np.arange(len(E_te))})
    pairs = []
    for acc, sub in df.groupby("acc"):
        env_list = sorted(sub["env"].unique())
        if len(env_list) < 2: 
            continue
        for e1, e2 in combinations(env_list, 2):
            i1 = int(sub.loc[sub["env"]==e1, "idx"].iloc[0])
            i2 = int(sub.loc[sub["env"]==e2, "idx"].iloc[0])
            pairs.append((i1, e1, i2, e2))
    if not pairs: return np.nan, None

    # observed deltas and model-implied counterfactual deltas (same genotype, toggle env)
    obs = []; cf = []
    for (i1, e1, i2, e2) in pairs:
        # observed Δ = y(e2)-y(e1)
        obs.append(Y_te[i2] - Y_te[i1])
        # counterfactual Δ = yhat(G_i1, env=e2) - yhat(G_i1, env=e1)
        Gi = G_te[i1:i1+1]; Pi = None if PCs_te is None else PCs_te[i1:i1+1]
        yhat_e1, _ = runner.predict(Gi, Pi, np.array([e1]), mc_samples=0)
        yhat_e2, _ = runner.predict(Gi, Pi, np.array([e2]), mc_samples=0)
        cf.append(yhat_e2[0] - yhat_e1[0])

    obs = np.vstack(obs)   # [pairs, genes]
    cf  = np.vstack(cf)
    # per-gene correlation across pairs
    rs = []
    for g in range(obs.shape[1]):
        m = np.isfinite(obs[:,g]) & np.isfinite(cf[:,g])
        if m.sum() < 3 or np.nanstd(obs[m,g]) < 1e-6 or np.nanstd(cf[m,g]) < 1e-6:
            rs.append(np.nan); continue
        rs.append(spearmanr(obs[m,g], cf[m,g]).correlation)
    return float(np.nanmedian(rs)), np.array(rs)

def cis_mass_for_env(attn_list, cis_map, sel_indices, E_batch, env_code):
    """
    attn_list: list over layers of [B,H,G,S] attention tensors
    cis_map:   dict {local_gene_idx -> list[int] of global SNP idx}
    sel_indices: 1D array of selected union SNP global indices used this fold
    E_batch:   1D array of environment codes for the batch (same B)
    env_code:  0=WW, 1=WS1, 2=WS2
    returns:   1D array [n_local_genes] of cis mass for this env
    """
    if not attn_list:
        return np.array([])
    A = np.mean([a.detach().cpu().numpy() for a in attn_list], axis=0)  # [B,H,G,S]
    A = A.mean(axis=1)  # [B,G,S]
    mask_env = (E_batch == env_code)
    if not np.any(mask_env):
        return np.full(A.shape[1], np.nan, dtype=float)
    Aenv = A[mask_env]  # [B_env, G, S]
    masses = []
    sel_indices = np.asarray(sel_indices, dtype=int)
    for g in range(Aenv.shape[1]):
        cis_idx = np.asarray(cis_map.get(g, []), dtype=int)
        if cis_idx.size == 0:
            masses.append(np.nan); continue
        mask_sel = np.isin(sel_indices, cis_idx)
        if not np.any(mask_sel):
            masses.append(np.nan); continue
        num = Aenv[:, g, mask_sel].sum(axis=1)
        den = Aenv[:, g, :].sum(axis=1) + 1e-8
        masses.append(np.nanmean(num/den))
    return np.asarray(masses, dtype=float)

def topk_within_cis(y, G_tr, cis_idx, k):
    if not cis_idx: return []
    X = G_tr[:, cis_idx]
    if X.shape[1]==0 or y.std()<1e-8: return []
    y0 = (y - y.mean()) / (y.std()+1e-8)
    xs = X.std(0); keep = xs>1e-8
    if not np.any(keep): return []
    X0 = (X[:,keep] - X[:,keep].mean(0)) / xs[keep]
    cor = np.abs(X0.T @ y0)
    k = min(k, len(cor))
    sel_sub = np.argpartition(-cor, k-1)[:k]
    return np.array(cis_idx)[keep][sel_sub].tolist()

def ensure_min_cis_per_gene(sel_indices, cis_map, min_per_gene=1):
    sel_set = set(int(i) for i in np.asarray(sel_indices).ravel())
    if min_per_gene <= 0:
        return np.array(sorted(sel_set), dtype=int)
    for g, idxs in cis_map.items():
        idxs = [int(i) for i in (idxs or [])]
        if idxs and not np.any(np.isin(list(sel_set), idxs)):
            sel_set.add(int(idxs[0]))
    return np.array(sorted(sel_set), dtype=int)

def _topk_union_cis_scored(G_tr, Y_tr, cis_map, k_per_gene):
    scores = {}
    for gi in range(Y_tr.shape[1]):
        sel = topk_within_cis(Y_tr[:,gi], G_tr, cis_map.get(gi,[]), k_per_gene)
        if not sel: continue
        y = (Y_tr[:,gi]-Y_tr[:,gi].mean())/(Y_tr[:,gi].std()+1e-8)
        X = G_tr[:, sel]; xs = X.std(0); m = xs>1e-8
        if not np.any(m): continue
        X0 = (X[:,m]-X[:,m].mean(0))/xs[m]
        cor = np.abs(X0.T @ y)
        for idx, c in zip(np.array(sel)[m], cor):
            scores[int(idx)] = max(scores.get(int(idx),0.0), float(c))
    ordered = [i for i,_ in sorted(scores.items(), key=lambda kv:(-kv[1], kv[0]))]
    return np.array(ordered, dtype=int), scores

def _enable_mc_dropout(model, on=True):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train(on)

class ECTRunner:
    def __init__(self, seed, lr, wd, epochs, patience, builder_fn, loss_cf_weight=0.2):
        self.seed=seed; self.lr=lr; self.wd=wd; self.epochs=epochs; self.patience=patience
        self.builder_fn=builder_fn
        self.loss_cf_weight = loss_cf_weight
        
        import random, numpy as _np
        random.seed(seed)
        _np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=None
        self.pi_scale = 1.0  # prediction-interval inflation (learned on val)
        self.g_mean = None   # NEW: per-SNP train means for imputation (pre-scaling)
        self.pi_add = 0.0     # NEW: additive (normalized) conformal half-width
        self.pi_add_scalar = 0.0  # NEW: same but in descaled units
        self._va_idx = None

    # NEW: simple column-mean imputer for G with sentinel <= -8 treated as missing
    def _impute_G_train(self, G):
        miss = (G <= -8)
        means = np.where(miss, np.nan, G).mean(axis=0)
        means = np.nan_to_num(means, nan=0.0)
        G_imp = np.where(miss, means, G)
        self.g_mean = means.astype(np.float32)
        return G_imp

    def _impute_G_infer(self, G):
        if self.g_mean is None:
            return G
        miss = (G <= -8)
        return np.where(miss, self.g_mean, G)

    def fit(self, G, PCs, E, Y_res, accessions, cis_mask=None):
        n_snps, n_genes = G.shape[1], Y_res.shape[1]
        n_pcs = 0 if PCs is None else PCs.shape[1]
        n_envs = int(E.max())+1
        self.model = self.builder_fn(n_snps=n_snps, n_genes=n_genes, n_pcs=n_pcs, n_envs=n_envs).to(self.dev)
        if cis_mask is not None:
            self.model.set_cis_mask(cis_mask)

        # split + scalers
        idx = np.arange(G.shape[0]); tr, va = train_test_split(idx, test_size=0.15, random_state=self.seed, stratify=E)
        self._va_idx = va  # expose for outer CV
        Gtr_imp = self._impute_G_train(G[tr])                 # NEW
        self.gsc = StandardScaler().fit(Gtr_imp)
        self.pcsc = None if PCs is None else StandardScaler().fit(PCs[tr])
        ymean = Y_res[tr].mean(0, keepdims=True); ystd = Y_res[tr].std(0, keepdims=True); ystd[ystd<1e-8]=1.0
        self.ymean, self.ystd = ymean, ystd

        def T(Gi, Pi, Ei, Yi=None):
            Gi_imp = self._impute_G_infer(Gi)                 # NEW
            Gt = torch.tensor(self.gsc.transform(Gi_imp), dtype=torch.float32, device=self.dev)
            Pt = None if Pi is None else torch.tensor(self.pcsc.transform(Pi), dtype=torch.float32, device=self.dev)
            Et = torch.tensor(Ei, dtype=torch.long, device=self.dev)
            Yt = None if Yi is None else torch.tensor((Yi-ymean)/ystd, dtype=torch.float32, device=self.dev)
            return Gt, Pt, Et, Yt

        Gtr, Ptr, Etr, Ytr = T(G[tr], PCs[tr] if PCs is not None else None, E[tr], Y_res[tr])
        Gva, Pva, Eva, Yva = T(G[va], PCs[va] if PCs is not None else None, E[va], Y_res[va])

        # Per-environment weights from the TRAIN split (inverse variance), normalised to mean 1.0
        with torch.no_grad():
            n_envs = int(E.max()) + 1
            env_w = torch.ones(n_envs, device=self.dev)
            for e in range(n_envs):
                mask = (Etr == e)
                v = Ytr[mask].var() if mask.any() else torch.tensor(1.0, device=self.dev)
                env_w[e] = 1.0 / (v + 1e-6)
            self.env_w = env_w / env_w.mean()

        # Pre-compute contrastive pairs from the training set
        acc_tr = accessions[tr]
        df_tr = pd.DataFrame({'acc': acc_tr, 'env': E[tr], 'idx': np.arange(len(acc_tr))})
        pairs = []
        for _, sub in df_tr.groupby('acc'):
            if sub['env'].nunique() < 2: continue
            for (idx1, env1), (idx2, env2) in combinations(sub[['idx', 'env']].values.tolist(), 2):
                pairs.append((idx1, idx2))
        
        if pairs:
            idx_e1, idx_e2 = zip(*pairs)
            idx_e1 = torch.tensor(idx_e1, dtype=torch.long, device=self.dev)
            idx_e2 = torch.tensor(idx_e2, dtype=torch.long, device=self.dev)

        # AdamW with param groups: exclude LayerNorm/bias from weight decay
        no_decay_keys = ("bias", "LayerNorm.weight", "LayerNorm.bias")
        decay_params, nodecay_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (nodecay_params if any(k in n for k in no_decay_keys) else decay_params).append(p)

        opt = torch.optim.AdamW(
            [{"params": decay_params,   "weight_decay": self.wd},
             {"params": nodecay_params, "weight_decay": 0.0}],
            lr=self.lr
        )
        best, bad, pat = 1e9, 0, self.patience
        for ep in range(self.epochs):
            self.model.train(); opt.zero_grad()
            pred = self.model(Gtr, Ptr, Etr)['predictions']
            w_tr = self.env_w[Etr].unsqueeze(1)        # [Ntr, 1]
            loss_mse = ((pred - Ytr)**2 * w_tr).mean()
            
            loss_cf = torch.tensor(0.0, device=self.dev)
            if pairs and self.loss_cf_weight > 0:
                delta_obs = Ytr[idx_e2] - Ytr[idx_e1]
                pred_e1 = self.model(Gtr[idx_e1], Ptr[idx_e1] if Ptr is not None else None, Etr[idx_e1])['predictions']
                pred_e2 = self.model(Gtr[idx_e1], Ptr[idx_e1] if Ptr is not None else None, Etr[idx_e2])['predictions']
                delta_pred = pred_e2 - pred_e1
                huber = torch.nn.SmoothL1Loss(beta=0.05)
                loss_cf = huber(delta_pred, delta_obs)

            loss = loss_mse + self.loss_cf_weight * loss_cf
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); opt.step()
            
            if (ep+1)%10==0:
                self.model.eval()
                with torch.no_grad(): 
                    pred_va = self.model(Gva, Pva, Eva)['predictions']
                    w_va = self.env_w[Eva].unsqueeze(1)         # [Nva, 1]
                    vloss = (((pred_va - Yva)**2) * w_va).mean().item()
                    if ep == 9: # First validation step
                        Eva0 = torch.zeros_like(Eva)
                        Eva1 = torch.ones_like(Eva)
                        yhat0 = self.model(Gva, Pva, Eva0)['predictions']
                        yhat1 = self.model(Gva, Pva, Eva1)['predictions']
                        mean_abs_delta = (yhat1 - yhat0).abs().mean().item()
                        print(f"  [Sanity Check] Mean abs(yhat(env=1)-yhat(env=0)) on val set: {mean_abs_delta:.4f}")

                        # Extended sanity check for all env pairs
                        envs = np.unique(E[va])
                        pairs = list(combinations(envs, 2))
                        deltas = []
                        
                        def predict_with_env(model, G, PCs, env_val):
                            env_tensor = torch.full_like(Eva, env_val)
                            return model(G, PCs, env_tensor)['predictions']

                        for e1, e2 in pairs:
                            yhat1 = predict_with_env(self.model, Gva, Pva, e1)
                            yhat2 = predict_with_env(self.model, Gva, Pva, e2)
                            deltas.append((yhat2 - yhat1).abs().mean().item())
                        
                        if deltas:
                            print(f"  [Sanity Check] Mean |Δŷ| over env pairs {pairs}: {np.mean(deltas):.4f}")


                if vloss < best: best, bad, state = vloss, 0, {k:v.detach().cpu() for k,v in self.model.state_dict().items()}
                else: bad += 1
                if bad >= pat: break
        if 'state' in locals(): self.model.load_state_dict(state)

        # === Calibrate MC-dropout PIs on validation split ===
        # Use modest sampling to estimate predictive spread vs residual spread.
        self.model.eval()
        _enable_mc_dropout(self.model, True)
        with torch.no_grad():
            S = 8  # small, fast
            samples = []
            for _ in range(S):
                samples.append(self.model(Gva, Pva, Eva)['predictions'].detach().cpu().numpy())
        _enable_mc_dropout(self.model, False)
        samples = np.stack(samples, axis=0)                           # [S, Nv, Gsub]
        mean_pred = samples.mean(axis=0)                              # [Nv, Gsub]
        std_pred  = samples.std(axis=0, ddof=1) + 1e-8                # [Nv, Gsub]
        # residuals in normalized space
        resid = (Yva.cpu().numpy() - mean_pred)
        std_resid = np.nanstd(resid, axis=0) + 1e-8                   # [Gsub]
        # global scale: robust ratio of observed/predicted spreads
        ratio = np.median(std_resid / std_pred)
        self.pi_scale = float(np.clip(ratio, 0.5, 5.0))

        # NEW: additive conformal half-width from val (central 80% ⇒ 90th percentile of |residual|)
        abs_res = np.abs(resid)                                       # normalized space
        q90 = np.nanpercentile(abs_res, 90, axis=0)                   # [Gsub]
        self.pi_add = float(np.nanmedian(q90))
        # Convert to descaled units using a robust gene-scale:
        ystd_med = float(np.nanmedian(self.ystd))
        self.pi_add_scalar = self.pi_add * ystd_med

    def predict(self, G, PCs, E, mc_samples=0, return_attention=False, return_samples=False, cis_mask="use_internal"):
        Gt = torch.tensor(self.gsc.transform(self._impute_G_infer(G)), dtype=torch.float32, device=self.dev)
        Pt = None if PCs is None else torch.tensor(self.pcsc.transform(PCs), dtype=torch.float32, device=self.dev)
        Et = torch.tensor(E, dtype=torch.long, device=self.dev)

        self.model.eval()

        if return_attention:
            if mc_samples and mc_samples > 0:
                raise ValueError("Cannot return attention with MC-dropout sampling.")
            # cis-masked run (normal)
            with torch.no_grad():
                out_dict = self.model(Gt, Pt, Et, return_attention=True, cis_mask="use_internal")
            out = out_dict['predictions'].detach().cpu().numpy()
            
            # unmasked diagnostic run
            with torch.no_grad():
                out_diag = self.model(Gt, Pt, Et, return_attention=True, cis_mask=None)
            attn_unmasked = out_diag.get('attention_weights', [])
            
            final_out = out * self.ystd + self.ymean
            return final_out, attn_unmasked

        if mc_samples and mc_samples > 0:
            if return_attention:
                raise ValueError("Cannot return attention with MC-dropout sampling.")
            
            _enable_mc_dropout(self.model, True)  # Activate dropout for MC-dropout
            samples = []
            with torch.no_grad():
                for _ in range(mc_samples):
                    samples.append(self.model(Gt, Pt, Et)['predictions'].detach().cpu().numpy())
            _enable_mc_dropout(self.model, False)  # Deactivate dropout after sampling

            samples = np.stack(samples, axis=0)                         # [S, B, Gsub]
            # Inflate around the mean: multiplicative (scale) + additive (conformal)
            mean_norm = samples.mean(axis=0, keepdims=True)             # [1, B, Gsub]
            dev_norm  = samples - mean_norm
            dev_scaled = dev_norm * self.pi_scale
            # push samples outward by an additive fudge in **descaled** units
            samples_descaled = (mean_norm + dev_scaled) * self.ystd + self.ymean
            mean = samples_descaled.mean(axis=0, keepdims=True)         # [1, B, Gsub]
            sign = np.sign(samples_descaled - mean)
            samples_descaled = mean + sign * (np.abs(samples_descaled - mean) + self.pi_add_scalar)
            mean = samples_descaled.mean(axis=0)

            if return_samples:
                return mean, samples_descaled
            return mean, None
        
        with torch.no_grad():
            out_dict = self.model(Gt, Pt, Et, return_attention=return_attention, cis_mask=cis_mask)
        
        out = out_dict['predictions'].detach().cpu().numpy()
        final_out = out * self.ystd + self.ymean
        
        if return_attention:
            return final_out, out_dict.get('attention_weights', [])
        return final_out, None

def run_ect_residual_cv(builder_fn, data, kfolds=5, seeds=(42,101,202), topk=5, union_cap=1000, mc_samples=16, lr=5e-4, wd=0.05, epochs=300, patience=20, loss_cf_weight=0.2, keep_genes=40, args=None, out_dir=None, save_attention=False):
    out_dir = Path(out_dir)
    G,Y,E,PCs,grp,cis = data["G"], data["Y"], data["E"], data["PCs"], data["groups"], data["cis_map"]

    # --- Assertions for data integrity ---
    assert PCs is not None and PCs.shape[1] > 0, "PCs must be provided (n_pcs > 0)"
    unique_envs = set(np.unique(E))
    expected_envs = {0, 1, 2}
    assert unique_envs == expected_envs, f"Environment mapping must be {{0, 1, 2}}, but found {unique_envs}"

    n_pcs = 0 if PCs is None else PCs.shape[1]
    gkf = GroupKFold(n_splits=kfolds)
    medians = []; union_pre=[]; union_post=[]; cf_align_scores = []; all_seeds_masses = []
    all_seeds_cis_WW, all_seeds_cis_WS1, all_seeds_cis_WS2 = [], [], []
    all_seeds_coverage = []; all_seeds_ci_width = []
    all_seeds_alphas = []
    all_seeds_r2_full, all_seeds_r2_null, all_seeds_delta, all_seeds_modeled = [], [], [], []
    _written_folds_ect = set()

    # Full-gene accumulators (global size G)
    G_total = Y.shape[1]  # number of genes in full matrix
    cis_acc = { "WW": np.full(G_total, np.nan),
                "WS1": np.full(G_total, np.nan),
                "WS2": np.full(G_total, np.nan) }
    cis_cnt = { k: np.zeros(G_total, dtype=int) for k in cis_acc.keys() }

    for sd in seeds:
        # ---- strict determinism per seed ----
        import random
        random.seed(sd)
        np.random.seed(sd)
        torch.manual_seed(sd)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sd)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        oof_pred = np.full_like(Y, np.nan, float)
        oof_null = np.full_like(Y, np.nan, float)
        global_keep_genes = None

        # Optional: determine a fixed gene subset per seed with no leakage.
        if getattr(args, "gene_select_mode", "rigorous_global") == "rigorous_global":
            resvar_sum = np.zeros(Y.shape[1], dtype=np.float64)
            counts = np.zeros(Y.shape[1], dtype=np.int32)
            for tr, _ in gkf.split(G, groups=grp):
                # We only need train-set predictions for this pre-pass
                yhat_null_tr_pre = np.zeros((len(tr), Y.shape[1]))
                pcs_tr_slice = PCs[tr] if PCs is not None else None
                
                # Using a dummy test set for the helper function's signature
                dummy_te_len = 1 
                dummy_E = np.array([0])
                dummy_PCs = pcs_tr_slice[0:1] if pcs_tr_slice is not None else None
                dummy_grp = np.array(["dummy"])

                results_pre = Parallel(n_jobs=max(1, (cpu_count() or 8) - 1))(
                    delayed(_fit_gene_null_model)(gi, Y[tr,:], E[tr], pcs_tr_slice, grp[tr],
                                                 dummy_E, dummy_PCs, dummy_grp, n_pcs)
                    for gi in range(Y.shape[1])
                )
                for gi, tr_hat, _, _ in results_pre:
                    rv = np.var(Y[tr, gi] - tr_hat)
                    if np.isfinite(rv):
                        resvar_sum[gi] += rv * len(tr) # Weight by number of samples
                        counts[gi] += len(tr)

            resvar_mean = np.divide(resvar_sum, np.maximum(counts, 1), where=counts>0)
            k = int(getattr(args, "keep_genes", 40))
            global_keep_genes = np.argsort(-resvar_mean)[:k]
            print(f"[ECT] Using RIGOROUS GLOBAL keep-genes list (k={k}) for all folds this seed.")

        align_scores_per_fold = []
        masses_per_fold = []
        cis_WW_per_fold, cis_WS1_per_fold, cis_WS2_per_fold = [], [], []
        coverage_per_fold = []; ci_width_per_fold = []
        alphas_per_fold = []
        for fold, (tr, te) in enumerate(gkf.split(G, groups=grp), 1):
            if args and args.dump_fold_sentinels and fold not in _written_folds_ect:
                outd = out_dir / "fold_sentinels" / "ect"
                outd.mkdir(parents=True, exist_ok=True)
                np.savetxt(outd / f"fold_{fold}_train.txt", np.asarray(grp)[tr], fmt="%s")
                np.savetxt(outd / f"fold_{fold}_test.txt",  np.asarray(grp)[te],  fmt="%s")
                print(f"[ECT] wrote sentinels: {outd / f'fold_{fold}_train.txt'} , {outd / f'fold_{fold}_test.txt'}")
                _written_folds_ect.add(fold)

            # 1) Null (train/test) per gene, no leakage
            yhat_null_tr = np.zeros((len(tr), Y.shape[1])); yhat_null_te = np.zeros((len(te), Y.shape[1]))
            
            pcs_tr_slice = PCs[tr] if PCs is not None else None
            pcs_te_slice = PCs[te] if PCs is not None else None
            
            results = Parallel(n_jobs=max(1, (cpu_count() or 8) - 1))(
                delayed(_fit_gene_null_model)(
                    gi, Y[tr, :], E[tr], pcs_tr_slice, grp[tr],
                    E[te], pcs_te_slice, grp[te], n_pcs
                ) for gi in range(Y.shape[1])
            )
            
            n_fb = 0
            for gi, tr_hat, te_hat, fallback_used in results:
                yhat_null_tr[:,gi] = tr_hat
                yhat_null_te[:,gi] = te_hat
                if fallback_used:
                    n_fb += 1
            
            print(f"[ECT] NULL MixedLM→OLS_FE fallbacks this fold: {n_fb}/{len(results)}")
            
            oof_null[te] = yhat_null_te
            assert np.isfinite(yhat_null_tr).all() and np.isfinite(yhat_null_te).all()
            
            r2_null_fold = np.nanmedian([_r2_safe(Y[te, i], yhat_null_te[:, i]) for i in range(Y.shape[1])])
            print(f"  [fold {fold}] median R² null-only: {r2_null_fold:.3f}")

            Ytr_res = Y[tr] - yhat_null_tr
            Yte_res = Y[te] - yhat_null_te  # Also calculate test residuals for export

            # 2) Scored union on residual targets (+cap)
            sel_scored, _ = _topk_union_cis_scored(G[tr], Ytr_res, cis, k_per_gene=topk)
            pre = len(sel_scored); cap = int(union_cap or 0)
            sel = sel_scored[:cap] if cap and pre>cap else sel_scored
            sel = ensure_min_cis_per_gene(sel, cis, min_per_gene=1)
            union_pre.append(pre); union_post.append(len(sel))
            print(f"[seed {sd}] fold {fold}: residual-union {pre}" + (f" (capped→{len(sel)})" if len(sel)<pre else ""))

            # Gene subsetting based on residual variance
            if global_keep_genes is not None:
                keep_genes_fold = global_keep_genes
            else: # Corresponds to per_fold mode
                res_var = Ytr_res.var(axis=0)
                keep_genes_fold = np.argsort(-res_var)[:keep_genes]

            # PATCH: Export per-gene baseline bundles before fitting the transformer
            if args.export_baselines:
                root = Path(args.export_dir or (out_dir / "baseline_npz")) / f"fold_{fold}"
                root.mkdir(parents=True, exist_ok=True)
                
                all_gene_ids = data.get('gene_ids', [f"GENE_{i}" for i in range(Y.shape[1])])

                print(f"  [Exporting] Generating baseline NPZ bundles for {len(keep_genes_fold)} genes in fold {fold}...")
                for gene_global_idx in keep_genes_fold:
                    gene_name = all_gene_ids[gene_global_idx]
                    
                    # This logic is now inside the loop, creating per-gene cis-SNP sets
                    cis_snps_for_gene = topk_within_cis(Ytr_res[:, gene_global_idx], G[tr], cis.get(int(gene_global_idx), []), k=topk)
                    
                    if not cis_snps_for_gene:
                        X_cis_tr = np.empty((len(tr), 0), dtype=np.float32)
                        X_cis_te = np.empty((len(te), 0), dtype=np.float32)
                    else:
                        X_cis_tr = G[tr][:, cis_snps_for_gene]
                        X_cis_te = G[te][:, cis_snps_for_gene]

                    np.savez_compressed(
                        root / f"{gene_name}.npz",
                        X_tr=X_cis_tr.astype(np.float32),
                        y_tr=Ytr_res[:, gene_global_idx].astype(np.float32),
                        env_tr=E[tr].astype(np.int16),
                        X_te=X_cis_te.astype(np.float32),
                        y_te=Yte_res[:, gene_global_idx].astype(np.float32),
                        env_te=E[te].astype(np.int16),
                        gene=np.array([gene_name])
                    )

            # 3) Train ECT on residuals (train/val split inside)
            runner = ECTRunner(seed=sd, lr=lr, wd=wd, epochs=epochs, patience=patience, builder_fn=builder_fn, loss_cf_weight=loss_cf_weight)

            sel = np.asarray(sel, dtype=int)
            keep_genes_arr = np.asarray(keep_genes_fold, dtype=int)
            sel_pos = {s: j for j, s in enumerate(sel)}
            cis_mask_sub = np.zeros((len(keep_genes_arr), len(sel)), dtype=bool)
            for new_g, g in enumerate(keep_genes_arr):
                for s in cis.get(int(g), []):
                    j = sel_pos.get(s)
                    if j is not None:
                        cis_mask_sub[new_g, j] = True
            
            runner.fit(G[tr][:,sel], PCs[tr] if PCs is not None else None, E[tr], Ytr_res[:, keep_genes_fold], grp[tr], cis_mask=torch.from_numpy(cis_mask_sub))

            # --- Validation predictions for uplift gating ---
            va = runner._va_idx
            yhat_res_val_sub, _ = runner.predict(G[tr][va][:, sel],
                                                 PCs[tr][va] if PCs is not None else None,
                                                 E[tr][va], mc_samples=0)
            # pad to all genes, keeping unmodeled as NaN
            yhat_res_val = np.full((len(va), Y.shape[1]), np.nan, dtype=float)
            yhat_res_val[:, keep_genes_fold] = yhat_res_val_sub
            # compare full vs null on validation, treating NaNs as zero only at addition time
            r2_full_val = np.array([
                _r2_safe(Y[tr][va, j], yhat_null_tr[va, j] + np.nan_to_num(yhat_res_val[:, j], nan=0.0))
                for j in range(Y.shape[1])
            ])
            r2_null_val = np.array([_r2_safe(Y[tr][va, j], yhat_null_tr[va, j])
                                    for j in range(Y.shape[1])])
            
            # ---- Per-gene shrinkage from validation uplift ----
            uplift_val = r2_full_val - r2_null_val  # shape [G]
            if args.alpha_mode == "none":
                alpha = np.ones_like(uplift_val, dtype=float)
            else:  # 'sigmoid' (softer gate)
                k = 3.0
                alpha = np.where(np.isfinite(uplift_val) & (uplift_val > 0.0),
                                 1.0/(1.0 + np.exp(-k*uplift_val)), 0.0)
            alphas_per_fold.append(alpha)
            alpha_sub = alpha[keep_genes_fold]  # map to the subset columns
            
            yhat_res_mean_subset, yhat_res_samples_subset = runner.predict(G[te][:,sel], PCs[te] if PCs is not None else None, E[te], mc_samples=mc_samples, return_samples=True)
            
            # apply to residual means/samples
            yhat_res_mean_subset *= alpha_sub[np.newaxis, :]
            if yhat_res_samples_subset is not None:
                yhat_res_samples_subset *= alpha_sub[np.newaxis, np.newaxis, :]

            # AFTER: keep unmodeled genes as NaN; write only modeled columns
            yhat_res_mean = np.full((len(te), Y.shape[1]), np.nan, dtype=float)
            yhat_res_mean[:, keep_genes_fold] = yhat_res_mean_subset

            yhat_res_samples = None
            if yhat_res_samples_subset is not None:
                yhat_res_samples = np.full((yhat_res_samples_subset.shape[0], len(te), Y.shape[1]), np.nan, dtype=float)
                yhat_res_samples[:, :, keep_genes_fold] = yhat_res_samples_subset

            # assemble full test preds for this fold, treating NaNs as zero contribution
            mean_total_te = yhat_null_te + np.nan_to_num(yhat_res_mean, nan=0.0)
            
            # Place the combined predictions into the OOF array for ALL genes.
            # The unmodeled genes will just have the null prediction.
            oof_pred[te] = mean_total_te

            # Sanity on modeled genes only
            cc_modeled_genes = keep_genes_fold
            if len(cc_modeled_genes) > 0:
                y_null_subset = yhat_null_te[:, cc_modeled_genes].ravel()
                y_total_subset = mean_total_te[:, cc_modeled_genes].ravel()
                if np.all(np.isfinite(y_null_subset)) and np.all(np.isfinite(y_total_subset)) and y_null_subset.size > 1 and y_total_subset.size > 1:
                    cc = np.corrcoef(y_null_subset, y_total_subset)[0, 1]
                    print(f"  [fold {fold}] corr(yhat_null, y_total)[modeled]: {cc:.3f}")

            # 5) Conformal calibration (per fold, per gene) on validation, then apply to test
            # Use validation absolute residuals of TOTAL prediction to get q_0.9 per gene
            q90_total = np.full(Y.shape[1], np.nan)
            if len(va) > 0:
                mean_val = yhat_null_tr[va, :] + yhat_res_val
                abs_res  = np.abs(Y[tr][va, :] - mean_val)
                q90_total = np.nanpercentile(abs_res, 90, axis=0)  # two-sided 80% PI

            # If we drew MC samples, use mean±q90_total; else just deterministic mean±q90_total
            mean_total_te = yhat_null_te + yhat_res_mean
            lower = mean_total_te - q90_total
            upper = mean_total_te + q90_total
            coverage = ((Y[te] >= lower) & (Y[te] <= upper)).mean()
            ci_width = (upper - lower).mean()
            coverage_per_fold.append(coverage)
            ci_width_per_fold.append(ci_width)

            # --- Padded runner for counterfactuals and attention ---
            class PaddedRunner:
                def __init__(self, runner, keep_indices, total_genes):
                    self.runner = runner
                    self.keep_indices = keep_indices
                    self.total_genes = total_genes
                
                def predict(self, G, PCs, E, mc_samples=0, return_attention=False, return_samples=False, cis_mask="use_internal"):
                    yhat_subset, attn_subset = self.runner.predict(G, PCs, E, mc_samples, return_attention, return_samples, cis_mask)
                    
                    yhat_padded = np.zeros((yhat_subset.shape[0], self.total_genes))
                    yhat_padded[:, self.keep_indices] = yhat_subset
                    
                    # Note: Attention weights are not padded as they correspond to the subset model's genes.
                    # The cis_attention_mass function handles this correctly by using the original cis_map.
                    return yhat_padded, attn_subset
            
            padded_runner = PaddedRunner(runner, keep_genes_fold, Y.shape[1])

            # 6) Counterfactual alignment
            median_alignment, _ = ect_counterfactual_alignment(
                padded_runner, G[te][:,sel], PCs[te] if PCs is not None else None,
                E[te], Y[te], grp[te]
            )
            if not np.isnan(median_alignment):
                align_scores_per_fold.append(median_alignment)

            # 7) Cis-attention mass (diagnostic pass on validation set without mask)
            if save_attention:
                _, attn_diag = runner.predict(
                    G[tr][va][:, sel],
                    PCs[tr][va] if PCs is not None else None,
                    E[tr][va],
                    mc_samples=0,
                    cis_mask=None,
                    return_attention=True
                )
                if attn_diag:
                    subset_cis_map = {new_idx: cis.get(original_idx, []) for new_idx, original_idx in enumerate(keep_genes_fold)}
                    cm_WW  = cis_mass_for_env(attn_diag, subset_cis_map, sel, E[tr][va], env_code=0)
                    cm_WS1 = cis_mass_for_env(attn_diag, subset_cis_map, sel, E[tr][va], env_code=1)
                    cm_WS2 = cis_mass_for_env(attn_diag, subset_cis_map, sel, E[tr][va], env_code=2)

                    for loc, glob in enumerate(keep_genes_fold):
                        for key, arr in (("WW", cm_WW), ("WS1", cm_WS1), ("WS2", cm_WS2)):
                            v = arr[loc]
                            if np.isfinite(v):
                                if np.isnan(cis_acc[key][glob]):
                                    cis_acc[key][glob] = v
                                else:
                                    cis_acc[key][glob] += v
                                cis_cnt[key][glob] += 1
                    
                    # Save attention weights
                    attn_dir = out_dir / "attention_weights"
                    attn_dir.mkdir(parents=True, exist_ok=True)
                    attn_path = attn_dir / f"seed_{sd}_fold_{fold}_attention.pt"
                    torch.save(attn_diag, attn_path)

            else:
                masses_per_fold.append(np.full(Y.shape[1], np.nan))
                cis_WW_per_fold.append(np.full(Y.shape[1], np.nan))
                cis_WS1_per_fold.append(np.full(Y.shape[1], np.nan))
                cis_WS2_per_fold.append(np.full(Y.shape[1], np.nan))

        r2_full_all  = np.array([_r2_safe(Y[:,i], oof_pred[:,i]) for i in range(Y.shape[1])])
        r2_null_all  = np.array([_r2_safe(Y[:,i], oof_null[:,i]) for i in range(Y.shape[1])])
        delta_all = r2_full_all - r2_null_all
        
        # A gene is "modeled" if its residual prediction is not NaN in at least one sample.
        oof_res = oof_pred - oof_null
        modeled = np.any(np.isfinite(oof_res), axis=0)

        # Optional but recommended belt-and-braces check:
        print(f"[seed {sd}] modeled genes this seed: {int(modeled.sum())}/{Y.shape[1]}")
        
        r2_full_modeled = r2_full_all[modeled]
        r2_null_modeled = r2_null_all[modeled]
        delta_modeled   = r2_full_modeled - r2_null_modeled

        print(f"\n[Metrics] OOF median R² (ALL):   full={np.nanmedian(r2_full_all):.3f}, null={np.nanmedian(r2_null_all):.3f}, Δ={np.nanmedian(delta_all):.3f}")
        print(f"[Metrics] OOF median R² (MODELED): full={np.nanmedian(r2_full_modeled):.3f}, null={np.nanmedian(r2_null_modeled):.3f}, Δ={np.nanmedian(delta_modeled):.3f}")
        
        medians.append(float(np.nanmedian(r2_full_all)))
        
        all_seeds_r2_full.append(r2_full_all)
        all_seeds_r2_null.append(r2_null_all)
        all_seeds_delta.append(delta_all)
        all_seeds_modeled.append(modeled)
        
        if alphas_per_fold:
            all_seeds_alphas.append(np.mean(alphas_per_fold, axis=0))

        if align_scores_per_fold:
            cf_align_scores.append(np.mean(align_scores_per_fold))
        if masses_per_fold:
            _arr = np.array(masses_per_fold, dtype=float)  # [n_folds, n_genes]
            avg_mass_for_seed = nanmean_or(_arr, axis=0, fallback=None)
            if avg_mass_for_seed is not None:
                all_seeds_masses.append(avg_mass_for_seed)
            # else: skip appending; we'll produce an all-NaN vector later
        if cis_WW_per_fold:
            avg_cis_WW_for_seed = nanmean_or(np.array(cis_WW_per_fold, dtype=float), axis=0, fallback=None)
            if avg_cis_WW_for_seed is not None:
                all_seeds_cis_WW.append(avg_cis_WW_for_seed)
        if cis_WS1_per_fold:
            avg_cis_WS1_for_seed = nanmean_or(np.array(cis_WS1_per_fold, dtype=float), axis=0, fallback=None)
            if avg_cis_WS1_for_seed is not None:
                all_seeds_cis_WS1.append(avg_cis_WS1_for_seed)
        if cis_WS2_per_fold:
            avg_cis_WS2_for_seed = nanmean_or(np.array(cis_WS2_per_fold, dtype=float), axis=0, fallback=None)
            if avg_cis_WS2_for_seed is not None:
                all_seeds_cis_WS2.append(avg_cis_WS2_for_seed)
        if coverage_per_fold:
            all_seeds_coverage.append(np.mean(coverage_per_fold))
        if ci_width_per_fold:
            all_seeds_ci_width.append(np.mean(ci_width_per_fold))

    final_gene_mass = nanmean_or(np.array(all_seeds_masses), axis=0,
                                     fallback=np.full(Y.shape[1], np.nan))
    final_cis_WW = nanmean_or(np.array(all_seeds_cis_WW), axis=0, fallback=np.full(Y.shape[1], np.nan))
    final_cis_WS1 = nanmean_or(np.array(all_seeds_cis_WS1), axis=0, fallback=np.full(Y.shape[1], np.nan))
    final_cis_WS2 = nanmean_or(np.array(all_seeds_cis_WS2), axis=0, fallback=np.full(Y.shape[1], np.nan))
    final_coverage_mean = np.mean(all_seeds_coverage) if all_seeds_coverage else np.nan
    final_coverage_sd = np.std(all_seeds_coverage, ddof=1) if len(all_seeds_coverage) > 1 else 0.0
    final_ci_width_mean = np.mean(all_seeds_ci_width) if all_seeds_ci_width else np.nan
    final_ci_width_sd = np.std(all_seeds_ci_width, ddof=1) if len(all_seeds_ci_width) > 1 else 0.0
    
    final_r2_full = nanmean_or(all_seeds_r2_full, axis=0,
                                     fallback=np.full(Y.shape[1], np.nan))
    final_r2_null = nanmean_or(all_seeds_r2_null, axis=0,
                                     fallback=np.full(Y.shape[1], np.nan))
    final_delta = nanmean_or(all_seeds_delta, axis=0,
                                     fallback=np.full(Y.shape[1], np.nan))
    final_modeled = (np.any(all_seeds_modeled, axis=0)
                           if all_seeds_modeled else np.full(Y.shape[1], False))
    final_alphas = nanmean_or(all_seeds_alphas, axis=0,
                                     fallback=np.full(Y.shape[1], np.nan))
    
    return {
        "r2_median_mean": float(np.mean(medians)),
        "r2_median_sd": float(np.std(medians, ddof=1) if len(medians)>1 else 0.0),
        "union_pre_mean": nanmean_or(union_pre, fallback=np.nan),
        "union_post_mean": nanmean_or(union_post, fallback=np.nan),
        "cf_align_mean": float(np.mean(cf_align_scores)) if cf_align_scores else np.nan,
        "cf_align_sd": float(np.std(cf_align_scores, ddof=1) if len(cf_align_scores) > 1 else 0.0),
        "cis_mass_per_gene": final_gene_mass,
        "coverage_mean": final_coverage_mean,
        "coverage_sd": final_coverage_sd,
        "ci_width_mean": final_ci_width_mean,
        "ci_width_sd": final_ci_width_sd,
        "r2_full_all": final_r2_full,
        "r2_null_all": final_r2_null,
        "delta_all": final_delta,
        "modeled": final_modeled,
        "alpha_mean_per_gene": final_alphas,
        "cis_mass_WW": final_cis_WW,
        "cis_mass_WS1": final_cis_WS1,
        "cis_mass_WS2": final_cis_WS2,
        "cis_acc": cis_acc,
        "cis_cnt": cis_cnt
    }

def _optuna_objective(trial, data, base_args, base_cfg, OUT_DIR, LMM_CSV):
    # ---- Hyperparam search space (tight, fast) ----
    topk        = trial.suggest_categorical("topk", [6, 8, 12])
    union_cap   = trial.suggest_categorical("union_cap", [topk*10, topk*20, 200, 400])
    n_pcs       = trial.suggest_categorical("n_pcs", [6, 10, 14])
    lr          = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    wd          = trial.suggest_float("wd", 1e-6, 1e-2, log=True)
    dropout     = trial.suggest_float("dropout", 0.0, 0.3)
    patience    = trial.suggest_categorical("patience", [8, 12, 16])
    epochs_cap  = trial.suggest_categorical("epochs", [80, 120, 160])

    # ---- Fast-mode config (3-fold, 1 seed, early stop) ----
    cfg = base_cfg.copy()
    cfg.update(
        KFOLDS=3,
        SEEDS=[42],
        EPOCHS=min(epochs_cap, cfg.get("EPOCHS", 200)),
        PATIENCE=patience,
        TOPK=topk,
        UNION_CAP=union_cap,
        LR=lr,
        WD=wd,
        DROPOUT=dropout,
        N_PCS=n_pcs,
    )

    trial_dir = Path(OUT_DIR) / "tune" / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run one training/eval with attention export ----
    res = run_ect_residual_cv(
        build_ect, data,
        kfolds=cfg['KFOLDS'],
        seeds=cfg['SEEDS'],
        topk=cfg['TOPK'],
        union_cap=cfg['UNION_CAP'],
        mc_samples=cfg['MC_SAMPLES'],
        lr=cfg['LR'],
        wd=cfg['WD'],
        epochs=cfg['EPOCHS'],
        patience=cfg['PATIENCE'],
        loss_cf_weight=cfg['LOSS_CF_WEIGHT'],
        keep_genes=getattr(base_args, "keep_genes", None),
        args=base_args,                 # use args.alpha-mode etc.
        out_dir=str(trial_dir),
        save_attention=True,            # needed for cis_mass_by_env
    )

    # ---- Score & log ----
    try:
        score, details = _score_trial(trial_dir, lmm_csv=LMM_CSV)
    except Exception as e:
        # absolute last resort — don't crash the study
        score, details = PENALTY_SCORE, {"status": "exception_in_score", "error": repr(e)}
    
    print(f"[tune] trial={trial.number:03d} score={score:.4f} details={details}")

    # Save config for the trial
    with open(trial_dir / "trial_config.json", "w") as f:
        json.dump(dict(cfg=cfg, score=score, **details), f, indent=2)

    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECT standalone runner with honest CV.")
    parser.add_argument("--data-file", type=str, default=r"C:\Users\ms\Desktop\gwas\output\transformer_data\transformer_data_100_genes.pt",
                        help="Path to the transformer data .pt file.")
    parser.add_argument("--out-dir", default=r"C:\Users\ms\Desktop\gwas\output\ect_v3m",
                        help="directory for ECT artifacts (must be ect_v3j for the assembler)")
    parser.add_argument("--tune", dest="auto", type=int, default=0,
        help="Run Optuna fast tuning with N trials (3-fold, seed=42, early stop). Alias for --auto")
    parser.add_argument("--study-out", type=str, default=None,
        help="Directory to save Optuna study (defaults to <out-dir>\\tune).")
    parser.add_argument("--keep-genes", type=int, default=40, help="Top-N residual-variance genes per fold to model")
    parser.add_argument("--gene-select-mode", choices=["rigorous_global", "per_fold"], default="rigorous_global",
        help="Gene subset mode: 'rigorous_global' (leak-free fixed set) or 'per_fold' (dynamic set per fold).")
    parser.add_argument("--smoke", action="store_true", help="3-fold, one-seed, short-epochs")
    parser.add_argument("--dump-fold-sentinels", action="store_true", help="write per-fold train/test IDs")
    parser.add_argument("--kfolds", type=int, default=None,
        help="Override number of CV folds (default from config; smoke forces 3).")
    parser.add_argument("--seed", type=int, default=None,
        help="Master seed; if set, overrides SEEDS to a single value for full reproducibility.")
    parser.add_argument("--top-k-cis", type=int, default=8,
                        help="Number of top cis SNPs to use per gene (default: 8)")
    parser.add_argument("--union-cap", type=int, default=None,
        help="Cap the size of the per-fold residual SNP union (lower => less GPU memory).")
    parser.add_argument("--alpha-mode", choices=["none","sigmoid"], default="none",
                        help="Residual shrinkage gate: 'none' (no shrink) or 'sigmoid'.")
    parser.add_argument("--lmm-csv", dest="lmm_results", type=str, default=r"C:\Users\ms\Desktop\gwas\output\robust_lmm_analysis\tables\robust_lmm_comprehensive_results.csv",
                        help="Path to the comprehensive LMM results CSV. Alias for --lmm-results")
    parser.add_argument("--save-attention", type=str, default="false", choices=["true", "false"],
                        help="Whether to save attention weights (default: false)")
    parser.add_argument("--n-pcs", type=int, default=10,
                        help="Number of principal components to use (default: 10)")
    parser.add_argument(
        "--genes-file",
        type=str,
        default=None,
        help="CSV with a 'gene_id' column (or single column) listing gene IDs to whitelist."
    )
    # === CLI overrides (safe to append at end of argparse block) ===
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Override model dropout rate; if unset, use config default."
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=None,
        help="Override optimizer weight decay (WD); if unset, use config default."
    )
    parser.add_argument("--loss-cf-weight", type=float, default=None,
                    help="Counterfactual loss weight (default uses config value)")
                    
    # PATCH: Add arguments for model capacity and training control
    parser.add_argument("--layers", type=int, default=None,
        help="Override number of transformer layers; if unset, use config default.")
    parser.add_argument("--d-model", dest="d_model", type=int, default=None,
        help="Override model hidden size (d_model); if unset, use config default.")
    parser.add_argument("--mc-samples", dest="mc_samples", type=int, default=None,
        help="Override number of MC samples at predict-time; if unset, use config default.")
    parser.add_argument("--patience", dest="patience", type=int, default=None,
        help="Override early-stopping patience; if unset, use config default.")

    # PATCH: Add arguments for baseline data bundle export
    parser.add_argument("--export-baselines", action="store_true",
        help="If set, export per-gene train/test bundles (cis features, residuals, env) per fold.")
    parser.add_argument("--export-dir", type=str, default=None,
        help="Directory to write NPZ bundles; defaults to <out-dir>\\baseline_npz")
        
    args = parser.parse_args()

    # --- Start of argument validation and setup ---
    import os
    if not args.data_file or not os.path.exists(args.data_file):
        raise SystemExit(
            f"[ARG ERROR] --data-file path missing or not found: {args.data_file!r}. "
            "Make sure the variable is set and quoted, e.g., --data-file \"$DATA\"."
        )
    if getattr(args, "dropout", None) is not None:
        if not (0.0 <= args.dropout < 1.0):
            raise SystemExit("[ARG ERROR] --dropout must be in [0, 1).")
    if getattr(args, "weight_decay", None) is not None:
        if args.weight_decay < 0:
            raise SystemExit("[ARG ERROR] --weight-decay must be >= 0.")
    if hasattr(args, "save_attention") and isinstance(args.save_attention, str):
        args.save_attention = args.save_attention.lower() == "true"
    if getattr(args, "lmm_csv", None) in (None, "") and getattr(args, "lmm_results", None):
        args.lmm_csv = args.lmm_results

    cfg = resolve_config(args)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load the full dataset ---
    data = load_data(args.data_file, KEYS)

    # --- Step 2 (MOVED UP): Perform gene whitelist subsetting BEFORE any other action ---
    if args.genes_file:
        import pandas as pd
        try:
            df_wh = pd.read_csv(args.genes_file)
        except Exception as e:
            raise SystemExit(f"[genes-file] Could not read {args.genes_file!r}: {e}")

        # Prefer explicit string names first, then fall back to numeric IDs or the first column.
        for cand in ("gene_name", "gene", "gene_id", "id"):
            if cand in df_wh.columns:
                col = cand
                break
        else:
            col = df_wh.columns[0]
        
        want = set(df_wh[col].astype(str).str.upper().tolist())
        
        gid = data.get('gene_ids')
        if gid is None:
            raise SystemExit("[data-file] The loaded data is missing 'gene_ids' or 'gene_names'. Cannot apply whitelist.")

        gid_uc = [str(x).upper() for x in gid]
        keep_idx = [i for i, g in enumerate(gid_uc) if g in want]

        if not keep_idx:
            raise SystemExit("[genes-file] None of the provided gene IDs matched the dataset gene_ids.")
        
        print(f"[genes-file] Whitelist applied. Subsetting from {len(gid)} to {len(keep_idx)} genes.")

        # --- Subsetting all relevant data structures ---
        data['Y'] = data['Y'][:, keep_idx]
        
        # Remap the cis_map dictionary to the new, smaller indices
        new_cis_map = {}
        for new_i, old_i in enumerate(keep_idx):
            if old_i in data['cis_map']:
                new_cis_map[new_i] = data['cis_map'][old_i]
        data['cis_map'] = new_cis_map
        
        data['gene_ids'] = [gid[i] for i in keep_idx]

    # builder for your EnvConditionalTransformer
    def build_ect(n_snps, n_genes, n_pcs, n_envs, d_model=cfg['D_MODEL'], layers=cfg['LAYERS'], heads=cfg['HEADS'], dropout=cfg['DROPOUT'], mc_dropout=True, use_film=True):
        return EnvironmentConditionalTransformer(n_snps=n_snps, n_genes=n_genes, n_pcs=n_pcs, n_envs=n_envs,
                                         d_model=d_model, n_layers=layers, n_heads=heads, dropout=dropout, mc_dropout=mc_dropout, residual_mode=True, use_film=use_film)

    # --- Step 3: Run Optuna tuning or the main analysis ---
    if args.auto and args.auto > 0:
        # (Optuna logic remains unchanged)
        OUT_DIR = Path(args.out_dir)
        STUDY_DIR = Path(args.study_out) if args.study_out else OUT_DIR / "tune"
        STUDY_DIR.mkdir(parents=True, exist_ok=True)
        cfg = resolve_config(args)
        if not hasattr(args, "alpha_mode"):
            args.alpha_mode = "none"
        study = optuna.create_study(direction="maximize",
                                    sampler=TPESampler(seed=42),
                                    pruner=MedianPruner(n_warmup_steps=4))
        study.optimize(lambda t: _optuna_objective(t, data, args, cfg, OUT_DIR, args.lmm_results),
                       n_trials=args.auto, show_progress_bar=False)
        df = study.trials_dataframe(attrs=("number","value","state","params","user_attrs"))
        df.to_csv(STUDY_DIR / "trials.csv", index=False)
        with open(STUDY_DIR / "best_trial.json", "w") as f:
            json.dump({
                "value": study.best_value,
                "params": study.best_trial.params,
                "attrs": study.best_trial.user_attrs
            }, f, indent=2)
        print("[TUNE] Done. Best score:", study.best_value, "params:", study.best_trial.params)
        print("[TUNE] Results saved to:", str(STUDY_DIR))
        exit()  # stop after tuning

    # --- Step 4: Run the main cross-validation ---
    res = run_ect_residual_cv(
        build_ect, data,
        kfolds=cfg['KFOLDS'],
        seeds=cfg['SEEDS'],
        topk=cfg['TOPK'],
        union_cap=cfg['UNION_CAP'],
        mc_samples=cfg['MC_SAMPLES'],
        lr=cfg['LR'],
        wd=cfg['WD'],
        epochs=cfg['EPOCHS'],
        patience=cfg['PATIENCE'],
        loss_cf_weight=cfg['LOSS_CF_WEIGHT'],
        keep_genes=args.keep_genes,
        args=args,
        out_dir=OUT_DIR,
        save_attention=args.save_attention
    )

    # =============================================================================
    #         FINAL ARTIFACT GENERATION (Corrected for Subsetting)
    # =============================================================================
    print("\n--- Generating final output artifacts ---")

    # The list of gene names corresponding to the results we have.
    # This was correctly subsetted if --genes-file was used.
    gene_ids_final = data.get('gene_ids', [f"GENE_{i}" for i in range(len(res["r2_full_all"]))])

    # Ensure all result arrays have the same length as the final gene list.
    num_genes_in_results = len(gene_ids_final)
    assert len(res["r2_full_all"]) == num_genes_in_results, "Mismatch between R2 results and gene list length"

    # --- 1. Save OOF R2 results ---
    df_r2 = pd.DataFrame({
        "gene_id": gene_ids_final,
        "r2_null": res["r2_null_all"],
        "r2_full": res["r2_full_all"],
        "delta_r2": res["delta_all"]
    })
    r2_path = OUT_DIR / "ect_oof_r2_by_gene.csv"
    df_r2.to_csv(r2_path, index=False)
    print(f"Saved OOF R2 results to {r2_path}")

    # --- 2. Save alpha shrinkage weights ---
    if res.get("alpha_mean_per_gene") is not None and len(res["alpha_mean_per_gene"]) == num_genes_in_results:
        df_alpha = pd.DataFrame({
            "gene_id": gene_ids_final,
            "alpha": res["alpha_mean_per_gene"]
        })
        alpha_path = OUT_DIR / "ect_alpha_by_gene.csv"
        df_alpha.to_csv(alpha_path, index=False)
        print(f"Saved alpha weights to {alpha_path}")

    # --- 3. Save per-environment cis-attention mass ---
    cis_acc = res['cis_acc']
    cis_cnt = res['cis_cnt']

    # Average per-env cis mass across folds
    for k in cis_acc:
        with np.errstate(invalid="ignore", divide="ignore"):
            cis_acc[k] = cis_acc[k] / np.maximum(cis_cnt[k], 1)

    if len(cis_acc["WW"]) == num_genes_in_results:
        df_cis = pd.DataFrame({
            "gene_id": gene_ids_final,
            "cis_mass_WW":  cis_acc["WW"],
            "cis_mass_WS1": cis_acc["WS1"],
            "cis_mass_WS2": cis_acc["WS2"],
        })
        cis_path = OUT_DIR / "ect_cis_mass_by_env.csv"
        df_cis.to_csv(cis_path, index=False)
        print(f"Saved per-env cis mass to {cis_path}")

        # --- 4. Create the joined file for downstream modules ---
        df_join = df_cis.copy()
        df_join["cis_mass"] = df_cis[["cis_mass_WW", "cis_mass_WS1", "cis_mass_WS2"]].median(axis=1)
        df_join = df_join.merge(df_r2.rename(columns={"delta_r2": "delta"}), on="gene_id", how="left")
        
        join_path = OUT_DIR / "ect_cis_mass_joined.csv"
        df_join.to_csv(join_path, index=False)
        print(f"Saved joined cis mass results to {join_path}")

        # --- 5. Optional: Correlate with LMM results ---
        if args.lmm_results and os.path.exists(args.lmm_results):
            try:
                lmm, join_key = _load_lmm_table(args.lmm_results)
                m = df_cis.copy()
                m["gene_id"] = m["gene_id"].astype(str).str.upper()
                M = m.merge(lmm[[join_key, "delta_r2_WW", "delta_r2_WS1", "delta_r2_WS2"]],
                            left_on="gene_id", right_on=join_key, how="inner")
                rows = []
                for env, c_ect, c_lmm in [("WW","cis_mass_WW","delta_r2_WW"),
                                          ("WS1","cis_mass_WS1","delta_r2_WS1"),
                                          ("WS2","cis_mass_WS2","delta_r2_WS2")]:
                    sub = M[[c_ect, c_lmm]].replace([np.inf,-np.inf], np.nan).dropna()
                    rho = _spearman(sub[c_ect], sub[c_lmm]) if len(sub) >= 5 else np.nan
                    rows.append({"env": env, "n": len(sub), "spearman_rho": rho})
                
                corr_path = OUT_DIR / "correlation_summary.csv"
                pd.DataFrame(rows).to_csv(corr_path, index=False)
                print(f"Saved LMM correlation summary to {corr_path}")
            except Exception as e:
                print(f"Could not generate LMM correlation summary: {e}")

        # --- 6. Generate minimal gene modules for Narrative gate ---
        try:
            mod_dir = OUT_DIR / "gene_modules"
            mod_dir.mkdir(parents=True, exist_ok=True)
            q = df_join["cis_mass"].quantile(0.75) if df_join["cis_mass"].notna().any() else np.nan
            if np.isfinite(q):
                df_join["module"] = np.where(df_join["cis_mass"] >= q, 1, 2)
                df_join["module_label"] = np.where(df_join["module"] == 1, "cis-simple", "mixed")
            else:
                df_join["module"] = 2
                df_join["module_label"] = "mixed"

            df_join[["gene_id", "module", "module_label"]].to_csv(mod_dir / "gene_modules.csv", index=False)
            print(f"Saved gene modules to {mod_dir / 'gene_modules.csv'}")
        except Exception as e:
            print(f"Failed to generate gene modules: {e}")

    else:
        print("Skipping final artifact generation due to length mismatch in results.")

    print("\n--- Script finished ---")