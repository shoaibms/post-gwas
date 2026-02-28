"""
Network Resilience Score (NRS) prediction using PC1 phenotype.

Uses 500-gene panel from transformer bundle for PC1 computation,
top SNPs for feature selection, and cross-validated Ridge/ElasticNet models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
from joblib import Parallel, delayed

BASE_DIR = Path(r"C:\Users\ms\Desktop\gwas")
DATA_DIR = BASE_DIR / "output" / "week5_nrs"
FILTERED_DIR = BASE_DIR / "output" / "data_filtered"
OUTPUT_DIR = DATA_DIR
FIGURES_DIR = DATA_DIR / "figures"

N_FOLDS = 5
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
# Keep BLAS threads bounded to avoid oversubscription when we parallelise folds
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
N_JOBS = max(1, min(os.cpu_count() - 2, 36))

print("="*60)
print("WEEK 5: NRS FOCUSED ANALYSIS")
print("="*60)


def compute_pc1_phenotype():
    """Compute PC1 of expression changes as phenotype."""
    print("\n1. Computing PC1 phenotype...")
    
    # Load 500-gene panel from transformer bundle (avoids circularity)
    import torch
    bundle_path = BASE_DIR / "output" / "transformer_data" / "transformer_data_win1Mb.pt"
    bundle = torch.load(bundle_path, map_location="cpu")
    panel_genes = bundle['gene_names']
    
    ww_expr = pd.read_csv(FILTERED_DIR / "WW_209-Uniq_FPKM.agpv4.txt.gz",
                          sep='\t', compression='gzip', index_col=0).T
    ws2_expr = pd.read_csv(FILTERED_DIR / "WS2_210-uniq_FPKM.agpv4.txt.gz",
                           sep='\t', compression='gzip', index_col=0).T
    
    common_samples = list(set(ww_expr.columns) & set(ws2_expr.columns))
    common_genes = list(set(panel_genes) & set(ww_expr.index) & set(ws2_expr.index))

    print(f"  Using {len(common_genes)} genes for PC1 computation")
    
    ww_sub = ww_expr.loc[common_genes, common_samples]
    ws2_sub = ws2_expr.loc[common_genes, common_samples]
    
    delta_expr = (ws2_sub - ww_sub).T
    
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(delta_expr).flatten()
    phenotype = pd.Series(pc1, index=delta_expr.index, name='pc1_drought_response')
    
    print(f"  PC1 variance explained: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"  PC1 mean: {phenotype.mean():.3f}")
    print(f"  PC1 std: {phenotype.std():.3f}")
    
    return phenotype


def load_data_focused():
    """Load genotype and PC1 phenotype."""
    print("\n2. Loading genotype data...")
    
    genotype_df = pd.read_csv(DATA_DIR / "genotype_matrix.csv", index_col=0)
    genotype_df = genotype_df.apply(pd.to_numeric, errors='coerce')
    
    phenotype = compute_pc1_phenotype()
    
    common_samples = genotype_df.index.intersection(phenotype.index)
    
    genotype_df = genotype_df.loc[common_samples]
    phenotype = phenotype.loc[common_samples]
    
    print(f"  Aligned: {len(common_samples)} samples")
    print(f"  Genotype: {genotype_df.shape}")
    
    return genotype_df, phenotype


def select_top_snps(X, y, n_snps=15):
    """Select top N most predictive SNPs."""
    print(f"\n3. Selecting top {n_snps} SNPs...")
    
    selector = SelectKBest(f_regression, k=n_snps)
    selector.fit(X, y)
    
    selected_mask = selector.get_support()
    selected_snps = X.columns[selected_mask].tolist()
    
    scores = selector.scores_
    top_snps_df = pd.DataFrame({
        'snp': X.columns,
        'f_score': scores,
        'selected': selected_mask
    }).sort_values('f_score', ascending=False)
    
    print(f"  Top {n_snps} SNPs selected")
    print(f"  F-scores range: {scores[selected_mask].min():.2f} - {scores[selected_mask].max():.2f}")
    
    return X[selected_snps], top_snps_df


def cross_validate_model(X, y, model, model_name="Model"):
    """Cross-validate model with proper metrics (parallel across folds)."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_indices = list(kf.split(X))

    def _one_fold(train_idx, test_idx):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        mdl = model.__class__(**model.get_params())
        mdl.fit(X_train_scaled, y_train)
        y_pred = mdl.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        return test_idx, r2, y_pred

    results = Parallel(n_jobs=N_JOBS, backend='loky', prefer='processes')(
        delayed(_one_fold)(tr, te) for tr, te in fold_indices
    )
    r2_scores = []
    predictions = np.zeros(len(y))
    for test_idx, r2, y_pred in results:
        r2_scores.append(r2)
        predictions[test_idx] = y_pred

    mean_r2 = float(np.mean(r2_scores))
    std_r2 = float(np.std(r2_scores))
    print(f"  {model_name}: R2 = {mean_r2:.4f} +/- {std_r2:.4f}")
    return {
        'model_name': model_name,
        'r2_scores': r2_scores,
        'mean_r2': mean_r2,
        'std_r2': std_r2,
        'predictions': predictions
    }


def run_models(X, y):
    """Run focused set of models."""
    print("\n4. Training models...")
    
    results = {}
    
    print("\nModel 1: Null (mean prediction)")
    null_pred = np.full(len(y), y.mean())
    null_r2_per_fold = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, test_idx in kf.split(y):
        y_test = y.iloc[test_idx]
        null_r2_per_fold.append(r2_score(y_test, np.full(len(y_test), y.iloc[train_idx].mean())))
    
    results['Null'] = {
        'model_name': 'Null',
        'r2_scores': null_r2_per_fold,
        'mean_r2': np.mean(null_r2_per_fold),
        'std_r2': np.std(null_r2_per_fold),
        'predictions': null_pred
    }
    print(f"  Null: R2 = {results['Null']['mean_r2']:.4f} +/- {results['Null']['std_r2']:.4f}")
    
    print("\nModel 2: Ridge (strong regularization)")
    results['Ridge'] = cross_validate_model(X, y, Ridge(alpha=10.0), "Ridge")
    
    print("\nModel 3: Ridge (moderate regularization)")
    results['Ridge_Mod'] = cross_validate_model(X, y, Ridge(alpha=1.0), "Ridge_Mod")
    
    print("\nModel 4: ElasticNet")
    results['ElasticNet'] = cross_validate_model(X, y, 
                                                  ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000),
                                                  "ElasticNet")
    
    return results


def compare_models(results, y):
    """Compare model performance."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    tcrit = stats.t.ppf(0.975, df=N_FOLDS-1)
    comparison_df = pd.DataFrame([
        {
            'model': k,
            'mean_r2': v['mean_r2'],
            'std_r2': v['std_r2'],
            'ci_lower': v['mean_r2'] - tcrit * v['std_r2'] / np.sqrt(N_FOLDS),
            'ci_upper': v['mean_r2'] + tcrit * v['std_r2'] / np.sqrt(N_FOLDS)
        }
        for k, v in results.items()
    ]).sort_values('mean_r2', ascending=False)
    
    print("\n" + comparison_df.to_string(index=False))
    
    best_name = comparison_df.iloc[0]['model']
    best = results[best_name]
    null = results['Null']
    
    delta_r2 = np.array(best['r2_scores']) - np.array(null['r2_scores'])
    mean_delta = np.mean(delta_r2)
    std_delta = np.std(delta_r2)
    
    t_stat, p_value = stats.ttest_rel(best['r2_scores'], null['r2_scores'])
    
    tcrit = stats.t.ppf(0.975, df=N_FOLDS-1)
    ci_lower = mean_delta - tcrit * std_delta / np.sqrt(N_FOLDS)
    ci_upper = mean_delta + tcrit * std_delta / np.sqrt(N_FOLDS)
    
    print(f"\n" + "="*60)
    print("BEST vs NULL")
    print("="*60)
    print(f"Best: {best_name}")
    print(f"  R2 = {best['mean_r2']:.4f} +/- {best['std_r2']:.4f}")
    print(f"Null:")
    print(f"  R2 = {null['mean_r2']:.4f} +/- {null['std_r2']:.4f}")
    print(f"\ndelta_R2 = {mean_delta:.4f} (95% CI [{ci_lower:.4f}, {ci_upper:.4f}]); "
          f"paired t-test p = {p_value:.3e}")
    
    if mean_delta >= 0.10 and p_value < 0.01 and ci_lower > 0:
        decision = "STRONG"
        target = "Nature Plants consideration"
    elif mean_delta >= 0.05 and p_value < 0.05 and ci_lower > 0:
        decision = "ADEQUATE"
        target = "Plant Cell / Genome Biology"
    else:
        decision = "WEAK"
        target = "Emphasize mechanistic validations"
    
    print(f"\n" + "="*60)
    print(f"GATE 5: {decision}")
    print(f"Target: {target}")
    print("="*60)
    
    return {
        'best_model': best_name,
        'delta_r2': mean_delta,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'decision': decision,
        'target': target
    }, comparison_df


def plot_results(results, y, comparison_df, output_path):
    """Create publication figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    best_name = comparison_df.iloc[0]['model']
    best = results[best_name]
    
    axes[0, 0].scatter(y, best['predictions'], alpha=0.6, s=50, 
                      edgecolors='black', linewidths=0.5, color='forestgreen')
    min_val, max_val = y.min(), y.max()
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('Observed PC1', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Predicted PC1', fontsize=12, fontweight='bold')
    axes[0, 0].set_title(f'Best Model: {best_name}\nR2 = {best["mean_r2"]:.3f}',
                        fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(y, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 1].set_xlabel('PC1 Value', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Phenotype Distribution', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, axis='y', alpha=0.3)
    
    sorted_df = comparison_df.sort_values('mean_r2', ascending=True)
    colors = ['gray' if m == 'Null' else 'forestgreen' for m in sorted_df['model']]
    
    y_pos = np.arange(len(sorted_df))
    axes[1, 0].barh(y_pos, sorted_df['mean_r2'], xerr=sorted_df['std_r2'],
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.8, capsize=5)
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels(sorted_df['model'], fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Cross-Validated R2', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Model Comparison', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, axis='x', alpha=0.3)
    axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    for i, (bar, mean) in enumerate(zip(axes[1, 0].patches, sorted_df['mean_r2'])):
        width = bar.get_width()
        if width >= 0:
            axes[1, 0].text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                          f'{mean:.3f}', ha='left', va='center', fontweight='bold')
        else:
            axes[1, 0].text(width - 0.01, bar.get_y() + bar.get_height()/2.,
                          f'{mean:.3f}', ha='right', va='center', fontweight='bold')
    
    fold_nums = list(range(1, N_FOLDS + 1))
    for name, res in results.items():
        color = 'gray' if name == 'Null' else 'forestgreen' if name == best_name else 'steelblue'
        linestyle = '--' if name == 'Null' else '-'
        linewidth = 2.5 if name == best_name else 1.5
        axes[1, 1].plot(fold_nums, res['r2_scores'], 'o-', 
                       label=name, color=color, linestyle=linestyle,
                       linewidth=linewidth, markersize=7)
    
    axes[1, 1].set_xlabel('Fold Number', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('R2 Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Cross-Validation Performance', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    axes[1, 1].set_xticks(fold_nums)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")


def save_results(results, comparison_df, summary, top_snps_df):
    """Save all results."""
    print("\n5. Saving results...")
    
    comparison_df.to_csv(OUTPUT_DIR / "nrs_performance_focused.csv", index=False)
    
    summary_df = pd.DataFrame([{
        'best_model': summary['best_model'],
        'delta_r2': summary['delta_r2'],
        'ci_lower': summary['ci_lower'],
        'ci_upper': summary['ci_upper'],
        'p_value': summary['p_value'],
        'decision': summary['decision'],
        'target': summary['target']
    }])
    summary_df.to_csv(OUTPUT_DIR / "nrs_comparison_focused.csv", index=False)
    
    gate_df = pd.DataFrame([{
        'gate': 'Gate 5',
        'approach': 'Focused (PC1 + Top SNPs)',
        'decision': summary['decision'],
        'delta_r2': summary['delta_r2'],
        'p_value': summary['p_value'],
        'target': summary['target']
    }])
    gate_df.to_csv(OUTPUT_DIR / "gate5_decision_focused.csv", index=False)
    
    top_snps_df.to_csv(OUTPUT_DIR / "top_snps_selected.csv", index=False)
    
    print("  Saved all results")


def main():
    """Main execution."""
    X_full, y = load_data_focused()
    
    X, top_snps_df = select_top_snps(X_full, y, n_snps=15)
    
    results = run_models(X, y)
    
    summary, comparison_df = compare_models(results, y)
    
    plot_results(results, y, comparison_df, 
                FIGURES_DIR / "figure5_nrs_focused.png")
    
    save_results(results, comparison_df, summary, top_snps_df)
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    if summary['decision'] == 'STRONG':
        print("\nSTRONG predictive signal detected")
        print("  Network modulator SNPs show significant breeding value")
    elif summary['decision'] == 'ADEQUATE':
        print("\nADEQUATE predictive signal detected")
        print("  Network modulator SNPs show meaningful breeding potential")
    else:
        print("\nLIMITED predictive signal")
        print("  This does NOT invalidate your mechanistic discoveries!")
        print("\n  Your Week 1-4 validations remain strong:")
        print("    - 7.85x window enrichment (p<1e-28)")
        print("    - 3.48x regulatory enrichment (p<1e-14)")
        print("    - 2.93x TF proximity enrichment (p=0.002)")
        print("    - Perfect functional decoupling (Jaccard=0)")
        print("\n  Weak prediction suggests:")
        print("    - Expression changes are downstream of genetic effects")
        print("    - Sample size limits polygenic prediction")
        print("    - Trans-acting effects not captured by cis-SNPs")
        print("\n  Publication strategy:")
        print("    - Lead with architectural discovery")
        print("    - Emphasize mechanistic validations")
        print("    - Present NRS as exploratory/developmental")
        print("    - Target: Plant Cell / Genome Biology")
    
    print("\n" + "="*60)
    print("WEEK 5 COMPLETE")
    print("="*60)
    print("\nNext: Proceed to Week 6 manuscript preparation")


if __name__ == "__main__":
    main()