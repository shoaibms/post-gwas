#!/usr/bin/env python3
"""
LMM Results Validation and Diagnostic Analysis
==============================================
Diagnose potential issues in LMM analysis results and provide recommendations
for strengthening publication claims.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class LMMResultsValidator:
    """Validate and diagnose LMM analysis results for publication quality."""
    
    def __init__(self, results_path: str, top_genes_path: str):
        self.results_path = Path(results_path)
        self.top_genes_path = Path(top_genes_path)
        self.load_data()
        
    def load_data(self):
        """Load LMM results."""
        print("Loading LMM results for validation...")
        self.results = pd.read_csv(self.results_path)
        self.top_genes = pd.read_csv(self.top_genes_path)
        
        print(f"Loaded {len(self.results)} genes from comprehensive results")
        print(f"Loaded {len(self.top_genes)} top genes")
        
    def diagnose_negative_r2(self):
        """Investigate negative R² values."""
        print("\n" + "="*60)
        print("DIAGNOSING NEGATIVE R² VALUES")
        print("="*60)
        
        # Analyze null model R²
        negative_null = self.results['r2_null'] < 0
        print(f"Genes with negative null R²: {negative_null.sum()}/{len(self.results)} ({100*negative_null.mean():.1f}%)")
        print(f"Range of null R²: {self.results['r2_null'].min():.3f} to {self.results['r2_null'].max():.3f}")
        
        # Analyze full model R²
        negative_full = self.results['r2_full'] < 0
        print(f"Genes with negative full R²: {negative_full.sum()}/{len(self.results)} ({100*negative_full.mean():.1f}%)")
        print(f"Range of full R²: {self.results['r2_full'].min():.3f} to {self.results['r2_full'].max():.3f}")
        
        # Investigate patterns
        print(f"\nMean null R² by improvement status:")
        print(f"  Improved genes:     {self.results[self.results['improved']]['r2_null'].mean():.3f}")
        print(f"  Not improved genes: {self.results[~self.results['improved']]['r2_null'].mean():.3f}")
        
        # Check if negative R² correlates with delta R²
        corr_null_delta = np.corrcoef(self.results['r2_null'], self.results['delta_r2'])[0,1]
        print(f"\nCorrelation between null R² and ΔR²: {corr_null_delta:.3f}")
        
        return negative_null, negative_full
    
    def validate_statistical_significance(self):
        """Validate statistical significance patterns."""
        print("\n" + "="*60)
        print("VALIDATING STATISTICAL SIGNIFICANCE")
        print("="*60)
        
        # Check p-value distribution
        pvals = self.results['pvalue'].dropna()
        print(f"P-value range: {pvals.min():.2e} to {pvals.max():.2e}")
        print(f"P-values < 0.05: {(pvals < 0.05).sum()}/{len(pvals)} ({100*(pvals < 0.05).mean():.1f}%)")
        print(f"P-values < 0.01: {(pvals < 0.01).sum()}/{len(pvals)} ({100*(pvals < 0.01).mean():.1f}%)")
        print(f"P-values < 0.001: {(pvals < 0.001).sum()}/{len(pvals)} ({100*(pvals < 0.001).mean():.1f}%)")
        
        # FDR correction validation
        fdr_vals = self.results['pvalue_fdr'].dropna()
        print(f"\nFDR-corrected p-values < 0.05: {(fdr_vals < 0.05).sum()}/{len(fdr_vals)} ({100*(fdr_vals < 0.05).mean():.1f}%)")
        
        # Check for uniformity under null (should be uniform if no true effects)
        ks_stat, ks_pval = stats.kstest(pvals, 'uniform')
        print(f"\nKolmogorov-Smirnov test for uniformity:")
        print(f"  KS statistic: {ks_stat:.3f}")
        print(f"  KS p-value: {ks_pval:.3e}")
        if ks_pval < 0.05:
            print("  Significant deviation from uniform (expected with true effects)")
        else:
            print("  Distribution not significantly different from uniform")
    
    def check_effect_size_consistency(self):
        """Check consistency of effect sizes."""
        print("\n" + "="*60)
        print("CHECKING EFFECT SIZE CONSISTENCY")
        print("="*60)
        
        # Delta R² distribution
        delta_r2 = self.results['delta_r2']
        print(f"ΔR² statistics:")
        print(f"  Mean: {delta_r2.mean():.4f}")
        print(f"  Median: {delta_r2.median():.4f}")
        print(f"  Std: {delta_r2.std():.4f}")
        print(f"  Range: {delta_r2.min():.4f} to {delta_r2.max():.4f}")
        
        # Check for outliers
        q75, q25 = np.percentile(delta_r2, [75, 25])
        iqr = q75 - q25
        outlier_threshold = q75 + 1.5 * iqr
        outliers = delta_r2 > outlier_threshold
        print(f"\nOutlier analysis (IQR method):")
        print(f"  Outlier threshold: {outlier_threshold:.3f}")
        print(f"  Outliers: {outliers.sum()}/{len(delta_r2)} ({100*outliers.mean():.1f}%)")
        
        # Confidence interval consistency
        ci_width = self.results['delta_r2_ci_upper'] - self.results['delta_r2_ci_lower']
        print(f"\nConfidence interval analysis:")
        print(f"  Mean CI width: {ci_width.mean():.4f}")
        print(f"  Median CI width: {ci_width.median():.4f}")
        
        # Check if point estimates fall within CIs
        in_ci = ((self.results['delta_r2'] >= self.results['delta_r2_ci_lower']) & 
                 (self.results['delta_r2'] <= self.results['delta_r2_ci_upper']))
        print(f"  Point estimates within CI: {in_ci.sum()}/{len(in_ci)} ({100*in_ci.mean():.1f}%)")
    
    def generate_diagnostic_plots(self, output_dir: str = "output/lmm_diagnostics"):
        """Generate diagnostic plots."""
        print("\n" + "="*60)
        print("GENERATING DIAGNOSTIC PLOTS")
        print("="*60)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: P-value histogram
        ax = axes[0, 0]
        ax.hist(self.results['pvalue'].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('P-value')
        ax.set_ylabel('Frequency')
        ax.set_title('P-value Distribution')
        ax.axhline(y=len(self.results)/50, color='red', linestyle='--', alpha=0.7, label='Expected uniform')
        ax.legend()
        
        # Plot 2: Q-Q plot for p-values
        ax = axes[0, 1]
        pvals_sorted = np.sort(self.results['pvalue'].dropna())
        expected = np.linspace(0, 1, len(pvals_sorted))
        ax.plot(expected, pvals_sorted, 'o', alpha=0.5)
        ax.plot([0, 1], [0, 1], 'r--', label='Expected uniform')
        ax.set_xlabel('Expected p-value')
        ax.set_ylabel('Observed p-value')
        ax.set_title('Q-Q Plot: P-values vs Uniform')
        ax.legend()
        
        # Plot 3: R² distributions
        ax = axes[0, 2]
        ax.hist(self.results['r2_null'], bins=30, alpha=0.5, label='Null model', color='red')
        ax.hist(self.results['r2_full'], bins=30, alpha=0.5, label='Full model', color='blue')
        ax.set_xlabel('R²')
        ax.set_ylabel('Frequency')
        ax.set_title('R² Distributions')
        ax.legend()
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 4: Delta R² vs Null R²
        ax = axes[1, 0]
        scatter = ax.scatter(self.results['r2_null'], self.results['delta_r2'], 
                           c=self.results['improved'], cmap='RdYlBu', alpha=0.6)
        ax.set_xlabel('Null R²')
        ax.set_ylabel('ΔR²')
        ax.set_title('Improvement vs Baseline')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=ax, label='Improved')
        
        # Plot 5: Effect size vs significance
        ax = axes[1, 1]
        ax.scatter(self.results['delta_r2'], -np.log10(self.results['pvalue'] + 1e-100), 
                  c=self.results['significant'], cmap='RdYlBu', alpha=0.6)
        ax.set_xlabel('ΔR²')
        ax.set_ylabel('-log₁₀(p-value)')
        ax.set_title('Volcano Plot: Effect vs Significance')
        ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.legend()
        
        # Plot 6: Confidence interval coverage
        ax = axes[1, 2]
        ci_coverage = ((self.results['delta_r2'] >= self.results['delta_r2_ci_lower']) & 
                      (self.results['delta_r2'] <= self.results['delta_r2_ci_upper']))
        coverage_rate = ci_coverage.mean()
        
        ax.barh(['Within CI', 'Outside CI'], 
               [ci_coverage.sum(), (~ci_coverage).sum()],
               color=['green', 'red'], alpha=0.7)
        ax.set_xlabel('Number of Genes')
        ax.set_title(f'95% CI Coverage: {coverage_rate:.1%}')
        
        plt.tight_layout()
        plt.savefig(output_path / 'lmm_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_path / 'lmm_diagnostics.pdf', bbox_inches='tight')
        print(f"Diagnostic plots saved to {output_path}")
        plt.close()
    
    def generate_recommendations(self):
        """Generate recommendations for improving analysis."""
        print("\n" + "="*60)
        print("RECOMMENDATIONS FOR PUBLICATION")
        print("="*60)
        
        # Check key issues
        negative_null, negative_full = self.diagnose_negative_r2()
        
        recommendations = []
        
        # Issue 1: Negative R²
        if negative_null.sum() > len(self.results) * 0.1:  # More than 10% negative
            recommendations.append(
                "[CRITICAL] CRITICAL: High frequency of negative R² values suggests model misspecification"
            )
            recommendations.append(
                "   → Investigate: sample size per fold, covariate scaling, model convergence"
            )
            recommendations.append(
                "   → Consider: simpler baseline models, different CV strategy"
            )
        
        # Issue 2: Universal significance
        if self.results['significant'].mean() > 0.95:  # More than 95% significant
            recommendations.append(
                "[WARNING] WARNING: Universal significance is unusual and may indicate issues"
            )
            recommendations.append(
                "   → Validate: permutation tests, independent dataset replication"
            )
        
        # Issue 3: Effect size validation
        mean_delta = self.results['delta_r2'].mean()
        if mean_delta > 0.15:  # Very large effect
            recommendations.append(
                f"[OK] STRENGTH: Large effect size (ΔR² = {mean_delta:.3f}) is impressive"
            )
            recommendations.append(
                "   → Validate: independent cohort, literature comparison"
            )
        
        # Environment-specific issues
        env_cols = ['env_WW_r2', 'env_WS1_r2', 'env_WS2_r2']
        if all(col in self.results.columns for col in env_cols):
            env_means = [self.results[col].mean() for col in env_cols]
            if min(env_means) < -0.5:  # Very negative environment performance
                recommendations.append(
                    "[CRITICAL] CRITICAL: Severe negative R² in some environments"
                )
                recommendations.append(
                    "   → Investigate: environment-specific overfitting, sample stratification"
                )
        
        # General recommendations
        recommendations.extend([
            "\n GENERAL RECOMMENDATIONS:",
            "   1. Validate results with permutation tests",
            "   2. Test on independent maize population",
            "   3. Compare against published eQTL effect sizes",
            "   4. Investigate model assumptions and convergence",
            "   5. Consider more conservative significance thresholds"
        ])
        
        for rec in recommendations:
            print(rec)
    
    def run_full_validation(self, output_dir: str = "output/lmm_diagnostics"):
        """Run complete validation pipeline."""
        print("[INFO] RUNNING COMPREHENSIVE LMM VALIDATION")
        print("="*80)
        
        # Run all diagnostic checks
        self.diagnose_negative_r2()
        self.validate_statistical_significance()
        self.check_effect_size_consistency()
        
        # Generate plots
        self.generate_diagnostic_plots(output_dir)
        
        # Provide recommendations
        self.generate_recommendations()
        
        # Final assessment
        print("\n" + "="*60)
        print("PUBLICATION READINESS ASSESSMENT")
        print("="*60)
        
        # Calculate quality score
        quality_factors = []
        
        # Factor 1: Effect size magnitude
        mean_delta = self.results['delta_r2'].mean()
        if mean_delta > 0.15:
            quality_factors.append(("Large effect size", 2))
        elif mean_delta > 0.05:
            quality_factors.append(("Moderate effect size", 1))
        else:
            quality_factors.append(("Small effect size", 0))
        
        # Factor 2: Statistical rigor
        if self.results['significant'].mean() < 0.8:
            quality_factors.append(("Reasonable significance rate", 1))
        else:
            quality_factors.append(("High significance rate", 0))
        
        # Factor 3: Negative R² prevalence
        negative_rate = (self.results['r2_null'] < 0).mean()
        if negative_rate < 0.1:
            quality_factors.append(("Low negative R² rate", 1))
        else:
            quality_factors.append(("High negative R² rate", -1))
        
        total_score = sum(score for _, score in quality_factors)
        max_score = 4
        
        print(f"Quality Assessment:")
        for factor, score in quality_factors:
            print(f"  {factor}: {score}")
        print(f"\nOverall Score: {total_score}/{max_score}")
        
        if total_score >= 3:
            print("[OK] ASSESSMENT: Strong candidate for Nature Methods/Communications")
        elif total_score >= 2:
            print("[WARNING] ASSESSMENT: Good candidate with revisions needed")
        else:
            print("[CRITICAL] ASSESSMENT: Requires substantial improvements before submission")


def main():
    """Run LMM validation analysis."""
    
    # File paths (adjust for your system)
    results_path = r"C:\Users\ms\Desktop\gwas\output\lmm_nature_methods\tables\lmm_comprehensive_results.csv"
    top_genes_path = r"C:\Users\ms\Desktop\gwas\output\lmm_nature_methods\tables\TableS1_all_genes.csv"
    output_dir = r"C:\Users\ms\Desktop\gwas\output\lmm_nature_methods\lmm_validation"
    
    # Run validation
    validator = LMMResultsValidator(results_path, top_genes_path)
    validator.run_full_validation(output_dir)


if __name__ == "__main__":
    main()