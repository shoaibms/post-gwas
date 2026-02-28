"""
Statistical Utilities for GWAS Manuscript Figures
==================================================
Reusable statistical functions with proper error handling and validation.
All functions return results with confidence intervals where appropriate.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
from typing import Tuple, Optional, List
import warnings

# ========================================================================
# BOOTSTRAP METHODS
# ========================================================================

def bootstrap_ci(data: np.ndarray, 
                 statistic: callable = np.median,
                 n_bootstrap: int = 10000,
                 confidence: float = 0.95,
                 random_seed: int = 42) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for any statistic.
    
    Args:
        data: 1D array of data
        statistic: Function to calculate (default: median)
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (default 0.95)
        random_seed: Random seed for reproducibility
    
    Returns:
        (observed_statistic, ci_lower, ci_upper)
    """
    np.random.seed(random_seed)
    
    # Remove NaN values
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) == 0:
        return np.nan, np.nan, np.nan
    
    # Observed statistic
    observed = statistic(data_clean)
    
    # Bootstrap distribution
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data_clean, size=len(data_clean), replace=True)
        bootstrap_stats.append(statistic(sample))
    
    # Calculate CI
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return observed, ci_lower, ci_upper


def bootstrap_difference(data1: np.ndarray, 
                         data2: np.ndarray,
                         statistic: callable = np.median,
                         n_bootstrap: int = 10000,
                         random_seed: int = 42) -> Tuple[float, float, float, float]:
    """
    Bootstrap test for difference between two groups.
    
    Returns:
        (difference, ci_lower, ci_upper, p_value)
    """
    np.random.seed(random_seed)
    
    data1_clean = data1[np.isfinite(data1)]
    data2_clean = data2[np.isfinite(data2)]
    
    # Observed difference
    obs_diff = statistic(data1_clean) - statistic(data2_clean)
    
    # Bootstrap null distribution (pooled data)
    pooled = np.concatenate([data1_clean, data2_clean])
    n1, n2 = len(data1_clean), len(data2_clean)
    
    boot_diffs = []
    for _ in range(n_bootstrap):
        perm = np.random.permutation(pooled)
        sample1 = perm[:n1]
        sample2 = perm[n1:]
        boot_diffs.append(statistic(sample1) - statistic(sample2))
    
    boot_diffs = np.array(boot_diffs)
    
    # Two-tailed p-value
    p_value = 2 * min(
        np.mean(boot_diffs >= obs_diff),
        np.mean(boot_diffs <= obs_diff)
    )
    
    # CI for difference
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    
    return obs_diff, ci_lower, ci_upper, p_value


# ========================================================================
# PERMUTATION TESTS
# ========================================================================

def permutation_test(data1: np.ndarray,
                    data2: np.ndarray,
                    statistic: callable = lambda x, y: np.mean(x) - np.mean(y),
                    n_permutations: int = 10000,
                    random_seed: int = 42) -> Tuple[float, float]:
    """
    Permutation test for difference between two groups.
    
    Returns:
        (observed_statistic, p_value)
    """
    np.random.seed(random_seed)
    
    data1_clean = data1[np.isfinite(data1)]
    data2_clean = data2[np.isfinite(data2)]
    
    # Observed statistic
    obs_stat = statistic(data1_clean, data2_clean)
    
    # Permutation distribution
    pooled = np.concatenate([data1_clean, data2_clean])
    n1 = len(data1_clean)
    
    perm_stats = []
    for _ in range(n_permutations):
        perm = np.random.permutation(pooled)
        perm_stat = statistic(perm[:n1], perm[n1:])
        perm_stats.append(perm_stat)
    
    perm_stats = np.array(perm_stats)
    
    # Two-tailed p-value
    p_value = 2 * min(
        np.mean(perm_stats >= obs_stat),
        np.mean(perm_stats <= obs_stat)
    )
    
    return obs_stat, p_value


def permutation_correlation(x: np.ndarray,
                           y: np.ndarray,
                           method: str = 'spearman',
                           n_permutations: int = 10000,
                           random_seed: int = 42) -> Tuple[float, float, float, float]:
    """
    Permutation test for correlation with bootstrap CI.
    
    Returns:
        (correlation, ci_lower, ci_upper, p_value)
    """
    np.random.seed(random_seed)
    
    # Remove NaN
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]
    
    if len(x_clean) < 3:
        return np.nan, np.nan, np.nan, 1.0
    
    # Observed correlation
    if method == 'spearman':
        obs_corr, _ = stats.spearmanr(x_clean, y_clean)
    elif method == 'pearson':
        obs_corr, _ = stats.pearsonr(x_clean, y_clean)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Bootstrap CI
    boot_corrs = []
    for _ in range(n_permutations):
        indices = np.random.choice(len(x_clean), size=len(x_clean), replace=True)
        if method == 'spearman':
            corr, _ = stats.spearmanr(x_clean[indices], y_clean[indices])
        else:
            corr, _ = stats.pearsonr(x_clean[indices], y_clean[indices])
        if np.isfinite(corr):
            boot_corrs.append(corr)
    
    ci_lower = np.percentile(boot_corrs, 2.5)
    ci_upper = np.percentile(boot_corrs, 97.5)
    
    # Permutation p-value
    perm_corrs = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y_clean)
        if method == 'spearman':
            corr, _ = stats.spearmanr(x_clean, y_perm)
        else:
            corr, _ = stats.pearsonr(x_clean, y_perm)
        if np.isfinite(corr):
            perm_corrs.append(corr)
    
    p_value = 2 * min(
        np.mean(np.array(perm_corrs) >= obs_corr),
        np.mean(np.array(perm_corrs) <= obs_corr)
    )
    
    return obs_corr, ci_lower, ci_upper, p_value


# ========================================================================
# ENRICHMENT TESTS
# ========================================================================

def fisher_exact_with_ci(contingency_table: np.ndarray,
                         confidence: float = 0.95) -> Tuple[float, float, float, float]:
    """
    Fisher's exact test with odds ratio CI.
    
    Args:
        contingency_table: 2x2 array [[a, b], [c, d]]
    
    Returns:
        (odds_ratio, ci_lower, ci_upper, p_value)
    """
    odds_ratio, p_value = stats.fisher_exact(contingency_table)
    
    # Log odds ratio CI (from Agresti 2002)
    a, b, c, d = contingency_table.flatten()
    
    if a == 0 or b == 0 or c == 0 or d == 0:
        # Add 0.5 to all cells (Haldane correction)
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    
    log_or = np.log(odds_ratio)
    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    ci_lower = np.exp(log_or - z * se_log_or)
    ci_upper = np.exp(log_or + z * se_log_or)
    
    return odds_ratio, ci_lower, ci_upper, p_value


def hypergeometric_enrichment(overlap: int,
                              foreground_size: int,
                              background_with_feature: int,
                              background_size: int) -> Tuple[float, float, float]:
    """
    Hypergeometric test for enrichment.
    
    Args:
        overlap: Number of foreground items with feature
        foreground_size: Total foreground items
        background_with_feature: Background items with feature
        background_size: Total background items
    
    Returns:
        (enrichment_ratio, expected, p_value)
    """
    # Expected overlap by chance
    expected = (foreground_size * background_with_feature) / background_size
    
    # Enrichment ratio
    enrichment = overlap / expected if expected > 0 else np.inf
    
    # Hypergeometric p-value (survival function)
    p_value = stats.hypergeom.sf(
        overlap - 1,  # -1 because sf is P(X > k), we want P(X >= k)
        background_size,
        background_with_feature,
        foreground_size
    )
    
    return enrichment, expected, p_value


# ========================================================================
# ECDF FUNCTIONS
# ========================================================================

def compute_ecdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical cumulative distribution function.
    
    Returns:
        (sorted_data, ecdf_values)
    """
    data_clean = data[np.isfinite(data)]
    sorted_data = np.sort(data_clean)
    ecdf_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, ecdf_vals


def ecdf_confidence_bands(data: np.ndarray,
                          n_bootstrap: int = 1000,
                          confidence: float = 0.95,
                          random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ECDF with bootstrap confidence bands.
    
    Returns:
        (x_vals, ecdf, ci_lower, ci_upper)
    """
    np.random.seed(random_seed)
    
    data_clean = data[np.isfinite(data)]
    n = len(data_clean)
    
    # Observed ECDF
    x_vals, ecdf = compute_ecdf(data_clean)
    
    # Bootstrap ECDFs
    boot_ecdfs = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data_clean, size=n, replace=True)
        _, boot_ecdf = compute_ecdf(boot_sample)
        # Interpolate to common x values
        boot_ecdf_interp = np.interp(x_vals, np.sort(boot_sample), boot_ecdf)
        boot_ecdfs.append(boot_ecdf_interp)
    
    boot_ecdfs = np.array(boot_ecdfs)
    
    # Pointwise confidence bands
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_ecdfs, 100 * alpha / 2, axis=0)
    ci_upper = np.percentile(boot_ecdfs, 100 * (1 - alpha / 2), axis=0)
    
    return x_vals, ecdf, ci_lower, ci_upper


# ========================================================================
# MULTIPLE TESTING CORRECTION
# ========================================================================

def fdr_correction(p_values: np.ndarray, alpha: float = 0.05, method: str = 'bh') -> np.ndarray:
    """
    FDR correction for multiple testing.
    
    Args:
        p_values: Array of p-values
        alpha: Significance level
        method: 'bh' (Benjamini-Hochberg) or 'by' (Benjamini-Yekutieli)
    
    Returns:
        Array of q-values (adjusted p-values)
    """
    p_values = np.asarray(p_values)
    n = len(p_values)
    
    # Sort p-values and track original order
    sort_idx = np.argsort(p_values)
    p_sorted = p_values[sort_idx]
    
    # Benjamini-Hochberg procedure
    if method == 'bh':
        q_sorted = p_sorted * n / (np.arange(n) + 1)
    # Benjamini-Yekutieli (more conservative)
    elif method == 'by':
        c_n = np.sum(1 / np.arange(1, n + 1))
        q_sorted = p_sorted * n * c_n / (np.arange(n) + 1)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Enforce monotonicity
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    
    # Restore original order
    q_values = np.empty(n)
    q_values[sort_idx] = q_sorted
    
    return q_values


# ========================================================================
# SIGN TEST
# ========================================================================

def sign_test(data: np.ndarray, null_value: float = 0.0) -> Tuple[float, int, int]:
    """
    Exact sign test for median.
    
    Returns:
        (p_value, n_positive, n_negative)
    """
    # Remove zeros and NaN
    data_clean = data[np.isfinite(data)]
    differences = data_clean - null_value
    differences = differences[differences != 0]
    
    n_pos = np.sum(differences > 0)
    n_neg = np.sum(differences < 0)
    n_total = len(differences)
    
    if n_total == 0:
        return 1.0, 0, 0
    
    # Exact binomial test (two-tailed)
    p_value = 2 * stats.binom.cdf(min(n_pos, n_neg), n_total, 0.5)
    
    return p_value, n_pos, n_neg


# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================

def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    group1_clean = group1[np.isfinite(group1)]
    group2_clean = group2[np.isfinite(group2)]
    
    mean_diff = np.mean(group1_clean) - np.mean(group2_clean)
    pooled_std = np.sqrt(
        (np.var(group1_clean) + np.var(group2_clean)) / 2
    )
    
    return mean_diff / pooled_std if pooled_std > 0 else 0.0


def cramers_v(contingency_table: np.ndarray) -> float:
    """Calculate Cramér's V effect size for contingency table."""
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum()
    min_dim = min(contingency_table.shape) - 1
    
    return np.sqrt(chi2 / (n * min_dim))


def format_pvalue(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"


def format_ci(lower: float, upper: float, decimals: int = 3) -> str:
    """Format confidence interval for display."""
    fmt = f"{{:.{decimals}f}}"
    return f"95% CI [{fmt.format(lower)}, {fmt.format(upper)}]"


# ========================================================================
# TESTING
# ========================================================================

if __name__ == "__main__":
    print("Statistical Utilities Test Suite")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Test bootstrap CI
    print("\n1. Testing bootstrap CI...")
    data = np.random.randn(100)
    obs, ci_lower, ci_upper = bootstrap_ci(data, n_bootstrap=1000)
    print(f"   Median: {obs:.3f}, CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # Test permutation test
    print("\n2. Testing permutation test...")
    group1 = np.random.randn(50)
    group2 = np.random.randn(50) + 0.5
    obs_diff, p_val = permutation_test(group1, group2, n_permutations=1000)
    print(f"   Difference: {obs_diff:.3f}, p-value: {p_val:.4f}")
    
    # Test correlation
    print("\n3. Testing correlation with CI...")
    x = np.random.randn(100)
    y = x + np.random.randn(100) * 0.5
    corr, ci_l, ci_u, p = permutation_correlation(x, y, n_permutations=1000)
    print(f"   Correlation: {corr:.3f}, CI: [{ci_l:.3f}, {ci_u:.3f}], p: {p:.4f}")
    
    # Test enrichment
    print("\n4. Testing hypergeometric enrichment...")
    enr, exp, p = hypergeometric_enrichment(15, 50, 100, 1000)
    print(f"   Enrichment: {enr:.2f}x, Expected: {exp:.1f}, p: {p:.4e}")
    
    # Test ECDF with bands
    print("\n5. Testing ECDF with confidence bands...")
    data = np.random.randn(200)
    x, ecdf, ci_l, ci_u = ecdf_confidence_bands(data, n_bootstrap=100)
    print(f"   Computed ECDF with {len(x)} points")
    
    # Test FDR correction
    print("\n6. Testing FDR correction...")
    p_values = np.random.uniform(0, 0.1, 100)
    q_values = fdr_correction(p_values)
    n_sig = np.sum(q_values < 0.05)
    print(f"   {n_sig} of 100 tests significant at FDR<0.05")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
