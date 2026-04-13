"""
This module contains all the functions used in the permutation test of RiboMARTA, as per the refactoring done in October 2025
Author: Gionmattia Carancini
Date: 02-04-2026
"""

import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

# needed to remove samples where we do not have info
def _clean_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Remove runs (samples) with zero coverage across all positions.

    Filters out columns where every position has zero counts, indicating
    no sequencing data or failed library preparation.

    Args:
        df: Coverage dataframe with positions as rows, runs (SRRs) as columns
        verbose: If True, print filtering statistics (default: False)

    Returns:
        Cleaned dataframe with all-zero columns removed
    """

    n_runs_original = df.shape[1]
    # Keeps columns with at least a non-0 value.
    df_clean = df.loc[:, (df != 0).any(axis=0)]

    n_runs_after = df_clean.shape[1]

    # Originally, we used to remove all columns were x was 0...now we are doing the permutation test

    if verbose:
        n_removed = n_runs_original - n_runs_after
        print(f"Runs originally: {n_runs_original}")
        print(f"Runs with zero coverage removed: {n_removed}")
        print(f"Runs remaining: {n_runs_after}")

    return df_clean


def _permute_values(
        base_array: np.ndarray,
        noise_array: np.ndarray,
        N: int,
        random_state: int | None = None
) -> np.ndarray:
    """
    Generate null distribution by permuting baseline and noise regions.

    Shuffles the combined baseline + noise arrays N times, recomputing
    the absolute difference between means for each permutation.

    Args:
        base_array: Array of shape (n_positions_base, n_runs) from baseline region
        noise_array: Array of shape (n_positions_noise, n_runs) from noise region
        N: Number of permutations to perform
        random_state: Seed for reproducibility (default: None)

    Returns:
        Array of shape (N, n_runs) containing permuted absolute differences
    """

    # Combine the two arrays
    combined = np.concatenate((base_array, noise_array), axis=0)
    # Retrieve the original size of x
    base_size = base_array.shape[0]
    # Set the random state for permutations
    rng = np.random.default_rng(random_state)

    # Set the empty array for the results
    perms_array = np.empty((N, base_array.shape[1]))

    for i in range(N):
        # Shuffles the labels
        shuffled = np.apply_along_axis(rng.permutation, 0, combined)
        # Gets permutated sets averages
        perm_base = shuffled[:base_size].mean(axis=0)
        perm_noise = shuffled[base_size:].mean(axis=0)
        # Computes the diff between perm x and n. Stores it in the output array.
        perms_array[i, :] = abs(perm_base - perm_noise)

    return perms_array


# Computes the p-value and confidence boundary
def test_x_region(
        diffs_df: pd.DataFrame,
        perms_array: np.ndarray,
        ci: int = 95
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute p-values and identify significant runs via permutation test.

    Calculates one-tailed p-values by comparing observed differences to the
    permutation-based null distribution, then applies FDR correction.

    Args:
        diffs_df: DataFrame with observed differences, must contain 'abs_xn_diff' column
        perms_array: Array of shape (N, n_runs) with null distribution
        ci: Confidence level for FDR correction (default: 95)

    Returns:
        results: DataFrame with columns ['x avg', 'n avg', 'padj_bh', 'reject']
        good_runs: List of run IDs where reject == True
    """

    # """
    # One-tailed p-values and upper confidence bound for each series:
    #   p = (count(perms_array >= observed) + 1) / (N + 1)
    #   ci_upper = percentile(perms_array, ci)
    # """

    N = perms_array.shape[0]
    observed = diffs_df["abs_xn_diff"]

    # Now we observe how many times the perms_array abs(x-n) were more extreme than the observed values.
    greater = (perms_array >= observed.values).sum(axis=0)
    pvals = (greater + 1) / (N + 1)  # Changed from  (greater + 0.5) / (N + 1)

    # FDR correction for multiple testing
    reject, pvals_bh, _, _ = multipletests(pvals, alpha=1 - (ci / 100), method='fdr_bh')

    # Compile results
    results = pd.DataFrame({
        "x avg": diffs_df["x avg"],
        "n avg": diffs_df["n avg"],
        'padj_bh': pvals_bh,
        'reject': reject,
    }, index=observed.index)

    good_runs = results[results["reject"] == True].index.to_list()

    return results, good_runs


def test_baseline_vs_noise(
        dataframe: pd.DataFrame,
        coordinates: dict[str, tuple[int, int]],
        N: int = 10000,
        random_state: int | None = None,
        ci: int = 95
) -> tuple[pd.DataFrame, list[str]]:
    """
    Test if baseline region (x) differs significantly from noise region (n).

    Uses permutation testing to identify runs (samples) where the baseline
    region shows average raw counts significantly different from the noise region.

    Biological reasoning: not everything is espressed.
    This test makes sure that baseline is indeed expressed to a significant level.

    Args:
        dataframe: Coverage data with positions as rows, runs (SRRs) as columns
        coordinates: Dictionary with 'x_slice' and 'n_slice' keys, each containing
                    (start, end) tuples defining region boundaries
        N: Number of permutations for null distribution (default: 1000)
        random_state: Seed for reproducibility (default: None)
        ci: Confidence level for statistical testing (default: 95)

    Returns:
        results: DataFrame with columns ['x avg', 'n avg', 'padj_bh', 'reject']
                indexed by run IDs
        good_runs: List of run IDs that passed the significance test
    """

    # computes an array of observed differences between baseline(x) and noise(n).
    diffs_df, reg_x, reg_n = _compute_xn_diffs(dataframe, coordinates)

    # permutes the values between x and n, to generate Null hypothesis distribution.
    perms_array = _permute_values(reg_x, reg_n, N, random_state=random_state)

    # does the actual test, computing the p-values and adjusting them
    results, good_runs = test_x_region(diffs_df, perms_array, ci=ci)

    return results, good_runs


def _compute_xn_diffs(df: pd.DataFrame,
                      coordinates: dict[str, tuple[int, int]]
                      ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Extract baseline (x) and noise (n) regions from the dataframe, and compute their differences in terms of average raw counts.

    Args:
        df: Coverage dataframe with positions as rows, runs as columns
        coordinates: Dictionary containing 'x_slice' and 'n_slice' keys with
                    (start, end) tuples

    Returns:
        diffs_df: DataFrame with columns ['x avg', 'n avg', 'abs_xn_diff'],
                 indexed by run IDs
        reg_x: Array of shape (n_positions_x, n_runs) with x region coverage
        reg_n: Array of shape (n_positions_n, n_runs) with n region coverage
    """

    # slice out each region
    reg_x = df.iloc[coordinates["x_slice"][0]: coordinates["x_slice"][1]]
    reg_n = df.iloc[coordinates["n_slice"][0]: coordinates["n_slice"][1]]

    # Calculated as it follows so that the Pandas series keeps the rownames
    x_avg = reg_x.mean(axis=0)
    n_avg = reg_n.mean(axis=0)

    # Compute the absolute difference between x and n
    abs_xn_dff = abs(x_avg - n_avg)

    # Stores the values into a dataframe

    diffs_df = pd.DataFrame({
        "x avg": x_avg,
        "n avg": n_avg,
        "abs_xn_diff": abs_xn_dff}
    )

    return diffs_df, reg_x.to_numpy(), reg_n.to_numpy()


def _compute_observed_ratios(
        df: pd.DataFrame,
        coordinates: dict[str, tuple[int, int]],
        good_runs: list[str]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute observed RTA ratios for valid runs and extract region arrays.

    Filters the input dataframe to include only runs that passed baseline vs
    noise testing, then calculates Relative Translational Activity (RTA) as:
    RTA = (Y_avg - N_avg) / (X_avg - N_avg)

    Args:
        df: Coverage dataframe with positions as rows, runs as columns
        coordinates: Dictionary defining region boundaries with keys:
            - 'x_slice': (start, end) for baseline region
            - 'y_slice': (start, end) for test region
            - 'n_slice': (start, end) for noise region
        good_runs: List of run IDs (SRRs) that passed baseline filtering

    Returns:
        ratios_df: DataFrame with columns ['x avg', 'y avg', 'n avg', 'RTA'],
                   indexed by run IDs
        y_array: Array of shape (n_positions_y, n_runs) with Y region coverage
        n_array: Array of shape (n_positions_n, n_runs) with N region coverage
        x_avg_array: Array of shape (n_runs,) with mean X region coverage per run

    Notes:
        - Only runs in good_runs are included in the output
        - x_avg_array is returned separately for use in permutation testing
        - Arrays are returned as numpy arrays for computational efficiency
    """

    # Filter to valid runs only.
    df_filtered = df[good_runs].copy()

    # Slice out each region.
    reg_x = df_filtered.iloc[coordinates["x_slice"][0]: coordinates["x_slice"][1]]
    reg_y = df_filtered.iloc[coordinates["y_slice"][0]: coordinates["y_slice"][1]]
    reg_n = df_filtered.iloc[coordinates["n_slice"][0]: coordinates["n_slice"][1]]

    # Compute averages per each srr, on each different slice.
    x_avg = reg_x.mean(axis=0)
    y_avg = reg_y.mean(axis=0)
    n_avg = reg_n.mean(axis=0)

    # Compute the observed RTA.
    rta = (y_avg - n_avg) / (x_avg - n_avg)

    # Stores the values in a dataframe.
    ratios_df = pd.DataFrame({
        "x avg": x_avg,
        "y avg": y_avg,
        "n avg": n_avg,
        "RTA": rta,
    })

    # Convert to arrays for permutation testing.
    x_avg_array = x_avg.to_numpy()

    return ratios_df, reg_y.to_numpy(), reg_n.to_numpy(), x_avg_array


def _permute_ratios(y_array: np.ndarray,
                    n_array: np.ndarray,
                    x_avg_array: np.ndarray,
                    N: int,
                    random_state: int | None = None
                    ) -> np.ndarray:
    """
    Generate null distribution for RTA via permutation of y and n regions.

    Shuffles the combined y and n region positions N times, recomputing
    permuted RTA ratios to establish the null distribution for significance testing.

    Args:
        y_array: Array of shape (n_positions_y, n_runs) with y region coverage
        n_array: Array of shape (n_positions_n, n_runs) with n region coverage
        x_avg_array: Array of shape (n_runs,) with mean X region coverage
        N: Number of permutations to perform
        random_state: Seed for reproducibility (default: None)

    Returns:
        Array of shape (N, n_runs) containing permuted RTA ratios.
        Each row represents one permutation's RTA values across all runs.

    Notes:
        - Permutation preserves run structure (shuffles positions, not runs)
        - X region average is held constant (not permuted)
        - RTA formula: (perm_y_avg - perm_n_avg) / (X_avg - perm_n_avg)
    """
    # Combine the two arrays with y and n regions
    combined = np.concatenate((y_array, n_array), axis=0)

    # Retrieve the original size of y for splitting and shuffle
    y_size = y_array.shape[0]

    # Initialise the random generator
    rng = np.random.default_rng(random_state)

    # Preallocate results array
    perms_array = np.empty((N, y_array.shape[1]))

    for i in range(N):
        # Shuffles the positions within each run (columnn-wise)
        shuffled = np.apply_along_axis(rng.permutation, 0, combined)

        # Compute averages for permuted regions
        perm_y = shuffled[:y_size].mean(axis=0)
        perm_n = shuffled[y_size:].mean(axis=0)

        # Compute the null-hypothesis RTAs and stores them
        perms_array[i, :] = (perm_y - perm_n) / (x_avg_array - perm_n)

    return perms_array


# Computes the p-value and confidence boundary
def compute_pvalues_and_ci(
        ratios_df: pd.DataFrame,
        perms_array: np.ndarray,
        ci: int = 95
) -> pd.DataFrame:
    """
    Compute p-values and confidence intervals for observed RTA values.

    Calculates one-tailed permutation p-values by comparing observed RTA
    to the null distribution, applies FDR correction, and determines
    confidence interval bounds for the null distribution.

    Args:
        ratios_df: DataFrame with observed RTA values and region averages
        perms_array: Array of shape (N, n_runs) with permuted RTA null distribution
        ci: Confidence level as percentage (default: 95)

    Returns:
        DataFrame with columns:
            - 'x avg', 'y avg', 'n avg': Mean coverage for each region
            - 'RTA': Observed Relative Translational Activity
            - 'log2RTA': Log2-transformed RTA
            - 'p-value': Raw permutation p-value
            - 'H0_ci_lower_XX%': Lower bound of null distribution CI
            - 'H0_ci_upper_XX%': Upper bound of null distribution CI
            - 'padj_bh': Benjamini-Hochberg adjusted p-value
            - 'significant': Boolean indicating statistical significance

    Notes:
        - P-values use standard permutation formula: (greater + 1) / (N + 1)
        - Confidence intervals represent the null distribution range
        - FDR correction via Benjamini-Hochberg method
    """
    N = perms_array.shape[0]

    # Extract observed RTA values
    observed = ratios_df["RTA"]

    # Count permutations where RTA >= observed
    greater = (perms_array >= observed.values).sum(axis=0)
    pvals = (greater + 1) / (N + 1)

    # Compute null distribution confidence intervals
    alpha = 100 - ci
    lower = np.percentile(perms_array, alpha / 2, axis=0)
    upper = np.percentile(perms_array, 100 - (alpha / 2), axis=0)

    # FDR correction
    reject, pvals_bh, _, _ = multipletests(pvals, alpha=1 - (ci / 100), method='fdr_bh')

    return pd.DataFrame({
        "x avg": ratios_df["x avg"],
        "y avg": ratios_df["y avg"],
        "n avg": ratios_df["n avg"],
        'RTA': ratios_df["RTA"],
        'log2RTA': np.log2(ratios_df["RTA"]),
        'p-value': pvals,
        f'H0_ci_lower_{ci}%': lower,
        f'H0_ci_upper_{ci}%': upper,
        'padj_bh': pvals_bh,
        'significant': reject,
    }, index=observed.index)


# New function for boostrap analysis and RTA CIs

def _bootstrap_rta_ci(
        df: pd.DataFrame,
        coordinates: dict[str, tuple[int, int]],
        good_runs: list[str],
        B: int = 10000,
        ci: int = 95,
        random_state: int | None = 42
) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals for observed RTA estimates.

    Resamples positions within each region (x, y, n) with replacement to
    estimate the sampling distribution of RTA, providing confidence intervals
    that quantify uncertainty in the point estimates.

    Args:
        df: Coverage dataframe with positions as rows, runs as columns
        coordinates: Dictionary defining region boundaries with keys:
            - 'x_slice': (start, end) for baseline region
            - 'y_slice': (start, end) for test region
            - 'n_slice': (start, end) for noise region
        good_runs: List of run IDs (SRRs) to compute bootstrap CIs for
        B: Number of bootstrap samples (default: 10000)
        ci: Confidence level as percentage (default: 95)
        random_state: Seed for reproducibility (default: 42)

    Returns:
        DataFrame indexed by run IDs with columns:
            - 'bootstrap_ci_lower_XX%': Lower bound of bootstrap CI for RTA
            - 'bootstrap_ci_upper_XX%': Upper bound of bootstrap CI for RTA

    Notes:
        - Bootstrap resamples positions independently within each region
        - Confidence intervals reflect uncertainty in the RTA estimate itself
        - Different from permutation test null distribution CIs
    """
    # Filter to good runs
    df_filtered = df[good_runs].copy()

    # Extract regions
    reg_x = df_filtered.iloc[coordinates["x_slice"][0]: coordinates["x_slice"][1]].to_numpy()
    reg_y = df_filtered.iloc[coordinates["y_slice"][0]: coordinates["y_slice"][1]].to_numpy()
    reg_n = df_filtered.iloc[coordinates["n_slice"][0]: coordinates["n_slice"][1]].to_numpy()

    # Initialize random generator
    rng = np.random.default_rng(random_state)

    # Get dimensions
    n_runs = len(good_runs)
    n_x = reg_x.shape[0]
    n_y = reg_y.shape[0]
    n_n = reg_n.shape[0]

    # Preallocate array for bootstrap RTAs
    bootstrap_rtas = np.empty((B, n_runs))

    for i in range(B):
        # Resample positions within each region (with replacement)
        boot_x_idx = rng.integers(0, n_x, size=n_x)
        boot_y_idx = rng.integers(0, n_y, size=n_y)
        boot_n_idx = rng.integers(0, n_n, size=n_n)

        # Get bootstrap samples, using the random indices created above.
        boot_x = reg_x[boot_x_idx, :]
        boot_y = reg_y[boot_y_idx, :]
        boot_n = reg_n[boot_n_idx, :]

        # Compute bootstrap RTA
        x_avg_boot = boot_x.mean(axis=0)
        y_avg_boot = boot_y.mean(axis=0)
        n_avg_boot = boot_n.mean(axis=0)

        bootstrap_rtas[i, :] = (y_avg_boot - n_avg_boot) / (x_avg_boot - n_avg_boot)

    # Mask infinite values as nan (so nanpercentile can ignore them -> saves from numerical instability).
    non_finite_mask = ~np.isfinite(bootstrap_rtas)
    bootstrap_rtas[non_finite_mask] = np.nan

    # If a bootstrapped run had too many NaN values, it is flagged.
    non_finite_counts = pd.Series(non_finite_mask.sum(axis=0), index=good_runs)
    non_finite_summary = pd.Series(
        [f"{int(count)}({count / B * 100:.1f}%)" for count in non_finite_counts.values],
        index=good_runs
    )

    # Compute percentile-based CI
    alpha = 100 - ci
    lower = np.nanpercentile(bootstrap_rtas, alpha / 2, axis=0)
    upper = np.nanpercentile(bootstrap_rtas, 100 - (alpha / 2), axis=0)

    # Return as DataFrame with proper index
    return pd.DataFrame({
        f'RTA_ci_lower_{ci}%': lower,
        f'RTA_ci_upper_{ci}%': upper,
        'bootstrap_non_finite': non_finite_summary,
        'bootstrap_ci_reliable': non_finite_counts <= 0.05 * B,
    }, index=good_runs)


# Function linking together the various bits to be run all together
def run_permutation_analysis(df: pd.DataFrame,
                             coordinates: dict[str, tuple[int, int]],
                             ci: int = 95,
                             N: int = 10000,
                             n_jobs: int = 3,
                             random_state: int | None = None) -> pd.DataFrame:
    """
    Execute RiboMARTA's "permutation_analysis" pipeline with two-stage permutation testing.

    This function performs a comprehensive analysis of ribosome profiling data to
    identify significant changes in translational activity. The pipeline consists of:

    1. Data cleaning: Remove runs with zero coverage
    2. Baseline filtering: Test x vs n regions to identify valid runs
    3. RTA calculation: Compute Relative Translational Activity ratios
    4. Significance testing: Permutation test to assess RTA significance

    Args:
        df: Coverage dataframe where rows represent nucleotide positions and
            columns represent individual runs (SRRs). Values are read counts.
        coordinates: Dictionary defining region boundaries with keys:
            - 'x_slice': (start, end) for baseline region
            - 'y_slice': (start, end) for test region
            - 'n_slice': (start, end) for noise/background region
        ci: Confidence level for statistical testing, as percentage (default: 95)
        N: Number of permutations for null distribution generation (default: 10000)
        n_jobs: Number of parallel jobs for permutation computation (default: 3)
        random_state: Random seed for reproducibility. If None, results are
            non-deterministic (default: None)

    Returns:
        DataFrame indexed by run IDs (SRRs) containing:
            - 'x avg', 'y avg', 'n avg': Mean coverage for each region
            - 'RTA': Relative Translational Activity ratio
            - 'log2RTA': Log2-transformed RTA
            - 'p-value': Raw permutation p-value
            - 'ci_lower_XX%', 'ci_upper_XX%': Confidence interval bounds
            - 'padj_bh': Benjamini-Hochberg adjusted p-value
            - 'significant': Boolean flag indicating statistical significance

        Only runs that passed the baseline vs noise filter are included.

    Notes:
        - RTA is calculated as: (Y_avg - N_avg) / (X_avg - N_avg)
        - Runs where X region shows no difference from N region are excluded
        - Parallelization uses independent random seeds per job for valid permutations
        - Higher N values provide finer p-value resolution but increase runtime

    Example:
        >>> coords = {
        ...     'x_slice': (0, 100),
        ...     'y_slice': (100, 200),
        ...     'n_slice': (200, 300)
        ... }
        >>> results = run_analysis(coverage_df, coords, ci=95, N=10000)
        >>> significant_runs = results[results['significant']]
    """

    # Clean input data.
    df = _clean_data(df)

    # Test baseline (x) vs noise region (n).
    results_x_vs_n, good_runs = test_baseline_vs_noise(df, coordinates, N=N, random_state=random_state, ci=ci)

    # Compute observed RTAs and extract arrays.
    ratios_df, y_array, n_array, x_avg_array = _compute_observed_ratios(df, coordinates, good_runs)

    # Do the permutation test again, this time testing y vs n. In parallel.
    perms_per_job = N // n_jobs
    perm_results = Parallel(n_jobs=n_jobs)(
        delayed(_permute_ratios)(y_array, n_array, x_avg_array, perms_per_job, random_state=i)
        for i in range(n_jobs)
    )
    # Stack the results into a single array (N, features).
    perms_array = np.vstack(perm_results)

    # Compiles results dataframe.
    results_df = compute_pvalues_and_ci(ratios_df, perms_array, ci=ci)

    # Computes CI for RTA by boostrapping.
    bootstrap_cis = _bootstrap_rta_ci(df, coordinates, good_runs, B=N, ci=ci, random_state=random_state)

    # Merge into results
    results_df = results_df.join(bootstrap_cis)

    return results_df