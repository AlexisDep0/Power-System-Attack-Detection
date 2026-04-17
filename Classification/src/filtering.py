import pandas as pd


def remove_low_variance_windows(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.1
):
    """
    Removes windows with low total variation using all *_std columns.

    Parameters
    ----------
    X : pd.DataFrame
        Window-based feature dataframe
    y : pd.Series
        Labels
    threshold : float
        Minimum total std across std-features required to keep a sample
    """
    std_cols = [col for col in X.columns if "_std" in col]

    if not std_cols:
        raise ValueError("No '_std' columns found in X. Cannot compute variation score.")

    variation_score = X[std_cols].sum(axis=1)
    mask = variation_score > threshold

    X_filtered = X.loc[mask].reset_index(drop=True)
    y_filtered = y.loc[mask].reset_index(drop=True)

    print("\n=== FILTERING REPORT ===")
    print(f"Original samples : {len(X)}")
    print(f"Filtered samples : {len(X_filtered)}")
    print(f"Removed samples  : {len(X) - len(X_filtered)}")

    print("\n=== NEW CLASS DISTRIBUTION ===")
    print(y_filtered.value_counts())

    return X_filtered, y_filtered