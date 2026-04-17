from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class WindowConfig:
    window_size: int = 10
    step_size: int = 1
    feature_columns: Tuple[str, ...] = (
        "Va", "Vb", "Vc",
        "Ia", "Ib", "Ic",
        "Pa", "Pb", "Pc", "P_Total",
    )
    target_column: str = "Fault_Type"
    time_column: str = "Time"
    label_strategy: str = "last"  # "last" or "majority"


def validate_window_config(df: pd.DataFrame, config: WindowConfig) -> None:
    required_cols = set(config.feature_columns) | {config.target_column, config.time_column}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if config.window_size <= 0:
        raise ValueError("window_size must be > 0")

    if config.step_size <= 0:
        raise ValueError("step_size must be > 0")

    if config.label_strategy not in {"last", "majority"}:
        raise ValueError("label_strategy must be either 'last' or 'majority'")

    if len(df) < config.window_size:
        raise ValueError(
            f"DataFrame has only {len(df)} rows, but window_size={config.window_size}."
        )


def sort_by_time(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
    return df.sort_values(by=time_column).reset_index(drop=True)


def get_window_label(window_df: pd.DataFrame, target_column: str, strategy: str) -> str:
    if strategy == "last":
        return str(window_df[target_column].iloc[-1])

    return str(window_df[target_column].mode().iloc[0])


def create_raw_windows(
    df: pd.DataFrame,
    config: WindowConfig,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Returns:
        X_seq: shape (num_windows, window_size, num_features)
        y: shape (num_windows,)
        indices: list of (start_idx, end_idx)
    """
    validate_window_config(df, config)
    df = sort_by_time(df, config.time_column)

    X_seq = []
    y = []
    indices = []

    feature_values = df.loc[:, config.feature_columns].to_numpy(dtype=np.float64)

    for start in range(0, len(df) - config.window_size + 1, config.step_size):
        end = start + config.window_size
        window_df = df.iloc[start:end]

        window_x = feature_values[start:end]
        window_y = get_window_label(window_df, config.target_column, config.label_strategy)

        X_seq.append(window_x)
        y.append(window_y)
        indices.append((start, end - 1))

    return np.asarray(X_seq, dtype=np.float64), np.asarray(y), indices


def summarize_windows(
    X_seq: np.ndarray,
    feature_columns: Tuple[str, ...],
) -> pd.DataFrame:
    """
    Converts raw windows into tabular features using:
    mean, std, min, max, range, first, last, delta
    """
    if X_seq.ndim != 3:
        raise ValueError(
            f"Expected X_seq shape (num_windows, window_size, num_features), got {X_seq.shape}"
        )

    _, _, num_features = X_seq.shape
    if num_features != len(feature_columns):
        raise ValueError(
            f"Feature dimension mismatch: X_seq has {num_features}, "
            f"but feature_columns has {len(feature_columns)}"
        )

    feature_dict = {}

    for i, col in enumerate(feature_columns):
        values = X_seq[:, :, i]

        mean_ = values.mean(axis=1)
        std_ = values.std(axis=1)
        min_ = values.min(axis=1)
        max_ = values.max(axis=1)
        range_ = max_ - min_
        first_ = values[:, 0]
        last_ = values[:, -1]
        delta_ = last_ - first_

        feature_dict[f"{col}_mean"] = mean_
        feature_dict[f"{col}_std"] = std_
        feature_dict[f"{col}_min"] = min_
        feature_dict[f"{col}_max"] = max_
        feature_dict[f"{col}_range"] = range_
        feature_dict[f"{col}_first"] = first_
        feature_dict[f"{col}_last"] = last_
        feature_dict[f"{col}_delta"] = delta_

    return pd.DataFrame(feature_dict)


def create_windowed_dataset(
    df: pd.DataFrame,
    config: WindowConfig,
) -> Tuple[pd.DataFrame, pd.Series, List[Tuple[int, int]], np.ndarray]:
    """
    Full pipeline:
    1. create raw windows
    2. summarize windows into tabular features
    """
    X_seq, y, indices = create_raw_windows(df, config)
    X_tabular = summarize_windows(X_seq, config.feature_columns)

    return X_tabular, pd.Series(y, name=config.target_column), indices, X_seq


def print_windowing_report(
    df: pd.DataFrame,
    X_tabular: pd.DataFrame,
    y: pd.Series,
    indices: List[Tuple[int, int]],
    config: WindowConfig,
) -> None:
    print("\n=== WINDOWING REPORT ===")
    print(f"Original rows           : {len(df)}")
    print(f"Window size             : {config.window_size}")
    print(f"Step size               : {config.step_size}")
    print(f"Label strategy          : {config.label_strategy}")
    print(f"Number of windows       : {len(X_tabular)}")
    print(f"Windowed feature shape  : {X_tabular.shape}")
    print(f"Target shape            : {y.shape}")

    if indices:
        print(f"First window row range  : {indices[0]}")
        print(f"Last window row range   : {indices[-1]}")

    print("\n=== WINDOW LABEL DISTRIBUTION ===")
    print(y.value_counts())