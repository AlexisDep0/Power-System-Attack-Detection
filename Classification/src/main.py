import pandas as pd

from windowing import WindowConfig, create_windowed_dataset, print_windowing_report
from filtering import remove_low_variance_windows
from train import train_model


def main():
    # =========================
    # 1. Load data
    # =========================
    df = pd.read_csv("data/sim_output_combined.csv")

    # =========================
    # 2. Windowing
    # =========================
    config = WindowConfig(
        window_size=10,
        step_size=1,
        feature_columns=(
            "Va", "Vb", "Vc",
            "Ia", "Ib", "Ic",
            "Pa", "Pb", "Pc", "P_Total",
        ),
        target_column="Fault_Type",
        time_column="Time",
        label_strategy="last",
    )

    X_tabular, y, indices, X_seq = create_windowed_dataset(df, config)

    print_windowing_report(df, X_tabular, y, indices, config)

    print("\n=== FIRST 5 WINDOWED SAMPLES ===")
    print(X_tabular.head())

    print("\n=== FIRST 5 LABELS ===")
    print(y.head())

    print("\n=== RAW WINDOW ARRAY SHAPE ===")
    print(X_seq.shape)

    # =========================
    # 3. Filtering
    # =========================
    X_filtered, y_filtered = remove_low_variance_windows(
        X_tabular,
        y,
        threshold=0.1
    )

    print("\n=== FINAL SHAPES AFTER FILTERING ===")
    print("X_filtered:", X_filtered.shape)
    print("y_filtered:", y_filtered.shape)

    # =========================
    # 4. Training
    # =========================
    model, label_encoder, importance_df = train_model(X_filtered, y_filtered)

    print("\n=== LABEL ENCODING MAP ===")
    label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(label_map)


if __name__ == "__main__":
    main()