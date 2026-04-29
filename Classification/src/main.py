import pandas as pd

from windowing import WindowConfig, create_windowed_dataset, print_windowing_report
from filtering import remove_low_variance_windows
from train import train_two_models


def main():
    # =========================
    # 1. Load data
    # =========================
    df = pd.read_csv("/Users/abhilashreddysomigari/Documents/hwsecurity/Power-System-Attack-Detection/Classification/data/train.csv")

    print("\n=== ORIGINAL COLUMNS ===")
    print(df.columns)

        # =========================
    # 1.5 Clean fault labels
    # =========================
    def clean_fault_label(x):
        x = str(x)
        if x.startswith("ABC"):
            return "ABC"
        if x.startswith("AG"):
            return "AG"
        if x.startswith("BC"):
            return "BC"
        if x.startswith("Normal"):
            return "Normal"
        return x

    df["physical_fault"] = df["Fault_Type"].apply(clean_fault_label)

    print("\n=== CLEAN PHYSICAL FAULT DISTRIBUTION ===")
    print(df["physical_fault"].value_counts())

    # =========================
    # 2. Fault Windowing
    # =========================
    fault_config = WindowConfig(
        window_size=20,
        step_size=10,
        feature_columns=(
            "Va", "Vb", "Vc",
            "Ia", "Ib", "Ic",
            "Pa", "Pb", "Pc", "P_Total",
            "V_Unbalance", "Va_Var", "Ia_Var"
        ),
        target_column="physical_fault",
        time_column="Time",
        label_strategy="last",
    )

    X_tabular, y_fault, indices, X_seq = create_windowed_dataset(df, fault_config)

    print_windowing_report(df, X_tabular, y_fault, indices, fault_config)

    # =========================
    # 3. Attack Labels
    # =========================
    attack_config = WindowConfig(
        window_size=20,
        step_size=10,
        feature_columns=fault_config.feature_columns,
        target_column="attack_label",
        time_column="Time",
        label_strategy="last",
    )

    _, y_attack, _, _ = create_windowed_dataset(df, attack_config)

    print("\n=== ATTACK LABEL DISTRIBUTION BEFORE FILTERING ===")
    print(y_attack.value_counts())

    # =========================
    # 4. Filtering
    # =========================
    X_filtered, y_fault_filtered = remove_low_variance_windows(
        X_tabular,
        y_fault,
        threshold=0.1
    )

    # Important: apply same filtering mask to attack labels
    std_cols = [col for col in X_tabular.columns if "_std" in col]
    variation_score = X_tabular[std_cols].sum(axis=1)
    mask = variation_score > 0.1

    y_attack_filtered = y_attack.loc[mask].reset_index(drop=True)

    print("\n=== ATTACK LABEL DISTRIBUTION AFTER FILTERING ===")
    print(y_attack_filtered.value_counts())

    # =========================
    # 5. Train Two Models
    # =========================
    fault_model, attack_model = train_two_models(
        X_filtered,
        y_fault_filtered,
        y_attack_filtered
    )


if __name__ == "__main__":
    main()