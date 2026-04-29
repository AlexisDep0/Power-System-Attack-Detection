import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def train_two_models(X, y_fault, y_attack):
    # =========================
    # Encode labels
    # =========================
    fault_encoder = LabelEncoder()
    attack_encoder = LabelEncoder()

    y_fault_encoded = fault_encoder.fit_transform(y_fault)
    y_attack_encoded = attack_encoder.fit_transform(y_attack)

    # =========================
    # Same split for both tasks
    # =========================
    X_train, X_test, yf_train, yf_test, ya_train, ya_test = train_test_split(
        X,
        y_fault_encoded,
        y_attack_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_fault_encoded
    )

    print("\n=== TRAIN / TEST SPLIT ===")
    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    # =========================
    # Fault model
    # =========================
    fault_model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    fault_model.fit(X_train, yf_train)
    yf_pred = fault_model.predict(X_test)

    print("\n==============================")
    print("FAULT CLASSIFICATION RESULTS")
    print("==============================")
    print("Fault Accuracy:", accuracy_score(yf_test, yf_pred))

    print("\n=== Fault Classification Report ===")
    print(classification_report(
        yf_test,
        yf_pred,
        target_names=fault_encoder.classes_
    ))

    fault_cm = confusion_matrix(yf_test, yf_pred)
    fault_cm_df = pd.DataFrame(
        fault_cm,
        index=fault_encoder.classes_,
        columns=fault_encoder.classes_
    )

    print("\n=== Fault Confusion Matrix ===")
    print(fault_cm_df)

    # =========================
    # Attack model
    # =========================
    attack_model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    attack_model.fit(X_train, ya_train)
    ya_pred = attack_model.predict(X_test)

    print("\n==============================")
    print("ATTACK CLASSIFICATION RESULTS")
    print("==============================")
    print("Attack Accuracy:", accuracy_score(ya_test, ya_pred))

    print("\n=== Attack Classification Report ===")
    print(classification_report(
        ya_test,
        ya_pred,
        target_names=attack_encoder.classes_
    ))

    attack_cm = confusion_matrix(ya_test, ya_pred)
    attack_cm_df = pd.DataFrame(
        attack_cm,
        index=attack_encoder.classes_,
        columns=attack_encoder.classes_
    )

    print("\n=== Attack Confusion Matrix ===")
    print(attack_cm_df)

    # =========================
    # Joint accuracy
    # =========================
    joint_correct = (yf_pred == yf_test) & (ya_pred == ya_test)
    joint_accuracy = joint_correct.mean()

    print("\n==============================")
    print("JOINT CLASSIFICATION RESULT")
    print("==============================")
    print("Joint Accuracy:", joint_accuracy)

    return fault_model, attack_model