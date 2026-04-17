import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder


def train_model(X: pd.DataFrame, y: pd.Series):
    """
    Trains a Random Forest model and prints:
    - train/test split
    - accuracy
    - classification report
    - confusion matrix
    - 5-fold CV weighted F1
    - top feature importances
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print("\n=== TRAIN / TEST SPLIT ===")
    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== FINAL MODEL RESULTS ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)

    print("\n=== CONFUSION MATRIX ===")
    print(cm_df)

    cv_scores = cross_val_score(
        model,
        X,
        y_encoded,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1
    )

    print("\n=== 5-FOLD CROSS-VALIDATION (F1-WEIGHTED) ===")
    print("Scores:", cv_scores)
    print("Mean F1-weighted:", cv_scores.mean())

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\n=== TOP 15 FEATURE IMPORTANCES ===")
    print(importance_df.head(15))

    return model, label_encoder, importance_df