import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = "kmeans_dyslexia_risk_dataset_40000.csv"

FEATURES = [
    "reading_accuracy",
    "word_omission_count",
    "pronunciation_error_count",
    "reading_speed_wpm",
]

TARGET = "risk_label"


def train_model() -> None:
    df = pd.read_csv(DATASET_PATH)

    missing_columns = [
        column for column in FEATURES + [TARGET]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df = df.dropna(subset=FEATURES + [TARGET]).copy()

    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=FEATURES)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES],
        encoded_labels,
        test_size=0.20,
        random_state=42,
        stratify=encoded_labels,
    )

    model = CatBoostClassifier(
        iterations=500,
        depth=7,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=42,
        verbose=100,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50,
    )

    predictions = model.predict(X_test).reshape(-1).astype(int)

    print("\nAccuracy :", accuracy_score(y_test, predictions))
    print(
        "Precision:",
        precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    )
    print(
        "Recall   :",
        recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    )
    print(
        "F1-score:",
        f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    )

    print(
        "\nClassification Report:\n",
        classification_report(
            y_test,
            predictions,
            target_names=label_encoder.classes_,
            zero_division=0,
        ),
    )

    model.save_model("catboost_dyslexia_model.cbm")
    joblib.dump(label_encoder, "risk_label_encoder.pkl")

    print("\nModel and encoder saved successfully.")


if __name__ == "__main__":
    train_model()