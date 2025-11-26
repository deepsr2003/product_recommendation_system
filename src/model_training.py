import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
import pickle
import joblib


def train_catboost_model(rfm_df):
    """
    Train CatBoostClassifier to predict high-value customers
    """
    # Features for training
    feature_columns = [
        "recency",
        "frequency",
        "monetary",
        "recency_score",
        "frequency_score",
        "monetary_score",
    ]

    X = rfm_df[feature_columns]
    y = rfm_df["is_high_value"]

    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(
        f"High-value customers in train: {y_train.sum()} ({y_train.mean() * 100:.1f}%)"
    )
    print(f"High-value customers in test: {y_test.sum()} ({y_test.mean() * 100:.1f}%)")

    # Train CatBoost model
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=False,
        loss_function="Logloss",
        eval_metric="Accuracy",
    )

    # Fit the model
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

    # Display classification report
    print("\nClassification Report:")
    print(
        classification_report(y_test, y_pred, target_names=["Low Value", "High Value"])
    )

    # Feature importance
    feature_importance = pd.DataFrame(
        {"feature": feature_columns, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    # Save model
    model.save_model("models/catboost_model.cbm")
    joblib.dump(model, "models/catboost_model.pkl")

    # Save feature columns for later use
    joblib.dump(feature_columns, "models/feature_columns.pkl")

    return model, X_test, y_test, y_pred, feature_importance


def load_model():
    """
    Load the trained CatBoost model
    """
    try:
        model = joblib.load("models/catboost_model.pkl")
        feature_columns = joblib.load("models/feature_columns.pkl")
        return model, feature_columns
    except FileNotFoundError:
        print("Model not found. Please train the model first.")
        return None, None


if __name__ == "__main__":
    # Load RFM data
    rfm_data = pd.read_csv("data/rfm_features.csv")

    # Train model
    model, X_test, y_test, y_pred, feature_importance = train_catboost_model(rfm_data)

    # Save test results for analysis
    test_results = pd.DataFrame(
        {
            "customer_id": rfm_data.loc[X_test.index, "customer_id"],
            "actual": y_test,
            "predicted": y_pred.flatten(),
            "recency": X_test["recency"],
            "frequency": X_test["frequency"],
            "monetary": X_test["monetary"],
        }
    )

    test_results.to_csv("outputs/test_results.csv", index=False)
    feature_importance.to_csv("outputs/feature_importance.csv", index=False)

    print(f"\nTest results saved to outputs/test_results.csv")
    print(f"Feature importance saved to outputs/feature_importance.csv")
