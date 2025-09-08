import os
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, precision_score,
    recall_score, f1_score)
from tensorflow import keras

def load_trained_model(model_name):

    h5_path = f"models/{model_name}.h5"
    pkl_path = f"models/{model_name}.pkl"

    if os.path.exists(h5_path):
        print(f"Loading Keras model from {h5_path}")
        return keras.models.load_model(h5_path)
    elif os.path.exists(pkl_path):
        print(f"Loading sklearn model from {pkl_path}")
        return joblib.load(pkl_path)
    else:
        raise FileNotFoundError(f"No saved model found for {model_name}")

def evaluate_model(model, X_test, y_test):

    # Predict
    if isinstance(model, keras.Model):
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    else:
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report
    }