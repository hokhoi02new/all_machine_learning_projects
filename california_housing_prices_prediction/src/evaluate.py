from src.data_process import load_data, split_data
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def evaluate_model(model_file="models/random_forest.pkl"):
    data = load_data()
    _, _, X_test, y_test = split_data(data, test_size=0.2)

    model = joblib.load(model_file)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE on test set: {rmse:.2f}")
    print(f"R² on test set: {r2:.4f}")


if __name__ == "__main__":
    evaluate_model("models/random_forest.pkl")
