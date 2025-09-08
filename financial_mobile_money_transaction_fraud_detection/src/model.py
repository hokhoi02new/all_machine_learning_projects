from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from tensorflow import keras

def build_model(model_name, input_dim=None):
    model_name = model_name.lower()

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=10, random_state=42, class_weight="balanced"
        )

    elif model_name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=10, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False,
            eval_metric="logloss"
        )

    elif model_name == "neural_net":
        if input_dim is None:
            raise ValueError("Need to pass input_dim for neural_net")
        model = keras.Sequential([
            keras.layers.Dense(15, input_shape=(input_dim,), activation='relu'),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    elif model_name == "svm":
        return SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    elif model_name == "naive_bayes":
        return GaussianNB()
    else:
        raise ValueError(f"Model {model_name} does not support" )
