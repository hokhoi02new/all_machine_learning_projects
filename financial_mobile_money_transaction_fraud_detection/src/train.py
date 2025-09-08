from src.model import build_model
from tensorflow import keras
import joblib


def train_model(model, X_train, y_train, model_name=""):
    print(f"Training...")
    if isinstance(model, keras.Model):
        model.fit(
            X_train, y_train,
            epochs=10, batch_size=32,
            validation_split=0.2, verbose=1
        )
        #save model
        path = f"models/{model_name}.h5"
        model.save(path)
    else:
        model.fit(X_train, y_train)
        #save model
        path = f"models/{model_name}.pkl"
        joblib.dump(model, path)
        
    return model

