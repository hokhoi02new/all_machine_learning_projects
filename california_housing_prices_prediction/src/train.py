from src.model import build_models
from src.data_process import split_data, load_data, create_pipe_process_data
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def train_model(model_type="linear", use_grid_search = False, save_path=None):

    data = load_data()
    X_train, y_train, _ , _ = split_data(data, test_size=0.2)

    processor = create_pipe_process_data()

    model = build_models(model_type)

    full_pipeline = Pipeline([
        ("preprocess", processor),
        ("model", model)
    ])
    if save_path is None:
        save_path = f"models/{model_type}.pkl"
        
    if use_grid_search==True:
        params_grid = {
            "random_forest": {
                "model__n_estimators" : [3, 10, 20],
                "model__max_features" : [2, 4, 6, 8],
                "boostrap" : [False, True],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2]},
            "svr": {
                'model__C': [0.1, 1, 10],
                'model__kernel': ['linear', 'rbf'],
                'model__gamma': ['scale', 'auto', 0.1]},
            "linear": {},
            "xgboost": {
                'model__n_estimators': [3, 10, 20],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.1, 0.3],
                'model__subsample': [0.8, 1.0]},
            "lightgbm": {
                'model__n_estimators': [3, 10, 20],
                'model__max_depth': [3, 5, 7, -1], 
                'model__learning_rate': [0.01, 0.1, 0.3],
                'model__num_leaves': [31, 50, 70]}
            }
        
        grid = GridSearchCV(full_pipeline,param_grid=params_grid[model_type], cv=3,scoring="neg_mean_squared_error",n_jobs=-1, verbose=2)
        grid.fit(X_train,y_train)
        print("Best Params:", grid.best_params_)
        best_model = grid.best_estimator_
        joblib.dump(best_model, save_path)
        print(f"Best model saved to {save_path}")
    else:
        full_pipeline.fit(X_train, y_train)
        joblib.dump(full_pipeline, save_path)
        print(f"model saved to {save_path}")

if __name__ == "__main__":
    train_model("random_forest",use_grid_search=False)  # train RandomForest