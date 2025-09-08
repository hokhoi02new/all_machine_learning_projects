from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR

def build_models(model_type="linear"):
    if model_type == "linear":
        model = LinearRegression()
    elif model_type=="random_forest":
        model = RandomForestRegressor(max_features = 8, n_estimators=10, random_state=42)
    elif model_type=="svm":
        model = SVR(kernel="linear")
    elif model_type=="xgboost":
        model = XGBRegressor(n_estimators=10, learning_rate=0.1, max_depth=3, random_state=42)
    elif model_type=="lightgbm":
        model = LGBMRegressor(n_estimators=10, learning_rate=0.1, max_depth=3, random_state=42, verbose=-1) 
    elif model_type=="stacking":
        estimators = [
        ('lr', LinearRegression()),
        ('dt', DecisionTreeRegressor(random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=10, random_state=42))]
        # Định nghĩa mô hình meta (meta-learner)
        model = LinearRegression()
        stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=model, cv=5, passthrough=False) 
    else:
        raise ValueError("Unsupported model type")
    return model

