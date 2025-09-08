import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(path="data/housing.csv"):
    return pd.read_csv(path)

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.total_rooms_idx = 3  
        self.households_idx = 6   
        self.population_idx = 5   
        self.total_bedrooms_idx = 4

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, self.total_rooms_idx] / X[:, self.households_idx]
        population_per_household = X[:, self.population_idx] / X[:, self.households_idx]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.total_bedrooms_idx] / X[:, self.total_rooms_idx]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def split_data(df, test_size=0.2):
    """chia dữ liệu bằng kỹ thuật stratified sampling"""
    df["income_cat"] = pd.cut(df["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_index, test_index in split.split(df, df["income_cat"]):
        data_train_set = df.loc[train_index]
        data_test_set = df.loc[test_index]

    X_train = data_train_set.drop("median_house_value",axis=1)
    y_train = data_train_set["median_house_value"].copy()

    X_test = data_test_set.drop("median_house_value",axis=1)
    y_test = data_test_set["median_house_value"].copy()

    return X_train, y_train, X_test, y_test

def create_pipe_process_data():
    """Tạo pipeline tiền xử lý cho dữ liệu numeric và categorical."""
    numeric_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
    categorical_features = ['ocean_proximity']

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder',CombinedAttributesAdder()),
        ('scaler', StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('onehot', OneHotEncoder())
    ])

    processor = ColumnTransformer([
    ("num_col",numeric_pipeline, numeric_features),
    ("cat_col",cat_pipeline, categorical_features)])
    
    return processor
