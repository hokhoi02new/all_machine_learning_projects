import joblib
import pandas as pd

def predict(input_data=None, model_file="models/random_forest.pkl"):
    full_pipeline = joblib.load(model_file)

    # Nếu không có input_data, dùng mẫu mặc định
    if input_data is None:
        input_data = pd.DataFrame({
            'longitude': [-122.23],
            'latitude': [37.88],
            'housing_median_age': [41.0],
            'total_rooms': [880.0],
            'total_bedrooms': [129.0],
            'population': [322.0],
            'households': [126.0],
            'median_income': [8.3252],
            'ocean_proximity': ['NEAR BAY']
        })
    
    # Nếu input là list, chuyển thành DataFrame
    if isinstance(input_data, list):
        if len(input_data) != 8:
            raise ValueError("Expected 8 numeric features in the list")
        feature_names = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income'
        ]
        input_data = pd.DataFrame([dict(zip(feature_names, input_data))])
        input_data['ocean_proximity'] = 'NEAR BAY'


    elif isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])


    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data must be a list, dict, or pandas DataFrame")

    predictions = full_pipeline.predict(input_data)
    print(predictions)
    return predictions
