
# California Houses Price Prediction

### Introduction
This project using machine learning techniques to predict housing prices in California. It leverages algorithms such as Linear Regression, SVM, ensemble model like Random Forest, XGBoost, LightGBM, Stacking model and some EDA, feature engineering and preprocess data techniques. 

### Project structure
california_housing_prices_prediction/

│── data/                # contain dataset

│── models/  			  # folder saved model (.pkl)

│── notebooks/ 			  # notebooks for EDA, data analysis and model development

    ├── notebook.ipynb         
    
│── src/				 scripts for

│   ├── data_process.py/ # data cleaning, transformation, feature engineering.

│   ├── model.py         # defining model

│   ├── evaluate.py     # model evaluation, metrics calculation, and validation.

│   ├── train.py        # training model

│   ├── predict.py      # predict on new data

│── main.py            #run all pipeline

│── requirements.txt     

│── README.md            

### dataset
For this project, we used the **California Housing Dataset** available on [Kaggle](https://www.kaggle.com/camnugent/california-housing-prices).  

The dataset contains information about California districts, including features such as:
- `longitude` and `latitude` – location coordinates
- `housing_median_age` – median age of houses in the district
- `total_rooms` – total number of rooms in the district
- `total_bedrooms` – total number of bedrooms
- `population` – population of the district
- `households` – number of households
- `median_income` – median income of households
- `median_house_value` – median house value (target variable)
- `ocean_proximity` – categorical variable indicating the district’s proximity to the ocean


### Requirements

#python
Python 3.8 or higher

#python library
pip install -r requirements.txt

### How to use

#### Train model
python main.py train --model_type <model_name> --model <model_path>

example:
python main.py train --model_type xgboost --model models/xgboost.pkl

#### Evaluate model
python main.py evaluate --model <model_path>

example: 
python main.py evaluate --model models/xgboost.pkl

#### Predict
python main.py predict --model <model_path> --sample <info_about_the_house>

example:
python main.py predict --model models/xgboost.pkl --sample -122.23 37.88 41 880 129 322 126 8.3252







