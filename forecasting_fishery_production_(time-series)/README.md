# forecasting_fishery_production_(time-series)

### Introduction
This project builds a model to predict fishery production over year based on attributes time-series such as farming area, number of fishing vessels, total fishery production. 
Using machine learning methods such as ARIMA model, VAR (vector autogression model), LSTM. Metrics to evalute is MAPE, RMSE. Hypothesis testing like Augmented Dickey Fuller Test (ADF Test), Granger causality test

### Dataset
The data about Vietnam's fishery production was collected by us from the General Statistics Office of Vietnam and some other sources.
The dataset containt information from 1990 to 2020, with a total of 25 data points.
Each sample has 5 attribute
"Nam": year
"ChiSoPhatTrien": development index, calculated by dividing current year's output by last year's output * 100.
"DienTichNuoiTrong":  cultivation area
"SoTauKhaiThac": number of vessels in operation
"TongSanLuong": total fishery production

### Repository Structure
data/: dataset

notebooks/: jupyter notebooks for EDA, Hypothesis testing, analyze data, development and evaluate model 

src/: contains any additional source code used in the project.










