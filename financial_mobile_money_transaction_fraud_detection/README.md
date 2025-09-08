# financial_mobile money_transaction_fraud_detection

### Introduction
For this project, we use a **Mobile Money Transaction Dataset** to detect fraudulent activities in financial transactions. Using model like Naive bayes, RandomForest, SVM, XGBoost, neural network (MLP). Metrics to evalute model is Accuracy, precision, Recall F1-score, ROC curve, AUC

#Project structure
financial_mobile money_transaction_fraud_detection/

│── data/                # contain dataset

│── models/  			  # folder saved model (.pkl)

│── notebooks/           # jupyter notebooks for EDA, model development and evaluation

    ├── notebook.ipynb

│── src/				  #scripts for

│   ├── data_process.py/ # process data, feature engineering and create pipeline  

│   ├── model.py         # build model

│   ├── evaluate.py     # evaluate model

│   ├── predict.py      # predict on new data

│   ├── train.py        # training model

│── main.py            #run all pipeline

│── requirements.txt     

│── README.md            

# dataset
We use PaySim dataset on kaggle. This is a simulation dataset that describes mobile money transactions.
There are 7003 transactions, The data set has 11 attributes which include is

- `transaction_id` – unique identifier for each transaction  
- `customer_id` – unique ID for the customer  
- `transaction_amount` – amount of the transaction  
- `transaction_type` – type of transaction (e.g., transfer, payment, cash-in, cash-out)  
- `timestamp` – date and time of the transaction  
- `location` – optional, location of the transaction  
- `device_info` – optional, device used for the transaction  
- `is_fraud` – target variable indicating whether the transaction is fraudulent (`1`) or legitimate (`0`)

# Requirements

#python
Python 3.8 or higher

#python library
pip install -r requirements.txt

# How to run

#Train and evaluate model type
example: Xgboost
python main.py --model xgboost








