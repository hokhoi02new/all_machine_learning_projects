# loan_approval_prediction_classification

### Introduction
This project build an ML model that the financial company can use to classify if a user can be approved a loan or not. based on the applicant’s features (income, credit score, loan history,...) 

Algorithm used
Logistic Regression, Decision Tree, KNN, Random Forest, ExtraTree, Gradient Boosting (XGBoost, LightGBM), Stacking model. Metrics to evalute model is Accuracy, precision, Recall F1-score, confusion matrix, ROC curve, AUC, 

### Project structure
loan_approval_prediction_classification/

│── data/                #dataset

│── notebooks/           #notebooks for EDA, process data, training and evaluate model

│── requirements.txt     

│── README.md            

### Dataset
The dataset used for this project is Loan Prediction Problem Dataset from Kaggle https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

Each applicant includes the following attributes:
Loan_ID:		Unique Loan ID
Gender: 		Male/Female
Married	Whether:  	Married: Yes/No
Dependents:		No. of people depending on the Applicant
Education:		Graduate/Undergraduate
Self_Employment:	Whether Self_Employment : Yes/No
ApplicantIncome:	Applicant Income
CoapplicantIncome:	Co-Applicant Income
LoanAmount:		Loan Amount (in thousands)
Loan_Amount_Term:	Loan Duration
Credit_History:		Credit History of the Applicant
Property_Area:		Urban/Semiurban/Rural
Loan_Status:		Whether Loan Approved: Yes/No

















