import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(path="C:\\Users\\HO KHOI\\Desktop\\FINAL project\\mobile money_transaction_fraud_detection\data\\PS_20174392719_1491204439457_log.csv"):
    return pd.read_csv(path)

def process_data(data, test_size=0.2):

    #add feature type2
    data_new = data.copy()
    data_new["Type2"] = np.nan
    data_new.loc[data_new.nameOrig.str.contains('C') & data_new.nameDest.str.contains('C'), "Type2"] = "CC"
    data_new.loc[data_new.nameOrig.str.contains('C') & data_new.nameDest.str.contains('M'), "Type2"] = "CM"
    data_new.loc[data_new.nameOrig.str.contains('M') & data_new.nameDest.str.contains('C'), "Type2"] = "MC"
    data_new.loc[data_new.nameOrig.str.contains('M') & data_new.nameDest.str.contains('M'), "Type2"] = "MM"
    data_new["Type2"].fillna("Other", inplace=True)


    #feature HourOfDay
    data_new["HourOfDay"] = data_new.step % 24

    #drop unused columns
    df = data_new.drop(["nameOrig", "nameDest", "isFlaggedFraud"], axis=1)

    #encode categorical columns
    for col in ["type", "Type2"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    #split features/target
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]
    
    #train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    scaler = MinMaxScaler()
    X_train=scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
