
import os
import pickle

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer

SCALER_PATH = "/app/data/weights/scaler.pkl"

def load_model(weights_path: str):
    with open(weights_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_scaler(scaler_path: str):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def load_csv_to_dataframe(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def split_features_target(df: pd.DataFrame):
    y = df.iloc[:, 1]
    x = df.iloc[:, 2:]
    return x, y

def split_train_test(x: pd.DataFrame, y: pd.Series, train_size: float = 0.8, seed: int = 42): 
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=seed)
    return x_train, x_test, y_train, y_test

def convert_labels_to_int(y_train: pd.Series, y_test: pd.Series):
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    return y_train, y_test

def train_model(x_train: pd.DataFrame, y_train: pd.Series):
    model = XGBClassifier(
        booster='gbtree',
        learning_rate=0.3,
        subsample=1,
        colsample_bytree=0.4,
        max_depth=1,
        reg_alpha=0.5,
        reg_lambda=0.05,
        eval_metric='logloss',
        n_estimators=50
    )
    model.fit(x_train, y_train)
    return model

def save_model(model, dir_path: str, filename: str = "xgboost_weights.pkl"):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved at: {file_path}')

def preprocess_features(xs: pd.DataFrame, scaler=None):
    if scaler is None:
        scaler = load_scaler(SCALER_PATH)
    xs = scaler.transform(xs)
    return scaler, xs

def train_e2e(input_csv: str, save_path: str):
    df = load_csv_to_dataframe(input_csv)
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = split_train_test(x, y)
    scaler, x_train = preprocess_features(x_train)
    x_test = scaler.transform(x_test)
    y_train, y_test = convert_labels_to_int(y_train, y_test)
    model = train_model(x_train, y_train)
    save_model(model, save_path, filename="xgboost_weights.pkl")
    return model, scaler, x_train, x_test, y_train, y_test

def predict(model, scaler, df: pd.DataFrame):
    x, y = split_features_target(df)
    x = scaler.transform(x)
    predictions = model.predict(x)
    prediction_scores = model.predict_proba(x)[:, 1]  # assuming binary classification
    return predictions, prediction_scores

if __name__ == "__main__":
    input_csv = "../../data/CRANK_MS.csv"
    save_path = "../../data/weights/"
    model, scaler, x_train, x_test, y_train, y_test = train_e2e(input_csv, save_path)
