import os
import pickle

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_model(weights_path: str):
    """
    Load the trained model from a pickle file.

    :param weights_path: The directory path to the saved model weights.
    :return: The loaded model.
    """
    with open(weights_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_csv_to_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load a csv file into a pandas DataFrame.

    :param csv_path: String path to the csv file.
    :return: A pandas DataFrame containing the csv data.
    """
    return pd.read_csv(csv_path)

def split_features_target(df: pd.DataFrame):
    """
    Split the data into independent and dependent features.

    :param df: The DataFrame to split.
    :return: A tuple (x, y), where x is a DataFrame of independent features and y is a Series of the dependent feature.
    """
    y = df.iloc[:, 1]
    x = df.iloc[:, 2:]
    return x, y

def preprocess_features(xs: pd.DataFrame):
    """
    Preprocess the independent features by fitting a StandardScaler.

    :param xs: Training data
    :return: Scaler object and transformed training data.
    """
    scaler = StandardScaler().fit(xs)
    xs = scaler.transform(xs)
    return scaler, xs

def train_e2e(input_csv, save_path):
    # Load the data
    df = load_csv_to_dataframe(input_csv)

    # Split into features and target
    x, y = split_features_target(df)

    # Split into training and test set
    x_train, x_test, y_train, y_test = split_train_test(x, y)

    # Preprocess the features
    scaler, x_train = preprocess_features(x_train)
    x_test = scaler.transform(x_test)

    # Convert labels to integers
    y_train, y_test = convert_labels_to_int(y_train, y_test)

    # Train the model
    model = train_model(x_train, y_train)

    # Save the model and scaler
    save_model(model, save_path, filename="xgboost_weights.pkl")
    save_model(scaler, save_path, filename="scaler.pkl")

    return model, scaler, x_train, x_test, y_train, y_test

# def preprocess_features(xs: pd.DataFrame):
#     """
#     Preprocess the independent features by fitting a StandardScaler.

#     :param x_train: Training data
#     :param x_test: Test data
#     :return: Transformed training and testing data.
#     """
#     norm_obj = StandardScaler().fit(xs)
#     xs = norm_obj.transform(xs)
#     return xs

def split_train_test(x: pd.DataFrame, y: pd.Series, train_size: float = 0.8, seed: int = 42): 
    """
    Split the data into a training set and a test set.

    :param x: Independent features.
    :param y: Dependent feature.
    :param train_size: The proportion of the data to include in the training set.
    :param seed: The seed for the random number generator.
    :return: x_train, x_test, y_train, y_test
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=seed)
    return x_train, x_test, y_train, y_test

def convert_labels_to_int(y_train: pd.Series, y_test: pd.Series):
    """
    Convert labels to integer type.

    :param y_train: Training labels
    :param y_test: Test labels
    :return: Labels converted to integers
    """
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    return y_train, y_test

def train_model(x_train: pd.DataFrame, y_train: pd.Series):
    """
    Train an XGBClassifier model on the given data.

    :param x_train: The independent features to train on.
    :param y_train: The dependent feature to train on.
    :return: The trained model.
    """
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
    """
    Save the trained model weights.

    :param model: The trained model.
    :param dir_path: The directory path to save the model.
    :param filename: The name of the file.
    """
    # Check if the directory exists
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)  # Create directory

    # Save the model
    file_path = os.path.join(dir_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

    print(f'Model saved at: {file_path}')


# def train_e2e(input_csv, save_path):
#     # Load the data
#     df = load_csv_to_dataframe(input_csv)

#     # Split into features and target
#     x, y = split_features_target(df)

#     # Split into training and test set
#     x_train, x_test, y_train, y_test = split_train_test(x, y)

#     # Preprocess the features
#     x_train = preprocess_features(x_train)
#     x_test = preprocess_features(x_test)

#     # Convert labels to integers
#     y_train, y_test = convert_labels_to_int(y_train, y_test)

#     # Train the model
#     model = train_model(x_train, y_train)

#     # Save the model
#     save_model(model, save_path)

#     # You can now use the model to make predictions
#     # predictions = model.predict(x_test)
#     return model, x_train, x_test, y_train, y_test


# def predict(model, df: pd.DataFrame):
#     # Split into features and target
#     x, y = split_features_target(df)

#     # Preprocess the features
#     x = preprocess_features(x)

#     # Make predictions
#     predictions = model.predict(x)
#     prediction_scores = model.predict_proba(x)[:, 1]  # assuming binary classification

#     # You can now use the model to make predictions
#     # predictions = model.predict(x_test)
#     return predictions, prediction_scores


# def predict(model, df: pd.DataFrame):
#     # Split into features and target
#     x, y = split_features_target(df)

#     # Preprocess the features
#     x = preprocess_features(x)

#     # Make predictions
#     predictions = model.predict(x)
#     prediction_scores = model.predict_proba(x)[:, 1]  # assuming binary classification

#     return predictions, prediction_scores

def predict(model, scaler, df: pd.DataFrame):
    # Split into features and target
    x, y = split_features_target(df)

    # Preprocess the features
    x = scaler.transform(x)

    # Make predictions
    predictions = model.predict(x)
    prediction_scores = model.predict_proba(x)[:, 1]  # assuming binary classification

    return predictions, prediction_scores
