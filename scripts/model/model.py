from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


def load_model(weights_path: str):
    """
    Load the trained model from a pickle file.

    :param weights_path: The directory path to the saved model weights.
    :return: The loaded model.
    """
    with open(Path(weights_path), 'rb') as file:
        model = pickle.load(file)
    return model


def load_csv_to_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    :param csv_path: String path to the CSV file.
    :return: A pandas DataFrame containing the CSV data.
    """
    return pd.read_csv(Path(csv_path))


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

    :param xs: Independent features.
    :return: Transformed independent features.
    """
    norm_obj = StandardScaler().fit(xs)
    xs = norm_obj.transform(xs)
    return xs


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

    :param y_train: Training labels.
    :param y_test: Test labels.
    :return: Labels converted to integers.
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
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Save the model
    file_path = dir_path / filename
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

    print(f'Model saved at: {file_path}')


def train_e2e(input_csv, save_path):
    """
    Load data, preprocess features, train the model, and save it.

    :param input_csv: Path to the input CSV file.
    :param save_path: Directory path to save the model.
    :return: The trained model, training and testing data.
    """
    # Load the data
    df = load_csv_to_dataframe(input_csv)

    # Split into features and target
    x, y = split_features_target(df)

    # Split into training and test set
    x_train, x_test, y_train, y_test = split_train_test(x, y)

    # Preprocess the features
    x_train = preprocess_features(x_train)
    x_test = preprocess_features(x_test)

    # Convert labels to integers
    y_train, y_test = convert_labels_to_int(y_train, y_test)

    # Train the model
    model = train_model(x_train, y_train)

    # Save the model
    save_model(model, save_path)

    # You can now use the model to make predictions
    # predictions = model.predict(x_test)
    return model, x_train, x_test, y_train, y_test


def predict(model, df: pd.DataFrame):
    """
    Use the trained model to make predictions.

    :param model: The trained model.
    :param df: Input DataFrame for predictions.
    :return: Predicted labels and prediction scores.
    """
    # Split into features and target
    x, y = split_features_target(df)

    # Preprocess the features
    x = preprocess_features(x)

    # Make predictions
    predictions = model.predict(x)
    prediction_scores = model.predict_proba(x)[:, 1]  # assuming binary classification

    return predictions, prediction_scores
