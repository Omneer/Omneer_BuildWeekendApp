from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle


def load_model(weights_path: str):
    """
    Load the trained model from a pickle file.

    :param weights_path: The directory path to the saved model weights.
    :return: The loaded model.
    """
    with open(Path(weights_path), 'rb') as f:
        model = pickle.load(f)
    return model


def save_model(model, dir_path: str, filename: str = "xgboost_weights.pkl"):
    """
    Save the trained model and scaler.

    :param model: The trained model.
    :param scaler: The QuantileTransformer object.
    :param dir_path: The directory path to save the model.
    :param filename: The name of the file.
    """
    # Check if the directory exists
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Save the model
    file_path = dir_path / filename
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

    print(f'Model saved at: {file_path}')


def load_csv_to_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load a csv file into a pandas DataFrame.

    :param csv_path: String path to the csv file.
    :return: A pandas DataFrame containing the csv data.
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


def preprocess_features(xs: pd.DataFrame, scaler: QuantileTransformer):
    """
    Preprocess the independent features using the given scaler.

    :param xs: Independent features.
    :param scaler: The QuantileTransformer object.
    :return: Transformed independent features.
    """
    xs = scaler.transform(xs)
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


def save_scaler(scaler, dir_path: str, filename: str = "scaler.pkl"):
    """
    Save the QuantileTransformer object.

    :param scaler: The QuantileTransformer object.
    :param dir_path: The directory path to save the scaler.
    :param filename: The name of the file.
    """
    # Check if the directory exists
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Save the scaler
    file_path = dir_path / filename
    with open(file_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f'Scaler saved at: {file_path}')


def load_scaler(scaler_path: str):
    """
    Load the QuantileTransformer object from a pickle file.

    :param scaler_path: The directory path to the saved scaler.
    :return: The loaded scaler object.
    """
    with open(Path(scaler_path), 'rb') as f:
        scaler = pickle.load(f)
    return scaler


def evaluate_model(model, x, y_true):
    """
    Evaluate the trained model on the test data.

    :param model: The trained model.
    :param x: The independent features for evaluation.
    :param y_true: The true labels for evaluation.
    :return: Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(x)
    y_prob = model.predict_proba(x)[:, 1]  # assuming binary classification

    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    metrics = {'Accuracy': accuracy, 'AUC': auc}
    return metrics


def train_e2e(input_csv, save_path):
    """
    Load data, preprocess features, train the model, and save the model and scaler.

    :param input_csv: Path to the input CSV file.
    :param save_path: Directory path to save the model and scaler.
    :return: The trained model, training and testing data, and the scaler.
    """
    # Load the data
    df = load_csv_to_dataframe(input_csv)

    # Split into features and target
    x, y = split_features_target(df)

    # Split into training and test set
    x_train, x_test, y_train, y_test = split_train_test(x, y)

    # Fit and save the scaler
    scaler = QuantileTransformer(output_distribution='normal')
    scaler.fit(x_train)
    save_scaler(scaler, save_path)

    # Preprocess the features
    x_train = preprocess_features(x_train, scaler)
    x_test = preprocess_features(x_test, scaler)

    # Convert labels to integers
    y_train, y_test = convert_labels_to_int(y_train, y_test)

    # Train the model
    model = train_model(x_train, y_train)

    # Evaluate the model
    train_metrics = evaluate_model(model, x_train, y_train)
    test_metrics = evaluate_model(model, x_test, y_test)
    print("Training Metrics:", train_metrics)
    print("Testing Metrics:", test_metrics)

    # Save the model
    save_model(model, save_path)

    # You can now use the model and scaler to make predictions
    # predictions = model.predict(x_test)
    return model, x_train, x_test, y_train, y_test, scaler


def predict(model, scaler, df: pd.DataFrame):
    """
    Use the trained model and scaler to make predictions.

    :param model: The trained model.
    :param scaler: The QuantileTransformer object.
    :param df: Input DataFrame for predictions.
    :return: Predicted labels and prediction scores.
    """
    # Split into features and target
    x, y = split_features_target(df)

    # Preprocess the features
    x = preprocess_features(x, scaler)

    # Make predictions
    predictions = model.predict(x)
    prediction_scores = model.predict_proba(x)[:, 1]  # assuming binary classification

    return predictions, prediction_scores


def main():
    input_csv = input("Enter the path to the input CSV file: ")
    save_path = input("Enter the directory path to save the model and scaler: ")

    model, x_train, x_test, y_train, y_test, scaler = train_e2e(input_csv, save_path)

    print("Training complete. Model and scaler saved.")
    print("You can now use the model and scaler to make predictions.")


if __name__ == "__main__":
    main()
