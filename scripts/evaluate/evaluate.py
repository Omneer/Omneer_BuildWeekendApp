from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from scripts.model.model import load_model, load_scaler, load_csv_to_dataframe, split_features_target, preprocess_features

def evaluate_model(model, x_test, y_test):
    """Evaluate the model on the test set and print the performance metrics."""
    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate AUC score
    auc_score = roc_auc_score(y_test, y_pred)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Generate classification report
    class_report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"AUC Score: {auc_score}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)


def main():
    # Load the model and scaler
    model = load_model("../../data/xgboost_weights.pkl")
    scaler = load_scaler("../../data/scaler.pkl")

    # Load the test data
    test_df = load_csv_to_dataframe("../../CRANK_MS.csv")

    # Split into features and target
    x_test, y_test = split_features_target(test_df)

    # Preprocess the features
    x_test = preprocess_features(x_test, scaler)

    # Evaluate the model
    evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    main()
