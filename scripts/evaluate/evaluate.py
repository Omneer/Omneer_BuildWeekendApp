from typing import Tuple
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier


def evaluate_model(model: XGBClassifier, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float]:
    """
    Evaluate the model's performance.

    :param model: The trained model.
    :param x_test: Test data
    :param y_test: Test labels
    :return: Precision, recall, and AUC score.
    """
    # Predict on test data
    y_pred = model.predict(x_test)

    # Print classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Calculate the AUC score
    auc_score = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    
    print("Classification Report:\n")
    print(report)

    print(f"AUC Score: {auc_score}")

    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    
    return precision, recall, auc_score