from fastapi import FastAPI, UploadFile, File
import pandas as pd

from scripts.model.model import load_model, predict
from scripts.apis.api_models import ModelPredictions


MODEL_PATH = "/app/data/weights/xgboost_weights.pkl"
model = load_model(MODEL_PATH)

app = FastAPI()

@app.post("/api/v1/predict", 
          response_model=ModelPredictions)
async def make_prediction(file: UploadFile = File(...)):
    """
    Endpoint accepting a CSV file, making predictions using a pretrained model, 
    and returning those predictions and prediction scores.

    :param file: File object representing the CSV file.
    :return: JSON object containing predictions and prediction_scores.
    """

    # Read the CSV file to a DataFrame
    df = pd.read_csv(file.file)
    
    predictions, prediction_scores = predict(model, df)

    return ModelPredictions(
        predictions=predictions.tolist(), 
        prediction_scores=prediction_scores.tolist()
    )
