from typing import List
from fastapi import UploadFile, File
from pydantic import BaseModel

# Pydantic model for request
class FileUpload(BaseModel):
    file: UploadFile = File(...)

# Pydantic model for responsexw
class ModelPredictions(BaseModel):
    predictions: List[int]
    prediction_scores: List[float]
