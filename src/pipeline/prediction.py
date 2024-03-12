import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path("artifacts/model_selection/best_model.pkl"))

    def predict(self,data):

        prediction = self.model.predict(data)

        return np.expm1(prediction)
    
    def preprocess_data(self,data):
        pass