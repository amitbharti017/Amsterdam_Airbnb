import os
import pandas as pd
import numpy as np
import joblib
from logger import logger
from entity.config_entity import ModelEvaluationConfig
from utils.common import rmse_metrics
from pathlib import Path
from utils.common import save_json

class ModelEvaluation:
    def __init__(self,config: ModelEvaluationConfig):
        self.config = config
    
    def save_result(self):
        try:
            self.X_test = joblib.load(self.config.test_data_X_path)
            self.y_test = joblib.load(self.config.test_data_y_path)
        except FileNotFoundError as e:
            logger.error("Error loading data: {}".format(e))
        
        try:
            self.model = joblib.load(self.config.model_path)
        except FileNotFoundError as e:
            logger.error("Model could not be loaded: {}".format(e))

        
        prediction = self.model.predict(self.X_test)
        rmse = rmse_metrics(self.y_test,prediction)

        scores = {"rmse" : rmse}
        save_json(path = Path(self.config.metric_file_name),data=scores)
