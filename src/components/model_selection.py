import os
import pandas as pandas
import numpy as np
import joblib
from logger import logger
from entity.config_entity import ModelSelectionConfig
from utils.common import rmse_metrics
from pathlib import Path
from utils.common import save_json

class ModelSelection:
    def __init__(self,config: ModelSelectionConfig):
        self.config = config
    
    def selection_process(self):
        try:
            self.X_val = joblib.load(self.config.val_data_X_path)
            self.y_val = joblib.load(self.config.val_data_y_path)
        except FileNotFoundError as e:
            logger.error("Error loading data: {}".format(e))
        
        try:
            self.xgboost_model = joblib.load(self.config.best_xgboost_model)
            self.lightgbm_model = joblib.load(self.config.best_lightgbm_model)
            self.linear_model = joblib.load(self.config.best_lightgbm_model)
        except FileNotFoundError as e:
            logger.error("Model could not be loaded: {}".format(e))

        xgboost_prediction = self.xgboost_model.predict(self.X_val)
        lightgbm_prediction = self.lightgbm_model.predict(self.X_val)
        linear_prediction = self.linear_model.predict(self.X_val)

        xgboost_rmse = rmse_metrics(self.y_val,xgboost_prediction)
        lightgbm_rmse = rmse_metrics(self.y_val,lightgbm_prediction)
        linear_rmse = rmse_metrics(self.y_val,linear_prediction)

        scores = {"xgboost_rmse": xgboost_rmse,"lightgbm_rmse": lightgbm_rmse,"linear_rmse": linear_rmse}
        save_json(path=Path(self.config.val_metric_file_name),data=scores)

        if ((xgboost_rmse < lightgbm_rmse) & (xgboost_rmse < linear_rmse)):
            joblib.dump(self.xgboost_model,os.path.join(self.config.root_dir,self.config.best_model))
        elif lightgbm_rmse < linear_rmse:
            joblib.dump(self.lightgbm_model,os.path.join(self.config.root_dir,self.config.best_model))
        else:
            joblib.dump(self.linear_model,os.path.join(self.config.root_dir,self.config.best_model))


        
