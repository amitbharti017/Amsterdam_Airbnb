import pandas as pd
import os
from logger import logger
import joblib
import mlflow
import optuna
import numpy as np
from entity.config_entity import DataTrainerConfig
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from optuna.integration.mlflow import MLflowCallback


class ModelTrainer:
    def __init__(self,config: DataTrainerConfig):
        self.config = config
    
    def objective(self,trial):
        X_train = joblib.load(self.config.data_X_train_path)
        y_train = joblib.load(self.config.data_y_train_path)

        params = {
            "booster": trial.suggest_categorical("booster",["gbtree","dart"]),
            # "device" : "cuda",
            "lambda": trial.suggest_float("lambda", 0.05,1.0),
            "alpha": trial.suggest_float("alpha",0.05,1.0),
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": trial.suggest_int("max_depth",3,14),
            "n_estimators": trial.suggest_int("n_estimators",100,1000),
            "subsample": trial.suggest_float("subsample",0.5,1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree",0.5,1.0),
            "gamma": trial.suggest_int("gamma",0,60),
            "min_child_weight": trial.suggest_float("min_child_weight",1,10)
        }
        # params = {
        #     'device': "cuda",
        #     'tree_method': 'hist',
        #     'objective':'reg:squarederror',
        #     'learning_rate': trial.suggest_categorical('learning_rate', [0.008, 0.01, 0.02, 0.05]),
        #     'random_state': 7,
        #     'max_depth': trial.suggest_int('max_depth', 3, 10), #change
        #     'min_child_weight': trial.suggest_int('min_child_weight', 1, 5), #change
        #     'lambda': trial.suggest_float('lambda', 1e-4, 10.0, log=True),
        #     'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
        # }

        model = XGBRegressor(**params)

        scores = cross_val_score(model,X_train,y_train,n_jobs=-1,cv=5,scoring="neg_root_mean_squared_error")
        rmse = -np.mean(scores)
        mlflow.log_params(params)
        mlflow.log_metric("rmse",rmse)
        mlflow.end_run()
    
    def optimize_xgboost(self):
        mlflow.set_tracking_uri('http://localhost:5000')
        mlflc = MLflowCallback(metric_name="rmse",create_experiment=False)
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective,n_trials=10,callbacks=[mlflc])
        # print('Number of finished trials:', len(study.trials))
        # print('Best trial:')
        # trial = study.best_trial
        # print('Value: {:.3f}'.format(trial.value))
        # print('Params: ')
        # for key, value in trial.params.items():
        #     print(f'{}: {}'.format(key, value))

        


        
