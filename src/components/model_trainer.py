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
from utils.common import get_mlflow_experiment
from utils.common import create_mlflow_experiment


class ModelTrainer:
    def __init__(self,config: DataTrainerConfig):
        self.config = config
    
    def objective(self,trial):
        X_train = joblib.load(self.config.data_X_train_path)
        y_train = joblib.load(self.config.data_y_train_path)

        regressor_name = trail.suggest_categorical('regressor',['XGBoost',"LightGBM"])
        if regressor_name == "XGBoost":
            params_xgboost = {
                "booster": trial.suggest_categorical("booster",["gbtree","dart"]),
                "device" : "cuda",
                "lambda": trial.suggest_float("lambda", 0.05,1.0),
                "alpha": trial.suggest_float("alpha",0.05,1.0),
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": trial.suggest_int("max_depth",3,10),
                "n_estimators": trial.suggest_int("n_estimators",100,1000),
                "subsample": trial.suggest_float("subsample",0.5,1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree",0.5,1.0),
                "gamma": trial.suggest_int("gamma",0,60),
                "min_child_weight": trial.suggest_float("min_child_weight",1,10)
            }
            model = XGBRegressor(**params_xgboost)
        else:
            params_lightgbm = {
                "num_leaves": trial.suggest_int("num_leaves",30,127),
                "max_depth": trial.suggest_int("max_depth",3,10),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf",5,15),
                "bagging_freq": trial.suggest_int("bagging_freq",4,8),
                "bagging_fraction": trial.suggest_float("bagging_fraction",0.6,0.9),
                "max_bin": trial.suggest_int("max_bin",150,250),
                "lambda_l1": trial.suggest.float("lambda_l1",0.1,0.3),
                "lambda_l2": trial.suggest.float("lambda_l2",0.1,0.3),
                "min_sum_hessian_in_leaf": trial.suggest.float("min_sum_hessian_in_leaf",1e-4,1e-2),
                "feature_fraction": trial.suggest.float("feature_fraction",0.6,0.9),
                "learning_rate": trial.suggest.float("learning_rate",0.05,0.2),
                "num_iterations": trial.suggest.int("num_iterations",70,130),
                "early_stopping_rounds": trial.suggest.int("early_stopping_rounds",30,50)
            }
            model = LightGBM(**params_lightgbm)
        scores = cross_val_score(model,X_train,y_train,n_jobs=-1,cv=5,scoring="neg_root_mean_squared_error")
        rmse = -np.mean(scores)
        # with mlflow.start_run():
        #     mlflow.log_params(params)
        #     mlflow.log_metric("rmse",rmse)
    
    # def optimize_xgboost(self):
    #     experiment_id = create_mlflow_experiment(experiment_name="XGBoost_trail",
    #                                              artifact_location = os.path.join(os.getcwd(), "xgboost_mlflow_artifacts"),
    #                                              tags={"env":"dev","version":"0.0.1"})
    #     mlflow.set_tracking_uri('http://localhost:5000')
    #     with mlflow.start_run(experiment_id=experiment_id):
        
    #         # mlflc = MLflowCallback(metric_name="rmse")
    #         study = optuna.create_study(direction="minimize")
    #     #     study.optimize(lambda trial: self.objective(trial, experiment_id),n_trials=10,callbacks=[mlflc])
        # print('Number of finished trials:', len(study.trials))
        # print('Best trial:')
        # trial = study.best_trial
        # print('Value: {:.3f}'.format(trial.value))
        # print('Params: ')
        # for key, value in trial.params.items():
        #     print(f'{}: {}'.format(key, value))
    def optimize_xgboost(self):
        mlflow.set_tracking_uri('http://localhost:5000')          
        study = optuna.create_study(direction="minimize")
        mlflc = MLflowCallback(metric_name="rmse")
        study.optimize(self.objective,n_trials=10,callbacks=[mlflc])



        


        
