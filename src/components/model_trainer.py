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
from lightgbm import LGBMRegressor
from sklearn.metrics import make_scorer


class ModelTrainer:
    def __init__(self,config: DataTrainerConfig):
        self.config = config
        self.best_params = None
        self.best_model = None
        self.best_rmse = np.inf
    
    def objective(self,trial):
        try:
            self.X_train = joblib.load(self.config.data_X_train_path)
            self.y_train = joblib.load(self.config.data_y_train_path)
        except FileNotFoundError as e:
            logger.error("Error loading data: {}".format(e))

        regressor_name = trial.suggest_categorical('regressor',['XGBoost',"LightGBM"])
        if regressor_name == "XGBoost":
            params = {
                "booster": trial.suggest_categorical("booster",["gbtree","dart"]),
                "device" : "cuda",
                "lambda": trial.suggest_float("lambda", 0.05,1.0),
                "alpha": trial.suggest_float("alpha",0.05,1.0),
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": trial.suggest_int("max_depth",3,10),
                "n_estimators": trial.suggest_int("n_estimators",100,500),
                "subsample": trial.suggest_float("subsample",0.5,1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree",0.5,1.0),
                "gamma": trial.suggest_int("gamma",0,60),
                "min_child_weight": trial.suggest_float("min_child_weight",1,10)
            }
            model = XGBRegressor(**params)
        else:
            params = {
                "num_leaves": trial.suggest_int("num_leaves",30,127),
                "max_depth": trial.suggest_int("max_depth",3,10),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf",5,15),
                "bagging_freq": trial.suggest_int("bagging_freq",4,8),
                "bagging_fraction": trial.suggest_float("bagging_fraction",0.6,0.9),
                "max_bin": trial.suggest_int("max_bin",150,250),
                "lambda_l1": trial.suggest_float("lambda_l1",0.1,0.3),
                "lambda_l2": trial.suggest_float("lambda_l2",0.1,0.3),
                "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf",1e-4,1e-2),
                "feature_fraction": trial.suggest_float("feature_fraction",0.6,0.9),
                "learning_rate": trial.suggest_float("learning_rate",0.05,0.2),
                "num_iterations": trial.suggest_int("num_iterations",70,130)
            }
            model = LGBMRegressor(**params)
        def rmse_function(y_true, y_pred):
            return -root_mean_squared_error(y_true, y_pred)
        rmse_scorer = make_scorer(rmse_function, greater_is_better=False)
        scores = cross_val_score(model,self.X_train,self.y_train,n_jobs=-1,cv=5,scoring=rmse_scorer)
        rmse = np.mean(scores)
        if rmse < self.best_rmse:
            self.best_rmse = rmse
            self.best_params = params
            self.best_model = model
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metric('rmse', rmse)

        return rmse
    def optimize_xgboost(self):
        mlflow.set_tracking_uri('http://localhost:5000')          
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective,n_trials=1000)
        with mlflow.start_run(run_name = "best_model"):
            best_model_trained = self.best_model.fit(self.X_train,self.y_train)
            mlflow.sklearn.log_model(best_model_trained, "best_model")
            mlflow.log_params(self.best_params)
            mlflow.log_metric('rmse', self.best_rmse)
            
        joblib.dump(best_model_trained,self.config.best_model)
        joblib.dump(self.best_rmse, self.config.best_model_rmse)
        joblib.dump(self.best_params, self.config.best_model_params)




        


        
